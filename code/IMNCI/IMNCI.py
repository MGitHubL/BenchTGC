import sys
import math
import torch
import ctypes
import datetime
import numpy as np

from collections import Counter
from torch.autograd import Variable
from dataset import MNCIDataSet
from sklearn.cluster import KMeans
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from evaluation import eva
import torch.nn.functional as F

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class MNCI:
    def __init__(self, directed=False):
        self.network = 'school'
        k_dict = {'arxivAI': 5, 'arxivCS': 40, 'arxivPhy': 53, 'arxivMath': 31, 'arxivLarge': 172, 'school': 9,
                  'dblp': 10, 'brain': 10, 'patent': 6, 'yelp': 5, 'tmall': 10, 'ml1m': 5, 'amms': 5,
                  'bitotc': 21, 'meta': 5, 'patent_noisy': 6}
        self.clusters = k_dict[self.network]
        self.file_path = '../../data/%s/%s.txt' % (self.network, self.network)
        self.emb_path = '../../emb/%s/%s_IMNCI.emb'
        self.label_path = '../../data/%s/label.txt' % (self.network)
        self.labels = self.read_label()
        self.feature_path = '../pretrain/%s_feature.emb' % self.network

        self.emb_size = 128
        self.neg_size = 10
        self.hist_len = 128  # This value is to store all neighbors, the extra positions are invalid in the calculation
        self.lr = 0.001
        self.batch = 16
        self.epochs = 1
        self.save_step = 1
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0
        self.best_epoch = 0

        self.data = MNCIDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, self.emb_size, directed)

        self.node_dim = self.data.get_node_dim()
        self.first_time = self.data.get_first_time()
        self.feature = self.data.get_feature()

        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(),
                                 requires_grad=True)
        self.final_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(),
                                 requires_grad=False)
        self.com_emb = Variable(MNCI.position_encoding_(self, self.clusters,
                                                        self.emb_size).type(FType).cuda(), requires_grad=False)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)

        self.delta_co = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
        self.delta_co = MNCI.truncated_normal_(self.delta_co, tensor=(self.delta_co))
        self.delta_ne = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
        self.delta_ne = MNCI.truncated_normal_(self.delta_ne, tensor=(self.delta_ne))

        self.w_node = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType).cuda(),
                               requires_grad=True)
        self.w_node = MNCI.truncated_normal_(self.w_node, tensor=(self.w_node))
        self.w_neighbor = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType).cuda(),
                                   requires_grad=True)
        self.w_neighbor = MNCI.truncated_normal_(self.w_neighbor, tensor=(self.w_neighbor))
        self.w_community = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType).cuda(),
                                    requires_grad=True)
        self.w_community = MNCI.truncated_normal_(self.w_community, tensor=(self.w_community))
        self.b = Variable((torch.zeros(4, self.node_dim, self.emb_size) + 1.).type(FType).cuda(),
                          requires_grad=True)
        self.b = MNCI.truncated_normal_(self.b, tensor=(self.b))

        self.time_omega = Variable((torch.zeros(self.emb_size // 2) + 1.).type(FType).cuda(), requires_grad=True)
        self.time_omega = MNCI.truncated_normal_(self.time_omega, tensor=(self.time_omega))

        self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(),
                                      requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        kmeans = KMeans(n_clusters=self.clusters, n_init=20)
        _ = kmeans.fit_predict(self.feature)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).type(FType).cuda()
        self.pre_cluster = self.cluster_layer.data.clone()
        self.v = 1.0

        self.opt = torch.optim.Adam(lr=self.lr, params=[self.node_emb, self.delta_co, self.delta_ne, self.w_node,
                                                        self.w_neighbor, self.w_community, self.b, self.com_emb,
                                                        self.time_omega, self.cluster_layer])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.98)

        self.loss = torch.FloatTensor()

    def read_label(self):
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                label = int(line)
                labels.append(label)
        return labels

    def position_encoding_(self, max_len, emb_size):
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)  # [node_number, 1]
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def the_GRU(self, s_nodes, node_emb, neighborhood_inf, community_inf):
        w_node = self.w_node.data  # [4, batch, batch]
        w_neighbor = self.w_neighbor.data
        w_community = self.w_community.data
        b = self.b.index_select(1, Variable(s_nodes.view(-1)))

        U_G = torch.sigmoid(torch.mm(node_emb, w_node[0]) + torch.mm(neighborhood_inf, w_neighbor[0]) +
                            torch.mm(community_inf, w_community[0]) + b[0])  # [batch, d]
        NR_G = torch.sigmoid(torch.mm(node_emb, w_node[1]) + torch.mm(neighborhood_inf, w_neighbor[1]) +
                             torch.mm(community_inf, w_community[1]) + b[1])
        CR_G = torch.sigmoid(torch.mm(node_emb, w_node[2]) + torch.mm(neighborhood_inf, w_neighbor[2]) +
                             torch.mm(community_inf, w_community[2]) + b[2])
        tem_node_emb = torch.tanh(torch.mm(node_emb, w_node[3]) +
                                  torch.mul(NR_G, torch.mm(neighborhood_inf, w_neighbor[3])) +
                                  torch.mul(CR_G, torch.mm(community_inf, w_community[3]) + b[3]))
        new_node_emb = torch.mul((1 - U_G), node_emb) + torch.mul(U_G, tem_node_emb)

        return new_node_emb

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()

        return p

    def dis_fun(self, x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    def no_diag(self, x, n):
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)  # [batch, d]
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        community_emb = self.com_emb.data  # [K, d]
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        h_time_mask = Variable(h_time_mask)  # [batch, hist_len]
        h_node_emb = torch.mul(h_node_emb, h_time_mask.unsqueeze(-1))
        # 'h_time_mask' mains that if there is a invalid neighbor in the sequence
        # we need to ensure that it does not play a role in the calculation

        delta_co = self.delta_co.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)  # [batch, 1]
        delta_ne = self.delta_ne.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        time_omega = self.time_omega.unsqueeze(0)  # [1, d/2]

        # neighborhood influence
        affinity_per_neighbor = torch.sigmoid(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(
            dim=-1, keepdim=False).neg()) * h_time_mask  # [batch, hist_len]
        affinity_sum = affinity_per_neighbor.sum(dim=-1, keepdim=True) + 1e-6
        affinity_weight = affinity_per_neighbor / affinity_sum  # [batch, hist_len]

        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # [batch, hist_len]
        time_emb = Variable(torch.zeros(batch, self.hist_len, self.emb_size).type(FType).cuda())  # [b,l,d]
        middle_value = torch.mul(time_omega.unsqueeze(0), d_time.unsqueeze(-1))  # [batch, hist_len, d/2]
        time_emb[:, :, 0::2] = torch.sin(middle_value)
        time_emb[:, :, 1::2] = torch.cos(middle_value)

        # all of these above parameters' shape are [batch, hist_len]
        neighborhood_param = torch.mul((affinity_weight * h_time_mask).unsqueeze(-1), time_emb)
        neighborhood_inf = torch.mul(delta_ne, torch.mul(neighborhood_param,
                                                         h_node_emb).sum(dim=1, keepdim=False))  # [batch, d]

        # community influence
        weight_per_community = torch.sigmoid(((community_emb.unsqueeze(1) - s_node_emb.unsqueeze(0))
                                              ** 2).sum(dim=-1, keepdim=False).neg())  # [K, batch]
        weight_sum = weight_per_community.sum(dim=0, keepdim=True)
        community_weight = weight_per_community / (weight_sum + 1e-6)  # [K, batch]
        community_inf = torch.mul(
            delta_co, torch.mul(community_weight.unsqueeze(-1),
                                community_emb.unsqueeze(1)).sum(dim=0, keepdim=False))  # [batch, d]

        # aggregate and lambda
        s_new_emb = self.the_GRU(s_nodes, s_node_emb, neighborhood_inf, community_inf)  # [batch, emb_size]

        p_lambda = ((s_new_emb - t_node_emb) ** 2).sum(dim=1).neg()  # [batch]
        n_lambda = ((s_new_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()  # [batch, neg]

        # update community emb
        emb_diff = s_new_emb - s_node_emb  # [batch, emb]
        weight_index = torch.max(community_weight, dim=0, keepdim=False)[1]  # [batch]
        for i in range(batch):
            community_emb[weight_index[i]].data += emb_diff[i].data.clone()

        self.final_emb[s_nodes].data = s_new_emb.data.clone()
        self.com_emb.data = community_emb.data.clone()

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)  # [b]

        l_x = torch.norm(s_node_emb - s_pre_emb, p=2) + 1e-6  # []

        s_p = self.target_dis(s_pre_emb)
        l_d = self.kl_loss(s_node_emb, s_p)

        l_c_node = -torch.log(((s_node_emb - s_node_emb) ** 2).sum(dim=1).neg().sigmoid() + 1e-6) / (-torch.log(
            ((s_node_emb - s_node_emb) ** 2).sum(dim=1).neg().sigmoid() + 1e-6) - torch.log(
            ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg().neg().sigmoid() + 1e-6).sum(dim=1))
        l_c_node = l_c_node.mean()
        l_c_cluster = -torch.log(torch.sum(torch.exp(
            torch.cosine_similarity(self.cluster_layer, self.cluster_layer)) / torch.sum(torch.exp(
            torch.cosine_similarity(self.cluster_layer.unsqueeze(1), self.cluster_layer.unsqueeze(0), dim=-1)), dim=1,
            keepdim=False)) / self.clusters + 1e-6)
        l_c = l_c_node + l_c_cluster

        center_distance = self.dis_fun(self.cluster_layer, self.cluster_layer)
        l_s_dilation = self.no_diag(center_distance, self.cluster_layer.shape[0])
        l_s_shrink = self.dis_fun(s_node_emb, self.cluster_layer.float())
        l_s_c = torch.sqrt(torch.sum(torch.pow(self.cluster_layer, 2), dim=1))
        l_s = torch.log(l_s_dilation.mean()) + l_s_shrink.mean() + l_s_c.sum()

        l_b = torch.norm(self.cluster_layer - self.pre_cluster, p=2) + 1e-6
        l_framework = l_d

        total_loss = loss.sum() + l_framework

        self.pre_cluster.data = self.cluster_layer.data.clone()

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()

        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        print('Training......')
        for epoch in range(self.epochs):
            once_start = datetime.datetime.now()
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=4)

            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings(self.emb_path % (self.network, self.network))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())

                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            once_end = datetime.datetime.now()
            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' + str(self.loss.cpu().numpy() / len(self.data))
                             + '\tonce_runtime: ' + str(once_end - once_start) + '\n')
            sys.stdout.write('ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc, nmi, ari, f1))
            sys.stdout.flush()

            self.scheduler.step()

        print('Best performance in %d epoch: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_epoch, self.best_acc, self.best_nmi, self.best_ari, self.best_f1))

        self.save_node_embeddings(self.emb_path % (self.network, self.network))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()


if __name__ == '__main__':
    start = datetime.datetime.now()
    MNCI = MNCI()
    MNCI.train()
    end = datetime.datetime.now()
    print('total runtime: %s' % str(end - start))
