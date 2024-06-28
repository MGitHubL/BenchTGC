import torch
from torch import nn, optim
from dgnn import DGNN
from film import Scale_4, Shift_4
from Emlp import EMLP
from node_relu import Node_edge
from global_prior import Global_emb, Global_w
from loss_w import Loss_weight
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.autograd import Variable

FType = torch.FloatTensor
LType = torch.LongTensor

class Model(nn.Module):
    def __init__(self, args, pre_feature):
        super(Model, self).__init__()
        self.args = args
        self.l2reg = args.l2_reg  # 0.001
        self.ncoef = args.ncoef  # 0.01
        self.EMLP = EMLP(args)  # [1,d],[1]
        # self.grow_f = E_increase(args.edge_grow_input_dim)
        self.gnn = DGNN(args)  # Dynamic Graph Neural Network
        self.scale_e = Scale_4(args)
        self.shift_e = Shift_4(args)
        self.node_edge = Node_edge(args)
        self.global_w = Global_w(args)
        self.global_emb = Global_emb(args)
        self.loss_w = Loss_weight(args)

        self.v = 1.0
        self.clusters = args.clusters
        self.pre_emb = Variable(torch.from_numpy(pre_feature).type(FType).cuda(), requires_grad=False)

        self.cluster_layer = (torch.zeros(args.clusters, args.feat_dim) + 1.).cuda()
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        kmeans = KMeans(n_clusters=args.clusters, n_init=20)
        _ = kmeans.fit_predict(pre_feature)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
        self.pre_cluster = self.cluster_layer.data.clone()

        # self.g_optim = optim.Adam(self.grow_f.parameters(), lr=args.lr)

        self.optim = optim.Adam([{'params': self.gnn.parameters()},
                                 {'params': self.EMLP.parameters()},
                                 {'params': self.scale_e.parameters()},
                                 {'params': self.shift_e.parameters()},
                                 {'params': self.node_edge.parameters()},
                                 {'params': self.global_w.parameters()},
                                 {'params': self.global_emb.parameters()},
                                 {'params': self.loss_w.parameters()},
                                 {'params': self.cluster_layer},
                                 ], lr=args.lr)


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

    def forward(self, s_node, t_node, s_self_feat, s_one_hop_feat,
                t_self_feat, t_one_hop_feat,
                neg_self_feat, neg_one_hop_feat,
                e_time, s_his_time,
                t_his_time,
                neg_his_time, s_edge_rate, t_edge_rate,
                training=True):
        batch = s_node.size()[0]
        s_gnn = self.gnn(s_self_feat, s_one_hop_feat,
                         e_time, s_his_time)  # [b,d]
        t_gnn = self.gnn(t_self_feat, t_one_hop_feat,
                         e_time, t_his_time)  # [b,d]
        neg_gnn = self.gnn(neg_self_feat, neg_one_hop_feat,
                            e_time, neg_his_time, neg=True)  # [b,1,d]
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_node.view(-1))).view(batch, -1)

        s_global_update, s_node_update = self.global_w(s_edge_rate)
        t_global_update, t_node_update = self.global_w(t_edge_rate)

        global_emb = self.global_emb(s_gnn, t_gnn, s_global_update, t_global_update)  # [1,d]

        s_emb = s_gnn + s_node_update.unsqueeze(-1) * global_emb
        t_emb = t_gnn + t_node_update.unsqueeze(-1) * global_emb
        # s_intensity = s_emb + s_node_update.unsqueeze(-1) * global_emb
        # t_intensity = t_emb + t_node_update.unsqueeze(-1) * global_emb
        # neg_embs = neg_gnn + (s_node_update.unsqueeze(-1) * global_emb).unsqueeze(1)
        neg_embs = neg_gnn + (((s_node_update + t_node_update) / 2).unsqueeze(-1) * global_emb).unsqueeze(1)

        ij_cat = torch.cat((s_emb, t_emb), dim=1)  # [128,32]
        alpha_ij = self.scale_e(ij_cat)
        beta_ij = self.shift_e(ij_cat)
        theta_e_new = []
        for s in range(2):
            theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (alpha_ij[s] + 1)) + beta_ij[s])

        p_dif = (s_emb - t_emb).pow(2)
        p_scalar = (p_dif * theta_e_new[0]).sum(dim=1, keepdim=True)
        p_scalar += theta_e_new[1]
        p_scalar_list = p_scalar
        # 公式5，注意到theta_e_new有两个位置的存放值，分别为W和b

        event_intensity = torch.sigmoid(p_scalar_list) + 1e-6  # [b,1]
        log_event_intensity = torch.mean(-torch.log(event_intensity))  # [1]

        dup_s_emb = s_emb.repeat(1, 1, self.args.neg_size)
        dup_s_emb = dup_s_emb.reshape(s_emb.size(0), self.args.neg_size, s_emb.size(1))

        neg_ij_cat = torch.cat((dup_s_emb, neg_embs), dim=2)
        neg_alpha_ij = self.scale_e(neg_ij_cat)
        neg_beta_ij = self.shift_e(neg_ij_cat)
        neg_theta_e_new = []
        for s in range(2):
            neg_theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (neg_alpha_ij[s] + 1)) + neg_beta_ij[s])

        neg_dif = (dup_s_emb - neg_embs).pow(2)
        neg_scalar = (neg_dif * neg_theta_e_new[0]).sum(dim=2, keepdim=True)
        neg_scalar += neg_theta_e_new[1]
        big_neg_scalar_list = neg_scalar

        neg_event_intensity = torch.sigmoid(- big_neg_scalar_list) + 1e-6

        neg_mean_intensity = torch.mean(-torch.log(neg_event_intensity))

        pos_l2_loss = [torch.norm(s, dim=1) for s in alpha_ij]
        pos_l2_loss = [torch.mean(s) for s in pos_l2_loss]
        pos_l2_loss = torch.sum(torch.stack(pos_l2_loss))
        pos_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=1)) for s in beta_ij]))
        neg_l2_loss = torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_alpha_ij]))
        neg_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_beta_ij]))

        l_theta = pos_l2_loss + neg_l2_loss

        h_intensity = self.gnn.hawkes(s_self_feat, t_self_feat, s_one_hop_feat, t_one_hop_feat, e_time, s_his_time, t_his_time)
        smooth_loss = nn.SmoothL1Loss()
        l_contra = smooth_loss(p_scalar_list, h_intensity)  # L_A

        l_emb = torch.mean(-torch.log(torch.sigmoid((s_emb - global_emb).pow(2)) + 1e-6)) +\
                torch.mean(-torch.log(torch.sigmoid((t_emb - global_emb).pow(2)) + 1e-6))  # L_G
        l_hawkes = torch.mean(-torch.log(torch.sigmoid(h_intensity) + 1e-6))

        loss = self.loss_w(l_contra, l_theta, l_hawkes, l_emb)
        L_model = log_event_intensity + neg_mean_intensity + loss

        l_x = torch.norm(s_emb - s_pre_emb, p=2) + 1e-6  # []

        new_st_adj = torch.cosine_similarity(s_emb, t_emb)  # [b]
        res_st_loss = torch.norm(1 - new_st_adj, p=1, dim=0)  # []
        new_sn_adj = torch.cosine_similarity(s_emb.unsqueeze(1), neg_embs, dim=2)  # [b,n]
        res_sn_loss = torch.norm(0 - new_sn_adj, p=1, dim=0).sum(dim=0, keepdims=False)  # []
        l_a = res_st_loss + res_sn_loss

        s_p = self.target_dis(s_pre_emb)
        l_d = self.kl_loss(s_emb, s_p)

        l_c_node = -torch.log(((s_emb - s_emb) ** 2).sum(dim=1).neg().sigmoid() + 1e-6) / (-torch.log(
            ((s_emb - s_emb) ** 2).sum(dim=1).neg().sigmoid() + 1e-6) - torch.log(
            ((s_emb.unsqueeze(1) - neg_embs) ** 2).sum(dim=2).neg().neg().sigmoid() + 1e-6).sum(dim=1))
        l_c_node = l_c_node.mean()
        l_c_cluster = -torch.log(torch.sum(torch.exp(
            torch.cosine_similarity(self.cluster_layer, self.cluster_layer)) / torch.sum(torch.exp(
            torch.cosine_similarity(self.cluster_layer.unsqueeze(1), self.cluster_layer.unsqueeze(0), dim=-1)), dim=1,
            keepdim=False)) / self.clusters + 1e-6)
        l_c = l_c_node + l_c_cluster

        center_distance = self.dis_fun(self.cluster_layer, self.cluster_layer)
        l_s_dilation = self.no_diag(center_distance, self.cluster_layer.shape[0])
        l_s_shrink = self.dis_fun(s_emb, self.cluster_layer.float())
        l_s_c = torch.sqrt(torch.sum(torch.pow(self.cluster_layer, 2), dim=1))
        l_s = torch.log(l_s_dilation.mean()) + l_s_shrink.mean() + l_s_c.sum()

        l_b = torch.norm(self.cluster_layer - self.pre_cluster, p=2) + 1e-6

        L_module = l_d + l_b + l_x
        L = L_model + L_module

        self.pre_cluster.data = self.cluster_layer.data.clone()

        if training == True:
            self.optim.zero_grad()
            L.backward()
            self.optim.step()

        return round((L.detach().clone()).cpu().item(), 4),\
               s_emb.detach().clone().cpu(),\
               t_emb.detach().clone().cpu(),\
               dup_s_emb.detach().clone().cpu(),\
               neg_embs.detach().clone().cpu(),\
               s_node.detach().clone().cpu(),\
               t_node.detach().clone().cpu()