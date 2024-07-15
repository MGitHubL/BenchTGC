import sys

import os.path as osp
import time
import torch
import numpy as np
import datetime
import random
import math
import time
import argparse
from data_dyn_cite import DataHelper
from torch.utils.data import DataLoader
from model import Model
from cluster_evaluation import eva

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
FType = torch.FloatTensor
LType = torch.LongTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    begin = time.time()
    setup_seed(args.seed)
    Data = DataHelper(args, args.file_path, args.node_feature_path, args.neg_size, args.hist_len, args.directed,
                      tlp_flag=args.tlp_flag)
    node_dim = Data.get_node_dim()
    pre_feature = Data.get_pre_feature()
    node_emb = torch.zeros(node_dim + 1, args.out_dim)

    model = Model(args, pre_feature).to(device)
    model.train()

    label_path = '../../data/%s/label.txt' % (args.data)
    labels = []
    with open(label_path, 'r') as reader:
        for line in reader:
            label = int(line)
            labels.append(label)

    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_epoch = 0

    for j in range(args.epoch_num):
        epoch_start = datetime.datetime.now()
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        for i_batch, sample_batched in enumerate(loader):
            loss, s_emb, t_emb, dup_s_emb, neg_embs, s_node, t_node = model.forward(
                sample_batched['s_node'].type(LType).to(device),
                sample_batched['t_node'].type(LType).to(device),
                sample_batched['s_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['s_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),

                sample_batched['t_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['t_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),

                sample_batched['neg_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['neg_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),

                sample_batched['event_time'].type(FType).to(device),
                sample_batched['s_history_times'].type(FType).to(device),
                sample_batched['t_history_times'].type(FType).to(device),
                sample_batched['neg_his_times_list'].type(FType).to(device),
            )

            node_emb[t_node] = t_emb
            node_emb[s_node] = s_emb
            if j == 0:
                if i_batch % 10 == 0:
                    print('batch_{} event_loss:'.format(i_batch), loss)
        if args.data == 'arxivLarge' or args.data == 'arxivPhy' or args.data == 'arxivMath' or args.data == 'arxivCS' or args.data == 'arxivAI':
            acc, nmi, ari, f1 = 0, 0, 0, 0
        else:
            acc, nmi, ari, f1 = eva(args.clusters, labels, node_emb)
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = j
            save_node_embeddings(args, node_emb, node_dim, args.emb_path)

        print('ep_{}_event_loss:'.format(j + 1), loss)

        epoch_end = datetime.datetime.now()
        print('One Epoch Complete with Time: %s' % str(epoch_end - epoch_start))
       
    print('Best performance in %d epoch: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
          (best_epoch, best_acc, best_nmi, best_ari, best_f1))
    end = time.time()
    print('Train Total Time: ' + str(round((end - begin)/60, 2)) + ' mins')


def save_node_embeddings(args, emb, node_dim, path):
    embeddings = emb.cpu().data.numpy()
    writer = open(path, 'w')
    writer.write('%d %d\n' % (node_dim, args.out_dim))
    for n_idx in range(node_dim):
        writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

    writer.close()


if __name__ == '__main__':
    print(device)
    parser = argparse.ArgumentParser()

    data = 'arxivCS'
    k_dict = {'arxivAI': 5, 'arxivCS': 40, 'arxivPhy': 53, 'arxivMath': 31, 'arxivLarge': 172, 'school': 9,
              'dblp': 10, 'brain': 10, 'patent': 6, 'yelp': 5, 'tmall': 10, 'ml1m': 5, 'amms': 5,
              'bitotc': 21, 'meta': 5}
    file_path = '../../data/%s/%s.txt' % (data, data)
    node_feature_path = '../../data/%s/feature.txt' % data
    emb_path = '../../emb/%s/%s_ITREND.emb' % (data, data)
    pre_feature_path = '../pretrain/%s_feature.emb' % data

    parser.add_argument('--data', type=str, default=data)
    parser.add_argument('--emb_path', type=str, default=emb_path)
    parser.add_argument('--file_path', type=str, default=file_path)
    parser.add_argument('--node_feature_path', type=str, default=node_feature_path)
    parser.add_argument('--pre_feature_path', type=str, default=pre_feature_path)
    parser.add_argument('--clusters', type=int, default=k_dict[data])
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--neg_size', type=int, default=1)
    parser.add_argument('--hist_len', type=int, default=10)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--epoch_num', type=int, default=20, help='epoch number')
    parser.add_argument('--tlp_flag', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ncoef', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    args = parser.parse_args()

    main(args)
