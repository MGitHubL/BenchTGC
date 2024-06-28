import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from evaluation import evaluation
import matplotlib.pyplot as plt


def test_external_score(emb_path, label_path, k):
    n2l = dict()
    with open(label_path, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            n2l[n_id] = l_id

    node_emb = dict()
    with open(emb_path, 'r') as reader:
        reader.readline()
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_id = int(embeds[0])
            if node_id in n2l:
                node_emb[node_id] = embeds[1:]
    Y = []
    X = []
    n2l_list = sorted(n2l.items(), key=lambda x: x[0])
    for (the_id, label) in n2l_list:
        Y.append(label)
        X.append(node_emb[the_id])

    model = KMeans(n_clusters=k, n_init=20)
    cluster_id = model.fit_predict(X)
    center = model.cluster_centers_
    acc, nmi, ari, f1 = evaluation(Y, cluster_id)
    print('ACC: %f, NMI: %f, ARI: %f, F1: %f' % (acc, nmi, ari, f1))

def test_internal_score(emb_path, k):
    node_emb = []
    with open(emb_path, 'r') as r:
        r.readline()
        for line in r:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_emb.append(embeds[1:])

    model = KMeans(n_clusters=k, n_init=20)
    model.fit(node_emb)
    y_pred = model.predict(node_emb)
    sci = (silhouette_score(node_emb, y_pred))
    chi = (calinski_harabasz_score(node_emb, y_pred))
    dbi = (davies_bouldin_score(node_emb, y_pred))
    print('SCI: ' + str(sci))  # -1到1，越大越好
    print('CHI: ' + str(chi))  # 0到正无穷，越大越好
    print('DBI: ' + str(dbi))  # 0到正无穷，越小越好


if __name__ == '__main__':
    data = 'patent'
    model = 'HTNE'

    emb_path = '../emb/%s/%s_%s.emb' % (data, data, model)
    label_path = '../data/%s/node2label.txt' % (data)
    # k for in: []
    ex_dict = {'arxivAI': 5, 'arxivCS': 40, 'arxivPhy': 53, 'arxivMath': 31, 'arxivLarge': 172, 'school': 9, 'dblp': 10,
              'brain': 10, 'patent': 6, 'yelp': 5, 'tmall': 10, 'ml1m': 5, 'amms': 5, 'bitotc': 21, 'meta': 6, 'patent_noisy': 6}

    k = ex_dict[data]
    test_external_score(emb_path, label_path, k)
    # test_internal_score(emb_path, k)
