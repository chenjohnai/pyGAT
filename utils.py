import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def train_doc2vec_feature(corpuss):
    data = []
    for corpus in corpuss:
        contents = corpus.split()
        data.append(TaggedDocument(words=contents[1:], tags=contents[0]))

    model = Doc2Vec(size=128,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab()
    for epoch in range(20):
        print('iteration {0}'.format(epoch))
        model.train(data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")
    return model

def load_dblp_data(path="./data/", dataset="dblp_4a/"):
    authors = open("{}{}id_author.txt".format(path, dataset), encoding="latin1").readlines()
    confs = open("{}{}id_conf.txt".format(path, dataset), encoding="latin1").readlines()
    G = nx.Graph()
    for author in authors:
        G.add_node("a{}".format(author.rstrip().split()[0]), type="author")
    a_idx = len(G.nodes())
    for conf in confs:
        G.add_node("c{}".format(conf.rstrip().split()[0]), type="conference")
    c_idx = len(G.nodes())

    maska = np.zeros([len(G.nodes()), ])
    maska[:a_idx] = 1
    maskc = np.zeros([len(G.nodes()), ])
    maskc[a_idx:c_idx] = 1

    row, column, values = [], [], []
    with open("{}{}AT.txt".format(path, dataset), encoding="latin1") as atfile:
        for line in atfile:
            i, j, value = line.split()
            row.append(int() - 1)
            column.append(int(j) - 1)
            values.append(int(value))
    author_text = sp.coo_matrix((values, (row, column)), shape=(max(row) + 1, max(column) + 1))

    row, column, values = [], [], []
    with open("./data/dblp_4a_p/CT.txt", encoding="latin1") as ctfile:
        for line in ctfile:
            i, j, value = line.split()
            row.append(int() - 1)
            column.append(int(j) - 1)
            values.append(int(value))
    conf_text = sp.coo_matrix((values, (row, column)), shape=(max(row) + 1, max(column) + 1))

    return G, maska, maskc, author_text, conf_text



def load_het_data(path="./data/", dataset="dbis/"):
    G = nx.Graph()
    print('Loading {} dataset...'.format(dataset))
    authors = open("{}{}id_author.txt".format(path, dataset), encoding="latin1").readlines()
    confs = open("{}{}id_conf.txt".format(path, dataset), encoding="latin1").readlines()
    papers = open("{}{}paper.txt".format(path, dataset), encoding="latin1").readlines()

    paper_d2v = train_doc2vec_feature(papers)

    for author in authors:
        G.add_node("a{}".format(author.rstrip().split()[0]), type="author")
    a_idx = len(G.nodes())
    for conf in confs:
        G.add_node("c{}".format(conf.rstrip().split()[0]), type="conference")
    c_idx = len(G.nodes())
    for paper in papers:
        G.add_node("p{}".format(paper.rstrip().split()[0]), type="paper")
    p_idx = len(G.nodes())
    paper_author = open("{}{}paper_author.txt".format(path, dataset), encoding="latin1").readlines()
    paper_conf = open("{}{}paper_conf.txt".format(path, dataset), encoding="latin1").readlines()
    maska = np.zeros([len(G.nodes()), ])
    maska[:a_idx] = 1
    maskc = np.zeros([len(G.nodes()), ])
    maskc[a_idx:c_idx] = 1
    maskp = np.zeros([len(G.nodes()), ])
    maskp[c_idx:p_idx] = 1
    for pa in paper_author:
        p_id, a_id = pa.rstrip().split()
        G.add_edge("p{}".format(p_id), "a{}".format(a_id))
    for pc in paper_conf:
        p_id, c_id = pc.rstrip().split()
        G.add_edge("p{}".format(p_id), "c{}".format(c_id))


    return G, maska, maskc, maskp, paper_d2v

    # author2id = {author_id: idx for idx, author_id in enumerate(author)}
    # conf2id = {conf_id: idx for idx, conf_id in enumerate(conf)}
    # paper2id = {paper_id: idx for idx, paper_id in enumerate(paper)}





def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

