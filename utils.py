from __future__ import print_function
import sys
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from collections import defaultdict

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_pubmed(path='data/'):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = []#np.empty((num_nodes, 1), dtype=np.int64)
    idx_map = {}
    idx=[]
    with open(path+"Pubmed-Diabetes/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            idx.append(i)
            info = line.split("\t")
            idx_map[int(info[0])] = i
            labels.append( int(info[1].split("=")[1])-1)
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    adj_array=[]
    with open(path+"Pubmed-Diabetes/Pubmed-Diabetes.Node.ids",'w') as fid:
        for each in idx_map:
            fid.write(str(each)+'\n')
    with open(path+"Pubmed-Diabetes/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            orig1=int(info[1].split(":")[1])
            orig2=int(info[-1].split(":")[1])
            paper1 = idx_map[orig1]
            paper2 = idx_map[orig2]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            adj_array.append([orig1,orig2])
    return idx, idx_map,feat_data, labels, adj_lists, adj_array


def load_data(path="data/", dataset="cora", layer2=False):
    """Load citation network dataset (cora only for now)"""
    path = path+dataset+'/'
    print('Loading {} dataset...'.format(dataset))


    # build graph
    #try:
    #    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #except:
    if dataset=='citeseer':
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        idx = np.array(idx_features_labels[:, 0], dtype=None)
        idx_map = {j: i for i, j in enumerate(idx)}
        print(idx)
        edges_unordered = np.genfromtxt("{}{}.cites2".format(path, dataset), dtype=None)
        flatten_edges=edges_unordered.flatten()
        flatten_edges2=[]
        for each in flatten_edges:
            flatten_edges2.append(idx_map[each])
        #print(flatten_edges2)
        edges = np.array(flatten_edges2,dtype=np.int32).reshape(edges_unordered.shape)
    elif 'pubmed' in dataset.lower():
        print('Loading pubmed')
        idx, idx_map, feat_data, labels_orig, adj_lists, adj_array=load_pubmed()
        labels=encode_onehot(labels_orig)
        features=sp.csr_matrix(feat_data, dtype=np.float32)
        edges_unordered=np.array(adj_array, dtype=np.int32)
        print(edges_unordered)
        flatten_edges=edges_unordered.flatten()
        flatten_edges2=[]
        print(idx_map.keys())
        for each in flatten_edges:
            flatten_edges2.append(idx_map[each])
        print(flatten_edges2)
        edges = np.array(flatten_edges2,dtype=np.int32).reshape(edges_unordered.shape)
        #edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    else:
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        flatten_edges=edges_unordered.flatten()
        flatten_edges2=[]
        for each in flatten_edges:
            flatten_edges2.append(idx_map[each])
        edges = np.array(flatten_edges2,dtype=np.int32).reshape(edges_unordered.shape)
    #print(edges)
                     #dtype=np.int32).reshape(edges_unordered.shape)
    print('Edges shape', edges.shape, 'np ones edges.shape', len(np.ones(edges.shape[0])))
    print('labels shape', labels.shape)
    #adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), dtype=np.float32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if layer2:
        edges_similarity = np.genfromtxt("{}{}_similarity_0.4.csv".format(path, dataset), dtype=np.int32)
        edges2 = np.array(list(map(lambda x:idx_map[x], edges_similarity.flatten())),
                         dtype=np.int32).reshape(edges_similarity.shape)
        #edges2 = np.array(list(map(idx_map.get, edges_similarity.flatten())),
        #                 dtype=np.int32).reshape(edges_similarity.shape)
        adj2 = sp.coo_matrix((np.ones(edges2.shape[0]), (edges2[:, 0], edges2[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
        print ('Similarity layer has {} nodes, {} edges'.format(adj2.shape[0], edges2.shape[0]))

        return features.todense(), adj, labels, adj2
    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(200)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
