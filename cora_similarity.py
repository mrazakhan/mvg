from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse as sp
import numpy as np
from collections import defaultdict

path="./data/cora/"
dataset="cora"
thresh=0.4

idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
rev_idx_map={i:j for i,j in enumerate(idx)}
features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32)

similarities = cosine_similarity(features)
similarities2 = similarities>thresh
row, col, data=sp.find(similarities2)

indices=[]
for k,v in zip(row, col):
    indices.append([rev_idx_map[k],rev_idx_map[v]])
np.savetxt('{}{}_similarity_{}.csv'.format(path, dataset,str(thresh)), indices, fmt='%d')

