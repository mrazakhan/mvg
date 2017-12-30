from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse as sp
import numpy as np

path="./"
dataset="cora"
thresh=0.4

idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32)

similarities = cosine_similarity(features)
similarities2 = similarities>0.4
row, col, data=sp.find(similarities2)

indices=[]
for k,v in zip(row, col):
    indices.append([k,v])
np.savetxt('similarity_{}.csv'.format(str(thresh)), indices, fmt='%d')

