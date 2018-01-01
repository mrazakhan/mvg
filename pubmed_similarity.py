from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse as sp
import numpy as np
from collections import defaultdict


def load_pubmed(path='./'):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = []#np.empty((num_nodes, 1), dtype=np.int64)
    idx_map = {}
    rev_idx_map={}
    idx=[]
    with open(path+"./Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            idx.append(i)
            info = line.split("\t")
            idx_map[int(info[0])] = i
            rev_idx_map[i]=int(info[0])
            labels.append( int(info[1].split("=")[1])-1)
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    adj_array=[]
    with open(path+"./Pubmed-Diabetes.Node.ids",'w') as fid:
        for each in idx_map:
            fid.write(str(each)+'\n')
    with open(path+"./Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
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
    return idx, idx_map,rev_idx_map,feat_data, labels, adj_lists, adj_array



path="./data/Pubmed-Diabetes/"
dataset='Pubmed-Diabetes'
thresh=0.4
idx, idx_map,rev_idx_map, feat_data, labels_orig, adj_lists, adj_array=load_pubmed(path)
features=sp.csr_matrix(feat_data, dtype=np.float32)

similarities = cosine_similarity(features)

similarities2 = similarities>thresh
row, col, data=sp.find(similarities2)

indices=[]
for k,v in zip(row, col):
    indices.append([rev_idx_map[k],rev_idx_map[v]])
np.savetxt('{}{}_similarity_{}.csv'.format(path, dataset,str(thresh)), indices, fmt='%d')




