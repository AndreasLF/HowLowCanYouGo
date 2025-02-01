from scipy.sparse.linalg import eigsh
import scipy
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from scipy.sparse import linalg

CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Spectral_clustering_init():
    def __init__(self,num_of_eig=7):
        
        self.num_of_eig=num_of_eig

    
    
    
    def spectral_clustering(self):
        
        sparse_i=self.sparse_i_idx.cpu().numpy()
        sparse_j=self.sparse_j_idx.cpu().numpy()
        
            
        V=np.ones(sparse_i.shape[0])
   
        self.Affinity_matrix=sparse.coo_matrix((V,(sparse_i,sparse_j)),shape=(self.input_size_1,self.input_size_2))
        u,_,v=linalg.svds(self.Affinity_matrix, k=self.num_of_eig)
        u=np.array(u)
        v=np.array(v)

        return torch.from_numpy(u).float().to(device),torch.from_numpy(v.transpose()).float().to(device)