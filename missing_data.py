# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:51:27 2021

@author: nnak
"""
import numpy as np
import torch

class Create_missing_data():
    def __init__(self,percentage=0.2):
        """
        Input: Edge indices, percentage of missing links,Number of total nodes
        Returns: Coordinates of missing points in the adjacency matrix (both links and non_links)
        
        """
   
        self.percentage=percentage
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
    def Missing_ij(self):
        #make place holder for link removal
        sparse_i=self.sparse_i_idx.cpu().numpy()
        sparse_j=self.sparse_j_idx.cpu().numpy()

        temp=np.ones((self.input_size,self.input_size))
        np.fill_diagonal(temp,0)
        
        #take upper diagonal indices
        non_sparse_idex=np.where(temp==1)
        temp=None
        non_sparse_i=non_sparse_idex[0]
        non_sparse_j=non_sparse_idex[1]
        cond=np.where(non_sparse_i<non_sparse_j)[0]
        
        #includes links and non-links
        non_sparse_i=non_sparse_i[cond]
        non_sparse_j=non_sparse_j[cond]
        amount_to_remove=int(self.percentage*non_sparse_i.shape[0])
        
        #samples to be removed
        samples=np.random.choice(non_sparse_i.shape[0], amount_to_remove,replace=False)
        non_sparse_i=non_sparse_i[samples]
        non_sparse_j=non_sparse_j[samples]
        
        full_rank=np.zeros((self.input_size,self.input_size))
        full_rank[sparse_i,sparse_j]=1
        # which are links
        g=full_rank[non_sparse_i,non_sparse_j]
        idx_link=np.asarray(np.where(g==1))
        sparse_i_idx_removed=non_sparse_i[idx_link[0]]
        sparse_j_idx_removed=non_sparse_j[idx_link[0]]
        
        
        sparse_i_rem=torch.from_numpy(sparse_i_idx_removed).long().to(self.device)
        sparse_j_rem=torch.from_numpy(sparse_j_idx_removed).long().to(self.device)
        
        #which are non links
        g=full_rank[non_sparse_i,non_sparse_j]
        idx_non_link=np.asarray(np.where(g==0))
        non_sparse_i_idx_removed=non_sparse_i[idx_non_link[0]]
        non_sparse_j_idx_removed=non_sparse_j[idx_non_link[0]]
        non_sparse_i=torch.from_numpy(non_sparse_i_idx_removed).long().to(self.device)
        non_sparse_j=torch.from_numpy(non_sparse_j_idx_removed).long().to(self.device)
        return sparse_i_rem,sparse_j_rem,non_sparse_i,non_sparse_j
   
        
        
        