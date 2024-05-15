# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:09:10 2021

@author: nnak
"""

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
            
        

# from blobs import *
# cl=Spectral_clustering_init(sparse_i=np.where(full_rank==1)[0],sparse_j=np.where(full_rank==1)[1],input_size=full_rank.shape[0])
# U_norm=cl.spectral_clustering()#kmeans over spectral clustering
# kmeans = KMeans(n_clusters=7).fit(U_norm)
# kmeans.labels_




# plt.scatter(X[:,0].cpu().data,X[:,1].cpu().data,c=kmeans.labels_,cmap="tab10")  # points in white (invisible)



# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov  4 16:28:28 2020

# @author: nnak
# """

# import timeit
# import torch
# import numpy as np
# from tqdm import trange
# from torch import Tensor
# import torch.nn as nn
# import torch.nn.functional as f
# from fractal_kmeans_cond import Euclidean_Kmeans
# #from fractal_kmeansSQ import Euclidean_Kmeans

# from torch_sparse import spspmm
# from copy import deepcopy
# import matplotlib.pyplot as plt
# class Tree_kmeans_recursion():
#     def __init__(self,minimum_points,init_layer_split):
#         """
#         Kmeans-Euclidean Distance minimization: Pytorch CUDA version
#         k_centers: number of the starting centroids
#         Data:Data array (if already in CUDA no need for futher transform)
#         N: Dataset size
#         Dim: Dataset dimensionality
#         n_iter: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
#         full_cuda_load: send the whole matrix into CUDA for faster implementations (memory requirements)
        
#         """
#         self.minimum_points=minimum_points
#         self.init_layer_split=init_layer_split
#         self.thetas=[]
#         self.hier_cl=[]

       

#     def kmeans_tree_recursively(self,Data,depth,init_first_layer_cent):
#         self.leaf_centroids=[]
#         self.data=Data
#         #first iteration
#         #initialize kmeans with number of clusters equal to logN
#         flag_uneven_split=False
#         model=Euclidean_Kmeans(k_centers=int(self.init_layer_split),dimensions= self.data.shape,init_cent=init_first_layer_cent)
#         sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(self.data.detach()),self.data)
#         first_centers=model.centroids.detach()
#         full_prev_cl=cl_idx
#         #flags shows if all clusters will be partioned later on
#         split_flag=1
#         global_cl=torch.zeros(self.data.shape[0]).long()
#         initial_mask=torch.arange(self.data.shape[0])
#         init_id=0
#         sum_leafs=0
#         #self.thetas.append(theta_approx)
#         for i in range(depth):
#             if i>29:
#                 print(i)
#             self.hier_cl.append(cl_idx) 
#         #datapoints in each of the corresponding clusters
#             #Compute points in each cluster
#             assigned_points= torch.cuda.FloatTensor(sparse_mask.shape[0]).fill_(0)
#             assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
#             #Splitting criterion decides if a cluster has enough points to be binary splitted
#             self.splitting_criterion=(assigned_points>self.minimum_points)
            
#             print(assigned_points)
#             #datapoints required for further splitting
#             self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())

#             self.mask_split=self.mask_split.squeeze(-1).bool()
#             #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree
#             split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
#             #if not all of the clusters have enough points to be partioned
#             # save them as leaf nodes and create the mask for the analytical calculations
#             if not split_flag:
               
#                 #erion shows the leaf nodes
#                 erion=(assigned_points<=self.minimum_points) & (assigned_points>0)
#                 #if we have leaf nodes in this tree level give them global ids
#                 if erion.sum()>0:
#                     #keep the leaf nodes for visualization
#                     with torch.no_grad():
#                         self.leaf_centroids.append(model.centroids[erion])
#                     #sum_leafs makes sure no duplicates exist in the cluster id
#                     sum_leafs=sum_leafs+erion.sum()
                    
#                     #global unique cluster ids to be used
#                     clu_ids=torch.arange(init_id,sum_leafs)
#                     #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
#                     cl_vec=torch.cuda.LongTensor(sparse_mask.shape[0]).fill_(0)
#                     cl_vec[erion]=clu_ids
#                     #initial starting point for next level of ids, so no duplicates exist
#                     init_id=sum_leafs
#                     # print(global_cl.unique(return_counts=True))
#                     # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
#                     mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
#                     # gl_idx takes the global node ids for the mask
#                     gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
#                     gl_idx=gl_idx.squeeze(-1).bool()
#                     # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
#                     gl_idx2=gl_idx
#                     if gl_idx.shape[0]!=global_cl.shape[0]:
#                         #in the case of uneven split the initial mask keeps track of the global node ids
#                         gl_idx=initial_mask[gl_idx].long()

#                     # give the proper cl ids to the proper nodes
#                     global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
#                     self.K_leafs=sum_leafs
           
 
#             centers=model.centroids
#             if self.splitting_criterion.sum()==0:
#                 break
            
#             #local clusters to be splitted
#             splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

#             if not split_flag:
#                 flag_uneven_split=True
#                 # rename ids so they start from 0 to total splitted
#                 splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
#                 # create sparse locations of K_old x K_new
#                 index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
#                 # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
#                 self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
#                 initial_mask=initial_mask[self.ind[0,:].long()]
#                 self.mask_split=initial_mask
#                 cl_idx=self.ind[1,:].long()
#             if flag_uneven_split:
#                 #translate the different size of splits to the total N of the network
#                 self.mask_split=initial_mask
                
            
#             model=Euclidean_Kmeans(k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.data[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0)
#             sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(self.data.detach()[self.mask_split]),self.data[self.mask_split])
#             full_prev_cl=cl_idx
            
       
#         return global_cl
        
   
   
    
#     def pairwise_squared_distance_trick(self,X,epsilon):
#         '''
#         Calculates the pairwise distance of a tensor in a memory efficient way
        
#         '''
#         Gram=torch.mm(X,torch.transpose(X,0,1))
#         dist=torch.diag(Gram,0).unsqueeze(0)+torch.diag(Gram,0).unsqueeze(-1)-2*Gram+epsilon
#         return dist
    
                    
                    
# Data=torch.from_numpy(U_norm).float().to(device)                  
# tree_k=Tree_kmeans_recursion(minimum_points=3*int(Data.shape[0]/(Data.shape[0]/np.log(Data.shape[0]))),init_layer_split=torch.round(torch.log(torch.tensor(Data.shape[0]).float())))

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()   

# global_cl=tree_k.kmeans_tree_recursively(Data, depth=25,init_first_layer_cent=torch.randn((7,7),device=device))
# centroids=torch.cat(tree_k.leaf_centroids)
# end.record()
# torch.cuda.synchronize()
# execution_time1 = start.elapsed_time(end)
# print(execution_time1/1000)
# plt.figure(figsize=(8,8))
# plt.scatter(X.cpu().detach()[:, 0].cpu(),X.cpu().detach()[:, 1].cpu(), c=global_cl.detach().cpu(), s= 30000 / len(Data.cpu().detach()), cmap="tab10")
# # plt.scatter(10*centroids[:, 0].detach().cpu(), 10*centroids[:, 1].detach().cpu(), c='black', s=100, alpha=.8)
# plt.show()

# new_x=Data.cpu().data
# plt.figure(figsize=(8,8))
# plt.scatter(new_x[:, 0],new_x[:, 1], c=global_cl.detach().cpu(), s= 30000 / len(Data.cpu().detach()), cmap="tab10")
# plt.scatter(centroids[:, 0].detach().cpu(), centroids[:, 1].detach().cpu(), c='black', s=100, alpha=.8)
# plt.show()


# # plt.scatter(centroids[:, 0].detach().cpu(), centroids[:, 1].detach().cpu(), c='black', s=100, alpha=.8)
# # plt.axis([0,1,0,1]) ; plt.tight_layout()
# # plt.show()
