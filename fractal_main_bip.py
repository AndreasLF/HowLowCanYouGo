# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:28:28 2020

@author: nnak
"""

import timeit
import torch
import numpy as np
from torch import Tensor
from fractal_kmeans_bip import Euclidean_Kmeans
#from fractal_kmeansSQ import Euclidean_Kmeans
import torch_sparse
from torch_sparse import spspmm
from copy import deepcopy
import matplotlib.pyplot as plt
class Tree_kmeans_recursion():
    def __init__(self,minimum_points,init_layer_split,device):
        """
        Kmeans-Euclidean Distance minimization: Pytorch CUDA version
        k_centers: number of the starting centroids
        Data:Data array (if already in CUDA no need for futher transform)
        N: Dataset size
        Dim: Dataset dimensionality
        n_iter: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
        full_cuda_load: send the whole matrix into CUDA for faster implementations (memory requirements)
        
        """
        self.minimum_points=minimum_points
        self.init_layer_split=init_layer_split
        self.thetas=[]
        self.max_layers=3*init_layer_split
        self.cond_control=0.001
        self.device=device
        


    def kmeans_tree_recursively(self,depth,initial_cntrs):
        self.leaf_centroids=[]
        self.general_mask1=[]
        self.general_cl_id1=[]
        self.general_mask2=[]
        self.general_cl_id2=[]
        self.total_sub_centroids=[]
        self.general_cl_id=[]
        #first iteration
        #initialize kmeans with number of clusters equal to logN
        flag_uneven_split=False
        self.number_of_remaining_zetas=self.latent_z.shape[0]
        bipartite_data=torch.cat([self.latent_z,self.latent_w])
        model=Euclidean_Kmeans(nzetas=self.number_of_remaining_zetas,cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= bipartite_data.shape,init_cent=initial_cntrs,device=self.device)
        sparse_mask,cl_idx,local_idx,aux_distance,sparse_mask_z,sparse_mask_w=model.Kmeans_run(deepcopy(bipartite_data.detach()),bipartite_data)
        full_prev_cl=cl_idx
        first_centers=model.centroids.detach()
        self.first_centers_bip=model.centroids.detach()

        #flags shows if all clusters will be partioned later on
        split_flag=1
        global_cl=torch.zeros(bipartite_data.shape[0]).long()
        initial_mask=torch.arange(bipartite_data.shape[0])
        initial_mask_z=torch.arange(self.latent_z.shape[0])
        initial_mask_w=torch.arange(self.latent_w.shape[0])

        init_id=0
        sum_leafs=0
        theta_approx=self.approximative_likelihood_fractal(cl_idx,model.centroids,sparse_mask,sparse_mask_z,sparse_mask_w,first_layer=True)
        self.general_mask1.append(torch.arange(self.input_size_1))       

        self.general_cl_id1.append(cl_idx[0:self.input_size_1])        #self.thetas.append(theta_approx)
        
        self.general_mask2.append(torch.arange(self.input_size_2))       

        self.general_cl_id2.append(cl_idx[self.input_size_1:]) 
        self.general_cl_id.append(cl_idx)
        for i in range(depth):
            if i==self.max_layers:
                self.cond_control=10*self.cond_control
                print(self.cond_control)
            if i>29:
                print(i)
        #datapoints in each of the corresponding clusters
            #Compute points in each cluster
            assigned_points= torch.zeros(sparse_mask.shape[0])
            assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            
            assigned_points_z=torch.zeros(sparse_mask_z.shape[0])
            assigned_points_z[torch.sparse.sum(sparse_mask_z,1)._indices()[0]]=torch.sparse.sum(sparse_mask_z,1)._values()
            assigned_points_w=torch.zeros(sparse_mask_w.shape[0])
            assigned_points_w[torch.sparse.sum(sparse_mask_w,1)._indices()[0]]=torch.sparse.sum(sparse_mask_w,1)._values()
            
           
            self.splitting_criterion=(assigned_points_w>self.minimum_points) & (assigned_points_z>self.minimum_points)
           
            #print(splits)
            #Splitting criterion decides if a cluster has enough points to be binary splitted
            #self.splitting_criterion=(assigned_points>self.minimum_points)
            #print(assigned_points)
            #datapoints required for further splitting
            self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())
            self.number_of_remaining_zetas=self.mask_split[0:int(self.number_of_remaining_zetas)].sum()
            self.mask_split_z=self.mask_split[0:self.number_of_remaining_zetas.long()].squeeze(-1).bool()
            self.mask_split_w=self.mask_split[self.number_of_remaining_zetas.long():].squeeze(-1).bool()
            self.mask_split=self.mask_split.squeeze(-1).bool()
            #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree
            split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
            #if not all of the clusters have enough points to be partioned
            # save them as leaf nodes and create the mask for the analytical calculations
            if not split_flag:
               
                #erion shows the leaf nodes
                erion=((assigned_points_w<=self.minimum_points) | (assigned_points_z<=self.minimum_points)) & (assigned_points>0)
                #if we have leaf nodes in this tree level give them global ids
                if erion.sum()>0:
                    #keep the leaf nodes for visualization
                    with torch.no_grad():
                        self.leaf_centroids.append(model.centroids[erion])
                    #sum_leafs makes sure no duplicates exist in the cluster id
                    sum_leafs=sum_leafs+erion.sum()
                    
                    #global unique cluster ids to be used
                    clu_ids=torch.arange(init_id,sum_leafs)
                    #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
                    cl_vec=torch.zeros(sparse_mask.shape[0])
                    cl_vec[erion]=clu_ids.float()
                    #initial starting point for next level of ids, so no duplicates exist
                    init_id=sum_leafs
                    # print(global_cl.unique(return_counts=True))
                    # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
                    mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
                    # gl_idx takes the global node ids for the mask
                    gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
                    gl_idx=gl_idx.squeeze(-1).bool()
                    # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
                    gl_idx2=gl_idx
                    if gl_idx.shape[0]!=global_cl.shape[0]:
                        #in the case of uneven split the initial mask keeps track of the global node ids
                        gl_idx=initial_mask[gl_idx].long()

                    # give the proper cl ids to the proper nodes
                    global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
                    self.K_leafs=sum_leafs
           
 
            centers=model.centroids
            if self.splitting_criterion.sum()==0:
                break
            
            #local clusters to be splitted
            splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

            if not split_flag:
                flag_uneven_split=True
                # rename ids so they start from 0 to total splitted
                splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
                # create sparse locations of K_old x K_new
                index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
                # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
                self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask=initial_mask[self.ind[0,:].long()]
                self.mask_split=initial_mask
                cl_idx=self.ind[1,:].long()
                
                #z
                self.ind, self.val = spspmm(sparse_mask_z._indices()[[1,0]],torch.ones(sparse_mask_z._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask_z.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask_z=initial_mask_z[self.ind[0,:].long()]
                self.mask_split_z=initial_mask_z
                
                #w
                self.ind, self.val = spspmm(sparse_mask_w._indices()[[1,0]],torch.ones(sparse_mask_w._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask_w.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask_w=initial_mask_w[self.ind[0,:].long()]
                self.mask_split_w=initial_mask_w
                
            if flag_uneven_split:
                #translate the different size of splits to the total N of the network
                self.mask_split=initial_mask
                self.mask_split_z=initial_mask_z
                self.mask_split_w=initial_mask_w

                
            
            model=Euclidean_Kmeans(nzetas=self.number_of_remaining_zetas,
                                   cond_control=self.cond_control,
                                   k_centers=2*int(splited_cl_ids_i.shape[0]),
                                   dimensions=bipartite_data[self.mask_split].shape,
                                   split_mask=self.mask_split,
                                   previous_cl_idx=cl_idx,
                                   full_prev_cl=full_prev_cl,
                                   prev_centers=centers.detach()[self.splitting_criterion],
                                   full_prev_centers=centers.detach(),
                                   centroids_split=self.splitting_criterion,
                                   assigned_points=assigned_points,
                                   aux_distance=aux_distance,
                                   local_idx=local_idx,
                                   initialization=0,
                                   device=self.device)
            sparse_mask,cl_idx,local_idx,aux_distance,sparse_mask_z,sparse_mask_w=model.Kmeans_run(deepcopy(bipartite_data.detach()[self.mask_split]),bipartite_data[self.mask_split])
            self.total_sub_centroids.append(model.centroids.detach())
            full_prev_cl=cl_idx
            theta_approx=theta_approx+self.approximative_likelihood_fractal(cl_idx,model.centroids,sparse_mask,sparse_mask_z,sparse_mask_w,first_layer=False)
            if cl_idx.shape[0]==int(self.input_size_1+self.input_size_2):
                self.general_mask1.append(torch.arange(self.input_size_1))       

                self.general_cl_id1.append(cl_idx[0:self.input_size_1])        #self.thetas.append(theta_approx)
        
                self.general_mask2.append(torch.arange(self.input_size_2))       

                self.general_cl_id2.append(cl_idx[self.input_size_1:])        
                self.general_cl_id.append(cl_idx)


        analytical_i,analytical_j=self.global_cl_likelihood_mask(global_cl)
        #mask=analytical_i!=analytical_j
        #return analytical_i[mask],analytical_j[mask],theta_approx,first_centers
        return analytical_i,analytical_j,theta_approx,first_centers

        
    def global_cl_likelihood_mask(self,global_cl):
        '''
        Returns the indexes of the mask required for the analytical evaluations of the last layer of the tree
        
        '''
        #make it to start from zero rather than 1
        # indexA=sparse_mask._indices()
        # values=sparse_mask._values()
        # indexB=sparse_mask.transpose(0,1)._indices()
        global_cl=global_cl
        N_values=torch.arange(global_cl.shape[0])
        indices_N_K=torch.cat([N_values.unsqueeze(0),global_cl.unsqueeze(0)],0)
        values=torch.ones(global_cl.shape[0])
        self.global_cl=global_cl
        # if matrices are not coalesced it does not work
        #Enforce it with 'coalesced=True'

        indexC, valueC = spspmm(indices_N_K,values,indices_N_K[[1,0]],values,global_cl.shape[0],self.K_leafs,global_cl.shape[0],coalesced=True)
#        mask_leafs=indexC[0]<self.latent_z,shape[0]
        
        analytical_i=indexC[0]
        
        analytical_j=indexC[1]
        mask=(analytical_i<self.latent_z.shape[0]) &(analytical_j>=self.latent_z.shape[0])
        analytical_i=analytical_i[mask]
        analytical_j=analytical_j[mask]-self.latent_z.shape[0]
        # mask=analytical_i<=analytical_j
        # return analytical_i[mask],analytical_j[mask]
        return analytical_i,analytical_j
    
    def approximative_likelihood_fractal(self,cluster_idx,layer_centroids,sparse_mask,sparse_mask_z,sparse_mask_w,first_layer=False):
      

       # max_link_dist=torch_sparse.SparseTensor(row=self.edges[0],col= self.edges[1],value=self.z_pdist.detach(),sparse_sizes=(self.input_size_1,self.input_size_2))
     

        if not self.scaling:
            if first_layer:
                
                #sum_cl_idx_i=torch.sparse.mm(sparse_mask_z,torch.exp(self.gamma).unsqueeze(-1))
                #print(sparse_mask_z._indices())
                sum_cl_idx_i=torch_sparse.spmm(sparse_mask_z._indices(), sparse_mask_z._values(), sparse_mask_z.shape[0], sparse_mask_z.shape[1],torch.ones(self.latent_z.shape[0]).unsqueeze(-1))

                #sum_cl_idx_j=torch.sparse.mm(sparse_mask_w,torch.exp(self.alpha).unsqueeze(-1))
                sum_cl_idx_j=torch_sparse.spmm(sparse_mask_w._indices(), sparse_mask_w._values(), sparse_mask_w.shape[0], sparse_mask_w.shape[1],torch.ones(self.latent_w.shape[0]).unsqueeze(-1))

                k_distance=self.pairwise_squared_distance_trick(layer_centroids, epsilon=1e-16)**0.5
                if self.link_function=='EXP':
                
                    dist_exp=torch.exp(-k_distance+self.bias)
                    theta_approx=(torch.mm(sum_cl_idx_i.transpose(0,1),(torch.mm((dist_exp-torch.diag(torch.diagonal(dist_exp))),sum_cl_idx_j))).sum())
                elif self.link_function=='SOFTPLUS':
                
                    dist_exp=self.softplus(-k_distance+self.bias)
                    theta_approx=(torch.mm(sum_cl_idx_i.transpose(0,1),(torch.mm((dist_exp-torch.diag(torch.diagonal(dist_exp))),sum_cl_idx_j))).sum())
                else:
                    raise ValueError('Invalid link function choice')
                    
                
            else:
               
                #K/2 X 1 
                #sum_cl_idx_i=torch.sparse.mm(sparse_mask_z,torch.exp(self.gamma[self.mask_split_z]).unsqueeze(-1))
                sum_cl_idx_i=torch_sparse.spmm(sparse_mask_z._indices(), sparse_mask_z._values(), sparse_mask_z.shape[0], sparse_mask_z.shape[1], torch.ones(self.mask_split_z.shape[0]).unsqueeze(-1))
               

                
                #sum_cl_idx_j=torch.sparse.mm(sparse_mask_w,torch.exp(self.alpha[self.mask_split_w]).unsqueeze(-1))                   
                sum_cl_idx_j=torch_sparse.spmm(sparse_mask_w._indices(), sparse_mask_w._values(), sparse_mask_w.shape[0], sparse_mask_w.shape[1],torch.ones(self.mask_split_w.shape[0]).unsqueeze(-1))

                k_distance=self.pdist(layer_centroids.view(-1,2,layer_centroids.shape[-1])[:,0,:],layer_centroids.view(-1,2,layer_centroids.shape[-1])[:,1,:])+self.bias
                
                if self.link_function=='EXP':

                    theta_approx=(torch.exp(-k_distance)*((sum_cl_idx_i.view(-1,2)[:,0]*sum_cl_idx_j.view(-1,2)[:,1]))).sum()+(torch.exp(-k_distance)*((sum_cl_idx_i.view(-1,2)[:,1]*sum_cl_idx_j.view(-1,2)[:,0]))).sum()
                elif self.link_function=='SOFTPLUS':
                    theta_approx=(self.softplus(-k_distance)*((sum_cl_idx_i.view(-1,2)[:,0]*sum_cl_idx_j.view(-1,2)[:,1]))).sum()+(self.softplus(-k_distance)*((sum_cl_idx_i.view(-1,2)[:,1]*sum_cl_idx_j.view(-1,2)[:,0]))).sum()

                else:
                    raise ValueError('Invalid link function choice')

        return theta_approx

    
    def pairwise_squared_distance_trick(self,X,epsilon):
        '''
        Calculates the pairwise distance of a tensor in a memory efficient way
        
        '''
        Gram=torch.mm(X,torch.transpose(X,0,1))
        dist=torch.diag(Gram,0).unsqueeze(0)+torch.diag(Gram,0).unsqueeze(-1)-2*Gram+epsilon
        return dist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #     def kmeans_tree_z_initialization(self,depth,initial_cntrs):
    #     self.leaf_centroids=[]

    #     #first iteration
    #     #initialize kmeans with number of clusters equal to logN
    #     flag_uneven_split=False
    #     model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= self.spectral_data.shape,init_cent=initial_cntrs)
    #     sparse_mask,cl_idx,local_idx,aux_distance,_,_=model.Kmeans_run(self.spectral_data)
    #     full_prev_cl=cl_idx
    #     first_centers=model.centroids.detach()
    #     #flags shows if all clusters will be partioned later on
    #     split_flag=1
    #     global_cl=torch.zeros(self.spectral_data.shape[0]).long()
    #     initial_mask=torch.arange(self.spectral_data.shape[0])
    #     init_id=0
    #     sum_leafs=0
    #     #self.thetas.append(theta_approx)
    #     for i in range(depth):
    #         if i==self.max_layers:
    #             self.cond_control=10*self.cond_control
    #             print(self.cond_control)
    #         if i>29:
    #             print(i)
    #     #datapoints in each of the corresponding clusters
    #         #Compute points in each cluster
    #         assigned_points= torch.cuda.FloatTensor(sparse_mask.shape[0]).fill_(0)
    #         assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            
          
    #         #Splitting criterion decides if a cluster has enough points to be binary splitted
    #         self.splitting_criterion=(assigned_points>self.minimum_points)
    #         #print(assigned_points)
    #         #datapoints required for further splitting
    #         self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())
            

    #         self.mask_split=self.mask_split.squeeze(-1).bool()
    #         #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree
    #         split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
    #         #if not all of the clusters have enough points to be partioned
    #         # save them as leaf nodes and create the mask for the analytical calculations
    #         if not split_flag:
               
    #             #erion shows the leaf nodes
    #             erion=(assigned_points<=self.minimum_points) & (assigned_points>0)
    #             #if we have leaf nodes in this tree level give them global ids
    #             if erion.sum()>0:
    #                 #keep the leaf nodes for visualization
    #                 with torch.no_grad():
    #                     self.leaf_centroids.append(model.centroids[erion])
    #                 #sum_leafs makes sure no duplicates exist in the cluster id
    #                 sum_leafs=sum_leafs+erion.sum()
                    
    #                 #global unique cluster ids to be used
    #                 clu_ids=torch.arange(init_id,sum_leafs)
    #                 #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
    #                 cl_vec=torch.cuda.LongTensor(sparse_mask.shape[0]).fill_(0)
    #                 cl_vec[erion]=clu_ids
    #                 #initial starting point for next level of ids, so no duplicates exist
    #                 init_id=sum_leafs
    #                 # print(global_cl.unique(return_counts=True))
    #                 # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
    #                 mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
    #                 # gl_idx takes the global node ids for the mask
    #                 gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
    #                 gl_idx=gl_idx.squeeze(-1).bool()
    #                 # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
    #                 gl_idx2=gl_idx
    #                 if gl_idx.shape[0]!=global_cl.shape[0]:
    #                     #in the case of uneven split the initial mask keeps track of the global node ids
    #                     gl_idx=initial_mask[gl_idx].long()

    #                 # give the proper cl ids to the proper nodes
    #                 global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
    #                 self.K_leafs=sum_leafs
           
 
    #         centers=model.centroids
    #         if self.splitting_criterion.sum()==0:
    #             break
            
    #         #local clusters to be splitted
    #         splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

    #         if not split_flag:
    #             flag_uneven_split=True
    #             # rename ids so they start from 0 to total splitted
    #             splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
    #             # create sparse locations of K_old x K_new
    #             index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
    #             # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
    #             self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
    #             initial_mask=initial_mask[self.ind[0,:].long()]
    #             self.mask_split=initial_mask
    #             cl_idx=self.ind[1,:].long()
    #         if flag_uneven_split:
    #             #translate the different size of splits to the total N of the network
    #             self.mask_split=initial_mask
                
            
    #         model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.spectral_data[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0)
    #         sparse_mask,cl_idx,local_idx,aux_distance,_,_=model.Kmeans_run(self.spectral_data[self.mask_split])
    #         full_prev_cl=cl_idx
    #     return global_cl,torch.cat(self.leaf_centroids)
        
    
    
    # def kmeans_tree_scaling(self,depth,initial_cntrs):
    #     self.leaf_centroids=[]

    #     #first iteration
    #     #initialize kmeans with number of clusters equal to logN
    #     flag_uneven_split=False
    #     self.number_of_remaining_zetas=self.latent_z.shape[0]
    #     bipartite_data=torch.cat([self.latent_z,self.latent_w])
    #     model=Euclidean_Kmeans(nzetas=self.number_of_remaining_zetas,cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= bipartite_data.shape,init_cent=initial_cntrs)
    #     sparse_mask,cl_idx,local_idx,aux_distance,sparse_mask_z,sparse_mask_w=model.Kmeans_run(deepcopy(self.scaling_factor.detach()*bipartite_data.detach()),self.scaling_factor*bipartite_data.detach())
    #     full_prev_cl=cl_idx
    #     first_centers=model.centroids.detach()
    #     #flags shows if all clusters will be partioned later on
    #     split_flag=1
    #     global_cl=torch.zeros(bipartite_data.shape[0]).long()
    #     initial_mask=torch.arange(bipartite_data.shape[0])
    #     initial_mask_z=torch.arange(self.latent_z.shape[0])
    #     initial_mask_w=torch.arange(self.latent_w.shape[0])

    #     init_id=0
    #     sum_leafs=0
    #     theta_approx=self.approximative_likelihood_fractal(cl_idx,model.centroids,sparse_mask,sparse_mask_z,sparse_mask_w,first_layer=True)
    #     #self.thetas.append(theta_approx)
    #     for i in range(depth):
    #         if i==self.max_layers:
    #             self.cond_control=10*self.cond_control
    #             print(self.cond_control)
    #         if i>29:
    #             print(i)
    #     #datapoints in each of the corresponding clusters
    #         #Compute points in each cluster
    #         assigned_points= torch.cuda.FloatTensor(sparse_mask.shape[0]).fill_(0)
    #         assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            
    #         assigned_points_z=torch.cuda.FloatTensor(sparse_mask_z.shape[0]).fill_(0)
    #         assigned_points_z[torch.sparse.sum(sparse_mask_z,1)._indices()[0]]=torch.sparse.sum(sparse_mask_z,1)._values()
    #         assigned_points_w=torch.cuda.FloatTensor(sparse_mask_w.shape[0]).fill_(0)
    #         assigned_points_w[torch.sparse.sum(sparse_mask_w,1)._indices()[0]]=torch.sparse.sum(sparse_mask_w,1)._values()
            
           
    #         self.splitting_criterion=(assigned_points_w>self.minimum_points) & (assigned_points_z>self.minimum_points)
           
    #         #print(splits)
    #         #Splitting criterion decides if a cluster has enough points to be binary splitted
    #         #self.splitting_criterion=(assigned_points>self.minimum_points)
    #         #print(assigned_points)
    #         #datapoints required for further splitting
    #         self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())
    #         self.number_of_remaining_zetas=self.mask_split[0:int(self.number_of_remaining_zetas)].sum()
    #         self.mask_split_z=self.mask_split[0:self.number_of_remaining_zetas.long()].squeeze(-1).bool()
    #         self.mask_split_w=self.mask_split[self.number_of_remaining_zetas.long():].squeeze(-1).bool()
    #         self.mask_split=self.mask_split.squeeze(-1).bool()
    #         #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree
    #         split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
    #         #if not all of the clusters have enough points to be partioned
    #         # save them as leaf nodes and create the mask for the analytical calculations
    #         if not split_flag:
               
    #             #erion shows the leaf nodes
    #             erion=((assigned_points_w<=self.minimum_points) | (assigned_points_z<=self.minimum_points)) & (assigned_points>0)
    #             #if we have leaf nodes in this tree level give them global ids
    #             if erion.sum()>0:
    #                 #keep the leaf nodes for visualization
    #                 with torch.no_grad():
    #                     self.leaf_centroids.append(model.centroids[erion])
    #                 #sum_leafs makes sure no duplicates exist in the cluster id
    #                 sum_leafs=sum_leafs+erion.sum()
                    
    #                 #global unique cluster ids to be used
    #                 clu_ids=torch.arange(init_id,sum_leafs)
    #                 #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
    #                 cl_vec=torch.cuda.LongTensor(sparse_mask.shape[0]).fill_(0)
    #                 cl_vec[erion]=clu_ids
    #                 #initial starting point for next level of ids, so no duplicates exist
    #                 init_id=sum_leafs
    #                 # print(global_cl.unique(return_counts=True))
    #                 # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
    #                 mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
    #                 # gl_idx takes the global node ids for the mask
    #                 gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
    #                 gl_idx=gl_idx.squeeze(-1).bool()
    #                 # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
    #                 gl_idx2=gl_idx
    #                 if gl_idx.shape[0]!=global_cl.shape[0]:
    #                     #in the case of uneven split the initial mask keeps track of the global node ids
    #                     gl_idx=initial_mask[gl_idx].long()

    #                 # give the proper cl ids to the proper nodes
    #                 global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
    #                 self.K_leafs=sum_leafs
           
 
    #         centers=model.centroids
    #         if self.splitting_criterion.sum()==0:
    #             break
            
    #         #local clusters to be splitted
    #         splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

    #         if not split_flag:
    #             flag_uneven_split=True
    #             # rename ids so they start from 0 to total splitted
    #             splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
    #             # create sparse locations of K_old x K_new
    #             index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
    #             # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
    #             self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
    #             initial_mask=initial_mask[self.ind[0,:].long()]
    #             self.mask_split=initial_mask
    #             cl_idx=self.ind[1,:].long()
                
    #             #z
    #             self.ind, self.val = spspmm(sparse_mask_z._indices()[[1,0]],torch.ones(sparse_mask_z._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask_z.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
    #             initial_mask_z=initial_mask_z[self.ind[0,:].long()]
    #             self.mask_split_z=initial_mask_z
                
    #             #w
    #             self.ind, self.val = spspmm(sparse_mask_w._indices()[[1,0]],torch.ones(sparse_mask_w._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask_w.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
    #             initial_mask_w=initial_mask_w[self.ind[0,:].long()]
    #             self.mask_split_w=initial_mask_w
                
    #         if flag_uneven_split:
    #             #translate the different size of splits to the total N of the network
    #             self.mask_split=initial_mask
    #             self.mask_split_z=initial_mask_z
    #             self.mask_split_w=initial_mask_w

                
            
    #         model=Euclidean_Kmeans(nzetas=self.number_of_remaining_zetas,cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions=bipartite_data[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0)
    #         sparse_mask,cl_idx,local_idx,aux_distance,sparse_mask_z,sparse_mask_w=model.Kmeans_run(deepcopy(self.scaling_factor.detach()*bipartite_data.detach()[self.mask_split]),self.scaling_factor*bipartite_data.detach()[self.mask_split])
    #         full_prev_cl=cl_idx
    #         theta_approx=theta_approx+self.approximative_likelihood_fractal(cl_idx,model.centroids,sparse_mask,sparse_mask_z,sparse_mask_w,first_layer=False)
    #     analytical_i,analytical_j=self.global_cl_likelihood_mask(global_cl)
    #     return analytical_i,analytical_j,theta_approx,first_centers  
    
                    

                    