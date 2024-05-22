import timeit
import torch
import numpy as np
from tqdm import trange
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable




class Euclidean_Kmeans():
    def __init__(self,cond_control,k_centers,dimensions,nzetas=None,init_cent=None,split_mask=None,previous_cl_idx=None,full_prev_cl=None,prev_centers=None,full_prev_centers=None,centroids_split=None,assigned_points=None,aux_distance=None,local_idx=None,initialization=1,CUDA=True,device = None, n_iter=300):
        """
        Kmeans-Euclidean Distance minimization: Pytorch CUDA version
        k_centers: number of the starting centroids
        Data:Data array (if already in CUDA no need for futher transform)
        N: Dataset size
        Dim: Dataset dimensionality
        n_iter: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
        full_cuda_load: send the whole matrix into CUDA for faster implementations (memory requirements)
        
        AVOID the use of dataloader module of Pytorch---every batch will be loaded from the CPU to GPU giving high order loading overhead
        """
        
        self.k_centers=k_centers
        self.N=dimensions[0]
        self.Dim=dimensions[-1]
        self.device=device
        self.CUDA=CUDA
        self.flag1=0
        self.previous_cl_idx=previous_cl_idx
        self.initialization=initialization
        self.cluster_idx=previous_cl_idx
        self.aux_distance=aux_distance
        #self.splitting_criterion=splitting_criterion
        #if not self.split_flag:
        self.pdist_tol=nn.PairwiseDistance(p=2,eps=0)
        self.collapse_flag=False
        self.cond_control=cond_control
        self.nzetas=nzetas
        
            
            
        if self.initialization:
            self.lambdas_full = torch.rand(self.N,self.k_centers,device=device)
        else:
            self.lambdas_full = torch.rand(self.N,2,device=device)
        self.local_cl_idx=torch.zeros(self.N,device=device)

        self.inv_lambdas_full=1/self.lambdas_full
        if self.initialization:
            self.centroids=init_cent
            self.cluster_idx= torch.zeros(self.N,device=device)
        else:
            self.centroids=torch.zeros(self.k_centers,self.Dim,device=device)
            even_idx=torch.arange(0, self.k_centers,2,device=self.device)
            odd_idx=torch.arange(1, self.k_centers,2,device=self.device)
   
            self.centroids[odd_idx,:]=prev_centers+0.01
            self.centroids[even_idx,:]=prev_centers-0.01
            collapse_control_avg_radius = torch.zeros(full_prev_centers.shape[0],device=device)
            
            collapse_control_avg_radius=collapse_control_avg_radius.index_add(0, full_prev_cl, 0.5*self.aux_distance[torch.arange(self.aux_distance.shape[0],device=self.device),local_idx])
            collapse_control_avg_radius=(collapse_control_avg_radius/assigned_points)[centroids_split]
            self.condensed_centers=torch.where(collapse_control_avg_radius<(self.Dim**0.5)*self.cond_control)[0]
            # ! commented out due to INT_MAX stuff
            #! if self.condensed_centers.shape[0]>0:
            #!     self.collapse_flag=True
            #!     self.collapses=torch.where(self.previous_cl_idx.unsqueeze(-1)==self.condensed_centers)
            #!     self.collapsed_nodes=self.collapses[0]
            #!     self.collapsed_cnts=self.condensed_centers[self.collapses[1]]
               
        




        self.n_iter=n_iter
       
    
    def Kmeans_run(self,Data,Data_grad=None):
        '''
        '''
        #self.kmeans_plus_plus()
        for t in range(300):
            if t==0:
                self.Kmeans_step(Data)
                self.previous_centers=self.centroids
            else:
                self.Kmeans_step(Data)
                if self.pdist_tol(self.previous_centers,self.centroids).sum()<1e-4:
                    break
                self.previous_centers=self.centroids
        if self.collapse_flag:
            self.cluster_idx[self.collapsed_nodes]=2*self.collapsed_cnts+torch.randint(0,2,(self.collapsed_cnts.shape[0],))
        if Data_grad==None:
            self.update_clusters(self.cluster_idx, self.sq_distance,Data)
        else:
            self.update_clusters(self.cluster_idx, self.sq_distance,Data_grad)

        #print('total number of iterations:',t)
        #create Z^T responsibility sparse matrix mask KxN 
        if self.initialization:
            sparse_mask=torch.sparse.FloatTensor(torch.cat((self.cluster_idx.unsqueeze(-1),torch.arange(self.N,device=self.device).unsqueeze(-1)),1).t(),torch.ones(self.N,device=self.device),torch.Size([self.k_centers,self.N]))
            
        else:
            sparse_mask=torch.sparse.FloatTensor(torch.cat((self.cluster_idx.unsqueeze(-1),torch.arange(self.N,device=self.device).unsqueeze(-1)),1).t(),torch.ones(self.N,device=self.device),torch.Size([self.k_centers,self.N]))
           
     
        if self.nzetas==None:
            sparse_mask_z=None
            sparse_mask_w=None
                
        else:
            self.nzetas=int(self.nzetas)
            sparse_mask_z=torch.sparse.FloatTensor(torch.cat((self.cluster_idx[0:self.nzetas].unsqueeze(-1),torch.arange(self.nzetas,device=self.device).unsqueeze(-1)),1).t(),torch.ones(self.nzetas,device=self.device),torch.Size([self.k_centers,self.nzetas]))
            sparse_mask_w=torch.sparse.FloatTensor(torch.cat((self.cluster_idx[self.nzetas:].unsqueeze(-1),torch.arange(int(self.N-self.nzetas),device=self.device).unsqueeze(-1)),1).t(),torch.ones(int(self.N-self.nzetas),device=self.device),torch.Size([self.k_centers,int(self.N-self.nzetas)]))

        return sparse_mask.coalesce(),self.cluster_idx,self.local_cl_idx,self.aux_distance,sparse_mask_z.coalesce(),sparse_mask_w.coalesce()
               

    def Kmeans_step(self,Data):
        cluster_idx,sq_distance=self.calc_idx(Data)
        self.update_clusters(cluster_idx,sq_distance,Data)

    def calc_idx(self,Data):
        aux_distance,sq_distance=self.calc_dis(Data)
        _, cluster_idx=torch.min(aux_distance,dim=-1)
        self.local_cl_idx=torch.zeros(self.N,device=self.device)
        if self.initialization:
            self.local_cl_idx=cluster_idx
            self.cluster_idx=cluster_idx
        else:
            self.local_cl_idx=cluster_idx
            self.cluster_idx=self.local_cl_idx+2*self.previous_cl_idx
           
                
        return self.cluster_idx,sq_distance
    
    def calc_dis(self,Data):
       

        with torch.no_grad():
            if self.initialization:
                sq_distance=(((Data.unsqueeze(dim=1)-self.centroids.unsqueeze(dim=0))**2).sum(-1))+1e-06
            else:
                
                sq_distance=((Data.unsqueeze(dim=1)-self.centroids.view(-1,2,self.Dim)[self.previous_cl_idx,:,:])**2).sum(-1)+1e-06
                
        aux_distance=(sq_distance)*self.inv_lambdas_full+self.lambdas_full

        self.aux_distance=aux_distance
        self.sq_distance=sq_distance
        return aux_distance,sq_distance
    

        
    def update_clusters(self,cluster_idx,sq_distance,Data):
       
 
        z = torch.zeros(self.k_centers, self.Dim,device=self.device)
        o = torch.zeros(self.k_centers,device=self.device)
   
        self.lambdas_full=sq_distance**0.5+1e-06
        self.inv_lambdas_full=1/self.lambdas_full
        lambdas=self.lambdas_full[torch.arange(self.N,device=self.device),self.local_cl_idx]
        inv_lambdas=1/lambdas
        self.lambdas_X=torch.zeros(self.N, self.Dim,device=self.device)
        self.lambdas_X=torch.mul(Data,inv_lambdas.unsqueeze(-1))
             
        # if not self.initialization:

        #     z=z.index_add(0, cluster_idx, self.lambdas_X)
        #     o=o.index_add(0, cluster_idx, inv_lambdas)
       
        idx=torch.ones(self.N,device=self.device).bool()
        z=z.index_add(0, cluster_idx[idx], self.lambdas_X[idx])
        o=o.index_add(0, cluster_idx[idx], inv_lambdas[idx])

        self.centroids=torch.mul(z,(1/(o+1e-06)).unsqueeze(-1))
