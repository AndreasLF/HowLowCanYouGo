# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:47:48 2020

@author: nnak
"""

# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch_sparse import spspmm
CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.neighbors import KDTree,BallTree

if device.type=='cuda':
    print('Running on GPU')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('Running on CPU (slow)')
    torch.set_default_tensor_type('torch.FloatTensor')

    
    
undirected=1

import matplotlib.pyplot as plt


class LSM(nn.Module):
    def __init__(self,link_function,sparse_i,sparse_j, input_size_1,input_size_2,latent_dim,sample_size):
        super(LSM, self).__init__()
        self.input_size_1=input_size_1
        self.input_size_2=input_size_2
        self.latent_dim=latent_dim
        self.device=device
        
        self.link_function=link_function
       
        # Initialize latent space with the centroids provided from the Fractal over the spectral clustering space
        #self.kmeans_tree_recursively(depth=80,init_first_layer_cent=self.first_centers)
        self.bias=nn.Parameter(torch.rand(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.softplus=nn.Softplus(beta=10)
        
        
        self.initialization=1
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j

        self.edges=torch.cat((self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)),0)
        #print(self.edges.shape)
        self.sample_size=sample_size

        
        self.flag1=0

        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        
        self.sampling_weights=torch.ones(N1)
        
       
        self.latent_z=nn.Parameter(torch.randn(self.input_size_1,self.latent_dim))
        self.latent_w=nn.Parameter(torch.randn(self.input_size_2,self.latent_dim))

        
    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm
 
        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,torch.ones(edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size_1,self.input_size_2,self.input_size_2,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size_1,self.input_size_2,self.input_size_1,coalesced=True)
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]

        
        return sample_idx,sparse_i_sample,sparse_j_sample
      

    
    def LSM_likelihood_bias(self,epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        '''
        self.epoch=epoch
        
        sample_idx,sparse_sample_i,sparse_sample_j=self.sample_network()
        mat=self.softplus(self.bias-((torch.cdist(self.latent_z[sample_idx],self.latent_w[sample_idx],p=2))+1e-06))
        z_pdist1=(mat-torch.diag(torch.diagonal(mat))).sum()
        z_pdist2=(self.bias-((((self.latent_z[sparse_sample_i]-self.latent_w[sparse_sample_j]+1e-06)**2).sum(-1)))**0.5).sum()

 

        log_likelihood_sparse=z_pdist2-z_pdist1

        return log_likelihood_sparse
    

    
    def final_analytical(self,i_link,j_link,i_non_link,j_non_link,hinge=True,margin=False):
        
        block_pdist_an=self.pdist(self.latent_z[i_link],self.latent_w[j_link])
        block_pdist_non=self.pdist(self.latent_z[i_non_link],self.latent_w[j_non_link])
        if margin:
            h1=1-(self.bias-block_pdist_an)
            h2=1+(self.bias-block_pdist_non)
        else:
            h1=-(self.bias-block_pdist_an)
            h2=+(self.bias-block_pdist_non)
        
        h_loss=(torch.clamp(h1, min=0).sum()+torch.clamp(h2, min=0).sum())

        return -h_loss


    
from joblib import Parallel, delayed


def radius_search(tree, query_point, radius):
     indices = tree.query_radius([query_point], r=radius)[0]
     count = len(indices)
     return count, indices

def check_reconctruction(edges,Z,W,beta,N1,N2):
    Z_np=Z.detach().cpu().numpy()
    W_np=W.detach().cpu().numpy()
    beta_np=beta.detach().cpu().numpy()
    X=np.concatenate((Z_np,W_np))

    node_i=torch.arange(N1)
   # node_j=torch.arange(N2)
   
    # Parameters
    leaf_size = int(4 * np.log(N1))
    
    # Build KDTree
    tree = KDTree(X, leaf_size=leaf_size)
    
    # Define the combined radius search function
   
    
    # Perform parallel radius searches for counts and indices
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(radius_search)(tree, X[i], beta_np) for i in range(N1)
    )
    
    # Separate the counts and indices from the results
    counts, indeces = zip(*results)
    counts = np.array(counts)
    indeces = list(indeces) # Keep indices as a list of arrays


    source_ind=torch.from_numpy(np.concatenate(indeces[0:N1]))
   # targets_ind=torch.from_numpy(np.concatenate(indeces[N1:]))
    
    source_counts=torch.from_numpy(counts[0:N1])
   # targets_counts=torch.from_numpy(counts[N1:])

    total_i=torch.repeat_interleave(node_i,source_counts)
   # total_j=torch.repeat_interleave(node_j,targets_counts)

    kd_indeces_i=torch.cat((total_i.unsqueeze(1),source_ind.unsqueeze(1)),1)
    #kd_indeces_j=torch.cat((total_j.unsqueeze(1),targets_ind.unsqueeze(1)),1)

    cleaned_kd_i=kd_indeces_i[kd_indeces_i[:,1]>=N1]
    cleaned_kd_i[:,1]=cleaned_kd_i[:,1]-N1
    
    #cleaned_kd_j=kd_indeces_j[kd_indeces_j[:,1]<N1]

    active_set_edges=cleaned_kd_i.T

    s1 = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (N1,N2)).coalesce()
    mask=active_set_edges[0]!=active_set_edges[1]
    active_set_edges=active_set_edges[:,mask]
    s2 = torch.sparse_coo_tensor(active_set_edges, torch.ones(active_set_edges.shape[1]), (N1,N2)).coalesce()
    overall_=(s1-s2).coalesce()
    elements=(overall_.values()!=0).sum()
    return (elements)/(N1*(N2-1)),elements,overall_


def check_reconctruction_analytical(edges,Z,W,beta,N1,N2):
    
    if (N1>5000) or (N2>5000):
        raise ValueError('Too large for analytical calcuation')

    with torch.no_grad():
        dist=torch.cdist(Z,W).detach()
        bool_m=(dist<=beta.detach()).long()
        bool_m[torch.arange(N1),torch.arange(N1)]=0
        
        
        s1 =torch.zeros(N1,N2) #torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (N1,N2)).to_dense()
        s1[edges[0],edges[1]]=1
        
    
        elements=((s1-bool_m)!=0).sum()
        i,j=torch.where((s1-bool_m)!=0)
    return (elements)/(N1*(N2-1)),elements,i,j

    
    
torch.autograd.set_detect_anomaly(True)

latent_dims=[32]


# Available choices are ['EXP','SOFTPLUS']
link_function='SOFTPLUS'

datasets=['cora']
for dataset in datasets:
    for latent_dim in latent_dims:
        print(latent_dim)
        rocs=[]
        prs=[]
        for cv_split in range(1):
            print(dataset)
            losses=[]
            ROC=[]
            PR=[]
            zetas=[]
            betas=[]
            scalings=[]
            num_of_el=[]
            num_of_ep=[]
            per_of_el=[]


           
    
    
    # ################################################################################################################################################################
    # ################################################################################################################################################################
    # ################################################################################################################################################################
            
            import pandas as pd

            sparse_i_=pd.read_csv('./'+dataset+'/sparse_i.txt')
            sparse_j_=pd.read_csv('./'+dataset+'/sparse_j.txt')

            
            sparse_i=torch.cat((torch.tensor(sparse_i_.values.reshape(-1),device=device).long(),torch.tensor(sparse_j_.values.reshape(-1),device=device).long()))       
            sparse_j=torch.cat((torch.tensor(sparse_j_.values.reshape(-1),device=device).long(),torch.tensor(sparse_i_.values.reshape(-1),device=device).long()))
            del sparse_i_
            del sparse_j_
            edges=torch.cat((sparse_i.unsqueeze(1),sparse_j.unsqueeze(1)),1).T


            sparse_i_rem=sparse_i#torch.from_numpy(np.loadtxt('./'+dataset+'/sparse_i_rem.txt')).long().to(device)
            sparse_j_rem=sparse_j#torch.from_numpy(np.loadtxt('./'+dataset+'/sparse_j_rem.txt')).long().to(device)
            non_sparse_i=torch.from_numpy(np.loadtxt('./'+dataset+'/non_sparse_i.txt')).long().to(device)
            non_sparse_j=torch.from_numpy(np.loadtxt('./'+dataset+'/non_sparse_j.txt')).long().to(device)
            
   
            N1=int(sparse_i.max()+1)
            N2=int(sparse_j.max()+1)
            #adj=torch.zeros(N1,N2)
            #adj[sparse_i,sparse_j]=1

            #non_sparse_i,non_sparse_j=torch.where(adj==0)
            sample_percentage=0.1
            sample_size=int(sample_percentage*N1)
           
            model = LSM(link_function,sparse_i,sparse_j,N1,N2,latent_dim=latent_dim,sample_size=sample_size).to(device)
    
            optimizer = optim.Adam(model.parameters(), 0.1)  
            #model.load_state_dict(torch.load(f'EE_model_{model}_{dataset}.pth'))

           
            model.scaling=0
            print('PHASE 1: Running HBDM for 1000 iterations')
            for epoch in range(10000):
                                  
                if epoch<10000:
                    loss=-model.LSM_likelihood_bias(epoch=epoch)/model.sample_size
                    optimizer.zero_grad() # clear the gradients.   
                    loss.backward() # backpropagate
                    optimizer.step() # update the weights
                          
            
                # scheduler.step()
                if epoch%1000==0:
                    print(loss.item())
                    print(epoch)
                    


percentage,num_elements,active=check_reconctruction(edges,model.latent_z,model.latent_w,model.bias,N1,N2)
print(f'Miss-classified percentage of total elements: {100*percentage}%, i.e. {num_elements} elements',)
#model.softplus=nn.Softplus(beta=5)

i_link,j_link=active.indices()[:,active.values()==1]

i_non_link,j_non_link=active.indices()[:,active.values()==-1]
mask=i_non_link!=j_non_link
i_non_link=i_non_link[mask]
j_non_link=j_non_link[mask]
num_of_el=[]
num_of_ep=[]
per_of_el=[]


for epoch in range(10001):
                      
    # if epoch%100==0:
    #     loss=-model.LSM_likelihood_bias(epoch=epoch)/N1
    # else:

    loss=-model.final_analytical(i_link, j_link, i_non_link, j_non_link)/(N1)
    
    

    optimizer.zero_grad() # clear the gradients.   

    loss.backward() # backpropagate

    #if epoch%5==0:

    optimizer.step() # update the weights

    
    # scheduler.step()
    if epoch%25==0:
        print(loss.item())
        print(epoch)
    if epoch%5==0:
        percentage,num_elements,active=check_reconctruction(edges,model.latent_z,model.latent_w,model.bias,N1,N2)
        
        i_link,j_link=active.indices()[:,active.values()==1]
        
        # i_link=torch.cat((i_link_,j_link_))
        # j_link=torch.cat((j_link_,i_link_))

        

        i_non_link,j_non_link=active.indices()[:,active.values()==-1]
        mask=i_non_link!=j_non_link
        i_non_link=i_non_link[mask]
        j_non_link=j_non_link[mask]
        
        # i_non_link=torch.cat((i_non_link_,j_non_link_))
        # j_non_link=torch.cat((j_non_link_,i_non_link_))

        print(f'Miss-classified percentage of total elements: {100*percentage}%, i.e. {num_elements} elements',)
        if num_elements==0:
            torch.save(model.state_dict(), f'EE_model_{model}_{dataset}.pth')

            print('Total reconstruction achieved')
            break
    #i_link,j_link=active.indices()[:,active.values()==1]

    

    if epoch%25==0:
        #i_non_link,j_non_link=active.indices()[:,active.values()==-1]
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Gradient for {name}: {param.grad.max()}')
            else:
                print(f'No gradient for {name}')
        
        num_of_el.append(num_elements.numpy())
        num_of_ep.append(epoch)
        per_of_el.append(100*percentage.numpy())
        #percentage,num_elements=check_reconctruction_analytical(edges,model.latent_z,model.latent_w,model.bias,N1,N2)
        #print(f'Miss-classified percentage of total elements: {percentage} %, i.e. {num_elements} number of elements',)
        #roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr
        plt.title('Number of elements')
        plt.plot(num_of_ep,num_of_el)
        plt.xlabel('epoch')
        plt.show()
        plt.title('% of total elements')
        plt.plot(num_of_ep,per_of_el)
        plt.xlabel('epoch')
        plt.show()
    #print(roc,pr)
    
    




