# Import all the packages
# import pdb
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from fractal_main_bip import Tree_kmeans_recursion
from missing_data import Create_missing_data
#from kmeans_cuda import Normal_Kmeans as Euclidean_Kmeans
# from blobs import *
CUDA = torch.cuda.is_available()
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
from joblib import Parallel, delayed
import pdb
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


from sklearn.neighbors import KDTree


if device.type=='cuda':
    print('Running on GPU')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('Running on CPU (slow)')
    torch.set_default_tensor_type('torch.FloatTensor')

    
    

import matplotlib.pyplot as plt


class LSM(nn.Module,Tree_kmeans_recursion,Create_missing_data,Spectral_clustering_init):
    def __init__(self,link_function,sparse_i,sparse_j, input_size_1,input_size_2,latent_dim,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,CVflag=True,graph_type='undirected',missing_data=False):
        super(LSM, self).__init__()
        Tree_kmeans_recursion.__init__(self,minimum_points=int(input_size_1/(input_size_1/np.log(input_size_1))),init_layer_split=torch.round(torch.log(torch.tensor(input_size_1).float())),device=device)
        Create_missing_data.__init__(self,percentage=0.2)
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim)
        self.input_size_1=input_size_1
        self.input_size_2=input_size_2
        self.latent_dim=latent_dim
        self.device=device
        
        self.link_function=link_function
       
        # Initialize latent space with the centroids provided from the Fractal over the spectral clustering space
        #self.kmeans_tree_recursively(depth=80,init_first_layer_cent=self.first_centers)
        self.bias=nn.Parameter(torch.rand(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.softplus = nn.Softplus(beta=10)
        
        
        self.graph_type=graph_type
        self.initialization=1
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j

        self.edges=torch.cat((self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)),0)
        #print(self.edges.shape)

        
        self.flag1=0

        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.missing_data=missing_data
        
        
        
        
        self.non_sparse_j_idx_removed=non_sparse_j
        self.non_sparse_i_idx_removed=non_sparse_i
           
        self.sparse_i_idx_removed=sparse_i_rem
        self.sparse_j_idx_removed=sparse_j_rem
        if sparse_i_rem!=None:
            self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
            self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))
             
          
    
            
      
           
        self.first_centers=torch.randn(int(torch.round(torch.log(torch.tensor(input_size_1).float()))),latent_dim,device=device)
      
        # spectral_centroids_to_z,spectral_centroids_to_w=self.spectral_clustering()
       
        # self.latent_z=nn.Parameter(spectral_centroids_to_z)
        # self.latent_w=nn.Parameter(spectral_centroids_to_w)
       
        self.latent_z=nn.Parameter(torch.randn(self.input_size_1,self.latent_dim))
        self.latent_w=nn.Parameter(torch.randn(self.input_size_2,self.latent_dim))
        # self.gamma=nn.Parameter(torch.randn(self.input_size_1,device=device))
       
        # self.alpha=nn.Parameter(torch.randn(self.input_size_2,device=device))
               
        # self.latent_z=nn.Parameter(torch.randn(self.input_size_1,self.latent_dim))
        # self.latent_w=nn.Parameter(torch.randn(self.input_size_2,self.latent_dim))
      


    def local_likelihood(self,analytical_i,analytical_j):
        '''

        Parameters
        ----------
        k_mask : data points belonging to the specific centroid

        Returns
        -------
        Explicit calculation over the box of a specific centroid

        '''
        #change the distance to matrix and then reuse the Z^T matrix to calculate everything
        #return
       
        self.analytical_i=analytical_i
        self.analytical_j=analytical_j
     
        block_pdist=self.pdist(self.latent_z[analytical_i],self.latent_w[analytical_j])+1e-08
                
        ## Block kmeans analytically#########################################################################################################
                
        lambda_block=-block_pdist+self.bias
        if self.link_function=='EXP':

            an_lik=torch.exp(lambda_block).sum()
        elif self.link_function=='SOFTPLUS':

            an_lik=self.softplus(lambda_block).sum()
            
        else:
            raise ValueError('Invalid link function choice')
        return an_lik
        
    
    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def LSM_likelihood_bias(self,epoch):
        '''

        Parameters
        ----------
        cent_dist : real
            distnces of the updated centroid and the k-1 other centers.
        count_prod : TYPE
            DESCRIPTION.
        mask : Boolean
            DESCRIBES the slice of the mask for the specific kmeans centroid.

        Returns
        -------
        None.

        '''
        self.epoch=epoch
        
        
        self.z_pdist=(((self.latent_z[self.sparse_i_idx]-self.latent_w[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5
  
        analytical_i,analytical_j,thetas,init_centroids=self.kmeans_tree_recursively(depth=80,initial_cntrs=self.first_centers)
        self.first_centers=init_centroids
        #theta_stack=torch.stack(self.thetas).sum()
        analytical_blocks_likelihood=self.local_likelihood(analytical_i,analytical_j)
        ##############################################################################################################################
         
        ####################################################################################################################################
                
                                
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #remember the indexing of the z_pdist vector
               
               
        logit_u=(-self.z_pdist+self.bias)
         #########################################################################################################################################################      
        log_likelihood_sparse=torch.sum(logit_u)-thetas-analytical_blocks_likelihood
        #############################################################################################################################################################        
                 
            
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
    
    
    
    def link_prediction(self):
        with torch.no_grad():
            z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_w[self.removed_j])**2).sum(-1))**0.5
            logit_u_miss=-z_pdist_miss+self.bias
            
            if self.link_function=='EXP':

            
                rates=torch.exp(logit_u_miss)
                
            elif self.link_function=='SOFTPLUS':

            
                rates=self.softplus(logit_u_miss)
            else:
                raise ValueError('Invalid link function choice')
            
                
                
            self.rates=rates

        
            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(recall,precision)
    

def radius_search(tree, query_point, radius):
     indices = tree.query_radius([query_point], r=radius)[0]
     count = len(indices)
     return count, indices

    
def check_reconctruction(edges,Z,W,beta,N1,N2):
    device = edges.device

    Z_np=Z.detach().contiguous().cpu().numpy()
    W_np=W.detach().contiguous().cpu().numpy()
    beta_np=beta.detach().contiguous().cpu().numpy()
    X=np.concatenate((Z_np,W_np))
 
    node_i=torch.arange(N1)
   # node_j=torch.arange(N2)
    # Parameters
    leaf_size = int(4 * np.log(N1))
    # Build KDTree
    tree = KDTree(X, leaf_size=leaf_size)
    # Define the combined radius search function

    # Perform parallel radius searches for counts and indices
    n_jobs = 16 # ! specify number of multiprocessing jobs
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(radius_search)(tree, X[i], beta_np) for i in range(N1)
    )
    # Separate the counts and indices from the results
    counts, indeces = zip(*results)
    counts = np.array(counts)
    indeces = list(indeces) # Keep indices as a list of arrays
 
    source_ind=torch.from_numpy(np.concatenate(indeces[0:N1])).to(device)
   # targets_ind=torch.from_numpy(np.concatenate(indeces[N1:]))
    source_counts=torch.from_numpy(counts[0:N1]).to(device)
   # targets_counts=torch.from_numpy(counts[N1:])
    
    total_i=torch.repeat_interleave(node_i,source_counts)
   # total_j=torch.repeat_interleave(node_j,targets_counts)
 
    kd_indeces_i=torch.cat((total_i.unsqueeze(1),source_ind.unsqueeze(1)),1)
    #kd_indeces_j=torch.cat((total_j.unsqueeze(1),targets_ind.unsqueeze(1)),1)
 
    cleaned_kd_i=kd_indeces_i[kd_indeces_i[:,1]>=N1]
    cleaned_kd_i[:,1]=cleaned_kd_i[:,1]-N1
    #cleaned_kd_j=kd_indeces_j[kd_indeces_j[:,1]<N1]
 
    active_set_edges=cleaned_kd_i.T
 
    s1 = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (N1,N2), device=device).coalesce()
    mask=active_set_edges[0]!=active_set_edges[1]
    active_set_edges=active_set_edges[:,mask]
    s2 = torch.sparse_coo_tensor(active_set_edges, torch.ones(active_set_edges.shape[1]), (N1,N2), device=device).coalesce()
    overall_=(s1-s2).coalesce()
    elements=(overall_.values()!=0).sum()
    return (elements)/(N1*(N2-1)),elements,overall_
    

def check_reconctruction_analytical(edges,Z,W,beta,N1,N2):
    
    if (N1>5000) or (N2>5000):
        raise ValueError('Too large for analytical calcuation')

    with torch.no_grad():
        dist=torch.cdist(Z,W).detach()
        bool_m=(dist<=beta.detach()).long()
        
        
        s1 =torch.zeros(N1,N2) #torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (N1,N2)).to_dense()
        s1[edges[0],edges[1]]=1
        
    
        elements=((s1-bool_m)!=0).sum()
    return (elements)/(N1*(N2-1)),elements



def create_model(dataset, latent_dim, link_function = "SOFTPLUS", device='cpu'):
    # Available choices for link_function are ['EXP','SOFTPLUS']
    # ! if we use full adj matrix, do not concatenate i->j, j->i
    sparse_i = torch.load(f'./{dataset}/sparse_i.pt', map_location=device)
    sparse_j = torch.load(f'./{dataset}/sparse_j.pt', map_location=device)
    edges=torch.vstack([sparse_i,sparse_j]).to(device)
    
    N1=int(sparse_i.max()+1)
    N2=int(sparse_j.max()+1)

    model = LSM(link_function,sparse_i,sparse_j,N1,N2,latent_dim=latent_dim,CVflag=True,graph_type='directed',missing_data=False).to(device)    
    return model, N1, N2, edges


def update_json(json_path, key, value):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    data[key] = value
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    return data


class Trainer:
    def __init__(self, model, N1, N2, edges, 
                 exp_id = None, 
                 phase_epochs = {1: 1_000, 2: 5_000, 3: 10_000}, 
                 kd_tree_freq = 5, 
                 learning_rate = 0.1, 
                 learning_rate_hinge = 0.1, 
                 dataset_name = None, 
                 model_path = "notset", 
                 wandb_logging = True, 
                 wandb_id = None,
                 results_dir = "results"):
        self.model = model
        self.N1 = N1
        self.N2 = N2
        self.edges = edges
        self.exp_id = exp_id
        self.phase_epochs = phase_epochs
        self.kd_tree_freq = kd_tree_freq
        self.learning_rate = learning_rate
        self.learning_rate_hinge = learning_rate_hinge
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.wandb_logging = wandb_logging
        self.wandb_id = wandb_id
        self.results_dir = results_dir

        # ! training state
        self.current_epoch = 0
        self.current_phase = 1
         
    def save_checkpoint(self):
        train_state = {"current_epoch": self.current_epoch, "current_phase": self.current_phase}

        rank = self.model.latent_dim
        save_state = {"model": self. model,
                      "optimizer_state": self.optimizer.state_dict(),
                      "optimizer_hinge_state": self.optimizer_hinge.state_dict() if self.optimizer_hinge is not None else None, 
                      "learning_rate_scheduler_hinge": self.learning_rate_scheduler_hinge.state_dict() if self.learning_rate_scheduler_hinge is not None else None,
                      "train_state": train_state,
                      "wandb_id": self.wandb_id}
        ckpt_path = f'{self.results_dir}/{self.exp_id}/checkpoints/state_rank{rank}_phase{self.current_phase}_epoch{self.current_epoch}.ckpt'
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(save_state, ckpt_path)
        # update json in results dir
        update_json(f'{self.results_dir}/{self.exp_id}/experiment_state.json', 'latest_checkpoint', ckpt_path)

    @classmethod
    def from_scratch(cls, model, N1, N2, edges, exp_id=None, phase_epochs={1: 1_000, 2: 5_000, 3: 10_000}, 
                     kd_tree_freq=5, learning_rate=0.1, learning_rate_hinge=0.1, dataset_name=None, 
                     model_path="notset", wandb_logging=True, results_dir="results"):
        instance = cls(model=model, N1=N1, N2=N2, edges=edges, exp_id=exp_id, phase_epochs=phase_epochs, 
                   kd_tree_freq=kd_tree_freq, learning_rate=learning_rate, learning_rate_hinge=learning_rate_hinge, 
                   dataset_name=dataset_name, model_path=model_path, wandb_logging=wandb_logging, results_dir=results_dir)

        instance.optimizer = optim.Adam(model.parameters(), learning_rate) 
        instance.optimizer_hinge = optim.Adam(model.parameters(), learning_rate_hinge)
        instance.learning_rate_scheduler_hinge = optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer_hinge, mode='min', factor=0.5, patience=15, verbose=True)

        return instance

    @classmethod
    def from_checkpoint(cls, checkpoint_path, exp_id=None, 
                        phase_epochs={1: 1_000, 2: 5_000, 3: 10_000}, kd_tree_freq=5, learning_rate=0.1, 
                        learning_rate_hinge=0.1, dataset_name=None, model_path="notset", wandb_logging=True, 
                        results_dir="results"):

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model = checkpoint['model']

            instance = cls(model=model, N1=model.input_size_1, N2=model.input_size_2, edges=model.edges,
                           exp_id=exp_id,
                           phase_epochs=phase_epochs,
                           kd_tree_freq=kd_tree_freq,
                           learning_rate=learning_rate, 
                           learning_rate_hinge=learning_rate_hinge, 
                           dataset_name=dataset_name, 
                           model_path=model_path, 
                           wandb_logging=wandb_logging, 
                           wandb_id=checkpoint.get('wandb_id', None),
                           results_dir=results_dir)
            
            
            optimizer = optim.Adam(model.parameters())  # Initialize the optimizer
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            instance.optimizer = optimizer

            # ! load training state
            train_state = checkpoint.get('train_state', None)
            if train_state is not None:
                instance.current_epoch = train_state.get('current_epoch', 0)
                instance.current_phase = train_state.get('current_phase', 1)
            
            if checkpoint.get('optimizer_hinge_state', None) is not None:
                instance.optimizer_hinge = optim.Adam(model.parameters(), lr=learning_rate_hinge if learning_rate_hinge is not None else checkpoint['learning_rate_hinge'])
                instance.optimizer_hinge.load_state_dict(checkpoint['optimizer_hinge_state'])

            if checkpoint.get('learning_rate_scheduler_hinge', None) is not None:
                instance.learning_rate_scheduler_hinge = optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer_hinge, mode='min', factor=0.5, patience=15, verbose=True)
                instance.learning_rate_scheduler_hinge.load_state_dict(checkpoint['learning_rate_scheduler_hinge'])

            print(f"Checkpoint loaded from {checkpoint_path}")
            return instance
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # ! main
    def train(self):        
        checkpoint_freq = 100
        # checkpoint_freq = 5

        torch.autograd.set_detect_anomaly(True)


        # ====================================== Initialize wandb ======================================
        rank = self.model.latent_dim
        if self.wandb_logging:
            wandb_run_id = self.wandb_id
            if wandb_run_id is None: 
                run = wandb.init(project="GraphEmbeddings", 
                                    config={'model_class': "LSM",
                                            'data': self.dataset_name,
                                            'rank': rank, 
                                            'phase1_epochs': self.phase_epochs[1],
                                            'phase2_epochs': self.phase_epochs[2],
                                            'phase3_epochs': self.phase_epochs[3],
                                            'kd_tree_freq': self.kd_tree_freq,
                                            'exp_id': self.exp_id,
                                            'learning_rate': self.learning_rate
                                            })   

                self.wandb_id = run.id
            else:   
                # Resume logging to another experiment
                wandb.init(project="GraphEmbeddings",
                resume="allow",  # Use "allow" to resume if possible, "must" to enforce resumption
                id=wandb_run_id  # Use the run_id from the previous run
                )     

    # ################################################################################################################################################################
    # ################################################################################################################################################################
    # ################################################################################################################################################################
    
        self.model.scaling=0
        # ! Phase 1
        print(f'PHASE 1: Running HBDM for {self.phase_epochs[1]} iterations')
        phase_str = "PHASE 1"
        self.current_phase = 1
        percentage, num_elements = torch.tensor(float('NaN')), torch.tensor(float('NaN'))
        last_hbdm_loss, last_hinge_loss = torch.tensor(float('NaN')), torch.tensor(float('NaN'))
        pbar = tqdm(range(self.current_epoch, self.phase_epochs[1] + self.phase_epochs[2]), initial=self.current_epoch, total=self.phase_epochs[1] + self.phase_epochs[2] + self.phase_epochs[3])
        for epoch in pbar:
            if self.current_phase == 3: break # if starting form a checkpoint in phase 3, break out of phase 1/2
            self.current_epoch = epoch
            metrics = {'epoch': epoch}
            # Save checkpoint
            if epoch % checkpoint_freq == 0 and epoch != 0: self.save_checkpoint() # ! Save checkpoint

            # ! PHASE 1
            if epoch < self.phase_epochs[1]: 
                loss = -self.model.LSM_likelihood_bias(epoch=epoch) / self.N1
                self.optimizer.zero_grad(set_to_none=True)  # clear the gradients.   
                loss.backward()  # backpropagate
                self.optimizer.step()  # update the weights
                last_hbdm_loss = loss.detach().cpu().item()
                metrics["hbdm_loss"] = last_hbdm_loss
            # ! PHASE 2
            else: 
                # TODO? reconstruction check based on size of edge list before going to phase 2
                # i.e. num_elements <= |edge_list|*log(|edge_list|)
                if epoch == self.phase_epochs[1]:
                    print(f'PHASE 2: Running HBDM and Hinge loss, for every HBDM iteration running {self.kd_tree_freq} iterations for the hinge loss')
                    phase_str = "PHASE 2"
                    self.current_phase = 2

                if epoch % 2 == 0:
                    loss = -self.model.LSM_likelihood_bias(epoch=epoch) / self.N1
                    self.optimizer.zero_grad(set_to_none=True)  # clear the gradients.   
                    loss.backward()  # backpropagate
                    self.optimizer.step()  # update the weights
                    last_hbdm_loss = loss.detach().cpu().item()
                    metrics["hbdm_loss"] = last_hbdm_loss
                else:
                    percentage, num_elements, active = check_reconctruction(self.edges, self.model.latent_z, self.model.latent_w, self.model.bias, self.N1, self.N2)
                    i_link, j_link = active.indices()[:, active.values() == 1]
                    i_non_link, j_non_link = active.indices()[:, active.values() == -1]
                    mask = i_non_link != j_non_link
                    i_non_link = i_non_link[mask]
                    j_non_link = j_non_link[mask]

                    if num_elements == 0: # ! PERFECT RECONSTRUCTION
                        print('Total reconstruction achieved')
                        self.save_checkpoint() # ! Save checkpoint
                        save_path = self.model_path.replace('.pt', '_FR.pt')
                        if self.wandb_logging:
                            wandb.config.update({'full_reconstruction': True, "model_path":save_path})
                            wandb.finish()
                        return True

                    for j in range(5):
                        loss = -self.model.final_analytical(i_link, j_link, i_non_link, j_non_link, hinge=True) / self.N1
                        self.optimizer.zero_grad(set_to_none=True)  # clear the gradients.
                        loss.backward()  # backpropagate
                        self.optimizer.step()  # update the weights
                    last_hinge_loss = loss.detach().cpu().item()

                    metrics["hinge_loss"] = last_hinge_loss
                    metrics["misclassified_dyads_perc"] = percentage.detach().cpu().item()*100
                    metrics["misclassified_dyads"] = num_elements

            pbar.set_description(f"[{phase_str}] [last HBDM loss={last_hbdm_loss:.4f}] [last Hinge loss={last_hinge_loss:.4f}] [misclassified dyads = {percentage.detach().cpu().item()*100 : .4f}% - i.e. {num_elements}]")
            if self.wandb_logging: wandb.log(metrics)

        # save checkpoint
        self.save_checkpoint() # ! Save checkpoint
                    
        print(f'PHASE 3: Running Hinge loss only (building kdtree every {self.kd_tree_freq} iterations)')
        phase_str = "PHASE 3"
        self.current_phase = 3

        # ! set to CPU for hinge loss
        torch.set_default_tensor_type('torch.FloatTensor') 
        device = torch.device('cpu')
        self.edges = self.edges.to(device)
        self.model = self.model.to(device)

        percentage,num_elements,active=check_reconctruction(self.edges,self.model.latent_z,self.model.latent_w,self.model.bias,self.N1,self.N2)
        print(f'Miss-classified percentage of total elements: {100*percentage}%, i.e. {num_elements} elements',)

        i_link,j_link=active.indices()[:,active.values()==1]

        i_non_link,j_non_link=active.indices()[:,active.values()==-1]
        mask=i_non_link!=j_non_link
        i_non_link=i_non_link[mask]
        j_non_link=j_non_link[mask]
        
        # ! LR scheduler for hinge loss
        optimizer_hinge = self.optimizer_hinge
        lr_patience = 15
        lr_scheduler = self.learning_rate_scheduler_hinge
        # lr_scheduler = None # comment out to use lr scheduling

        pbar = tqdm(range(self.current_epoch, self.phase_epochs[3] + self.phase_epochs[2] + self.phase_epochs[1]), initial=self.current_epoch, total=self.phase_epochs[3] + self.phase_epochs[2] + self.phase_epochs[1])
        # ! PHASE 3
        for epoch in pbar: 
            self.current_epoch = epoch
            metrics = {"epoch": epoch}

            if epoch % checkpoint_freq == 0 and epoch != 0: self.save_checkpoint() # ! Save checkpoint

            loss= - self.model.final_analytical(i_link, j_link, i_non_link, j_non_link)/self.N1
            last_hinge_loss = loss.detach().cpu().item()
        
            optimizer_hinge.zero_grad(set_to_none=True) # clear the gradients.   
            loss.backward() # backpropagate
            metrics["hinge_loss"] = last_hinge_loss
            optimizer_hinge.step() # update the weights
            if lr_scheduler is not None and epoch > lr_patience:
                lr_scheduler.step(loss.item())
            
            # scheduler.step()
            if epoch%self.kd_tree_freq==0: # ! evalute every 5 or 25? etc.
                percentage,num_elements,active=check_reconctruction(self.edges, self.model.latent_z,self.model.latent_w,self.model.bias,self.N1,self.N2)

                # ! log reconstruction metrics after every update
                metrics["misclassified_dyads_perc"] = percentage.detach().cpu().item()*100
                metrics["misclassified_dyads"] = num_elements
                if self.wandb_logging: wandb.log(metrics)

                i_link,j_link=active.indices()[:,active.values()==1]

                i_non_link,j_non_link=active.indices()[:,active.values()==-1]
                mask=i_non_link!=j_non_link
                i_non_link=i_non_link[mask]
                j_non_link=j_non_link[mask]

                if num_elements==0: # ! PERFECT RECONSTRUCTION
                    print('Total reconstruction achieved!')
                    self.save_checkpoint() # ! Save checkpoint
                    save_path = self.model_path.replace('.pt', '_FR.pt')
                    if self.wandb_logging:
                        wandb.config.update({'full_reconstruction': True, "model_path":save_path})
                        wandb.finish()

                    return True

            pbar.set_description(f"[{phase_str}] [last hinge loss={last_hinge_loss}] [misclassified dyads = {percentage.detach().cpu().item()*100 : .4f}% - i.e. {num_elements}]")

        if self.wandb_logging:
            wandb.config.update({"model_path": self.model_path})
            wandb.finish()
        return False
            

# !!!!! FOR EED SEARCH, RUN find_optimal_rank.py
if __name__ == '__main__':
    latent_dim = 50
    # dataset_relpath = "datasets"
    # dataset = 'Cora'
    # model, N1, N2, edges = create_model(f"{dataset_relpath}/{dataset}", latent_dim)
    # is_reconstructed = train(model, N1, N2, edges)
    # torch.save(model, "")
    # pdb.set_trace()            
