# %% 
import torch

from sklearn.neighbors import KDTree
import numpy as np
from joblib import Parallel, delayed
import time

device = "cpu"

# %%
def radius_search(tree, query_point, radius):
     indices = tree.query_radius([query_point], r=radius)[0]
     count = len(indices)
     return count, indices



# %%
frac_to_keep = [1, 1/2, 1/4]

# %% 
# Load statedict 

path_to_statedict = "com-amazon_statedict.pt"
statedict = torch.load(path_to_statedict)
latent_z = statedict["latent_z"]
latent_w = statedict["latent_w"]
beta_np = statedict["bias"]  

# %%

for frac in frac_to_keep:

    # Pick to_remove_n random indices to remove
    to_remove_n = int(latent_z.shape[0] - frac * latent_z.shape[0])

    indices_to_remove = np.random.choice(latent_z.shape[0], to_remove_n, replace=False)

    print(indices_to_remove.shape)

    # Remove the indices
    latent_z_new = np.delete(latent_z, indices_to_remove, axis=0)
    latent_w_new = np.delete(latent_w, indices_to_remove, axis=0)


    X=np.concatenate((latent_z_new, latent_w_new))

    dataset = "com-amazon"
    sparse_i = torch.load(f'data/hbdm-data/{dataset}/sparse_i.pt', map_location=device)    
    N1=int(sparse_i.max()+1)

    node_i=torch.arange(N1)
    # node_j=torch.arange(N2)
    # Parameters
    leaf_size = int(4 * np.log(N1))
    # Build KDTree
    tree = KDTree(X, leaf_size=leaf_size)


# %%

    print("Starting radius search")
    t1 = time.time()

    # Perform parallel radius searches for counts and indices
    n_jobs = 16 # ! specify number of multiprocessing jobs
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(radius_search)(tree, X[i], beta_np) for i in range(N1)
    )

    t2 = time.time()

    print("Time taken for radius search: ", t2-t1, "Denominator", denom)


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

    print("Active set size:", active_set_edges.shape)

# %%
