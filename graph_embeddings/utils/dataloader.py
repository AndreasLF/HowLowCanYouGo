import torch
from torch import Tensor
import torch_geometric
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import to_dense_adj
from functools import cached_property
from torch_geometric.utils import subgraph
import copy
from torch_geometric.utils import select, subgraph
from torch_geometric.data import Data
import pdb

from graph_embeddings.utils.set_ops import difference_set, intersection_set

def edge_index_to_adjacency_matrix(edge_index):
    # Identify all unique nodes and their new indices
    unique_nodes, new_indices = torch.unique(edge_index, return_inverse=True)
    
    # Number of nodes in the reduced graph
    num_nodes = len(unique_nodes)
    
    # Reshape new_indices to match the shape of edge_index for mapping
    new_edge_index = new_indices.view(edge_index.shape)
    
    # Create a tensor of ones with the same length as the new edge index to represent edge weights
    edge_weights = torch.ones(new_edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    
    # Create a sparse tensor with new edge indices and weights using torch.sparse_coo_tensor
    adjacency_matrix = torch.sparse_coo_tensor(new_edge_index, edge_weights, (num_nodes, num_nodes)).to_dense()
    
    # Set the diagonal elements to 1
    adjacency_matrix.fill_diagonal_(1)
    
    return adjacency_matrix

# Monkey patch the subgraph method on the Data class to not relabel nodes
def subgraph_custom(self, subset: Tensor) -> 'Data':
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        out = subgraph(subset, self.edge_index, relabel_nodes=False,
                       num_nodes=self.num_nodes, return_edge_mask=True)
        
        edge_index, _, edge_mask = out

        data = copy.copy(self)

        for key, value in self:
            if key == 'edge_index':
                data.edge_index = edge_index
            elif key == 'num_nodes':
                if subset.dtype == torch.bool:
                    data.num_nodes = int(subset.sum())
                else:
                    data.num_nodes = subset.size(0)
            elif self.is_node_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, subset, dim=cat_dim)
            elif self.is_edge_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, edge_mask, dim=cat_dim)

        return data

Data.subgraph = subgraph_custom

class Batch:
    def __init__(self, 
                 sub_graph: torch_geometric.data.Data, 
                 indices: torch.Tensor):
        self.indices = indices.sort()[0]
        self.sub_graph = sub_graph

    def to(self, device):
        self.sub_graph = self.sub_graph.to(device)
        self.indices = self.indices.to(device)
        return self

    @cached_property
    def adj(self):
        num_nodes = self.indices.shape[0]
        indices = self.indices

        # pdb.set_trace()
        # Initialize the adjacency matrix with zeros
        A = self.sub_graph.edge_index.new_zeros((num_nodes,num_nodes))

        src = self.sub_graph.edge_index[0]
        tgt = self.sub_graph.edge_index[1]

        # torch.bucketize is similar to numpy's searchsorted
        src_idx = torch.bucketize(src, indices)
        tgt_idx = torch.bucketize(tgt, indices)

        # Verify if src and tgt exactly match the indices. Just for testing
        # assert torch.all(indices[src_idx] == self.sub_graph.edge_index[0])
        # assert torch.all(indices[tgt_idx] == self.sub_graph.edge_index[1])

        # Fill in the adjacency matrix with ones where there are edges
        A[src_idx, tgt_idx] = 1

        # add 1 to diagonal
        A.fill_diagonal_(1)
        return A

    @cached_property
    def adj_s(self):
        A_tilde = self.adj*2 - 1
        return A_tilde

class RandomNodeDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data: Data,
        batch_size: int,
        dataset_name: str = None,
        shuffle: bool = True, # shuffle is True by default, because we want to cover the whole graph across epochs
        **kwargs,
    ):
        self.data = data
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        edge_index = data.edge_index

        self.edge_index = edge_index
        self.num_nodes = data.num_nodes

        super().__init__(
            range(self.num_nodes),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            **kwargs,
        )

    @cached_property
    def num_total_nodes(self):
        return self.data.num_nodes
    
    @cached_property
    def full_adj(self):
        A = to_dense_adj(self.data.edge_index).squeeze(0)
        # add 1 to diagonal
        A.fill_diagonal_(1)
        return A
    
    @cached_property
    def edge_index_with_selfloops(self):
        edge_index = self.data.edge_index
        src_tgt = torch.arange(self.num_nodes)
        return torch.hstack([edge_index,torch.vstack([src_tgt,src_tgt])])

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)

        return Batch(self.data.subgraph(index), index)

class CaseControlBatch:
    def __init__(self, 
                 edge_index: torch.Tensor, 
                 indices: torch.Tensor, 
                 negative_sampling_ratio: int,
                 num_nodes: int = None):
        self.edge_index = edge_index
        self.indices = indices.sort()[0]    
        self.negative_sampling_ratio = negative_sampling_ratio
        self.num_nodes = num_nodes

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.device = device
        return self
    
    @cached_property
    def links(self):
        return self.edge_index

    @cached_property
    def nonlinks(self):
        """
        Given a set of links, sample (d x |links|) nonlinks,
            where d is self.negative_sampling_ratio.
        """
        num_nodes = self.num_nodes
        srcs, targets = self.links
        unique_srcs = srcs.unique()

        # Calculate the number of samples to take for non-links
        num_samples_per_src = (srcs.bincount(minlength=num_nodes) * self.negative_sampling_ratio).long()

        all_non_links = []

        for src in unique_srcs:
            # Identify all targets connected to the current source
            current_targets = targets[srcs == src]

            # Create a tensor of all nodes and mask out the current targets
            mask = torch.ones(num_nodes, dtype=torch.bool)
            mask[current_targets] = False
            
            # Possible non-links are where mask is True
            possible_non_links = torch.masked_select(torch.arange(num_nodes), mask)

            # Number of non-links to sample
            num_samples = min(num_samples_per_src[src].item(), possible_non_links.numel())
            # num_samples = possible_non_links.numel()

            # Sample without replacement from the possible non-links
            if num_samples > 0:
                sampled_indices = torch.randperm(possible_non_links.numel())[:num_samples]
                sampled_non_links = possible_non_links[sampled_indices]
                # sampled_non_links = possible_non_links

                # Form the (2, num_samples) tensor of [src, non-link]
                nonlinks = torch.stack((torch.full((num_samples,), src, dtype=src.dtype), sampled_non_links))
                all_non_links.append(nonlinks)

        # Concatenate all results along the second dimension if any non-links were sampled
        if all_non_links:
            non_links_tensor = torch.cat(all_non_links, dim=1)
        else:
            non_links_tensor = torch.empty((2, 0), dtype=torch.long)  # return an empty tensor if no non-links


        return non_links_tensor.to(self.device)

    @cached_property
    def coeffs(self):
        
        weighting_coeffs = [] # weighting coeff for each source for nonlinks, since we subsample these

        non_links = self.nonlinks
        links = self.links

        unique_srcs = torch.unique(links[0])


        for idx,src in enumerate(unique_srcs):
            num_links = links[1][links[0] == src].shape[0]
            num_nonlinks = non_links[1][non_links[0] == src].shape[0]
            weighting_coeffs.append(torch.ones(num_nonlinks) * (self.num_nodes - num_links) / num_nonlinks) # weighting for nonlinks -> #{total nonlinks} div by #{sampled nonlinks}


        return torch.hstack(weighting_coeffs).to(self.device)

class CaseControlDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data: Data,
        batch_size: int,
        negative_sampling_ratio: int = 5,
        dataset_name: str = None,
        shuffle: bool = True, # shuffle is True by default, because we want to cover the whole graph across epochs
        **kwargs,
    ):
        self.data = data
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.negative_sampling_ratio = negative_sampling_ratio

        edge_index = data.edge_index

        self.edge_index = edge_index
        self.num_nodes = data.num_nodes

        super().__init__(
            range(self.num_nodes),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            **kwargs,
        )

    @cached_property
    def num_total_nodes(self):
        return self.data.num_nodes
    
    @cached_property
    def full_adj(self):
        A = to_dense_adj(self.data.edge_index).squeeze(0)
        # add 1 to diagonal
        A.fill_diagonal_(1)
        return A
    
    @cached_property
    def edge_index_with_selfloops(self):
        edge_index = self.data.edge_index
        src_tgt = torch.arange(self.num_nodes)
        return torch.hstack([edge_index,torch.vstack([src_tgt,src_tgt])])

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)

        src = self.data.edge_index[0]
        tgt = self.data.edge_index[1]

        # get indices where index is in src 
        matches = torch.where(src.unsqueeze(0) == index.unsqueeze(1))

        new_edge_index = torch.vstack([src[matches[1]], tgt[matches[1]]])

        return CaseControlBatch(edge_index=new_edge_index, indices=index, 
                                negative_sampling_ratio=self.negative_sampling_ratio, num_nodes=self.num_nodes)

class CaseControlDataLoaderOLD:
    def __init__(self, 
                 data: torch_geometric.data.Data, 
                 batch_size: int, 
                 negative_sampling_ratio: int = 5,
                 dataset_name: str = None):
        self.data = data
        self.batch_size = batch_size
        self.num_total_nodes = data.num_nodes
        self.num_parts = (self.num_total_nodes + self.batch_size - 1) // self.batch_size
        self.neg_ratio = negative_sampling_ratio
        self.dataset_name = dataset_name

    @cached_property
    def full_adj(self):
        """
        get full adjacency matrix for testing.
        """
        A = to_dense_adj(self.data.edge_index).squeeze(0)
        # add 1 to diagonal
        A = A.fill_diagonal_(1)
        return A

    def __iter__(self):
        # sample source indices without replacement
        sources = torch.randperm(self.num_total_nodes)
        for i in range(self.num_total_nodes // self.batch_size):
            src_i = sources[i*self.batch_size : (i+1)*self.batch_size]
            yield self.sample(src_i)

    def __len__(self):
        return self.num_parts
    

    @cached_property
    def all_possible_edges(self):
        return torch.arange(self.num_total_nodes)

    def sample_non_links(self, links):
        """
        Given a set of links, sample (d x |links|) nonlinks,
            where d is self.neg_ratio.
        """
        num_links = len(links)
        possible_non_links =  difference_set(self.all_possible_edges, links)
        non_links = possible_non_links[torch.randperm(possible_non_links.size(0))]
        non_links = non_links[:self.neg_ratio*num_links]
        return non_links

    def sample(self, sources: torch.Tensor):
        """
        Case control sampling, i.e. a calibrated version of positive/negative node sampling.
            Samples a batch of nodes, for which it returns all their links, and a fixed ratio of nonlinks (hence negative samples).
            By default the ratio of nonlinks to links is set to 5, meaning we sample (5 x |links|) nonlinks.
        """
        link_idxs = torch.cat([torch.where(self.data.edge_index[0] == src)[0] for src in sources]) # maybe more memory efficient? 
        # matches = self.data.edge_index[0].unsqueeze(0) == sources.unsqueeze(1) # broadcast formulation of above - i.e. parallelized
        # matches = matches.nonzero(as_tuple=False)        # get pairwise positions in edge_index tensor
        # link_idxs = matches[:,1] 

        srcs = self.data.edge_index[0][link_idxs]
        tgts = self.data.edge_index[1][link_idxs]
        link_edge_index = torch.stack([srcs, tgts], dim=0)
        
        # # # weighting_coeffs = torch.zeros((2, len(sources))) # weighting coeff for each source, for respectively links and nonlinks
        weighting_coeffs = [] # weighting coeff for each source for nonlinks, since we subsample these
        all_non_links_target = []
        all_non_links_source = []
        for idx,src in enumerate(sources): # TODO: refactor to vectorized instead of for loop (if slow)
            links = link_edge_index[1][link_edge_index[0] == src]
            nonlinks = self.sample_non_links(links)

            # TODO: remove this assertion, only for testing
            assert len(intersection_set(links, nonlinks)) == 0, "Intersection between set of links and of non-links is not empty"

            num_links = links.shape[0]
            num_nonlinks = nonlinks.shape[0]
            # # # weighting_coeffs[0,idx] = 1                                           # weighting for links
            all_non_links_source.append(torch.ones_like(nonlinks)*src)
            all_non_links_target.append(nonlinks)
            weighting_coeffs.append(torch.ones(num_nonlinks) * (self.num_total_nodes - num_links) / num_nonlinks) # weighting for nonlinks -> #{total nonlinks} div by #{sampled nonlinks}

        # link_masks = link_edge_index[0].unsqueeze(0) == sources # ? could be part of broadcasting formulation

        nonlink_edge_index = torch.vstack([torch.hstack(all_non_links_source), torch.hstack(all_non_links_target)])
        weighting_coeffs = torch.hstack(weighting_coeffs)
        
        return link_edge_index, nonlink_edge_index, weighting_coeffs # ! return (links, nonlinks, weighting_coeffs)
     

if __name__ == '__main__':
    from graph_embeddings.models.L2Model import L2Model
    from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
    from graph_embeddings.utils.config import Config

    import time

    torch.manual_seed(42)

    cfg = Config("configs/config.yaml")
   
    raw_path = cfg.get("data", "raw_path")
    adj_matrices_path = cfg.get("data", "adj_matrices_path")

    # %%
    dataset = get_data_from_torch_geometric("SNAPDataset", "ca-HepPh", raw_path)
    print(dataset.name)

    data = dataset[0]
    adj_true = torch.load(f"{adj_matrices_path}/Cora.pt")


    loader = CaseControlDataLoader(data, 10)

    for b_idx, batch in enumerate(loader):
        print("Batch:", b_idx)
        links, nonlinks, coeffs = batch.links, batch.non_links, batch.coeffs
        print("  Links shape:", links.shape)
        print("  Nonlinks shape:", nonlinks.shape)
        print("  Coeffs shape:", coeffs.shape)
        break

    # loaderOLD = CaseControlDataLoaderOLD(data, 10)
    # links, nonlinks, coeffs = next(iter(loaderOLD))
    # print("  Links shape:", links.shape)
    # print("  Nonlinks shape:", nonlinks.shape)
    # print("  Coeffs shape:", coeffs.shape)


    # dataloader = RandomNodeDataLoader(data,batch_size=512, datasetname=dataset.name)
    # print(dataloader.data)
    # model = L2Model.init_random(dataloader.num_total_nodes, dataloader.num_total_nodes, 50)

    # fulladj = dataloader.full_adj
    # compare = torch.all(fulladj == adj_true)
    # print("Adjacency matrices are equal:", compare.item())


    # for b_idx, batch in enumerate(dataloader):
    #     print("Batch:", b_idx)
    #     idx = batch.indices

    #     # only take out a submatrix of the adj_true matrix
    #     adj_true_sub = adj_true[idx][:,idx]
    #     batch_adj = batch.adj

    #     # compare the adjacency matrices
    #     compare = torch.all(adj_true_sub == batch_adj)
    #     print("  Adjacency matrices in batch are equal:", compare.item())

    #     print("  Batch adj shape:", batch.adj.shape)
    #     print("  Batch adj_s shape:", batch.adj_s.shape)
    #     print("  Batch indices shape:", batch.indices.shape)
    #     print("  Reconstruct shape:", model.reconstruct(idx).shape)
    #     break