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
                 retain_non_connected: bool = False):
        self.sub_graph = sub_graph
        self.retain_non_connected = retain_non_connected # ! temp fix for training on full adj matrix

    def to(self, device):
        self.sub_graph = self.sub_graph.to(device)
        return self

    @cached_property
    def indices(self):
        if self.retain_non_connected: self.sub_graph.edge_index # ! temp fix for training on full adj matrix
        else: return self.sub_graph.edge_index.unique()

    @cached_property
    def adj(self):
        A = to_dense_adj(self.sub_graph.edge_index).squeeze(0)
        if not self.retain_non_connected: # ! temp fix for training on full adj matrix
            A = A[~torch.all(A == 0, dim=1)]
            A = A[:, ~torch.all(A == 0, dim=0)]
        # add 1 to diagonal
        A = A.fill_diagonal_(1)
        return A

    @cached_property
    def adj_s(self):
        A_tilde = self.adj*2 - 1
        return A_tilde

class CustomGraphDataLoader:
    def __init__(self, data: torch_geometric.data.Data, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.num_nodes = data.num_nodes
        self.num_parts = (self.num_nodes + self.batch_size - 1) // self.batch_size
        self.sampler = RandomNodeLoader(data, num_parts=self.num_parts, shuffle=True)

    def __iter__(self):
        for sub_data in self.sampler:
            cond = (self.batch_size == self.num_total_nodes)
            yield Batch(sub_data, retain_non_connected=cond)

    def __len__(self):
        return self.num_parts
    
    @cached_property
    def num_total_nodes(self):
        return self.data.num_nodes
    
    @cached_property
    def full_adj(self):
        A = to_dense_adj(self.data.edge_index).squeeze(0)
        # add 1 to diagonal
        A = A.fill_diagonal_(1)
        return A

class CaseControl:
    def __init__(self, data: torch_geometric.data.Data, batch_size: int, negative_sampling_ratio: int = 5):
        self.data = data
        self.batch_size = batch_size
        self.num_nodes = data.num_nodes
        self.num_parts = (self.num_nodes + self.batch_size - 1) // self.batch_size
        self.neg_ratio = negative_sampling_ratio

    def __iter__(self):
        for sub_data in self.sampler:
            yield Batch(sub_data)

    def __len__(self):
        return self.num_parts
    

    def __difference_set(self, a, b):
        combined = torch.cat((a, b))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]

        return difference
    
    def __intersection_set(self, a, b):
        combined = torch.cat((a, b))
        uniques, counts = combined.unique(return_counts=True)
        intersection = uniques[counts > 1]

        return intersection

    @cached_property
    def all_possible_edges(self):
        return torch.arange(self.num_nodes)

    def sample_non_links(self, links):
        num_links = len(links)
        possible_non_links =  self.__difference_set(self.all_possible_edges, links)

        non_links = possible_non_links[torch.randperm(possible_non_links.size(0))]
        non_links = non_links[:self.neg_ratio*num_links]

        return non_links

    def sample(self, batch_size: int):
        d = 5
        # without replacemen
        sources = torch.randperm(self.num_nodes)[:batch_size]
        srcs_idxs = torch.cat([torch.where(self.data.edge_index[0] == src)[0] for src in sources])
        trgts = self.data.edge_index[1][srcs_idxs]
        srcs = self.data.edge_index[0][srcs_idxs]
        link_edge_index = torch.stack([srcs, trgts], dim=0)

        
        all_non_links_target = []
        all_non_links_source = []
        for src in sources:
            links = link_edge_index[1][link_edge_index[0] == src]
            # difference between all possible edges and targets_
            non_links = self.sample_non_links(links)

            all_non_links_source.append(torch.ones_like(non_links)*src)
            all_non_links_target.append(non_links)

            intersection = self.__intersection_set(links, non_links) # TODO: remove this line
            assert len(intersection) == 0, "Intersection between links and non-links is not empty"
            
        non_link_edge_index = torch.stack([torch.cat(all_non_links_source), torch.cat(all_non_links_target)], dim=0)
     

if __name__ == '__main__':
    from graph_embeddings.models.L2Model import L2Model
    from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
    from graph_embeddings.utils.config import Config

    cfg = Config("configs/config.yaml")
   
    raw_path = cfg.get("data", "raw_path")
    adj_matrices_path = cfg.get("data", "adj_matrices_path")

    # %%
    dataset = get_data_from_torch_geometric("Planetoid", "Cora", raw_path)

    data = dataset[0]
    adj_true = torch.load(f"{adj_matrices_path}/Cora.pt")


    loader = CaseControl(data, 10)
    print(loader.sample(10))

    # dataloader = CustomGraphDataLoader(data,batch_size=data.num_nodes)
    # model = L2Model.init_random(dataloader.num_total_nodes, dataloader.num_total_nodes, 50)


    # fulladj = dataloader.full_adj
    # compare = torch.all(fulladj == adj_true)
    # print("Adjacency matrices are equal:", compare.item())

    # data = dataset[0]

    # dataloader = CustomGraphDataLoader(data,batch_size=520)
    # model = L2Model.init_random(dataloader.num_total_nodes, dataloader.num_total_nodes, 50)

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