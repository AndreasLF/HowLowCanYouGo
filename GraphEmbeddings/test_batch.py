# %%
# import torch geometric 
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, NeighborSampler, RandomNodeLoader
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.sampler import NodeSamplerInput, BaseSampler
from functools import cached_property

from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
import os

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index

from torch import Tensor
import pdb

# def collate_fn_custom(self, index):
#         if not isinstance(index, Tensor):
#             index = torch.tensor(index)

#         if isinstance(self.data, Data):
#             return self.data.subgraph(index)

#         elif isinstance(self.data, HeteroData):
#             node_dict = {
#                 key: index[(index >= start) & (index < end)] - start
#                 for key, (start, end) in self.node_dict.items()
#             }
#             return self.data.subgraph(node_dict)

# RandomNodeLoader.collate_fn = collate_fn_custom

# import config
from graph_embeddings.utils.config import Config

cfg = Config("configs/config.yaml")

raw_path = cfg.get("data", "raw_path")

# %%
dataset = get_data_from_torch_geometric("Planetoid", "Cora", raw_path)


data = dataset[0]

if "edge_index" in data:
    edge_index = data.edge_index
    print(edge_index.shape)

# %%

class Batch:
    def __init__(self, sub_graph: torch_geometric.data.Data):
        self.sub_graph = sub_graph

    @cached_property
    def indices(self):
        print(self.sub_graph.edge_index.shape)
        return self.sub_graph.edge_index.unique()

    @cached_property
    def adj(self):
        # relabel nodes in subgraph
        A = to_dense_adj(self.sub_graph.edge_index).squeeze(0)
        return A

    @cached_property
    def adj_s(self):
        A_tilde = self.adj*2 - 1
        return A_tilde

class CustomGraphDataLoader:
    def __init__(self, data: torch_geometric.data.Data, batch_size: int = 256):
        self.data = data
        self.batch_size = batch_size
        self.num_nodes = data.num_nodes
        self.num_parts = (self.num_nodes + self.batch_size - 1) // self.batch_size
        self.sampler = RandomNodeLoader(data, num_parts=self.num_parts, shuffle=True)

    def __iter__(self):
        for sub_data in self.sampler:
            yield Batch(sub_data)

    def __len__(self):
        return self.num_parts
    
    @cached_property
    def num_total_nodes(self):
        return self.data.num_nodes

    
# %%
from graph_embeddings.models.L2Model import L2Model


dataloader = CustomGraphDataLoader(data,batch_size=data.num_nodes)
model = L2Model.init_random(dataloader.num_total_nodes, dataloader.num_total_nodes, 50)

for b_idx, batch in enumerate(dataloader):
    print("Batch", b_idx)
    idx = batch.indices
    print(idx.shape)
    print(batch.adj_s.shape)
    print(batch.adj)

    adj = batch.adj
    # remove any rows or columns that are all zeros
    adj = adj[~torch.all(adj == 0, dim=1)]
    adj = adj[:, ~torch.all(adj == 0, dim=0)]

    print(adj.shape)
    print(model.reconstruct(idx).shape)
    break


# %%

# Testing subgraph
from torch_geometric.utils import subgraph

edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 100, 2],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 2, 2]])

subset = torch.tensor([3, 4, 5, 6, 2, 100])
subg, attr = subgraph(subset, edge_index, relabel_nodes=True)
print(subg)
print(subg.shape)



# %%
