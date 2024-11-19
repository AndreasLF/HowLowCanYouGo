# import pytorch geometric
import pdb
import torch
import torch_geometric.datasets
from torch_geometric.datasets import SNAPDataset
from torch_geometric.utils import to_dense_adj

from graph_embeddings.utils.config import Config
import requests 
import os
import os.path as osp
import scipy.io

from torch_geometric.datasets.snap_dataset import read_wiki
from typing import List
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import coalesce


extra_datasets = {
    'ca-hepph': ['ca-HepPh.txt.gz'], # Nodes: 12,008
    'ca-grqc': ['ca-GrQc.txt.gz'], # Nodes: 5,242
    'ca-astroph': ['ca-AstroPh.txt.gz'], # Nodes: 18,772
    'ca-condmat': ['ca-CondMat.txt.gz'], # Nodes: 23,133
    'ca-hepth': ['ca-HepTh.txt.gz'], # Nodes: 9,877
    'email-enron': ['email-Enron.txt.gz'], # Nodes: 36,692
    'email-euall': ['email-EuAll.txt.gz'], # Nodes: 265,214
    'soc-epinions1': ['soc-Epinions1.txt.gz'], # Nodes: 75,879
    'soc-livejournal1': ['soc-LiveJournal1.txt.gz'], # Nodes: 4,847,571
    'soc-pokec-relationships': ['soc-pokec-relationships.txt.gz'], # Nodes: 1,632,803
    'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'], # Nodes: 77,360
    'soc-slashdot0902': ['soc-Slashdot0902.txt.gz'], # Nodes: 82,168
    'com-orkut': ['bigdata/communities/com-orkut.ungraph.txt.gz'], # Nodes: 3,072,441
    'com-youtube': ['bigdata/communities/com-youtube.ungraph.txt.gz'], # Nodes: 1,134,890
    'com-dblp': ['bigdata/communities/com-dblp.ungraph.txt.gz'], # Nodes: 317,080
    'com-amazon': ['bigdata/communities/com-amazon.ungraph.txt.gz'], # Nodes: 334,863
    'wiki-talk': ['wiki-Talk.txt.gz'], # Nodes: 2,394,385
    'cit-hepph': ['cit-HepPh.txt.gz'], # Nodes: 34,546
    'cit-hepth': ['cit-HepTh.txt.gz'], # Nodes: 27,770
    'cit-patents': ['cit-Patents.txt.gz'], # Nodes: 3,774,768
    'web-berkstan': ['web-BerkStan.txt.gz'], # Nodes: 685,230
    'web-google': ['web-Google.txt.gz'], # Nodes: 875,713
    'web-notredame': ['web-NotreDame.txt.gz'], # Nodes: 325,729
    'web-stanford': ['web-Stanford.txt.gz'], # Nodes: 281,903
    'amazon0302': ['amazon0302.txt.gz'], # Nodes: 262,111
    'amazon0312': ['amazon0312.txt.gz'], # Nodes: 400,727
    'amazon0505': ['amazon0505.txt.gz'], # Nodes: 410,236
    'amazon0601': ['amazon0601.txt.gz'], # Nodes: 403,394
    'p2p-gnutella04': ['p2p-Gnutella04.txt.gz'], # Nodes: 10,876
    'p2p-gnutella05': ['p2p-Gnutella05.txt.gz'], # Nodes: 8,846
    'p2p-gnutella06': ['p2p-Gnutella06.txt.gz'], # Nodes: 8,717
    'p2p-gnutella08': ['p2p-Gnutella08.txt.gz'], # Nodes: 6,301
    'p2p-gnutella09': ['p2p-Gnutella09.txt.gz'], # Nodes: 8,114
    'p2p-gnutella24': ['p2p-Gnutella24.txt.gz'], # Nodes: 26,518
    'p2p-gnutella25': ['p2p-Gnutella25.txt.gz'], # Nodes: 22,687
    'p2p-gnutella30': ['p2p-Gnutella30.txt.gz'], # Nodes: 36,682
    'p2p-gnutella31': ['p2p-Gnutella31.txt.gz'], # Nodes: 62,586
    'roadnet-ca': ['roadNet-CA.txt.gz'], # Nodes: 1,965,206
    'roadnet-pa': ['roadNet-PA.txt.gz'], # Nodes: 1,088,092
    'roadnet-tx': ['roadNet-TX.txt.gz'], # Nodes: 1,379,917
    'as-733': ['as20000102.txt.gz'], # Nodes: 103-6,474
    'as-skitter': ['as-skitter.txt.gz'], # Nodes: 1,696,415
    'as-caida': ['as-caida20071105.txt.gz'], # Nodes: 8,020-26,475
    'loc-gowalla': ['loc-gowalla_edges.txt.gz'], # Nodes: 196,591
    'loc-brightkite': ['loc-brightkite_edges.txt.gz'], # Nodes: 58,228
    'syn-euc-2': ['Euclidean_dimension_2.txt'],  # Nodes: 1,000
    'syn-euc-3': ['Euclidean_dimension_3.txt'],  # Nodes: 1,000
    'syn-euc-8': ['Euclidean_dimension_8.txt'],  # Nodes: 1,000
    'syn-hyp-2': ['Hyperbolic_dimension_2.txt'], # Nodes: 1,000
    'syn-hyp-3': ['Hyperbolic_dimension_3.txt'], # Nodes: 1,000
    'syn-hyp-8': ['Hyperbolic_dimension_8.txt'], # Nodes: 1,000
    'erdos-renyi': ['erdos-renyi']
    }

# Monekypatch: Add the datasets to the available datasets in SNAPDataset
SNAPDataset.available_datasets.update(extra_datasets)
# print(SNAPDataset.available_datasets)

# Save the original process method
original_process = SNAPDataset.process


def read_txt_dataset(files: List[str], name: str) -> List[Data]:
    import pandas as pd

    synth_datasets = ['syn-euc-2', 'syn-euc-3', 'syn-euc-8', 'syn-hyp-2', 'syn-hyp-3', 'syn-hyp-8']
    sep = None if name in synth_datasets else '\t'
    
    edge_index = pd.read_csv(files[0], sep=sep, header=None, comment="#", dtype=np.int64)
    
    edge_index = torch.from_numpy(edge_index.values).t()

    if name in (["com-amazon","com-dblp","com-orkut","com-youtube"] + synth_datasets):
        # make the graph undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    idx = torch.unique(edge_index.flatten())
    idx_assoc = torch.full((edge_index.max() + 1, ), -1, dtype=torch.long)
    idx_assoc[idx] = torch.arange(idx.size(0))

    edge_index = idx_assoc[edge_index]
    num_nodes = edge_index.max().item() + 1
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]

# Make a new process method, monkey patching the original one
def process_wrapper(self):
    print(f"Processing dataset: {self.name}")
    # The ca-HepPh dataset has the same format as the wiki-Vote dataset. Therefore, we can use the same processing function. 
    # This is a temporary fix until the SNAPDataset class is updated to include the ca-HepPh dataset. Should probably do a pull request to the torch_geometric repo to include this dataset.
    # ! Note: It seems like there is bug in the read_soc in snap_dataset.py. Looks like the shapes do not match with the graph data. Omit using it for now.
    if self.name.lower() in extra_datasets.keys():
        raw_dir = osp.join(self.root, self.name, 'raw')
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])
        data_list = read_txt_dataset(raw_files, self.name.lower())

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    else:
        # Otherwise, use the original process method
        original_process(self)

# Apply the monkey patch
SNAPDataset.process = process_wrapper


def get_data_from_torch_geometric(paper, dataset_name, root='./data/raw'):
    """
    Get dataset from torch_geometric
    :param paper: name of paper
    :param dataset_name: name of dataset
    :return: dataset
    """

    try:
        dataset_class = getattr(torch_geometric.datasets, paper)
        dataset = dataset_class(root=root, name=dataset_name)
    except AttributeError:
        raise Exception("Unknown paper: {}".format(paper))
    
    return dataset

def get_adjacency_matrix(dataset):
    """
    Get the adjacency matrix from a PyTorch Geometric dataset
    :param dataset: PyTorch Geometric dataset
    :return: Adjacency matrix as a torch.Tensor
    """
    data = dataset[0]  # Assuming we're interested in the first graph in the dataset
    edge_index = data.edge_index
    adj_matrix = to_dense_adj(edge_index)[0]  # Convert edge indices to dense adjacency matrix
    # Set diagonal to 1
    adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])
    # make all values in adj_matrix 0 or 1
    adj_matrix[adj_matrix > 0] = 1

    return adj_matrix

def save_adjacency_matrix(adj_matrix, file_path):
    """
    Save the adjacency matrix to a file in PyTorch format (.pt)
    :param adj_matrix: Adjacency matrix as a torch.Tensor
    :param file_path: Path to the file where the adjacency matrix will be saved
    """
    torch.save(adj_matrix, file_path)
    print(f"Adjacency matrix ({adj_matrix.shape}) saved to {file_path}")

def make_toy_dataset(n_tri, cycle=True, self_loops=True):
    # Create toy graph. Source: https://github.com/schariya/exact-embeddings/blob/master/ExactEmbeddings.ipynb
    import scipy as sp
    import numpy as np
    # Create the toy graph with n_tri triangles
    tri_block = sp.sparse.coo_matrix([[0,1,1],[1,0,1],[1,1,0]])
    mat = sp.sparse.block_diag((tri_block,)*n_tri)
    diag = np.tile([0,0,1], n_tri)[:-1]
    mat += sp.sparse.diags([diag, diag], [-1,1])
    if cycle:
        mat[0,-1] = 1
        mat[-1,0] = 1
    if self_loops:
        mat += sp.sparse.identity(n_tri*3)

    # convert to tensor dense
    adj_toy = torch.tensor(mat.todense(), dtype=torch.float32)
    return adj_toy


if __name__ == "__main__":
    cfg = Config("configs/config.yaml")

    raw_path = cfg.get("data", "raw_path")
    adj_matrix_path = cfg.get("data", "adj_matrices_path")
    datasets_cfg = cfg.get("data", "dataset_src") 

    # loop through datasets_cfg
    for key, val in datasets_cfg.items():
        dataset_name = key 
        src = val

        print(f"Processing dataset: {dataset_name} from source: {src}")

        # if the sorce is pytorch-geometric 
        if src and ("pytorch-geometric" in src.lower()):
            # split src string by /
            src_split = src.split("/")
            # get dataset
            dataset = get_data_from_torch_geometric(src_split[1], src_split[2], raw_path)
            # get adjacency matrix
            adj_matrix = get_adjacency_matrix(dataset)
            # save adjacency matrix
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(adj_matrix, new_file_path)

        elif src and "snapdataset" in src.lower():
            src_split = src.split("/")
            # get dataset
            dataset = get_data_from_torch_geometric(src_split[0], src_split[1], raw_path)
            # get adjacency matrix
            adj_matrix = get_adjacency_matrix(dataset)
            # save adjacency matrix
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(adj_matrix, new_file_path)
        elif src and ".mat" in src.lower():
            file_path = f"{raw_path}/{dataset_name}/{dataset_name}.mat"
            # Check if the file is already downloaded
            if not os.path.exists(file_path):
                # remove file_name from the path 
                file_folder = "/".join(file_path.split("/")[:-1])
                os.makedirs(file_folder, exist_ok=True)

                # download the file
                response = requests.get(src)

                with open(file_path, "wb") as f:
                    f.write(response.content)

            mat = scipy.io.loadmat(file_path)

            network = mat["network"]
            # convert to adjacency matrix
            adj_matrix = torch.tensor(network.todense(), dtype=torch.float32)
            adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])
            # make all values in adj_matrix 0 or 1
            adj_matrix[adj_matrix > 0] = 1

            # save adjacency matrix
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(adj_matrix, new_file_path)

        # toy dataset   
        elif src and "generate-toy-data" in src.lower():
            src_split = src.split("/")
            n_triangles = int(src_split[1])
            # generate a toy dataset
            toy_dataset = make_toy_dataset(n_triangles)
            # save adjacency matrix
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(toy_dataset, new_file_path)
        
        elif src and "syn" in dataset_name.lower():
            # pdb.set_trace()
            filepath = f"{raw_path}/{dataset_name}/raw/{src}"
            data = read_txt_dataset([filepath], name=dataset_name)[0]
            adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze()
            # save adjacency matrix
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(adj, new_file_path)
            processed_data_path = f"{raw_path}/{dataset_name}/processed"
            os.makedirs(processed_data_path, exist_ok=True)
            torch.save(data, f"{processed_data_path}/data.pt")
        
        elif src and 'erdos-renyi' in dataset_name.lower():
            import networkx
            import torch
            from torch_geometric.utils import from_networkx, to_dense_adj
            import random
            
            # seed **everything** for ER generation
            seed = 123
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            cora = (2_708, 10_556)
            citeseer = (3_327, 9_116)
            facebook = (4_039, 88_253)

            N,E = facebook
            edge_prob = 2 * E / (N*(N-1))
            er_graph = networkx.erdos_renyi_graph(N, p=edge_prob)

            data = from_networkx(er_graph)
            adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze()
            
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(adj, new_file_path)
            processed_data_path = f"{raw_path}/{dataset_name}/processed"
            os.makedirs(processed_data_path, exist_ok=True)
            torch.save(data, f"{processed_data_path}/data.pt")
