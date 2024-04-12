# import pytorch geometric
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
    'ca-hepph': ['ca-HepPh.txt.gz'], # Add ca-HepPh dataset to available datasets.
    'ca-grqc': ['ca-GrQc.txt.gz'], # Add ca-GrQc dataset to available datasets.
    'ca-astroph': ['ca-AstroPh.txt.gz'],
    'ca-condmat': ['ca-CondMat.txt.gz'],
    'ca-hepth': ['ca-HepTh.txt.gz'],
    'p2p-gnutella04': ['p2p-Gnutella04.txt.gz'], # Add p2p-Gnutella04 dataset to available datasets.
    'email-enron': ['email-Enron.txt.gz'], # Add email-Enron dataset to available datasets.
    'email-euall': ['email-EuAll.txt.gz'], # Add email-EuAll dataset to available datasets.
    'soc-epinions1': ['soc-Epinions1.txt.gz'],
    'soc-livejournal1': ['soc-LiveJournal1.txt.gz'],
    'soc-pokec-relationships': ['soc-pokec-relationships.txt.gz'],
    'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'],
    'soc-slashdot0902': ['soc-Slashdot0902.txt.gz'],
    'com-orkut': ['bigdata/communities/com-orkut.ungraph.txt.gz'],
    'com-youtube': ['bigdata/communities/com-youtube.ungraph.txt.gz'],
    'com-dblp': ['bigdata/communities/com-dblp.ungraph.txt.gz'],
    'com-amazon': ['bigdata/communities/com-amazon.ungraph.txt.gz'],
    'wiki-talk': ['wiki-Talk.txt.gz'],
    'cit-hepph': ['cit-HepPh.txt.gz'],
    'cit-hepth': ['cit-HepTh.txt.gz'],
    'cit-patents': ['cit-Patents.txt.gz'],
    'web-berkstan': ['web-BerkStan.txt.gz'],
    'web-google': ['web-Google.txt.gz'],
    'web-notredame': ['web-NotreDame.txt.gz'],
    'web-stanford': ['web-Stanford.txt.gz'],
    'amazon0302': ['amazon0302.txt.gz'],
    'amazon0312': ['amazon0312.txt.gz'],
    'amazon0505': ['amazon0505.txt.gz'],
    'amazon0601': ['amazon0601.txt.gz'],
    'p2p-gnutella04': ['p2p-Gnutella04.txt.gz'],
    'p2p-gnutella05': ['p2p-Gnutella05.txt.gz'],
    'p2p-gnutella06': ['p2p-Gnutella06.txt.gz'],
    'p2p-gnutella08': ['p2p-Gnutella08.txt.gz'],
    'p2p-gnutella09': ['p2p-Gnutella09.txt.gz'],
    'p2p-gnutella24': ['p2p-Gnutella24.txt.gz'],
    'p2p-gnutella25': ['p2p-Gnutella25.txt.gz'],
    'p2p-gnutella30': ['p2p-Gnutella30.txt.gz'],
    'p2p-gnutella31': ['p2p-Gnutella31.txt.gz'],
    'roadnet-ca': ['roadNet-CA.txt.gz'],
    'roadnet-pa': ['roadNet-PA.txt.gz'],
    'roadnet-tx': ['roadNet-TX.txt.gz'],
    'as-733': ['as20000102.txt.gz'],
    'as-skitter': ['as-skitter.txt.gz'],
    'as-caida': ['as-caida20071105.txt.gz'],
    'loc-gowalla': ['loc-gowalla_edges.txt.gz'],
    'loc-brightkite': ['loc-brightkite_edges.txt.gz']
    }

# Monekypatch: Add the datasets to the available datasets in SNAPDataset
SNAPDataset.available_datasets.update(extra_datasets)
# print(SNAPDataset.available_datasets)

# Save the original process method
original_process = SNAPDataset.process


def read_txt_dataset(files: List[str], name: str) -> List[Data]:
    import pandas as pd

    edge_index = pd.read_csv(files[0], sep='\t', header=None, comment="#",
                             dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()

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
        data_list = read_txt_dataset(raw_files, self.name[5:])

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
        
