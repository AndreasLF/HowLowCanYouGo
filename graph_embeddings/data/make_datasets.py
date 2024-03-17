# import pytorch geometric
import torch
import torch_geometric.datasets
from torch_geometric.datasets import SNAPDataset
from torch_geometric.utils import to_dense_adj

from graph_embeddings.utils.config import Config
import requests 
import os
import os.path as osp

from torch_geometric.datasets.snap_dataset import read_wiki


# Monekypatch: Add the ca-HepPh, ca-GrQc and p2p-Gnutella04 datasets to the available datasets in SNAPDataset
SNAPDataset.available_datasets.update({
    'ca-hepph': ['ca-HepPh.txt.gz'], # Add ca-HepPh dataset to available datasets.
    'ca-grqc': ['ca-GrQc.txt.gz'], # Add ca-GrQc dataset to available datasets.
    'p2p-gnutella04': ['p2p-Gnutella04.txt.gz'] # Add p2p-Gnutella04 dataset to available datasets.
})
print(SNAPDataset.available_datasets)

# Save the original process method
original_process = SNAPDataset.process

# Make a new process method, monkey patching the original one
def process_wrapper(self):
    # The ca-HepPh dataset has the same format as the wiki-Vote dataset. Therefore, we can use the same processing function. 
    # This is a temporary fix until the SNAPDataset class is updated to include the ca-HepPh dataset. Should probably do a pull request to the torch_geometric repo to include this dataset.
    # ! Note: It seems like there is bug in the read_soc in snap_dataset.py. Looks like the shapes do not match with the graph data. Omit using it for now.
    if self.name.startswith('ca-') or self.name.startswith('p2p-'):
        raw_dir = osp.join(self.root, self.name, 'raw')
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])
        data_list = read_wiki(raw_files, self.name[5:])

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
    Get the adjacency matrix from the first graph in the dataset
    :param dataset: PyTorch Geometric dataset
    :return: Adjacency matrix as a torch.Tensor
    """
    data = dataset[0]  # Assuming we're interested in the first graph in the dataset
    edge_index = data.edge_index
    adj_matrix = to_dense_adj(edge_index)[0]  # Convert edge indices to dense adjacency matrix
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
