# import pytorch geometric
import torch
import torch_geometric.datasets
from torch_geometric.utils import to_dense_adj

from graph_embeddings.utils.config import Config


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
    print(f"Adjacency matrix saved to {file_path}")

if __name__ == "__main__":
    cfg = Config("configs/config.yaml")

    raw_path = cfg.get("data", "raw_path")
    adj_matrix_path = cfg.get("data", "adj_matrices_path")
    datasets_cfg = cfg.get("data", "dataset_src") 

    # loop through datasets_cfg
    for key, val in datasets_cfg.items():
        dataset_name = key 
        src = val

        # if the sorce is pytorch-geometric 
        if src and "pytorch-geometric" in src.lower():
            # split src string by /
            src_split = src.split("/")
            # get dataset
            dataset = get_data_from_torch_geometric(src_split[1], src_split[2], raw_path)
            # get adjacency matrix
            adj_matrix = get_adjacency_matrix(dataset)
            # save adjacency matrix
            new_file_path = f"{adj_matrix_path}/{dataset_name}.pt"
            save_adjacency_matrix(adj_matrix, new_file_path)
