from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
import torch
import os

if __name__ == "__main__":
    raw_path = "data/raw"
    os.makedirs(raw_path, exist_ok=True)

    dataset_name = "Cora"
    paper = "Planetoid"
    dataset = get_data_from_torch_geometric(paper, dataset_name, raw_path)
    data = dataset[0]
    edge_index = data.edge_index


    save_folder = "datasets"
    save_path = f"datasets/{dataset_name}"

    os.makedirs(save_path, exist_ok=True)

    torch.save(edge_index[0], save_path + "/sparse_i.pt")
    torch.save(edge_index[1], save_path + "/sparse_j.pt")


