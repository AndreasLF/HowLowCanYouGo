# %%
from graph_embeddings.data.make_datasets import make_toy_dataset
from graph_embeddings.examples import create_1D_example, create_2D_example

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import torch
import os

from graph_embeddings.models.HyperbolicModel import HyperbolicModel
from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.LatentEigenModel import LatentEigenModel
from graph_embeddings.utils.load_data import load_adj
from graph_embeddings.utils.loss import CaseControlLogisticLoss, LogisticLoss, HingeLoss, PoissonLoss
from graph_embeddings.utils.trainer import Trainer

import argparse

# import loggers
from graph_embeddings.utils.logger import JSONLogger
from graph_embeddings.utils.dataloader import RandomNodeDataLoader, CaseControlDataLoader
import matplotlib.pyplot as plt
import pdb

# %%
def save_adj_img(adj, folder="results/block_graphs"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(10,10))
    plt.spy(adj)
    plt.savefig(f"{folder}/adjacency_matrix.pdf", format="pdf", bbox_inches='tight')
    plt.savefig(f"{folder}/adjacency_matrix.png", format="png", bbox_inches='tight')
    plt.close()

def adjacency_to_data(adj_matrix):
    """
    Converts a dense adjacency matrix to a PyTorch Geometric Data object without node features.
    
    Args:
        adj_matrix (torch.Tensor): The dense adjacency matrix of shape [num_nodes, num_nodes].
        
    Returns:
        torch_geometric.data.Data: The corresponding Data object.
    """
    # Convert the dense adjacency matrix to edge_index (sparse format)
    edge_index, edge_attr = dense_to_sparse(adj_matrix)
    
    # Create the Data object
    data = Data(
        edge_index=edge_index, 
        edge_attr=edge_attr)
    
    return data


# %%
results_folder = "results/block_graphs"
N = [1000]
num_blocks = list(range(10,1000, 10))


toy_data = create_1D_example(10, num_blocks=2)

toy_data_geom = adjacency_to_data(toy_data)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='PCA', choices=['PCA','L2','LatentEigen','Hyperbolic'], help='Type of reconstruction model to use {LPCA, L2} (default: %(default)s)')
    parser.add_argument('--loss-type', type=str, default='logistic', choices=['logistic','hinge', 'poisson'], help='Type of loss to use {logistic, hinge, poisson} (default: %(default)s)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--rank', type=int, default=32, metavar='R', help='dimension of the embedding space (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--load-ckpt', type=str, default='none', help='path to load model checkpoint (ckpt) from (default: %(default)s)')
    parser.add_argument('--save-ckpt', type=str, default='results/model.pt', help='path to save model checkpoint (ckpt) to (default: %(default)s)')
    parser.add_argument('--model-init', type=str, default='random', choices=['random', 'load', 'pre-svd', 'post-svd'], help='how to initialize the model (default: %(default)s)')
    parser.add_argument('--recons-check', type=str, default='frob', choices=['frob', 'neigh', 'both'], help='how to check reconstruction quality (default: %(default)s)')
    parser.add_argument('--dataset', type=str, default='Planetoid/Cora', help='dataset to train on (default: %(default)s)')
    parser.add_argument('--batchsize-percentage', type=float, default=1.0, help='percentage of the dataset to use as batch size (default: %(default)s)')
    parser.add_argument('--batching-type', type=str, default='random', choices=['random', 'casecontrol'], help='which type of batching to use (default: %(default)s)')
    parser.add_argument('--negatives-ratio', type=int, default=5, help='ratio of negative samples to positive samples in case control batching (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    for n in N:
        for b in num_blocks:

            print(f"Running experiment on block network with n={n} and b={b}...")

            exp_folder_name = f"{results_folder}/n_{n}_b_{b}"   
            toy_data = create_1D_example(n, num_blocks=b)
            save_adj_img(toy_data, folder=exp_folder_name)
            data = adjacency_to_data(toy_data)

            dataset_name = f"blocks_N{n}_B{b}"

            if args.batching_type == 'random':
                print("Using random batching with batch size: ", int(args.batchsize_percentage*data.num_nodes))
                dataloader = RandomNodeDataLoader(data, batch_size=int(args.batchsize_percentage*data.num_nodes), dataset_name=dataset_name, shuffle=True)
            elif args.batching_type == 'casecontrol': 
                print("Using case control batching with batch size: ", int(args.batchsize_percentage*data.num_nodes), " and negative sampling ratio: ", args.negatives_ratio)
                dataloader = CaseControlDataLoader(data, batch_size=int(args.batchsize_percentage*data.num_nodes), dataset_name=dataset_name, negative_sampling_ratio=args.negatives_ratio)
                assert args.loss_type == 'logistic', "Case Control batching only implemented for logistic loss!"

            model_init = args.model_init

            model = {"PCA": PCAModel, 
                    "L2": L2Model, 
                    "Hyperbolic": HyperbolicModel, 
                    "LatentEigen": LatentEigenModel}[args.model_type]
            loss_fn = {"logistic": CaseControlLogisticLoss if args.batching_type == 'casecontrol' else LogisticLoss,
                    "hinge": HingeLoss, 
                    "poisson": PoissonLoss}[args.loss_type]()

            # Initialize the trainer
            trainer = Trainer(dataloader=dataloader, model_class=model, loss_fn=loss_fn, model_init=model_init,
                            threshold=1e-10, num_epochs=args.num_epochs, save_ckpt=args.save_ckpt,
                            load_ckpt=args.load_ckpt, device=args.device, 
                            loggers=[], reconstruction_check=args.recons_check)
            
            # Train one model model
            model = trainer.init_model(args.rank)


            optimal_rank = trainer.find_optimal_rank(min_rank=1, max_rank=args.rank, lr=args.lr, eval_recon_freq=10)




            # ===== SAVE RESUKLTS =====
            # write to json in results folder
            import json

            # just a quick fix :)
            b_str = str(b)
            n_str = str(n)

            json_results = f"{results_folder}/results.json"

            # if json does not exist, create it
            if not os.path.exists(json_results):
                with open(json_results, 'w') as f:
                    json.dump({"schema": "n: {b: optimal_rank}", "results": {}}, f)

            # load json into dict 
            with open(json_results, 'r') as f:
                results = json.load(f)

            if args.model_type not in results["results"]:
                results["results"][args.model_type] = {}

            # add new result, results[n][b] = optimal_rank
            if n_str not in results["results"][args.model_type]:
                results["results"][args.model_type][n_str] = {}


            if b_str not in results["results"][args.model_type][n_str]:
                results["results"][args.model_type][n_str][b_str] = optimal_rank
                results["results"][args.model_type][n_str]["res_folder"] = exp_folder_name 

            # write back to json
            with open(json_results, 'w') as f:
                json.dump(results, f)



# %%
