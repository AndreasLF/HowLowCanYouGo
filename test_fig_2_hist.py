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
results_folder = "results/fig_2"
N = 50
rank = 12
num_runs = 100
num_blocks = [2, 5, 10, 25]


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-type', type=str, default='logistic', choices=['logistic','hinge', 'poisson'], help='Type of loss to use {logistic, hinge, poisson} (default: %(default)s)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
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

    # for model in [PCAModel, L2Model, LatentEigenModel, HyperbolicModel]:
    for model in [HyperbolicModel]:

        n = N
        for b in num_blocks:
            counts = {}
            for i in range(num_runs):
                print(f"Running experiment on block network with n={n} and b={b}...")

                toy_data = create_1D_example(n, num_blocks=b)
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

                loss_fn = {"logistic": CaseControlLogisticLoss if args.batching_type == 'casecontrol' else LogisticLoss,
                        "hinge": HingeLoss, 
                        "poisson": PoissonLoss}[args.loss_type]()

                # Initialize the trainer
                trainer = Trainer(dataloader=dataloader, model_class=model, loss_fn=loss_fn, model_init=model_init,
                                threshold=1e-10, num_epochs=args.num_epochs, save_ckpt=args.save_ckpt, device=args.device, 
                                loggers=[], reconstruction_check=args.recons_check)
                
                optimal_rank = trainer.find_optimal_rank(min_rank=1, max_rank=rank, lr=args.lr, eval_recon_freq=10)

                if optimal_rank not in counts:
                    counts[optimal_rank] = 0
                counts[optimal_rank] += 1


                print("++++++++++++++++++++++++++++++++++++++++++++")
                print(f"Run {i} of {num_runs} done!")
                print("++++++++++++++++++++++++++++++++++++++++++++")


            # save counts as json
            import json
            # convert class to name

            if not os.path.exists(f"{results_folder}/{dataset_name}"):
                os.makedirs(f"{results_folder}/{dataset_name}")

            model_name = model.__name__
            json_results = f"{results_folder}/{dataset_name}/counts_{model_name}.json"

            with open(json_results, 'w') as f:
                json.dump(counts, f)


            

