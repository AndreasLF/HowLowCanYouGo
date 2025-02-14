import torch
import torch.optim as optim
import argparse

from graph_embeddings.models.HyperbolicModel import HyperbolicModel
from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.LatentEigenModel import LatentEigenModel
from graph_embeddings.utils.load_data import load_adj
from graph_embeddings.utils.loss import CaseControlLogisticLoss, LogisticLoss, HingeLoss, PoissonLoss
from graph_embeddings.utils.trainer import Trainer

import pdb

# import loggers
import wandb
from graph_embeddings.utils.logger import JSONLogger

from graph_embeddings.utils.dataloader import CaseControlDataLoader, RandomNodeDataLoader
from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
from graph_embeddings.utils.config import Config

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
    # torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load and prepare your data
    # adj = load_adj(f'./data/adj_matrices/{args.data}.pt').to(device)
    
    cfg = Config("./configs/config.yaml")
    raw_path = cfg.get("data", "raw_path")


    dataset_split = args.dataset.split("/") # e.g. Planetoid/Cora
    dataset = get_data_from_torch_geometric(dataset_split[0], dataset_split[1], raw_path)
    # Get first graph in dataset
    data = dataset[0]

    if args.batching_type == 'random':
        print("Using random batching with batch size: ", int(args.batchsize_percentage*data.num_nodes))
        dataloader = RandomNodeDataLoader(data, batch_size=int(args.batchsize_percentage*data.num_nodes), dataset_name=dataset.name, shuffle=True)
    elif args.batching_type == 'casecontrol': 
        print("Using case control batching with batch size: ", int(args.batchsize_percentage*data.num_nodes), " and negative sampling ratio: ", args.negatives_ratio)
        dataloader = CaseControlDataLoader(data, batch_size=int(args.batchsize_percentage*data.num_nodes), dataset_name=dataset.name, negative_sampling_ratio=args.negatives_ratio)
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
    trainer.train(args.rank, 
                  model=model,
                  lr=args.lr, 
                  eval_recon_freq=10,
                  adjust_lr_patience=100, # ! changed from none
                  save_path=args.save_ckpt)