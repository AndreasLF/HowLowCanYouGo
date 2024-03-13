import torch
import torch.optim as optim
import argparse

from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.utils.load_data import load_adj
from graph_embeddings.utils.loss import lpca_loss, L2_loss
from graph_embeddings.utils.trainer import Trainer

import pdb

# import loggers
import wandb
from graph_embeddings.utils.logger import JSONLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='LPCA', choices=['LPCA','L2'], help='Type of model and loss to use {LPCA, L2} (default: %(default)s)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--max-eval', type=int, default=25, metavar='me', help='max_eval parameter of LBFGS optimizer [set to 1 for L2-loss] (default: %(default)s)')
    parser.add_argument('--rank', type=int, default=32, metavar='R', help='dimension of the embedding space (default: %(default)s)')
    parser.add_argument('--optim-type', type=str, default='lbfgs', choices=['lbfgs', 'adam', 'sgd'], help='which optimizer to use (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--train-mode', type=str, default='reconstruct', choices=['reconstruct', 'reconstruct-from-svd', 'pretrain'], 
                        help='mode of training, {reconstruct: ordinary training, pretrain: learn parameters for better initialization} (default: %(default)s)')
    parser.add_argument('--load-ckpt', type=str, default='none', help='path to load model checkpoint (ckpt) from (default: %(default)s)')
    parser.add_argument('--save-ckpt', type=str, default='results/model.pt', help='path to save model checkpoint (ckpt) to (default: %(default)s)')
    parser.add_argument('--data', type=str, default='Cora', choices=['Cora', 'Citeseer', 'Facebook', 'Pubmed'], help='dataset to train on (default: %(default)s)')
    parser.add_argument('--model-init', type=str, default='random', choices=['random', 'load', 'svd', 'loadsvd', 'mds'], help='how to initialize the model (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load and prepare your data
    adj = load_adj(f'./data/adj_matrices/{args.data}.pt').to(device)

    model_init = args.model_init

    model = LPCAModel if args.model_type == 'LPCA' else L2Model
    loss_fn = lpca_loss if args.model_type == 'LPCA' else L2_loss

    # Initialize the trainer
    trainer = Trainer(adj=adj, model_class=model, loss_fn=loss_fn, model_init=model_init,
                      threshold=10e-5, num_epochs=args.num_epochs, save_ckpt=args.save_ckpt,
                      load_ckpt=args.load_ckpt, optim_type=args.optim_type,
                      device=args.device, max_eval=args.max_eval, loggers=[JSONLogger])#, wandb])
    
    # Train one model model
    model = trainer.init_model(args.rank)
    trainer.train(args.rank, model=model,lr=args.lr, save_path=args.save_ckpt)

    # Find the optimal rank within a range
    # trainer.find_optimal_rank(1, 50)
    # Initialize model and move it to GPU

