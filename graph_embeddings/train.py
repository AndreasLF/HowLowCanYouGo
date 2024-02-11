import torch
import torch.optim as optim
import argparse

from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss, L2_loss
from graph_embeddings.utils.trainer import Trainer


# Ensure CUDA is available and select device, if not check for Macbook Pro support (MPS) and finally use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='LPCA', choices=['LPCA','L2'], help='Type of model and loss to use {LPCA, L2} (default: %(default)s)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--max-eval', type=int, default=25, metavar='me', help='max_eval parameter of LBFGS optimizer [set to 1 for L2-loss] (default: %(default)s)')
    parser.add_argument('--rank', type=int, default=32, metavar='R', help='dimension of the embedding space (default: %(default)s)')
    parser.add_argument('--optim-type', type=str, default='lbfgs', choices=['lbfgs', 'adam', 'sgd'], help='which optimizer to use (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    # parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    # parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    # parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    # parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    # parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Load and prepare your data
    adj = torch.load('./data/adj_matrices/Cora.pt').to(device)

    model = LPCAModel if args.model_type == 'LPCA' else L2Model
    loss_fn = lpca_loss if args.model_type == 'LPCA' else L2_loss

    # Initialize the trainer
    trainer = Trainer(adj=adj, model_class=model, loss_fn=loss_fn, 
                      threshold=10e-5, num_epochs=args.num_epochs, optim_type=args.optim_type, 
                      device=device)
    
    # Train one model model
    trainer.train(args.rank)

    # Find the optimal rank within a range
    # trainer.find_optimal_rank(1, 50)
    # Initialize model and move it to GPU

