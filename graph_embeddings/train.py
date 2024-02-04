import torch
import torch.nn as nn
import torch.optim as optim

from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss
from graph_embeddings.utils.trainer import train


# Ensure CUDA is available and select device, if not check for Macbook Pro support (MPS) and finally use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')


if __name__ == '__main__':

    # ======================== Hyperparameters ========================
    optim_type = 'lbfgs'
    rank = 32
    num_epochs = 1000
    print_loss_interval = 10
    # =================================================================


    # Load and prepare your data
    adj = torch.load('./data/adj_matrices/Cora.pt').to(device)

    n_row, n_col = adj.size()
    # Initialize model and move it to MPS
    model = LPCAModel(n_row, n_col, rank).to(device)

    # Train the model
    train(adj, model, lpca_loss, "lbfgs", num_epochs, print_loss_interval, device)