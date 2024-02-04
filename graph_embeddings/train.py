import torch
import torch.nn as nn
import torch.optim as optim

from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss
from graph_embeddings.utils.trainer import Trainer


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

    # Initialize the trainer
    trainer = Trainer(adj=adj, model_class=LPCAModel, loss_funct=lpca_loss, 
                      threshold=10e-5, num_epochs=num_epochs, optim_type=optim_type, 
                      device=device)
    
    # Train the model
    trainer.train(rank, print_loss_interval)
