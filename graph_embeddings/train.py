import torch
import torch.nn as nn
import torch.optim as optim

from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss


# ======================== Hyperparameters ========================
optim_type = 'lbfgs'
rank = 32
num_epochs = 1000
print_loss_interval = 100
# =================================================================

# Ensure CUDA is available and select device, if not check for Macbook Pro support (MPS) and finally use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Load and prepare your data
adj = torch.load('./data/adj_matrices/Cora.pt').to(device)
# shift adj matrix to -1's and +1's
adj_s = adj*2 - 1

n_row, n_col = adj.size()

# Initialize model and move it to GPU
model = LPCAModel(n_row, n_col, rank).to(device)

def closure():
    optimizer.zero_grad()
    logits = model.forward()  # Or simply model() if you have defined the forward method
    loss = lpca_loss(logits, adj_s)  # Ensure lpca_loss is compatible with PyTorch and returns a scalar tensor
    loss.backward()
    return loss

if optim_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=0.1)
elif optim_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=0.01)
elif optim_type == 'lbfgs':
    optimizer = optim.LBFGS(model.parameters(), lr=0.01)

for epoch in range(num_epochs):

    if optim_type == 'lbfgs':
        # LBFGS optimizer step takes the closure function and internally calls it multiple times
        loss = optimizer.step(closure)
    else: 
        # Forward pass
        optimizer.zero_grad()
        logits = model.forward()  # Or simply model() if you have defined the forward method
        loss = lpca_loss(logits, adj_s)  # Ensure lpca_loss is compatible with PyTorch and returns a scalar tensor
        loss.backward()
        optimizer.step()

        
    if epoch % print_loss_interval == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        # Compute and print the Frobenius norm for diagnostics
        with torch.no_grad():  # Ensure no gradients are computed in this block
            logits = model.forward()
            frobenius_loss = torch.norm(adj - logits, p='fro').item()
            clipped_logits = torch.clip(logits, min=0, max=1)
            frob_error_norm = torch.linalg.norm(clipped_logits - adj) / torch.linalg.norm(adj)
            print(f'Frobenius error: {frob_error_norm}')


# After training, retrieve parameters
with torch.no_grad():  # Ensure no gradients are computed in this block
    U, V = model.U.cpu().numpy(), model.V.cpu().numpy()

    