import torch
import torch.nn as nn
import torch.optim as optim


# Ensure CUDA is available and select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
class LPCAModel(nn.Module):
    def __init__(self, n_row, n_col, rank):
        super(LPCAModel, self).__init__()
        self.U = nn.Parameter(torch.randn(n_row, rank).to(device))
        self.V = nn.Parameter(torch.randn(rank, n_col).to(device))

    def forward(self):
        logits = self.U @ self.V
        return logits



def lpca_loss(logits, adj_s): # adj_s = shifted adj with -1's and +1's
    # loss function: eq.1  from Chanpuriya et al. (2020)
    z = logits*adj_s
    # log((1 + exp(-z))**(-1)) = log(1) - log(exp(-z)) = exp(0) + exp(-z)
    zero = torch.zeros_like(z)
    loss = (torch.logaddexp(zero,-z)).sum()
    return loss

# Load and prepare your data
adj = torch.load('./data/adj_matrices/Cora.pt').to(device)
# shift adj matrix to -1's and +1's
adj_s = adj*2 - 1

n_row, n_col = adj.size()
rank = 32

# Initialize model and move it to GPU
model = LPCAModel(n_row, n_col, rank).to(device)

# Define optimizer
optimizer = optim.LBFGS(model.parameters(), lr=0.01)

def closure():
    optimizer.zero_grad()
    logits = model.forward()  # Or simply model() if you have defined the forward method
    loss = lpca_loss(logits, adj_s)  # Ensure lpca_loss is compatible with PyTorch and returns a scalar tensor
    loss.backward()
    return loss

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # LBFGS optimizer step takes the closure function and internally calls it multiple times
    loss = optimizer.step(closure)

    if epoch % 1 == 0:
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
