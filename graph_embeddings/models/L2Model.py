import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model, in this case our Euclidean embedding model
class L2Model(nn.Module):
    def __init__(self, n_row, n_col, rank):
        super(L2Model, self).__init__()
        self.X = nn.Parameter(torch.randn(n_row, rank).to(device))
        self.Y = nn.Parameter(torch.randn(rank, n_col).to(device))
        self.beta = nn.Parameter(torch.randn(1).to(device)) # scalar free parameter bias term

    def reconstruct(self, model_output):
        X,Y,beta = model_output
        norms = torch.norm(X[:,None] - Y.T, p=2, dim=-1)
        z = - norms + beta
        return z

    def forward(self):
        return self.X, self.Y, self.beta
    
