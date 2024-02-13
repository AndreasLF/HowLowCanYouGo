import torch
import torch.nn as nn

# Define the model
class LPCAModel(nn.Module):
    def __init__(self, n_row, n_col, rank, device="cpu"):
        super(LPCAModel, self).__init__()
        self.X = nn.Parameter(torch.randn(n_row, rank).to(device))
        self.Y = nn.Parameter(torch.randn(rank, n_col).to(device))

    def reconstruct(self):
        A_hat = self.X @ self.Y
        return A_hat

    def forward(self):
        return self.X, self.Y