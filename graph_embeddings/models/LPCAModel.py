import torch
import torch.nn as nn

# Define the model
class LPCAModel(nn.Module):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, device: str="cpu"):
        super(LPCAModel, self).__init__()
        self.X = nn.Parameter(X.to(device))
        self.Y = nn.Parameter(Y.to(device))

    @classmethod    
    def init_random(cls, n_row: int, n_col: int, rank: int):
        X = torch.randn(n_row, rank)
        Y = torch.randn(rank, n_col)
        return cls(X,Y)

    def reconstruct(self):
        A_hat = self.X @ self.Y
        return A_hat

    def forward(self):
        return self.X, self.Y