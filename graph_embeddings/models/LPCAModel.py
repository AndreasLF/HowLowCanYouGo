import torch
import torch.nn as nn

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