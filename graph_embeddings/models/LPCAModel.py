import torch
import torch.nn as nn
import torch.nn.functional as F

# Logistic PCA model
class LPCAModel(nn.Module):
    def __init__(self, 
                 X: torch.Tensor, 
                 Y: torch.Tensor, 
                 device: str="cpu",
                 inference_only: bool=False):
        super(LPCAModel, self).__init__()
        if X.shape == Y.shape: Y = Y.t() # 2D transpose
        X = X.to(device)
        Y = Y.to(device)
        self.X = nn.Parameter(X) if not inference_only else X
        self.Y = nn.Parameter(Y) if not inference_only else Y
        self.S = None # ! only set if pretraining on SVD objective

    @classmethod    
    def init_random(cls, n_row: int, n_col: int, rank: int):
        X = torch.randn(n_row, rank)
        Y = torch.randn(rank, n_col)
        return cls(X,Y)

    """
    Method 1 of 2. [together with init_post_svd]
    Initialize a model for pre-training (to improve initialization point), 
        with unitary matrices U and V from SVD on A, for learning S.
    """
    @classmethod
    def init_pre_svd(cls, 
                     U: torch.Tensor, 
                     V: torch.Tensor, 
                     device: str="cpu"):
        assert U.shape == V.shape, "U & V must be dimensions (n,r) & (n,r), respectively, r: emb. rank, n: # of nodes"
        model = cls(U, V, device=device, inference_only=True) # we only learn S in A = USV^T
        S = torch.randn(U.shape[1]).to(device)
        model.S = nn.Parameter(S)
        return model
    
    """
    Method 2 of 2. [together with init_pre_svd]
    Initialize a model for further training, using U, V and learned S to 
        compute an improved initialization point.
    """
    @classmethod
    def init_post_svd(cls, 
                      U: torch.Tensor, 
                      V: torch.Tensor, 
                      S: torch.Tensor):
        S_inv_sqrt = torch.diag(torch.sqrt(F.softplus(S)) ** (-1))
        X = U @ S_inv_sqrt
        Y = S_inv_sqrt @ V
        return cls(X,Y)

    def reconstruct(self):
        if self.S is not None:
            # _S = softplus(_S) for nonneg
            _S = torch.diag(F.softplus(self.S))
            A_hat = self.X @ _S @ self.Y
        else:
            A_hat = self.X @ self.Y
        return A_hat

    def forward(self):
        if self.S is not None: # during pretraining, i.e. SVD target
            return self.X, self.Y, self.S
        return self.X, self.Y