import torch
import torch.nn as nn

# Define the model, in this case our Euclidean embedding model
class L2Model(nn.Module):
    def __init__(self, 
                 X: torch.Tensor, 
                 Y: torch.Tensor, 
                 device: str="cpu",
                 inference_only: bool=False):
        super(L2Model, self).__init__()
        X = X.to(device)
        Y = Y.to(device)
        self.X = nn.Parameter(X) if not inference_only else X
        self.Y = nn.Parameter(Y) if not inference_only else Y
        self.beta = nn.Parameter(torch.randn(1).to(device)) # scalar free parameter bias term
        self.S = None # ! only set if pretraining on SVD objective

    @classmethod
    def init_random(cls, 
                    n_row: int, 
                    n_col: int, 
                    rank: int):
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
        assert U.shape[1] == V.shape[0], "U & V must be dimensions (n,r) & (r,n), respectively, r: emb. rank, n: # of nodes"
        model = cls(U, V, inference_only=True) # we only learn S in A = USV^T
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
        S_inv_sqrt = S**(-1/2)
        X = S_inv_sqrt @ U
        Y = S_inv_sqrt @ V.T
        return cls(X,Y)

    def reconstruct(self):
        if self.S is not None:
            _S = torch.diag(self.S)
            norms = torch.norm((self.X@_S)[:,None] - (self.Y.T@_S), p=2, dim=-1)
        else:
            norms = torch.norm(self.X[:,None] - self.Y.T, p=2, dim=-1)
        A_hat = - norms + self.beta
        return A_hat

    def forward(self):
        return self.X, self.Y, self.beta
    
