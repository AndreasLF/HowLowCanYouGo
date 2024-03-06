import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# Euclidean embedding model
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
        self.beta = nn.Parameter(torch.randn(1).to(device)) # (scalar) free parameter bias term
        self.S = None # ! only set if pretraining on SVD objective

    @classmethod
    def init_random(cls, 
                    n_row: int, 
                    n_col: int, 
                    rank: int):
        """
        Initializes the low rank approximation tensors,
            with values drawn from std. gaussian distribution.
        """
        X = torch.randn(n_row, rank)
        Y = torch.randn(n_col, rank)
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
    
    @classmethod
    def init_pre_mds(cls,
                     A: torch.Tensor,
                     rank: int,
                     device: str="cpu"):
        n, _ = A.size()

        dist = 1. - A
        C = torch.eye(n).to(device) - torch.ones((n, n)).to(device) / n        
        B = - C @ torch.square(dist) @ C

        eigenvalues, eigenvectors = torch.linalg.eigh(B) # eigh because B symmetric
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        L = torch.diag(torch.sqrt(eigenvalues[:rank]))
        E = eigenvectors[:, :rank]

        Y = E @ L
        # X = Y.detach().clone() 
        X = Y.detach().clone() + torch.randn_like(Y) * 1e-1 # randn for breaking symmetry - otherwise identical gradient updates are computed at each training step.
        return cls(X=X, Y=Y, device=device, inference_only=False)

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
        Y = V @ S_inv_sqrt
        return cls(X,Y)

    # ? multi-dimensional scaling for L2 model instead? 
    def reconstruct(self):
        if self.S is not None:
            # _S = softplus(_S) for nonneg # _S = _S**(1/2) as it is mult on both matrices
            _S = torch.diag(torch.sqrt(F.softplus(self.S)))
            norms = torch.norm((self.X@_S)[:,None] - (self.Y@_S), p=2, dim=-1)
            # norms = torch.cdist(self.X@_S, self.Y@_S, p=2)
        else:
            norms = torch.norm(self.X[:,None] - self.Y, p=2, dim=-1) # ? seems like better training than with cdist, why?
            # norms = torch.cdist(self.X, self.Y, p=2)
        A_hat = - norms + self.beta
        return A_hat

    def forward(self):
        if self.S is not None: # during pretraining, i.e. SVD target
            return self.X, self.Y, self.S
        
        return self.X, self.Y, self.beta
    
