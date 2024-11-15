import pdb
import torch
import torch.nn as nn

# Hyperbolic embedding model
class HyperbolicModel(nn.Module):
    def __init__(self, 
                 X: torch.Tensor, 
                 Y: torch.Tensor, 
                 inference_only: bool=False):
        super(HyperbolicModel, self).__init__()
        self.X = nn.Parameter(X) if not inference_only else X
        self.Y = nn.Parameter(Y) if not inference_only else Y
        self.beta = nn.Parameter(torch.rand(1)) # (scalar) free parameter bias term

    @classmethod
    def init_random(cls, 
                    n_row: int, 
                    n_col: int, 
                    rank: int,
                    **kwargs):
        """
        Initializes the low rank approximation tensors,
            with values drawn from std. gaussian distribution.
        """
        X = torch.randn(n_row, rank)
        Y = torch.randn(n_col, rank)
        return cls(X,Y, **kwargs)
    

    def reconstruct(self, node_indices: torch.Tensor = None):
        if node_indices is not None:
            X = self.X[node_indices]
            Y = self.Y[node_indices]
        else:
            X = self.X
            Y = self.Y

        dists = self.poincare_distance(X, Y)
        A_hat = - dists + self.beta
    
        return A_hat
    
    
    def poincare_distance(self,u, v, eps=1e-5, all_pairs=True):
        """
        Compute the Poincaré distance between points u and v in the Poincaré ball model.
        
        Args:
            u (torch.Tensor): First point on the Poincaré ball (batch_size, dim).
            v (torch.Tensor): Second point on the Poincaré ball (batch_size, dim).
            eps (float): A small value to prevent division by zero.
            
        Returns:
            torch.Tensor: Poincaré distance between u and v.
        """
        # Norms of the vectors
        norm_u = torch.clamp(torch.norm(u, p=2, dim=-1), max=1 - eps)
        norm_v = torch.clamp(torch.norm(v, p=2, dim=-1), max=1 - eps)
        
        # Squared Euclidean distance between points
        if all_pairs:
            euclidean_dist_sq = torch.cdist(u,v)#torch.sum((u - v) ** 2, dim=-1)
        
            # Compute the Poincaré distance
            denom = (1 - norm_u**2).unsqueeze(1) * (1 - norm_v**2)
        else:
            euclidean_dist_sq = torch.sum((u - v) ** 2, dim=-1)
            denom = (1 - norm_u**2) * (1 - norm_v**2)
    
        arg = 1 + 2 * euclidean_dist_sq / denom
        poincare_dist = torch.acosh(torch.clamp(arg, min=1 + eps))
        
        return poincare_dist
    

    def forward(self):
        return self.X, self.Y, self.beta