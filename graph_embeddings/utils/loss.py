import pdb
import torch

class BaseLoss:
    def __init__(self):
        self.zero = torch.tensor([])

class LogisticLoss(BaseLoss):
    def __call__(self, 
                 A_hat: torch.Tensor, 
                 adj_s: torch.Tensor
                 ):
        z = adj_s * A_hat
        zero = self.zero if self.zero.shape == z.shape else torch.zeros_like(z)
        self.zero = zero
        l_loss = (torch.logaddexp(zero,-z)).sum()
        return l_loss

class HingeLoss(BaseLoss):
    def __call__(self, 
                 A_hat: torch.Tensor, 
                 adj_s: torch.Tensor
                 ):
        h1 = 1 - adj_s*A_hat
        h2 = self.zero if self.zero.shape == h2.shape else torch.zeros_like(h2)
        self.zero = h2
        h_loss = torch.max(h1, h2).sum()
        return h_loss


class PoissonLoss(BaseLoss):
    def __call__(self, 
                 A_hat: torch.Tensor, 
                 adj: torch.Tensor
                 ):
        """
        A_hat: torch.Tensor - reconstruction of adj. matrix.
        adj: torch.Tensor - adj. matrix.
        """
        p1 = adj*A_hat
        p2 = torch.exp(A_hat)
        p_loss = -(p1 + p2).sum()
        return p_loss