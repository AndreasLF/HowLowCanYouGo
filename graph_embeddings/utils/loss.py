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
        h2 = self.zero if self.zero.shape == h1.shape else torch.zeros_like(h1)
        self.zero = h2
        h_loss = torch.max(h1, h2).sum()
        return h_loss


class PoissonLoss(BaseLoss):
    def __call__(self, 
                 A_hat: torch.Tensor, 
                 A: torch.Tensor
                 ):
        """
        A_hat: torch.Tensor - reconstruction of adj. matrix.
        A: torch.Tensor - adj. matrix.
        """
        p1 = A*A_hat
        p2 = torch.exp(A_hat)
        p_loss = (p2 - p1).sum()
        return p_loss

class SimpleLoss(BaseLoss):
    def __call__(self,
                 A_hat: torch.Tensor,
                 adj: torch.Tensor
                 ):
        """
        Does not work.
        """
        adj_sum = adj.sum()
        weight = 1 + (adj.numel() - adj_sum) / adj_sum
        return ((1. - weight * adj)*A_hat).sum()