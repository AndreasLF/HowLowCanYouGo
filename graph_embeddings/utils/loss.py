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
    def __call__(self, A_hat, adj_s):
        h2 = 1 - adj_s*A_hat
        h1 = self.zero if self.zero.shape == h2.shape else torch.zeros_like(h2)
        self.zero = h1
        h_loss = torch.max(h1, h2).sum()
        return h_loss