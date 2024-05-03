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

class CaseControlLogisticLoss(BaseLoss):
    def __call__(self, 
                preds: torch.Tensor,        # link predictions: ordered according to [links, nonlinks] as output from CaseControlDataLoader
                num_links: int,             # number of links
                weight_coeffs: torch.Tensor
                ):
        
        device = preds.device
        # links
        z1 = torch.logaddexp(torch.zeros(num_links, device=device), -preds[:num_links])

        # nonlinks
        # z0 = -preds[num_links:] # mult by -1 => same as mult by shifted adjacency matrix index for nonlink
        # pdb.set_trace()
        z0 = torch.logaddexp(torch.zeros(preds.shape[0] - num_links, device=device), preds[num_links:]) 
        z0 *= weight_coeffs
        
        return z0.sum() + z1.sum()

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
    

class CaseControlHingeLoss(BaseLoss):
    def __call__(self, 
                preds: torch.Tensor,        # link predictions: ordered according to [links, nonlinks] as output from CaseControlDataLoader
                num_links: int,             # number of links
                weight_coeffs: torch.Tensor
                ):
        
        h_nonlinks = 1+preds[num_links:]
        h_links = 1-preds[:num_links]

        links_zero = torch.zeros_like(h_links)
        nonlinks_zero = torch.zeros_like(h_nonlinks)

        h_loss1 = (torch.max(h_nonlinks, nonlinks_zero) * weight_coeffs).sum()
        h_loss2 = torch.max(h_links, links_zero).sum()
        
        return h_loss1 + h_loss2


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