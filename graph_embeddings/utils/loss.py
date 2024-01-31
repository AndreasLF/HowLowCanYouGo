import torch

def lpca_loss(logits, adj_s): # adj_s = shifted adj with -1's and +1's
    # loss function: eq.1  from Chanpuriya et al. (2020)
    z = logits*adj_s
    # log((1 + exp(-z))**(-1)) = log(1) - log(exp(-z)) = exp(0) + exp(-z)
    zero = torch.zeros_like(z)
    loss = (torch.logaddexp(zero,-z)).sum()
    return loss
