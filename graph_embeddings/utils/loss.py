import torch

def lpca_loss(logits, adj_s): # adj_s = shifted adj with -1's and +1's
    # loss function: eq.1  from Chanpuriya et al. (2020)
    z = logits*adj_s
    # ? log((1 + exp(-z))**(-1)) = log(1) - log(exp(-z)) = exp(0) + exp(-z)
    zero = torch.zeros_like(z)
    loss = (torch.logaddexp(zero,-z)).sum()
    return loss

def L2_loss(A_hat, adj_s):
    z = adj_s * A_hat
    zero = torch.zeros_like(z)
    loss = (torch.logaddexp(zero,-z)).sum()

    return loss

# ? naïve (but closely resembling the mathematical terms) 
# ? code used to test the more efficient L2_loss(·,·) function above.
# def L2_loss_naive(logits, adj_s):
#     X,Y,beta = logits
    
#     norm_acc = torch.zeros((X.shape[0], Y.shape[1]))
#     for i in range(X.shape[0]):
#         for j in range(Y.shape[1]):
#             norm_acc[i,j] = (X[i,:] - Y[:,j]).norm(p=2)

#     z = adj_s * (beta - norm_acc)

#     zero = torch.zeros_like(z)
#     loss = (torch.logaddexp(zero,-z)).sum()

#     return loss