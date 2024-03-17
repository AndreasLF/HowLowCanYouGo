import torch

def pairwise_jaccard_similarity(A: torch.Tensor):
    """
    Compute the pairwise Jaccard similarity for an adjacency matrix A represented as a torch tensor.

    Args:
    A (torch.Tensor): An adjacency matrix.

    Returns:
    torch.Tensor: A matrix of pairwise Jaccard similarities.
    """
    # Compute the intersection and union for each pair of rows (nodes) in the matrix
    intersection = torch.mm(A, A.t())
    row_sums = A.sum(dim=1).reshape(-1, 1)
    union = row_sums + row_sums.t() - intersection

    # Handle division by zero in case of zero union
    jaccard_similarity = intersection / union
    jaccard_similarity[torch.isnan(jaccard_similarity)] = 0

    return jaccard_similarity

def norm_and_rescale(X: torch.Tensor, scale: float|None = None):
    X = X.float()

    norms = torch.sqrt(torch.sum(X**2, dim=1)) + 1e-7

    X_normalized = X / norms.unsqueeze(1)

    if scale is None:
        scale = torch.sqrt(torch.tensor(X.shape[1]))

    X_rescaled = X_normalized * scale

    return X_rescaled
