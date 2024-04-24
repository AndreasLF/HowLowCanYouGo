import torch

def difference_set(a, b):
    """
    Treat two tensors as sets, and compute the difference.
    """
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]

    return difference

def intersection_set(a: torch.Tensor, b: torch.Tensor):
    """
    Treat two tensors as sets, and compute the intersection.
    """
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]

    return intersection

def equals_set(a: torch.Tensor, b: torch.Tensor, return_frac: bool = False):
    """
    Treat two tensors (a,b) as sets, and compute whether they are equal.
    If return_percentage is set to True, return also how many percent of the 
        elements are identical.
    """
    combined = torch.hstack([a, b])
    uniques, counts = combined.unique(return_counts=True, dim=1)
    exists_in_both = counts == 2
    is_equal = torch.all(exists_in_both)
    
    if return_frac: return is_equal, exists_in_both.sum() / uniques.shape[1]
    else: return is_equal
