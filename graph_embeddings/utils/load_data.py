import torch


def load_adj(adj_path):
    """
    Loads (unweighted) adjacency matrix (dense repr.) and ensures diagonal is ones.
    """
    adj = torch.load(adj_path)
    adj.fill_diagonal_(1)
    return adj