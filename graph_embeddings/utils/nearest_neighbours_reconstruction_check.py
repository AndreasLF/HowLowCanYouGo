import torch
from sklearn.neighbors import NearestNeighbors

def get_edge_index_embeddings(X, Y, beta):

    # Define nearest neighbors object
    nn = NearestNeighbors(radius=beta.item(), metric='euclidean', algorithm='kd_tree', n_jobs=-1)
    # fit the model
    nn.fit(Y.cpu().detach().numpy())

    # Get the neighbors of Y
    neighbors = nn.radius_neighbors(X.cpu().detach().numpy(), return_distance=False)

    # Convert neighbors to a padded tensor
    # Assuming neighbors_X is a list of tensors or lists
    neighbors_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(neighbors, dtype=torch.long) for neighbors in neighbors], batch_first=True, padding_value=-1)

    #  Create a tensor of source indices
    num_neighbors = neighbors_tensor.size(1)
    sources_expanded = torch.arange(neighbors_tensor.size(0)).unsqueeze(-1).expand(-1, num_neighbors).reshape(-1)

    # Flatten the neighbors tensor and filter out padding values (-1)
    targets_flattened = neighbors_tensor.reshape(-1)
    valid_mask = targets_flattened != -1  # Mask to filter out padding
    sources_filtered = sources_expanded[valid_mask]
    targets_filtered = targets_flattened[valid_mask]

    # Remove self-loops
    self_loop_mask = sources_filtered != targets_filtered
    sources_final = sources_filtered[self_loop_mask]
    targets_final = targets_filtered[self_loop_mask]

    # Construct the edge index matrix
    edge_index_from_neighbors = torch.stack((sources_final, targets_final), dim=0)

    return edge_index_from_neighbors

def compare_edge_indices(edge_index1, edge_index2):
    # Convert edge indices to sets of tuples
    edge_set1 = set(tuple(edge) for edge in edge_index1.t().tolist())
    edge_set2 = set(tuple(edge) for edge in edge_index2.t().tolist())

    # Find common edges
    common_edges = edge_set1.intersection(edge_set2)

    return common_edges