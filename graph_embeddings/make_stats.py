import numpy as np
import scipy.sparse
import torch
import networkx as nx
from graph_embeddings.utils.config import Config
import pandas as pd
import os 


def get_stats(adj): 
    # subtract diagonal
    adj = adj - torch.eye(adj.shape[0])

    G = nx.Graph()

    # Add edges from the sparse matrix
    rows, cols = scipy.sparse.find(adj)[:2]  # Get the non-zero indices
    for row, col in zip(rows, cols):
        if row != col:
            G.add_edge(row, col)

    # Calculate the number of edges
    adj_ = adj - torch.eye(adj.shape[0])
    # calculate number of edges, directed graph
    num_edges = adj_.sum()

    # Get a list of all degrees
    degrees = [degree for node, degree in G.degree()]

    # Calculate the average degree
    average_degree = np.mean(degrees)

    #Calculate the 95th percentile
    percentile_95 = np.percentile(degrees, 95)

    # Global Clustering Coefficient
    clustering_coefficient = nx.average_clustering(G)

    # Assortativity
    assortativity = nx.degree_pearson_correlation_coefficient(G)

    # Calculate the number of triangles for each node
    triangles_per_node = nx.triangles(G)

    # Calculate the total number of triangles in the graph
    total_triangles = sum(triangles_per_node.values()) // 3


    return {
        "num_nodes": adj.shape[0],
        "num_edges": num_edges.item(),
        "average_degree": average_degree.item(),
        "95_percentile_degree": percentile_95,
        "density": nx.density(G),
        "num_connected_components": nx.number_connected_components(G),
        "clustering_coefficient": clustering_coefficient,
        "assortativity": assortativity,
        "total_triangles": total_triangles
    }

if __name__ == "__main__":
    cfg = Config("configs/config.yaml")

    datasets = cfg.get("data", "dataset_src").keys()
    adj_matrix_path = cfg.get("data", "adj_matrices_path")
    stats_path = cfg.get("results", "stats_path")

    os.makedirs(stats_path, exist_ok=True)

    # Try to load the existing CSV file, if it doesn't exist, create an empty DataFrame
    try:
        existing_df = pd.read_csv("adj_matrix_stats.csv")
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Check if this dataset is already in the existing DataFrame
        if not existing_df.empty and dataset in existing_df['Dataset'].values:
            print(f"Stats for {dataset} already exist in CSV, skipping...")
            continue  # Skip this dataset

        adj_matrix = torch.load(f"{adj_matrix_path}/{dataset}.pt")
        stats = get_stats(adj_matrix)  # get_stats returns a dictionary with the stats

        # Convert the stats dictionary to a DataFrame
        df = pd.DataFrame.from_dict({dataset: stats}, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Dataset"}, inplace=True)

        # Append the DataFrame to the CSV file
        if existing_df.empty:
            df.to_csv(f"{stats_path}/adj_matrix_stats.csv", index=False)  # Write with header if file was just created
        else:
            df.to_csv(f"{stats_path}/adj_matrix_stats.csv", mode='a', header=False, index=False)  # Append without header if file already exists

        # Optionally, you can read the CSV again to update the existing_df variable
        existing_df = pd.read_csv(f"{stats_path}/adj_matrix_stats.csv")

    # Load and print the CSV file content to verify
    df_loaded = pd.read_csv(f"{stats_path}/adj_matrix_stats.csv")
    print(df_loaded.head())