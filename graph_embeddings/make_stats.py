import numpy as np
import scipy.sparse
import torch
import networkx as nx
from graph_embeddings.utils.config import Config
import pandas as pd
import os 
import torch_geometric
import pdb

from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
from torch_geometric.utils import to_dense_adj

def get_stats(edge_index, num_nodes):
    # Create a sparse tensor representation from edge_index
    adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj.setdiag(0)  # Remove self-loops
    adj.eliminate_zeros()  # Clean up matrix after modification

    is_symmetric = (adj != adj.T).nnz == 0
    
    # Create a NetworkX graph from the adjacency matrix, based on whether it is symmetric or not
    if is_symmetric:
        G = nx.Graph(adj)  # Convert to a NetworkX graph by passing the matrix directly  
    else:
        G = nx.DiGraph(adj)  

    # make undirected graph for some stats
    G_undirected = G.to_undirected()

    # Number of edges
    num_edges = adj.nnz / 2  # Since it's undirected

    # Calculate the average degree
    degrees = np.array(adj.sum(axis=1)).flatten()
    average_degree = np.mean(degrees)
    
    # 95 percentile degree
    percentile_95 = np.percentile(degrees, 95)
    max_degree = int(max(degrees))

    # Global Clustering Coefficient (NetworkX)
    clustering_coefficient = nx.average_clustering(G)
    
    # Assortativity
    assortativity = nx.degree_pearson_correlation_coefficient(G)

    # Calculate the number of triangles for each node
    triangles_per_node = nx.triangles(G_undirected)

    # Calculate the total number of triangles in the graph
    total_triangles = sum(triangles_per_node.values()) // 3

    # Calculate graph density and number of connected components
    density = nx.density(G)
    num_connected_components = nx.number_connected_components(G_undirected)



    ab = []
    # loop through connected components
    for cc in nx.connected_components(G_undirected):
        G_sub = G_undirected.subgraph(cc)
        # calculate the number of nodes in the connected component
        n_nodes = len(G_sub.nodes)
        # calculate density of the connected component
        density = nx.density(G_sub)

        ab.append(density * n_nodes)

    # arboricity bound
    ab_max = max(ab)
    
    return {
        "directed/undirected": "undirected" if is_symmetric else "directed", # "directed" if directed else "undirected
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": average_degree,
        "95_percentile_degree": percentile_95,
        "density": density,
        "num_connected_components": num_connected_components,
        "clustering_coefficient": clustering_coefficient,
        "assortativity": assortativity,
        "total_triangles": total_triangles,
        "max_degree": max_degree,
        "arbocity_bound": ab_max
    }

if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser(description='Arguments for make_stats.py')
    parser.add_argument('--print-latex', action='store_true', help='Print the stats in LaTeX format')
    args = parser.parse_args()

    cfg = Config("configs/config.yaml")

    datasets = cfg.get("data", "dataset_src").values()
    adj_matrix_path = cfg.get("data", "adj_matrices_path")
    stats_path = cfg.get("results", "stats_path")

    os.makedirs(stats_path, exist_ok=True)

    # Try to load the existing CSV file, if it doesn't exist, create an empty DataFrame
    try:
        existing_df = pd.read_csv(f"{stats_path}/adj_matrix_stats.csv")
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    for dataset in datasets:
        # split the dataset name
        datasetsplit = dataset.split("/")
        dataset_name = datasetsplit[-1]

        print(f"Processing dataset: {dataset}")

        # Check if this dataset is already in the existing DataFrame
        if not existing_df.empty and dataset_name in existing_df['Dataset'].values:
            print(f"Stats for {dataset} already exist in CSV, skipping...")
            continue  # Skip this dataset

        # adj_matrix = torch.load(f"{adj_matrix_path}/{dataset}.pt")
        graph = get_data_from_torch_geometric(datasetsplit[-2], dataset_name, cfg.get("data", "raw_path"))[0]
        # adj_matrix = to_dense_adj(graph.edge_index).squeeze(0)

        stats = get_stats(graph.edge_index, graph.num_nodes)

        # Convert the stats dictionary to a DataFrame
        df = pd.DataFrame.from_dict({dataset_name: stats}, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Dataset"}, inplace=True)

        # Append the DataFrame to the CSV file
        if existing_df.empty:
            df.to_csv(f"{stats_path}/adj_matrix_stats.csv", index=False)  # Write with header if file was just created
        else:
            df.to_csv(f"{stats_path}/adj_matrix_stats.csv", mode='a', header=False, index=False)  # Append without header if file already exists

        # Optionally, read the CSV again to update the existing_df variable
        existing_df = pd.read_csv(f"{stats_path}/adj_matrix_stats.csv")

    # Load and print the CSV file content to verify
    df_loaded = pd.read_csv(f"{stats_path}/adj_matrix_stats.csv")
    print(df_loaded.head())


    if args.print_latex:

        keep_columns = ["Dataset", "num_nodes", "directed/undirected", "average_degree", 
                        "max_degree", "num_connected_components", "total_triangles", "arbocity_bound"]

        df_loaded = df_loaded[keep_columns]


        # update max degree to string and add ae in parenthesis to this column string
        df_loaded["max_degree"] = df_loaded.apply(
            lambda row: f"{row['max_degree']} / {np.round(row['arbocity_bound'],2)}" if row['directed/undirected'] == "undirected" else row['max_degree'], axis=1
        )

        # drop arbicity bound column 
        df_loaded.drop(columns=["arbocity_bound"], inplace=True)

        # rename columns
        df_loaded.rename(columns={
            "num_nodes": "Nodes", 
            "directed/undirected": "Type", 
            "average_degree": "Avg.\\newline Degree",
            # "95_percentile_degree": "95th Percentile\\newline Degree",
            "max_degree": "Max Degree",
            "num_connected_components": "Connected\\newline Components",
            "total_triangles": "Total\\newline Triangles"
        }, inplace=True)


        dataset_citations = {
            "Cora": r"Cora \citep{Yang2016RevisitingEmbeddings}",
            "PubMed": r"Pubmed \citep{Yang2016RevisitingEmbeddings}",
            "CiteSeer": r"Citeseer \citep{Yang2016RevisitingEmbeddings}",
            "Facebook": r"Facebook \citep{Yang2020PANE:Embedding}",
            "ca-HepPh": r"ca-HepPh \citep{Leskovec2005GraphsExplanations, Gehrke2003OverviewCup}",
            "p2p-Gnutella04": r"p2p-Gnutella04 \citep{Leskovec2007GraphEvolution, Ripeanu2002MappingDesign}",
            "wiki-Vote": r"Wiki-Vote \citep{Leskovec2010SignedMedia, Leskovec2010PredictingNetworks}",
            "ca-GrQc": r"ca-GrQc \cite{Leskovec2007GraphEvolution}"
        }

        # NOTE uncomment this to add the citation to the dataset
        # df_loaded["Dataset"] = df_loaded["Dataset"].replace(dataset_citations)

        # sort dataframe by Nodes
        df_loaded.sort_values(by="Nodes", inplace=True)

        col_format = "l|rcrrrr"
        # dont't use toprule, midrule, bottomrule, instead use \hline
        # latex_code = df_loaded.to_latex(index=False, float_format="%.2f", column_format="l|cccccc")
        latex_code = df_loaded.to_latex(index=False, column_format=col_format, escape=False, header=False, float_format="%.2f")

        # Construct the custom header
        custom_header = r"""
        \toprule
        \multicolumn{1}{c|}{Dataset} & \multicolumn{1}{c}{Nodes} & \multicolumn{1}{c}{Type} & \multicolumn{1}{c}{Avg.} & \multicolumn{1}{c}{Max Degree /} & \multicolumn{1}{c}{Connected} & \multicolumn{1}{c}{Total} \\
        \multicolumn{1}{c|}{} & \multicolumn{1}{c}{} & \multicolumn{1}{c}{} & \multicolumn{1}{c}{Degree} & \multicolumn{1}{c}{Componentwise Arboricity} & \multicolumn{1}{c}{Components} & \multicolumn{1}{c}{Triangles} \\
        \midrule
        """


        # Adjust LaTeX code to remove default top and mid rules added by pandas
        # Split on first occurrence of '\midrule' and take the second part (data rows and below)
        body_start_index = latex_code.find('\\midrule') + 7  # 7 is the length of '\midrule'
        latex_code_body = latex_code[body_start_index:]

        # Remove any other top or mid rules from the body
        latex_code_body = latex_code_body.replace('\\toprule', '').replace('\\midrule', '')

        # Combine custom header with the body
        full_latex_code = f"\\begin{{tabular}}{{{col_format}}}" + custom_header + latex_code_body

        print(full_latex_code)