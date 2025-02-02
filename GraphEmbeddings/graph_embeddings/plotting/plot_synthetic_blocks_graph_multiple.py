import os
import json
import numpy as np
import matplotlib.pyplot as plt

from graph_embeddings.examples import create_1D_example
import argparse



# argument
parser = argparse.ArgumentParser()
parser.add_argument('--graph-grid', type=bool, default=False, help='Whether to plot the graph grid')

args = parser.parse_args()

colors = plt.cm.Paired.colors  # Use a color map for different subfolders


if not args.graph_grid:
    # Define the directory containing the data
    data_dir = "results/fig_2"  # Replace with the path to your folder

    error_bar_style = {
        "capsize": 5,          # Length of the error bar caps
        "capthick": 1.5,       # Thickness of the caps
        "elinewidth": 1.5,     # Thickness of the error bar lines
        "color": "black",      # Color of the error bars
    }

    # Initialize data storage
    data = {}
    # Traverse the folder structure and read JSON files
    for subfolder in sorted(os.listdir(data_dir), key=lambda x: int(x.split("_B")[-1])):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            data[subfolder] = {}
            for file in os.listdir(subfolder_path):
                if file.endswith(".json"):
                    model_name = file.replace("counts_", "").replace(".json", "")
                    file_path = os.path.join(subfolder_path, file)
                    with open(file_path, "r") as f:
                        counts = json.load(f)
                    # Store counts for the model in the subfolder
                    data[subfolder][model_name] = counts


    # Prepare data for plotting
    model_names = list({model for subfolder in data.values() for model in subfolder})
    x = np.arange(len(model_names))  # x positions for the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    offset = 0

    for idx, (subfolder, models) in enumerate(data.items()):
        # extract everythin after B from subfolder
        label = subfolder.split("B")[-1]
        label = f"{label} blocks"
        means = []
        std_devs = []
        for model in model_names:
            counts = models.get(model, {})
            if counts:
                # Create a list of ranks repeated by their counts
                data_points = []
                for rank, count in counts.items():
                    data_points.extend([int(rank)] * count)
                mean_rank = np.mean(data_points) if data_points else 0
                std_rank = np.std(data_points) if data_points else 0
                means.append(mean_rank)
                std_devs.append(std_rank)
            else:
                means.append(0)
                std_devs.append(0)
        # Plot bars for this subfolder
        ax.bar(x + offset, means, bar_width, label=label, color=colors[idx % len(colors)], yerr=std_devs, edgecolor='white',error_kw=error_bar_style )
        offset += bar_width

    # Customizing the plot
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("EED", fontsize=12)
    ax.set_title("Mean EED observations with standard deviation", fontsize=14)
    ax.set_xticks(x + bar_width / 2 * (len(data) - 1))
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend(title="Number of blocks")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.savefig("figures/fig_2_hist.pdf", format="pdf", bbox_inches='tight')

else:
    # create synthetic grahps and plot in 4x4 grid

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    # flatten axs to make it easier to iterate over
    axs = axs.flatten()

    num_blocks = [2, 5, 10, 25]
    N = 50
    for i, b in enumerate(num_blocks):
        toy_data = create_1D_example(N, num_blocks=b)
        ax = axs[i]
        ax.spy(toy_data, markersize=1.8, color=colors[i % len(colors)])
        ax.set_title(f"{b} blocks")

    plt.tight_layout()
    plt.savefig("figures/fig_2_grid.pdf", format="pdf", bbox_inches='tight')
