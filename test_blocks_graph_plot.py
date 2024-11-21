import json
import matplotlib.pyplot as plt
import numpy as np


# Load json data
with open("results/block_graphs/results.json", "r") as f:
    data = json.load(f)

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over models and create plots
for model, n_values in data["results"].items():
    for n, b_values in n_values.items():
        x = []
        y = []
        y_err = []
        
        for b, values in b_values.items():
            if b == "res_folder":  # Skip non-numeric keys
                continue
            
            # Calculate mean and standard deviation
            x.append(int(b))  # Convert key to integer for plotting
            mean = np.mean(values)
            std = np.std(values)
            y.append(mean)
            y_err.append(std)
        
        # Sort by x for a proper line plot
        x, y, y_err = zip(*sorted(zip(x, y, y_err)))
        
        # Plot points, error bars, and connecting line
        ax.errorbar(x, y, yerr=y_err, label=f"{model}", capsize=5, marker="o", linestyle="-", alpha=0.5)

# Customize the plot
ax.set_xlabel("Number of blocks", fontsize=12)
ax.set_ylabel("Mean Optimal Rank", fontsize=12)
ax.set_title("Mean EED on a synthetic graph of 100x100 with different number of blocks", fontsize=14)
ax.legend(title="Model and n")
ax.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
