# %% 
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm

def get_data_from_wandb(data="Cora", model_class="L2Model", rank=6):
        
    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    # %%
    print("Fetching runs...")
    # filter on all runs in the project
    runs = api.runs(path=project_name)

    # data should be = Cora, model_class = L2, rank = 6
    print("Filtering runs...")
    matching_runs = [run for run in tqdm(runs) if run.config.get("data") == data and run.config.get("model_class") == model_class and run.config.get("rank") == rank]

    # %%

    # get all unique batch_size values
    batch_sizes = set([run.config.get("batch_size") for run in matching_runs])


    print("Fetching data on each run...")
    frob_error_norms = {}
    for run in tqdm(matching_runs):
        batch_size = run.config.get("batch_size")
        if batch_size not in frob_error_norms:
            frob_error_norms[batch_size] = []

        # append frob_error_norm history
        history_df = run.history()
        # get the epcoh column and frob_error_norm column

        frobs = history_df["frob_error_norm"]
        epochs = history_df["epoch"]

        frob_error_norms[batch_size].append((list(epochs), list(frobs)))

    return frob_error_norms, batch_sizes
# %%

if __name__ == "__main__":
    rank = 6
    model = "L2Model"
    data = "Cora"
    frob_error_norms, batch_sizes = get_data_from_wandb(data=data, model_class=model, rank=rank)

    import matplotlib.pyplot as plt
    from cycler import cycler
    from graph_embeddings.plotting.plotter import PaperStylePlotter
    import numpy as np

    with PaperStylePlotter().apply():
        fig, ax = plt.subplots()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']

        colors = [colors[0], colors[5]]


        for b, batch_size in enumerate(batch_sizes):

            color = colors[b]
            linestyle = linestyles[b]

            frob_error_norms_list = frob_error_norms[batch_size]
            epochs_list = [frob_error_norms_list[i][0] for i in range(len(frob_error_norms_list))]
            frobs_list = [frob_error_norms_list[i][1] for i in range(len(frob_error_norms_list))]

            max_length = max(len(frobs) for frobs in frobs_list)

            # Pad frobenius errors to have the same length
            padded_frobs_list = np.array([np.pad(frobs, (0, max_length - len(frobs)), 'constant', constant_values=np.nan) for frobs in frobs_list])

            # Calculate mean and standard deviation ignoring NaN
            mean_frobs = np.nanmean(padded_frobs_list, axis=0)
            std_frobs = np.nanstd(padded_frobs_list, axis=0)

            # Plot each individual run with lower opacity
            for epochs, frobs in zip(epochs_list, frobs_list):
                ax.plot(epochs, frobs, color=color, linestyle=linestyle, alpha=0.5, linewidth=0.5)

            # Plot mean line and fill the confidence area around the mean
            # get longest epoch list
            longest_epoch_list = epochs_list[np.argmax([len(epochs) for epochs in epochs_list])]

            label = f"{batch_size} (full)" if batch_size == max(batch_sizes) else f"{batch_size}"
            ax.plot(longest_epoch_list, mean_frobs, label=label, color=color, linestyle=linestyle, linewidth=1)

        ax.set_title(f"Frobenius Errors at Different Batch Sizes\n({data}, Rank {rank}, {model})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Frobenius Error")
        ax.legend(title="Batch Size Mean")

        # save figure 
        fig_name = "frob_error_norms"
        print("Saving figure to: ", fig_name)
        PaperStylePlotter().save_fig(fig, fig_name)


# %%
