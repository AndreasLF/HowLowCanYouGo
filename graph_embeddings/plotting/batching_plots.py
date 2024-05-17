# %% 
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def fetch_run_data(run):
    batch_size = run.config.get("batch_size")
    batch_type = run.config.get("batch_type")
    rank = run.config.get("rank")
    
    history_df = run.history()
    frobs = history_df["frob_error_norm"]
    epochs = history_df["epoch"]

    # Check the epoch differences
    if len(epochs) > 1:
        epoch_diff = [epochs[i+1] - epochs[i] for i in range(len(epochs)-1)]
        if any([diff != eval_recon_freq for diff in epoch_diff]):
            return None

    return rank, (batch_type, batch_size), (list(epochs), list(frobs))

def get_data_from_wandb(data="Cora", model_class="L2Model", loss_fn=["LogisticLoss", "CaseControlLogisticLoss"], start_date=None, eval_recon_freq=20):
        
    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    print("Fetching runs...")
    # filter on all runs in the project

    # Define the filters for the config parameters
    filters = {
        "config.data": data,
        "config.model_class": model_class,
        "config.loss_fn": {"$in": loss_fn}
    }    
    if start_date:
        filters["created_at"] = {"$gt": start_date}
    matching_runs = api.runs(path=project_name, filters=filters)

    # get all unique batch_size values
    batch_sizes = set([run.config.get("batch_size") for run in matching_runs])


    print("Fetching data on each run...")

    frob_error_norms = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_run_data, run): run.id for run in matching_runs}
        for future in tqdm(futures):
            result = future.result()
            if result is None:
                continue
            rank, batch_info, frob_data = result
            if rank not in frob_error_norms:
                frob_error_norms[rank] = {}
            if batch_info not in frob_error_norms[rank]:
                frob_error_norms[rank][batch_info] = []
            frob_error_norms[rank][batch_info].append(frob_data)

    print("Data fetching complete.")

    return frob_error_norms, batch_sizes
# %%

if __name__ == "__main__":
    model = "L2Model"
    data = "PubMed"
    loss_fn = ["LogisticLoss", "CaseControlLogisticLoss"]
    eval_recon_freq = 20
    start_date = "2024-05-17" # NOTE V2 of experiments, no diagonal
    frob_error_norms, batch_sizes = get_data_from_wandb(data=data, model_class=model, loss_fn=loss_fn, start_date=start_date, eval_recon_freq=eval_recon_freq)


# %%
    import matplotlib.pyplot as plt
    from cycler import cycler
    from graph_embeddings.plotting.plotter import PaperStylePlotter
    import numpy as np


    def pad_list(lst, target_len, pad_value=np.nan):
        return lst + [pad_value] * (target_len - len(lst))


    with PaperStylePlotter().apply():
        # fig, ax = plt.subplots()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']

        # colors = [colors[0], colors[5], colors[2]]

        colors_dict = {('RandomNodeDataLoader', 1971): colors[0], 
                    ('RandomNodeDataLoader', 19717): colors[5], 
                    ('CaseControlDataLoader', 1971): colors[2]}
        
        linestyles_dict = {('RandomNodeDataLoader', 1971): linestyles[0],
                    ('RandomNodeDataLoader', 19717): linestyles[1],
                    ('CaseControlDataLoader', 1971): linestyles[2]}

        # Define the number of plots
        n_ranks = len(frob_error_norms)
        cols = 3  # Number of columns in the grid
        rows = (n_ranks + cols - 1) // cols  # Calculate rows needed

        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()  # Flatten to easily iterate


        # reverse sort frob_error_norms by key
        frob_error_norms_sorted = dict(sorted(frob_error_norms.items(), reverse=True))
    
        max_epocs = 0

        for i, (rank, batch_type_exp) in enumerate(frob_error_norms_sorted.items()):
            
            ax = axes[i]
            ax.set_title(f"Rank: {rank}")
            
            for j, (unique_tuple, value_list) in enumerate(batch_type_exp.items()):

                max_len = max(len(y) for _, y in value_list)

                color = colors_dict[unique_tuple]
                linestyle = linestyles_dict[unique_tuple]
                all_y_values = []
                
                longest_epoch_list = []

                for (epoch_list_x, loss_y) in value_list:
                    # ax.plot(epoch_list_x, loss_y, color=color, alpha=0.5)

                    # update max epochs
                    max_epocs = max(max_epocs, max(epoch_list_x))

                    # update longest epoch list
                    longest_epoch_list = epoch_list_x if len(epoch_list_x) > len(longest_epoch_list) else longest_epoch_list

                    ax.plot(epoch_list_x, loss_y, color=color, alpha=0.3, linestyle=linestyle)
                    
                    padded_loss_y = pad_list(loss_y, max_len)
                    padded_epoch_x = pad_list(epoch_list_x, max_len)
                    all_y_values.append(np.array(padded_loss_y))
                # Calculate and plot the mean
                # mean_y = np.mean(all_y_values, axis=0)
                # ax.plot(epoch_list_x, mean_y, color=color, linewidth=2, label=f'{unique_tuple} mean' if i == 0 else "")

                # Calculate and plot the mean

                mean_y = np.nanmean(np.array(all_y_values), axis=0)
                lab = "Random Node Sampling (10%)" if unique_tuple[0] == "RandomNodeDataLoader" and unique_tuple[1] != 19717 else "Negative Sampling (10%)" if unique_tuple[0] == "CaseControlDataLoader" else "Full dataset"
                ax.plot(longest_epoch_list, mean_y, color=color, label=lab, linestyle=linestyle)
                
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Frob error norm')
            ax.legend()
            
        # Handle legend outside of loop
        first_ax = axes[0]
        handles, labels = first_ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='lower right')

        # set all x limits to max epoch
        for ax in axes:
            ax.set_xlim([0, max_epocs])

        # plot title 
        fig.suptitle(f"Frobenius Errors for different batch types at all the ranks in the search\n({data}, {model})", fontsize=16)
        plt.tight_layout()

        plt.show()
        PaperStylePlotter().save_fig(fig, "batching_frob_error_norms")

# %%
