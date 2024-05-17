# %% 
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pdb

def fetch_run_data(run, eval_recon_freq=20):
    # print(run.name)
    batch_size = run.config.get("batch_size")
    batch_type = run.config.get("batch_type")
    rank = run.config.get("rank")
    
    history_df = run.history()
    frobs = history_df["frob_error_norm"]
    epochs = history_df["epoch"]

    # convert to list
    epochs = list(epochs)
    frobs = list(frobs)

    epochs = [int(epoch) for epoch in epochs]
    # Check the epoch differences
    if len(epochs) > 1:
        epoch_diff = [epochs[i+1] - epochs[i] for i in range(len(epochs)-1)]

    return rank, (list(epochs), list(frobs))

def get_data_from_wandb(data="Cora", model_class="L2Model", loss_fn=["LogisticLoss", "CaseControlLogisticLoss"], start_date=None, eval_recon_freq=20, cold_start=False):
        
    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    # filter on all runs in the project

    # Define the filters for the config parameters
    filters = {
        "config.data": data,
        "config.model_class": model_class,
        "config.loss_fn": {"$in": loss_fn},
        "config.batch_type": "RandomNodeDataLoader",
        "config.batch_size": 19717,
    }    

    if cold_start:
        filters["tags"] = "cold_start"
    # else tag not cold_start
    else:
        filters["tags"] = {"$ne": "cold_start"}
    if start_date:
        filters["created_at"] = {"$gt": start_date}
    matching_runs = api.runs(path=project_name, filters=filters)

    # get all unique batch_size values
    batch_sizes = set([run.config.get("batch_size") for run in matching_runs])


    print("Fetching data on each run...")

    frob_error_norms = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_run_data, run, eval_recon_freq): run.id for run in matching_runs}
        for future in tqdm(futures):
            result = future.result()
            if result is None:
                continue
            rank, frob_data = result
            if rank not in frob_error_norms:
                frob_error_norms[rank] = []
            frob_error_norms[rank].append(frob_data)

    print("Data fetching complete.")

    return frob_error_norms, batch_sizes

# %%

if __name__ == "__main__":
    model = "L2Model"
    data = "PubMed"
    loss_fn = ["LogisticLoss", "CaseControlLogisticLoss"]
    eval_recon_freq = 20
    start_date = "2024-05-17" # NOTE V2 of experiments, no diagonal

    print("Fetching warm start runs...")
    frob_error_norms_warm, batch_sizes = get_data_from_wandb(data=data, model_class=model, loss_fn=loss_fn, start_date=start_date, eval_recon_freq=eval_recon_freq, cold_start=False)
    # slice out the highest rank in frob_error_norms_warm
    frob_error_norms_warm = {k: v for k, v in frob_error_norms_warm.items() if k == max(frob_error_norms_warm.keys())}

    print("Fetching cold start runs...")
    frob_error_norms_cold, _ = get_data_from_wandb(data=data, model_class=model, loss_fn=loss_fn, start_date=start_date, eval_recon_freq=eval_recon_freq, cold_start=True)
    # %%
    import matplotlib.pyplot as plt
    from cycler import cycler
    from graph_embeddings.plotting.plotter import PaperStylePlotter
    import numpy as np

    def pad_with_nans(epochs, frob_errors, target_len):
        # Find the minimum and maximum values in the list
        min_val = min(epochs)
        max_val = max(epochs)
        
        epoch_frob_dict = dict(zip(epochs, frob_errors))

        # Generate the complete list of numbers including all multiples of 20 within the range
        complete_list = list(range(min_val, max_val + 1))
        
        # Generate all multiples of 20 within the range
        multiples_of_20 = list(range(min_val - min_val % 20 + 20, max_val + 1, 20))
        
        # Initialize the result list
        result = []
        
        # Iterate through the complete list and pad with nan where needed
        for num in complete_list:
            if num in epochs:
                result.append(epoch_frob_dict[num])
            elif num in multiples_of_20:
                result.append(np.nan)

        # Pad the result list with nan to reach the target length
        result += [np.nan] * (target_len - len(result))
        
        return result

    with PaperStylePlotter().apply():
        # fig, ax = plt.subplots()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']

        # colors = [colors[0], colors[5], colors[2]]

        colors_list = [colors[0], colors[5]]

        linestyles_list = [linestyles[0], linestyles[1]]
        
        # Define the number of plots
        n_ranks = max(len(frob_error_norms_cold), len(frob_error_norms_warm))
        cols = 3  # Number of columns in the grid
        rows = (n_ranks + cols - 1) // cols  # Calculate rows needed


        figsize = plt.rcParams['figure.figsize']

        fig, axes = plt.subplots(rows, cols)
        
        axes = axes.flatten()  # Flatten to easily iterate


        # reverse sort frob_error_norms by key
        frob_error_norms_warm_sorted = dict(sorted(frob_error_norms_warm.items(), reverse=True))
        frob_error_norms_cold_sorted = dict(sorted(frob_error_norms_cold.items(), reverse=True))
    
        all_rank = set(frob_error_norms_warm.keys()).union(set(frob_error_norms_cold.keys()))
        all_rank = sorted(list(all_rank), reverse=True)
        # sort and assign value from 0 to n_ranks
        rank_axis_dict = dict(zip(all_rank, range(n_ranks)))


        max_epochs = 0

        for i, (rank, frob_data) in enumerate(frob_error_norms_warm_sorted.items()):
            ax = axes[rank_axis_dict[rank]]
            ax.set_title(f"Rank: {rank}")

            all_frob_errors = []


            for j, (epochs, frob_erros) in enumerate(frob_data):
                max_epochs = max(max_epochs, max(epochs))
                ax.plot(epochs, frob_erros, label="Warm", color=colors_list[0], linestyle=linestyles_list[0], alpha=0.5, linewidth=.5)
                all_frob_errors.append(pad_with_nans(epochs, frob_erros, max([len(frob_erros) for epochs, frob_erros in frob_data])))

            mean_frobs = np.nanmean(all_frob_errors, axis=0)
            std_frobs = np.nanstd(all_frob_errors, axis=0)
            mean_epoch_list = list(range(0, max_epochs+1, eval_recon_freq))

            ax.plot(mean_epoch_list, mean_frobs, label="Warm start (mean)", color=colors_list[0], linestyle=linestyles_list[0], linewidth=1)


        for i, (rank, frob_data) in enumerate(frob_error_norms_cold_sorted.items()):
            ax = axes[rank_axis_dict[rank]]
            ax.set_title(f"Rank: {rank}")

            all_frob_errors = []

            for j, (epochs, frob_erros) in enumerate(frob_data):
                # update max_epochs
                max_epochs = max(max_epochs, max(epochs))
                ax.plot(epochs, frob_erros, color=colors_list[1], linestyle=linestyles_list[1], alpha=0.5, linewidth=.5)
                all_frob_errors.append(pad_with_nans(epochs, frob_erros, max([len(frob_erros) for epochs, frob_erros in frob_data])))

            mean_frobs = np.nanmean(all_frob_errors, axis=0)
            std_frobs = np.nanstd(all_frob_errors, axis=0)

            # longest_epoch_list length of mean_frobs with 20 spaced epochs
            mean_epoch_list = list(range(0, max_epochs+1, eval_recon_freq))

            ax.plot(mean_epoch_list, mean_frobs, label="Cold start (mean)", color=colors_list[1], linestyle=linestyles_list[1], linewidth=1)



        for ax in axes:
            # ax.set_xlim(0, max_epochs+1000)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Frobenius Error")
            # smaller legend
            ax.legend(loc='upper right', fontsize='small')

        plt.suptitle(f"Frobenius Errors for different warm and cold start runs\n({data}, {model})")
        plt.tight_layout()
        # plt.legend()
        # plt.show()
        PaperStylePlotter().save_fig(fig, "hot_n_cold")
