# %% 
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm

def get_data_from_wandb(data="Cora", model_class="L2Model", loss_fn="LogisticLoss", start_date=None):
        
    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    print("Fetching runs...")
    # filter on all runs in the project
    runs = api.runs(path=project_name)

    # data should be = Cora, model_class = L2, rank = 6
    print("Filtering runs...")
    if start_date == None:
        matching_runs = [run for run in tqdm(runs) if run.config.get("data") == data and run.config.get("model_class") == model_class]
    else: 
        matching_runs = [run for run in tqdm(runs) if run.config.get("data") == data and run.config.get("model_class") == model_class and run.created_at > start_date]

    # get all unique batch_size values
    batch_sizes = set([run.config.get("batch_size") for run in matching_runs])


    print("Fetching data on each run...")
    frob_error_norms = {}
    for run in tqdm(matching_runs):
        batch_size = run.config.get("batch_size")
        batch_type = run.config.get("batch_type")
        rank = run.config.get("rank")
        if rank not in frob_error_norms:
            frob_error_norms[rank] = {}
        if (batch_type, batch_size) not in frob_error_norms[rank]:
            frob_error_norms[rank][(batch_type, batch_size)] = []

        # append frob_error_norm history
        history_df = run.history()
        # get the epcoh column and frob_error_norm column

        frobs = history_df["frob_error_norm"]
        epochs = history_df["epoch"]

        frob_error_norms[rank][(batch_type, batch_size)].append((list(epochs), list(frobs)))

    return frob_error_norms, batch_sizes
# %%

if __name__ == "__main__":
    model = "L2Model"
    data = "PubMed"
    loss_fn = "LogisticLoss"
    start_date = "2024-05-05" # NOTE V2 of experiments, no diagonal
    frob_error_norms, batch_sizes = get_data_from_wandb(data=data, model_class=model, loss_fn=loss_fn, start_date=start_date)


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

                for (epoch_list_x, loss_y) in value_list:
                    # ax.plot(epoch_list_x, loss_y, color=color, alpha=0.5)

                    # update max epochs
                    max_epocs = max(max_epocs, max(epoch_list_x))


                    padded_loss_y = pad_list(loss_y, max_len)
                    padded_epoch_x = pad_list(epoch_list_x, max_len)
                    ax.plot(padded_epoch_x, padded_loss_y, color=color, alpha=0.2, linestyle=linestyle)
                    all_y_values.append(padded_loss_y)
                # Calculate and plot the mean
                # mean_y = np.mean(all_y_values, axis=0)
                # ax.plot(epoch_list_x, mean_y, color=color, linewidth=2, label=f'{unique_tuple} mean' if i == 0 else "")

                # Calculate and plot the mean
                mean_y = np.nanmean(all_y_values, axis=0)

                lab = "Random Node Sampling (10%)" if unique_tuple[0] == "RandomNodeDataLoader" and unique_tuple[1] != 19717 else "Negative Sampling (10%)" if unique_tuple[0] == "CaseControlDataLoader" else "Full dataset"
                ax.plot(padded_epoch_x, mean_y, color=color, linewidth=1, label=lab, linestyle=linestyle)
                
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

        plt.tight_layout()
        plt.show()

    # import matplotlib.pyplot as plt
    # from cycler import cycler
    # from graph_embeddings.plotting.plotter import PaperStylePlotter
    # import numpy as np

    # with PaperStylePlotter().apply():
    #     fig, ax = plt.subplots()

    #     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     linestyles = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']

    #     colors = [colors[0], colors[5]]

    #     global_longest_epoch_list = []

    #     for b, batch_size in enumerate(batch_sizes):

    #         color = colors[b]
    #         linestyle = linestyles[b]

    #         frob_error_norms_list = frob_error_norms[batch_size]
    #         epochs_list = [frob_error_norms_list[i][0] for i in range(len(frob_error_norms_list))]
    #         frobs_list = [frob_error_norms_list[i][1] for i in range(len(frob_error_norms_list))]

    #         max_length = max(len(frobs) for frobs in frobs_list)

    #         # Pad frobenius errors to have the same length
    #         padded_frobs_list = np.array([np.pad(frobs, (0, max_length - len(frobs)), 'constant', constant_values=np.nan) for frobs in frobs_list])

    #         # Calculate mean and standard deviation ignoring NaN
    #         mean_frobs = np.nanmean(padded_frobs_list, axis=0)
    #         std_frobs = np.nanstd(padded_frobs_list, axis=0)

    #         # Plot each individual run with lower opacity
    #         for epochs, frobs in zip(epochs_list, frobs_list):
    #             ax.plot(epochs, frobs, color=color, linestyle=linestyle, alpha=0.5, linewidth=0.5)

    #         # Plot mean line and fill the confidence area around the mean
    #         # get longest epoch list
    #         longest_epoch_list = epochs_list[np.argmax([len(epochs) for epochs in epochs_list])]

    #         # get global longest epoch list
    #         global_longest_epoch_list = longest_epoch_list if len(longest_epoch_list) > len(global_longest_epoch_list) else global_longest_epoch_list


    #         label = f"{batch_size} (full)" if batch_size == max(batch_sizes) else f"{batch_size}"
    #         ax.plot(longest_epoch_list, mean_frobs, label=label, color=color, linestyle=linestyle, linewidth=1)

    #     ax.set_title(f"Frobenius Errors at Different Batch Sizes\n({data}, Rank {rank}, {model})")
    #     ax.set_xlabel("Epoch")
    #     ax.set_ylabel("Frobenius Error")
    #     ax.legend(title="Batch Size Mean")

    #     # max from global longest epoch list
    #     max_epoch = max(global_longest_epoch_list)+101
    #     ticks = np.arange(0, max_epoch, 5000)
    #     # replace 0 with 100 
    #     ticks[0] = 100
    #     # set x-ticks to max epoch
    #     ax.set_xticks(ticks)

    #     # save figure 
    #     fig_name = "frob_error_norms"
    #     print("Saving figure to: ", fig_name)
    #     plt.show()
    #     PaperStylePlotter().save_fig(fig, fig_name)


# %%
