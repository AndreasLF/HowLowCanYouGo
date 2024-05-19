# %% 
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from matplotlib import pyplot as plt

def fetch_run_data(run):
    rank = run.config.get("rank")
    
    history_df = run.history()
    missclass_num = history_df["misclassified_dyads"]
    missclass_perc = history_df["misclassified_dyads_perc"]
    # hbdm_loss = history_df["hbdm_loss"]
    # hinge_loss = history_df["hinge_loss"]
    epochs = history_df["epoch"]
    data = run.config.get("data")
    FR = run.config.get("full_reconstruction") or False

    create_date = run.created_at

    phase1_epochs = run.config.get("phase1_epochs")
    phase2_epochs = run.config.get("phase2_epochs")
    phase3_epochs = run.config.get("phase3_epochs")

    return_data = {"missclass_num": list(missclass_num), 
                   "missclass_perc": list(missclass_perc), 
                #    "hbdm_loss": list(hbdm_loss), 
                #    "hinge_loss": list(hinge_loss), 
                   "epochs": list(epochs)}
    
    return rank, data, create_date, (phase1_epochs, phase2_epochs, phase3_epochs), return_data, FR

# %%
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Plot the misclassified dyads for each search step")
    parser.add_argument("--data", type=str, default="Cora", help="The dataset to plot")
    parser.add_argument("--ymax", type=float, default=1.0, help="The maximum y value for the plot")
    args = parser.parse_args()

    # =================== Get the data ===================
    exp_id_dict = {
        "Citeseer": "a2447153-cacd-417b-b1a2-136848e4aea1",
        "Cora": "a795119a-cf6c-4ee0-bbeb-5ccec2f86fd3"
    }

    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    print("Fetching runs...")
    # filter on all runs in the project
    runs = api.runs(path=project_name)

    # Define the filters for the config parameters
    filters = {
        "config.exp_id": {"$in": list(exp_id_dict.values())}
    }    
    matching_runs = api.runs(path=project_name, filters=filters)


    data = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_run_data, run): run.id for run in matching_runs}
        for future in tqdm(futures):
            result = future.result()
            if result is None:
                continue
            rank, dataset, create_date, phase_epochs, run_data, FR = result

            if dataset not in data:
                data[dataset] = {}
            # if rank not in data[dataset]:
            data[dataset][rank] = {"create_date": create_date, "phase_epochs": phase_epochs, "data": run_data, "FR": FR}
    


    # %%
    # =================== Plot the data ===================

    import matplotlib.pyplot as plt

    plot_dataset = str(args.data)
    plot_data = data[plot_dataset]
    

        # Function to convert 'create_date' string to datetime object
    def get_create_date(item):
        key, value = item
        # create date is a key in the sub-dictionary
        create_date = value["create_date"]
        return datetime.fromisoformat(create_date)

    # Sort the dictionary items by 'create_date'
    sorted_items = sorted(plot_data.items(), key=get_create_date)

    # Convert the sorted items back to a dictionary
    sorted_dict = {key: value for key, value in sorted_items}

    num_plots = len(sorted_dict)

    max_x = 0
    max_y = 0

    # get maximum value for percentage recon

    fig, ax = plt.subplots(1, num_plots, figsize=(3*num_plots, 3))
    
    for i, (rank, values) in enumerate(sorted_dict.items()):
        epochs = values["data"]["epochs"]
        phase1_epochs, phase2_epochs, phase3_epochs = values["phase_epochs"]
        FR = values["FR"]
        print("Rank", rank)
        print("Phase1 epochs", phase1_epochs)
        print("Phase2 epochs", phase2_epochs)
        print("Phase3 epochs", phase3_epochs)

        # add phase1_epochs to epochs
        # epochs = [epoch + phase1_epochs for epoch in epochs]


        missclass_num = values["data"]["missclass_perc"]

        ax[i].plot(epochs, missclass_num)

        # if FR add star to title
        if FR:
            ax[i].set_title(f"Search step {i+1} *\n(Rank {rank})")
        else:
            ax[i].set_title(f"Search step {i+1}\n(Rank {rank})")

        # plot a horizontal line at 0 and phase1_epochs and fill between

        print("Phase1 epochs", phase1_epochs)
        # ax[i].axvline(phase1_epochs, color="salmon", linestyle="--")
        # ax[i].axvline(0, color="salmon", linestyle="--")
        # fill between
        y_min, y_max = ax[i].get_ylim()
        max_y = max(max_y, y_max)
        # Fill between from the bottom to the top of the plot
        if phase1_epochs - 0 > 0:
            ax[i].fill_betweenx([y_min, args.ymax], 0, phase1_epochs, color="salmon", alpha=0.2, label="HBDM Phase")

        x_min, x_max = ax[i].get_xlim()
        ax[i].set_xlim(0, x_max)
        ax[i].fill_betweenx([y_min, args.ymax], phase1_epochs, x_max, color="white", alpha=0, label="Hinge Loss Phase")

        if i == 0:
            ax[i].set_ylabel("Misclassified Dyads (%)")
            ax[i].legend()
        # break


    for i in range(num_plots):
        ax[i].set_xlabel("Epoch")
        if i == 0:
            ax[i].set_ylabel("Misclassified Dyads (%)")
        # ax[i].grid()
        # ax[i].set_xlim(0, max_x)
        ax[i].set_ylim(0, args.ymax)
    
    # plt.suptitle(f"{plot_dataset} Misclassified Dyads (%) for each search step")
    plt.tight_layout()

    # make all text sizes a bit larger
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'xtick.labelsize': 'medium'})
    plt.rcParams.update({'ytick.labelsize': 'medium'})


    plt.savefig(f"figures/{plot_dataset}_misclassified_dyads_search.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05, dpi=300)

        # %%
