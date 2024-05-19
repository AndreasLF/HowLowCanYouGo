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
    frobs = history_df["frob_error_norm"]
    epochs = history_df["epoch"]
    # hbdm_loss = history_df["hbdm_loss"]
    # hinge_loss = history_df["hinge_loss"]
    data = run.config.get("data")
    FR = run.config.get("full_reconstruction") or False

    create_date = run.created_at

    return_data = {"frob": frobs,
                   "epochs": list(epochs)}
    
    return rank, data, create_date, return_data, FR

def get_data(filters):

    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    matching_runs = api.runs(path=project_name, filters=filters)

    data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_run_data, run): run.id for run in matching_runs}
        for future in tqdm(futures):
            result = future.result()
            if result is None:
                continue
            rank, dataset, create_date, run_data, FR = result
            # if rank not in data[dataset]:
            data[rank] = {"create_date": create_date, "data": run_data, "FR": FR}



        # Function to convert 'create_date' string to datetime object
    def get_create_date(item):
        key, value = item
        # create date is a key in the sub-dictionary
        create_date = value["create_date"]
        return datetime.fromisoformat(create_date)

    # Sort the dictionary items by 'create_date'
    sorted_items = sorted(data.items(), key=get_create_date)

    # Convert the sorted items back to a dictionary
    sorted_dict = {key: value for key, value in sorted_items}

    return sorted_dict



# %%
if __name__ == "__main__":
    # %%

    # Define the filters for the config parameters
    filters = {
        "config.exp_id": "6f5d7264-f05f-433d-b49d-09957a192f39"
    }    
    data_hot_start = get_data(filters)
    # remove the first key
    data_hot_start = {k: v for k, v in data_hot_start.items() if k != max(data_hot_start.keys())}

    filters = {
        "tags": "cold_start",
        "config.data": "PubMed"
    }
    data_cold_start = get_data(filters)


    # %%
    import matplotlib.pyplot as plt
    from cycler import cycler
    from graph_embeddings.plotting.plotter import PaperStylePlotter

    with PaperStylePlotter().apply():
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']

    colors = [colors[0], colors[5]]
    linestyles = [linestyles[0], linestyles[1]]


    # import matplotlib.pyplot as plt
    # grid of 7 pltos
    fig, ax = plt.subplots(3, 3, figsize=(12, 8))
    ax = ax.flatten()   # Flatten to easily iterate


    for i, (rank, values) in enumerate(data_hot_start.items()):
        epochs = values["data"]["epochs"]
        frobs = values["data"]["frob"]

        FR = values["FR"] or False

        lab = f"SVD Hot Start *" if FR else f"SVD Hot Start"
        ax[i].plot(epochs, frobs, label=lab, color=colors[0], linestyle=linestyles[0])
        ax[i].set_title(f"Rank {rank}")
        
        try:
            coldstart_epochs = data_cold_start[rank]["data"]["epochs"]
            coldstart_frobs = data_cold_start[rank]["data"]["frob"]
            FR_cold = data_cold_start[rank]["FR"] or False
            lab = f"Cold Start *" if FR_cold else f"Cold Start"
            ax[i].plot(coldstart_epochs, coldstart_frobs, label=lab, color=colors[1], linestyle=linestyles[1])
            ax[i].set_title(f"Rank {rank}")
        except: 
            print(f"Rank {rank} not in cold start data")

        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Frobenius Error")

        ax[i].legend()

        # make tick text smaller
        ax[i].tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()

    # remove last two plots 
    ax[-1].axis("off")
    ax[-2].axis("off")


    plt.savefig("figures/hot_n_cold_start.pdf", format="pdf")



# %%
