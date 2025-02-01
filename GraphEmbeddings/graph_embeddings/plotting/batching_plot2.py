# %% 
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

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

            if rank not in data:
                data[rank] = []
            # if rank not in data[dataset]:

            # Check the epoch differences
            data[rank].append({"create_date": create_date, "data": run_data, "FR": FR})

    #     # Function to convert 'create_date' string to datetime object
    # def get_create_date(item):
    #     key, value = item
    #     # create date is a key in the sub-dictionary
    #     create_date = value["create_date"][0]
    #     return datetime.fromisoformat(create_date)


    # # Sort the dictionary items by 'create_date'
    # sorted_items = sorted(data.items(), key=get_create_date)

    # Convert the sorted items back to a dictionary
    # sorted_dict = {key: value for key, value in sorted_items}

    return data


# %%
if __name__ == "__main__":
    
    model = "L2Model"
    data = "PubMed"
    start_date = "2024-05-05" # NOTE V2 of experiments, no diagonal

    filters = {
        "config.model_class": model,
        "config.data": data,
        "created_at": {"$gt": start_date}
    }

    # ----- Full dataset -----
    filters["config.loss_fn"]= "LogisticLoss"
    filters["config.batch_type"] = "RandomNodeDataLoader"
    filters["config.batch_size"] = 19717
    data_rn_full = get_data(filters)

    # ----- Random Node Data Loader (10%) -----
    filters["config.batch_size"] = 1971
    data_rn_10 = get_data(filters)

    # ----- Case Control Data Loader (10%) -----
    filters["config.loss_fn"] = "CaseControlLogisticLoss"
    filters["config.batch_type"] = "CaseControlDataLoader"
    filters["config.batch_size"] = 1971
    data_cc_10 = get_data(filters)
    


    # %%
    from matplotlib import pyplot as plt
    from cycler import cycler
    from graph_embeddings.plotting.plotter import PaperStylePlotter


    
    def pad_loss_with_nan(epoch_list, loss_list, intended_epochs):        
        # Initialize the padded loss list
        padded_loss_list = []
        
        # Initialize an iterator for the original epoch and loss lists
        epoch_iter = iter(epoch_list)
        loss_iter = iter(loss_list)
        
        # Get the first values
        current_epoch = next(epoch_iter, None)
        current_loss = next(loss_iter, None)
        
        for intended_epoch in intended_epochs:
            if current_epoch == intended_epoch:
                padded_loss_list.append(current_loss)
                current_epoch = next(epoch_iter, None)
                current_loss = next(loss_iter, None)
            else:
                padded_loss_list.append(np.nan)
        
        return padded_loss_list

    with PaperStylePlotter().apply():
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']

    colors = [colors[0], colors[5], colors[2]]
    linestyles = [linestyles[0], linestyles[1], linestyles[2]]

    
    # ranks_rn_full = list(data_rn_full.keys())
    # ranks_rn_10 = list(data_rn_10.keys())
    # ranks_cc_10 = list(data_cc_10.keys())
    # rank_list = [*ranks_rn_full, *ranks_rn_10, *ranks_cc_10]
    # ranks = ranks_rn_full
    # # append what is not in full from rn10 and cc10 
    # for i, rank in enumerate(rank_list):
    #     if rank not in ranks:
    #         ranks.append(rank)

    ranks = [80, 20, 10, 15, 12, 13, 14, 17, 18, 19] # ! Just hardcoded the order

    # plot three colums for each rank
    fig, ax = plt.subplots(4, 3, figsize=(12, 8))
    ax = ax.flatten()   # Flatten to easily iterate

    for i, rank in enumerate(ranks):
        values = data_rn_full.get(rank, None)

        if values:


            max_epoch = max([len(run["data"]["epochs"]) for run in values])
            intended_epochs = np.arange(0, max_epoch, 20)

            padded_frobs = []
            FR = False
            for j, run in enumerate(values):
                epochs = run["data"]["epochs"]

                frobs = run["data"]["frob"]
                if run["FR"]: FR = True
                lab = "Full Dataset" if j == 0 else None
                ax[i].plot(epochs, frobs, label=lab, color=colors[0], linestyle=linestyles[0], alpha=0.4)
                ax[i].set_title(f"Rank {rank}")

        values = data_rn_10.get(rank, None)
        if values:
            for j, run in enumerate(values):
                epochs = run["data"]["epochs"]
                frobs = run["data"]["frob"]
                FR = run["FR"] or False
                lab = "RN (10%)" if j == 0 else None
                ax[i].plot(epochs, frobs, label=lab, color=colors[1], linestyle=linestyles[1], alpha=0.4)
                ax[i].set_title(f"Rank {rank}")

        values = data_cc_10.get(rank, None)
        lab_set = None
        if values:
            for j, run in enumerate(values):
                epochs = run["data"]["epochs"]
                frobs = run["data"]["frob"]
                FR = run["FR"] or False
                lab = "CC (10%)" if j == 0 else None
                ax[i].plot(epochs, frobs, label=lab, color=colors[2], linestyle=linestyles[2], alpha=0.4)
                ax[i].set_title(f"Rank {rank}")


        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Frobenius Error")

        ax[i].legend()

        # make tick text smaller
        ax[i].tick_params(axis='both', which='major', labelsize=8)


plt.tight_layout()
# %%
