# %%
from graph_embeddings.utils.config import Config
import wandb
from tqdm import tqdm
import torch
from graph_embeddings.models import L2Model, PCAModel, LatentEigenModel
from graph_embeddings.utils.loss import LogisticLoss, HingeLoss, PoissonLoss, CaseControlLogisticLoss, CaseControlHingeLoss
from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
from graph_embeddings.utils.dataloader import RandomNodeDataLoader
from graph_embeddings.utils.trainer import Trainer

def compute_svd_target(model):
    """
    Computes SVD on the the centered concatenation of our X and Y embedding matrices.
    The SVD will thus correspond to PCA.
    """
    X,Y,*_ = model.forward() # get X and Y, disregard other parameters (i.e. beta)
    svd_target = torch.concatenate([X,Y], dim=0)

    # Only center if we use the L2Model
    if model.__class__.__name__ == "L2Model":
        return svd_target - svd_target.mean(dim=0).unsqueeze(0) # center svd_target -> PCA
    return svd_target
    

def get_data_from_wandb(data="Cora", model_class="L2Model", loss_fn="LogisticLoss", start_date=None):
        
    api = wandb.Api()

    # Specify your project and run
    project_name = "GraphEmbeddings"

    print("Fetching runs...")
    # filter on all runs in the project
    runs = api.runs(path=project_name)

    # data should be = Cora, model_class = L2, rank = 6
    print("Filtering runs...")
    if start_date:
        matching_runs = [run for run in tqdm(runs) if run.config.get("data") == data and run.config.get("model_class") == model_class and run.config.get("loss_fn") == loss_fn and run.created_at > start_date]
    else:
        matching_runs = [run for run in tqdm(runs) if run.config.get("data") == data and run.config.get("model_class") == model_class and run.config.get("loss_fn") == loss_fn]

    # get all unique batch_size values
    ranks = set([run.config.get("rank") for run in matching_runs])

    print("Fetching data on each run...")
    losses_rank = {}
    model_paths = {}
    full_recons = {}
    for run in tqdm(matching_runs):
        rank = run.config.get("rank")
        if rank not in losses_rank:
            losses_rank[rank] = []

        # append frob_error_norm history
        history_df = run.history()
        # get the epcoh column and frob_error_norm column

        frobs = history_df["frob_error_norm"]
        loss = history_df["loss"]
        epochs = history_df["epoch"]

        losses_rank[rank].append((list(epochs), list(loss)))

        # from the run also get the model_path
        model_paths[rank] = run.config.get("model_path")
        full_recons[rank] = run.config.get("full_reconstruction")

    return losses_rank, full_recons, model_paths
# %%

if __name__ == "__main__":
    model = "L2Model"
    loss_fn = "LogisticLoss"
    data = "Cora"
    start_date = "2024-05-05" # NOTE V2 of experiments, no diagonal

    # get experiments from wandb
    losses_rank, full_recons, model_paths = get_data_from_wandb(data=data, model_class=model, loss_fn=loss_fn, start_date=start_date)

    # %%
    # cfg_path = "../../configs/config.yaml"
    # cfg = Config(cfg_path)

    # # Get the data
    # dataset_path = cfg.get("data", "dataset_src", data).split("/")
    # dat = get_data_from_torch_geometric(dataset_path[-2], dataset_path[-1])
    # dataloader = RandomNodeDataLoader(dat[0], batch_size=dat[0].num_nodes, shuffle=True)
    # batch = next(iter(dataloader))




#     # %%
#     # sort keys in full_recons
#     full_recons = dict(sorted(full_recons.items()))

#     # get the max key
#     upper_bound = max(full_recons.keys())
#     lower_bound = 1
#     optimal_rank = upper_bound


#     if full_recons.get(upper_bound) is None:
#         assert False, f"Full reconstruction not found for rank {upper_bound}" 

#     first_model_path = model_paths[upper_bound]    

#     # binary search loop starting at max_key
#     X,Y,*_ = torch.load(f"../../{first_model_path}", map_location='cpu')
#     # get the model
#     if model in globals():
#         mdl = getattr(globals()[model], model).init_random(dat[0].num_nodes, dat[0].num_nodes, rank=upper_bound, inference_only=True)
#     else:
#         assert False, f"Model {model} not found in globals"
    
#     # initialize the svd_inits dict
#     svd_inits = {optimal_rank: (mdl)}

#     # get the svd_target
#     svd_target = compute_svd_target(mdl)
    

#     while lower_bound <= upper_bound:
#         current_rank = (upper_bound + lower_bound) // 2

#         _,_,V = torch.svd_lowrank(svd_target, q=current_rank)
#         X,Y = torch.chunk(svd_target, 2, dim=0)
#         # get the model
#         if model in globals():
#             mdl = getattr(globals()[model], model)(X, Y, inference_only=True)
#         else:
#             assert False, f"Model {model} not found in globals"

#         svd_inits[optimal_rank] = svd_target


#         # binary search scheme
#         if full_recons.get(current_rank):
#             optimal_rank = current_rank
#             upper_bound = current_rank - 1
#             svd_target = compute_svd_target(mdl)
#         else:
#             lower_bound = current_rank + 1
       
        




#     # %%
#     # get dataset path from config
#     cfg_path = "../../configs/config.yaml"
#     cfg = Config(cfg_path)

#     # Get the data
#     dataset_path = cfg.get("data", "dataset_src", data).split("/")
#     dat = get_data_from_torch_geometric(dataset_path[-2], dataset_path[-1])
#     dataloader = RandomNodeDataLoader(dat[0], batch_size=dat[0].num_nodes, shuffle=True)
#     batch = next(iter(dataloader))
#     # intialize the model empty trainer object, used to calculate frob_error
#     trainer = Trainer(*[None]*6)
#     if loss_fn == 'PoissonLoss': trainer.thresh = torch.log(torch.tensor(.5))  # for the Poisson objective, we threshold our logits at log(1/2), corresponding to 0.5 under Poisson formulation
#     else: trainer.thresh = 0 # for the Bernoulli objectives, we threshold logits at 0, corresponding to 0.5 under Bernoulli formulation
                

#     if loss_fn in globals():
#         loss = globals()[loss_fn]()
#     else:
#         assert False, f"Loss {loss_fn} not found in globals"

#     # %%
#     for rank, model_path in model_paths.items():
#         print(f"Rank {rank} has model_path {model_path}")

#         # load torch model
#         # Add two steps up to the path 
#         model_path = f"../../{model_path}"
#         X,Y,*_ = torch.load(model_path, map_location='cpu')

#         # get the model
#         if model in globals():
#             mdl = getattr(globals()[model], model)(X, Y, rank)
#         else: 
#             assert False, f"Model {model} not found in globals"
        
#         A_hat = mdl.reconstruct()

#         if loss_fn == 'PoissonLoss': A = batch.adj # use normal adjacency matrix (indices in {0,1})
#         else: A = batch.adj_s        

#         ls = loss(A_hat, A)
#         # frob = trainer.calc_frob_error_norm(A_hat, A)

#         break

# # %%
