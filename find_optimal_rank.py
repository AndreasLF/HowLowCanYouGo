import torch 
from main import create_model
from main import train
import uuid
import os
from wandb_api_utils import WANDbAPIUtils


def make_model_save_path(dataset, rank,results_folder="results", exp_id="_"):
    os.makedirs(results_folder, exist_ok=True)
    return f"{results_folder}/EE_model_LSM_{dataset}_{rank}_{exp_id}.pt"


def find_optimal_rank(min_rank: int, 
                    max_rank: int, 
                    dataset: str,
                    phase_epochs: dict,
                    results_folder='results',
                    load_ckpt: str = None, # TODO add load-ckpt
                    device: str = 'cpu',
                    wandb_logging: bool = False):
    """
    Find the optimal rank for the model by starting on a 
        high guess (i.e. upper bound) at the optimal rank, followed 
        by a binary search in the range [lower, upper]

    Args:
        min_rank (int): The minimum rank
        max_rank (int): The maximum rank
    
    Returns:
        optimal_rank (int): The optimal rank
    
    """
    dataset_name = dataset.split("/")[-1]

    exp_id = str(uuid.uuid4())

    lower_bound = min_rank
    upper_bound = max_rank
    optimal_rank = upper_bound  # Assume the worst case initially

    print(f'{"="*50}\nFinding optimal rank between {min_rank} and {max_rank}\n{"="*50}')

    # 1. Train initial high guess
    save_path = None # ! just ignore initial model


    def compute_svd_target(model):
        """
        Computes SVD on the the centered concatenation of our X and Y embedding matrices.
        The SVD will thus correspond to PCA.
        """
        X = model.latent_z
        Y = model.latent_w
        svd_target = torch.vstack([X,Y])
        # Center
        return svd_target - svd_target.mean(dim=0).unsqueeze(0) # center svd_target -> PCA

    

    if load_ckpt is not None:
        # TODO change to loading search state
        search_state = torch.load(load_ckpt)

        # ! ensure that embedding dims in {model}.pt-file match max_rank params
        print(f'Initializing FIRST model from {load_ckpt}')
        lower_bound = search_state['lb']
        upper_bound = search_state['ub']
        current_rank = search_state['cur_rank']
        model = create_model(dataset=dataset, latent_dim=current_rank, device=device)
        model.load_state_dict(search_state['current_model'])

        full_recon_model, N1, N2, edges = create_model(dataset=dataset, latent_dim=upper_bound, device=device)
        full_recon_model.load_state_dict(search_state['full_recon_model'])

        svd_target = compute_svd_target(full_recon_model)
    else:
        model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=upper_bound, device=device)
        print(f'Training FIRST model with rank {upper_bound}')
        search_state = {'lb': lower_bound, 'ub': upper_bound, 'full_recon_model': None}
        save_path = make_model_save_path(dataset=dataset_name, rank=upper_bound, results_folder=results_folder, exp_id=exp_id)
        is_fully_reconstructed = train(model, N1, N2, edges, exp_id=exp_id, phase_epochs=phase_epochs, dataset_name=dataset_name, model_path=save_path, wandb_logging=wandb_logging)
        torch.save(model, save_path)
        
        svd_target = svd_target or compute_svd_target(model)

    if is_fully_reconstructed:
        print("Full reconstruction not found with random initialization")
        return -1 # ! -1 <=> no rank found


    while lower_bound <= upper_bound:
        if load_ckpt is None:
            current_rank = (lower_bound + upper_bound) // 2

            # 2. Perform SVD estimate from higher into lower rank approx.
            _,_,V = torch.svd_lowrank(svd_target, q=current_rank)
            X,Y = torch.chunk(svd_target, 2, dim=0)
            model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=current_rank, device=device)
            model.latent_z = torch.nn.Parameter(X@V, requires_grad=True)
            model.latent_w = torch.nn.Parameter(Y@V, requires_grad=True)

            del X; del Y; del V; del _;

        load_ckpt = None # set load_ckpt to None for enabling SVD for next iterations
 
        print(f'Training model (SVD initialized) with rank {current_rank}')

        search_state['cur_rank'] = current_rank
        search_state['lb'] = lower_bound
        search_state['ub'] = upper_bound

        save_path = make_model_save_path(dataset=dataset_name, rank=current_rank,results_folder=results_folder, exp_id=exp_id)
        is_fully_reconstructed = train(model, N1, N2, edges, 
                                       exp_id=exp_id, 
                                       phase_epochs=phase_epochs, 
                                       dataset_name=dataset_name, 
                                       model_path=save_path, 
                                       wandb_logging=wandb_logging,
                                       search_state=search_state)
        torch.save(model, save_path)

        # Check if the reconstruction is within the threshold
        if is_fully_reconstructed:
            search_state['full_recon_model'] = model.state_dict() # save in search state

            print(f'Full reconstruction at rank {current_rank}\n')
            optimal_rank = current_rank  # Update the optimal rank if reconstruction is within the threshold
            upper_bound = current_rank - 1  # Try to find a smaller rank
            
            # compute new svd_target for next iteration
            svd_target = compute_svd_target(model)
        else:
            print(f'Full reconstruction NOT found for rank {current_rank}\n')
            lower_bound = current_rank + 1

    if wandb_logging:  
        print("Tagging best rank in wandb")
        wandb_api = WANDbAPIUtils("GraphEmbeddings")
        wandb_api.tag_best_rank(exp_id) 

    return optimal_rank


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--dataset", type=str, default='Cora', help="Dataset (graph) to run search on for.")
    parser.add_argument("--load-ckpt", type=str, default=None, help="Specify which model state dict to initialize the search.")
    parser.add_argument("--phase1", type=int, default=1_000,  help="How many steps to run phase 1 for.")
    parser.add_argument("--phase2", type=int, default=5_000,  help="How many steps to run phase 2 for.")
    parser.add_argument("--phase3", type=int, default=10_000, help="How many steps to run phase 3 for.")
    parser.add_argument("--max", type=int, default=100, help="Max rank, i.e. upper bound of search range.")
    parser.add_argument("--min", type=int, default=1, help="Min rank, i.e. lower bound of search range.")
    parser.add_argument("--wandb", action='store_true', help='Flag for logging to wandb')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    dataset_relpath = "datasets"
    # find_optimal_rank(3,100,f"{dataset_relpath}/Cora")
    find_optimal_rank(args.min, args.max,
                      f"{dataset_relpath}/{args.dataset}", 
                      phase_epochs={1: args.phase1, 
                                    2: args.phase2 + args.phase1, # ? due to the way the loop ranges are defined inside train()
                                    3: args.phase3},
                      load_ckpt=args.load_ckpt,
                      device=args.device, 
                      wandb_logging=args.wandb)

