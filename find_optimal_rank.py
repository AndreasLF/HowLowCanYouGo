import torch 
from main import create_model
import uuid
import os
from wandb_api_utils import WANDbAPIUtils
import json
from main import update_json, Trainer

import pdb

def make_model_save_path(dataset, rank,results_folder="results", exp_id="_"):
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/{exp_id}", exist_ok=True)
    return f"{results_folder}/{exp_id}/EE_model_LSM_{dataset}_{rank}.pt"

def find_optimal_rank(min_rank: int, 
                    max_rank: int, 
                    dataset: str,
                    phase_epochs: dict,
                    results_folder='results',
                    device: str = 'cpu',
                    wandb_logging: bool = False,
                    exp_id: str = None):
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

    # if exp_id is set, we continue an experiment
    if exp_id:
        print(f"Continuing experiment with id: {exp_id}")

        experiment_folder = f"{results_folder}/{exp_id}"
        exp_json_path = f"{experiment_folder}/experiment_state.json"

        # load JSON file from experiment folder
        with open(exp_json_path, "r") as f:
            exp_state_json = json.load(f)
        lower_bound = exp_state_json['min_rank']
        upper_bound = exp_state_json['max_rank']
        current_rank = exp_state_json['current_rank']
        orig_min_rank = exp_state_json['orig_min_rank']
        orig_max_rank = exp_state_json['orig_max_rank']
        phase_epochs = exp_state_json['phase_epochs']
        phase_epochs = {int(k): v for k, v in phase_epochs.items()}
        latest_checkpoint = exp_state_json['latest_checkpoint']
        num_trainings = exp_state_json['num_trainings']

        # We skip the HBDM phase if we have already trained a model 
        if num_trainings > 0 and latest_checkpoint is not None:
            phase_epochs[1] = 0
    else:
        exp_id = str(uuid.uuid4())
        # create folder for experiment
        experiment_folder = f"{results_folder}/{exp_id}"
        exp_json_path = f"{experiment_folder}/experiment_state.json"
        os.makedirs(experiment_folder, exist_ok=True)

        lower_bound = min_rank
        upper_bound = max_rank
        orig_min_rank = min_rank
        orig_max_rank = max_rank

        optimal_rank = upper_bound  # Assume the worst case initially

        # create experiment_state.json
        latest_checkpoint = None
        num_trainings = 0
        exp_state_json = {
            "dataset": dataset_name,
            "min_rank": min_rank,
            "max_rank": max_rank,
            "current_rank": upper_bound,
            "orig_min_rank": min_rank,
            "orig_max_rank": max_rank,
            "phase_epochs": phase_epochs,
            "orig_phase_epochs": phase_epochs,
            "results_folder": results_folder,
            "device": device,
            "wandb_logging": wandb_logging,
            "exp_id": exp_id,
            "latest_checkpoint": latest_checkpoint,
            "num_trainings": num_trainings
        }

        current_rank = upper_bound
        # Save to json file
        with open(exp_json_path, "w") as f:
            json.dump(exp_state_json, f)

    print("Experiment ID:   ", exp_id)
    print(phase_epochs)
    print(f'{"="*50}\nFinding optimal rank between {orig_min_rank} and {orig_max_rank}\n{"="*50}')

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


    print(f'Training model with rank {current_rank}')
    save_path = make_model_save_path(dataset=dataset_name, rank=current_rank, results_folder=results_folder, exp_id=exp_id)

    if latest_checkpoint: # continue training from checkpoint
        print("Defining trainer from checkpoint")
        trainer = Trainer.from_checkpoint(checkpoint_path=latest_checkpoint, 
                                    exp_id=exp_id, 
                                    phase_epochs=phase_epochs, 
                                    dataset_name=dataset_name,
                                    model_path=latest_checkpoint, 
                                    wandb_logging=wandb_logging,
                                    learning_rate=0.1,
                                    learning_rate_hinge=0.25)
    else: # train from scratch
        print("Defining trainer from scratch")
        model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=upper_bound, device=device)
        trainer = Trainer.from_scratch(model, N1, N2, edges, 
                                    exp_id=exp_id, 
                                    phase_epochs=phase_epochs, 
                                    dataset_name=dataset_name,
                                    model_path=save_path, 
                                    wandb_logging=wandb_logging,
                                    learning_rate=0.1,
                                    learning_rate_hinge=0.25)
        
    
    is_fully_reconstructed = trainer.train()

    if is_fully_reconstructed:
        save_path = save_path.replace('.pt', '_FR.pt')
    torch.save(trainer.model, save_path)
    
    svd_target = compute_svd_target(trainer.model)

    if not is_fully_reconstructed:
        print("Full reconstruction not found with random initialization")
        return -1 # ! -1 <=> no rank found

    while lower_bound <= upper_bound:
        current_rank = (lower_bound + upper_bound) // 2
        update_json(exp_json_path, "current_rank", current_rank)

        # 2. Perform SVD estimate from higher into lower rank approx.
        _,_,V = torch.svd_lowrank(svd_target, q=current_rank)
        # _bias = model.bias
        model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=current_rank, device=device)
        # X,Y = torch.chunk(svd_target, 2, dim=0)
        X,Y = torch.split(svd_target, [N1, N2])
        model.latent_z = torch.nn.Parameter(X@V, requires_grad=True)
        model.latent_w = torch.nn.Parameter(Y@V, requires_grad=True)
        # model.bias = _bias

        del X; del Y; del V; del _;

        print(f'Training model (SVD initialized) with rank {current_rank}')

        save_path = make_model_save_path(dataset=dataset_name, rank=current_rank,results_folder=results_folder, exp_id=exp_id)

        # After SVD initialization, we train only the last phase (hinge)
        new_phase_epochs = {1: 0, 2: 0, 3: phase_epochs[3]}
        # update JSON file
        update_json(exp_json_path, "phase_epochs", new_phase_epochs)

        trainer = Trainer.from_scratch(model, N1, N2, edges, 
                                    exp_id=exp_id, 
                                    phase_epochs=new_phase_epochs, 
                                    dataset_name=dataset_name,
                                    model_path=save_path, 
                                    wandb_logging=wandb_logging,
                                    learning_rate=0.1,
                                    learning_rate_hinge=0.25)
    
        is_fully_reconstructed = trainer.train()

        if is_fully_reconstructed:
            save_path = save_path.replace('.pt', '_FR.pt')
            torch.save(trainer.model, save_path)

        # Check if the reconstruction is within the threshold
        if is_fully_reconstructed:
            print(f'Full reconstruction at rank {current_rank}\n')
            optimal_rank = current_rank  # Update the optimal rank if reconstruction is within the threshold
            upper_bound = current_rank - 1  # Try to find a smaller rank
            update_json(exp_json_path, "max_rank", upper_bound)

            # compute new svd_target for next iteration
            svd_target = compute_svd_target(model)
        else:
            print(f'Full reconstruction NOT found for rank {current_rank}\n')
            lower_bound = current_rank + 1
            update_json(exp_json_path, "min_rank", lower_bound)

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
    parser.add_argument("--phase2", type=int, default=0,  help="How many steps to run phase 2 for.")
    parser.add_argument("--phase3", type=int, default=10_000, help="How many steps to run phase 3 for.")
    parser.add_argument("--max", type=int, default=100, help="Max rank, i.e. upper bound of search range.")
    parser.add_argument("--min", type=int, default=1, help="Min rank, i.e. lower bound of search range.")
    parser.add_argument("--wandb", action='store_true', help='Flag for logging to wandb')
    parser.add_argument("--cexp", type=str, default=None, help="Continue an experiment")

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    dataset_relpath = "datasets"

    find_optimal_rank(args.min, args.max,
                      f"{dataset_relpath}/{args.dataset}", 
                      phase_epochs={1: args.phase1, 
                                    2: args.phase2,
                                    3: args.phase3},
                      device=args.device, 
                      wandb_logging=args.wandb,
                      exp_id=args.cexp)

