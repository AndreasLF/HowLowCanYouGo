import torch 
from main import create_model
from main import train
import uuid
import os


def make_model_save_path(dataset, rank,results_folder="results", exp_id="_"):
    os.makedirs(results_folder, exist_ok=True)
    return f"{results_folder}/EE_model_LSM_{dataset}_{rank}_{exp_id}.pt"


def find_optimal_rank(min_rank: int, 
                    max_rank: int, 
                    dataset: str,
                    results_folder='results',
                    device='cpu'):
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
        svd_target = torch.concatenate([X,Y], dim=0)
        # Center
        return svd_target - svd_target.mean(dim=0).unsqueeze(0) # center svd_target -> PCA


    print(f'Training FIRST model with rank {upper_bound}')
    model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=upper_bound, device=device)
    save_path = make_model_save_path(dataset=dataset, rank=upper_bound, results_folder=results_folder, exp_id=exp_id)
    is_fully_reconstructed = train(model, N1, N2, edges, exp_id=exp_id, dataset_name=dataset_name, model_path=save_path)
    torch.save(model, save_path)

    if is_fully_reconstructed:
        print("Full reconstruction not found with random initialization")
        return False

    svd_target = compute_svd_target(model)

    while lower_bound <= upper_bound:
        current_rank = (lower_bound + upper_bound) // 2
        print(f'Training model (SVD initialized) with rank {current_rank}')

        # 2. Perform SVD estimate from higher into lower rank approx.
        _,_,V = torch.svd_lowrank(svd_target, q=current_rank)
        X,Y = torch.chunk(svd_target, 2, dim=0)
        model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=current_rank)
        model.latent_z = X
        model.latent_w = Y

        save_path = make_model_save_path(dataset=dataset, rank=current_rank,results_folder=results_folder, exp_id=exp_id)
        is_fully_reconstructed = train(model, N1, N2, edges, exp_id=exp_id, dataset_name=dataset_name, model_path=save_path)
        torch.save(model, save_path)

        # Check if the reconstruction is within the threshold
        if is_fully_reconstructed:
            print(f'Full reconstruction at rank {current_rank}\n')
            optimal_rank = current_rank  # Update the optimal rank if reconstruction is within the threshold
            upper_bound = current_rank - 1  # Try to find a smaller rank
            
            # compute new svd_target for next iteration
            svd_target = compute_svd_target(model)
        else:
            print(f'Full reconstruction NOT found for rank {current_rank}\n')
            lower_bound = current_rank + 1


    return optimal_rank


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument("--dataset", type=str, default='Cora', help="Dataset to run experiment for")
    parser.add_argument("--max", type=int, default=100, help="Max rank to search for")
    parser.add_argument("--min", type=int, default=1, help="Max rank to search for")

    args = parser.parse_args()
    device = args.device

    dataset = args.dataset

    dataset_relpath = "datasets"
    # find_optimal_rank(3,100,f"{dataset_relpath}/Cora")
    find_optimal_rank(args.min,args.max,f"{dataset_relpath}/{dataset}", device=device)

