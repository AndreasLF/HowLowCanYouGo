import torch 
from main import create_model
from main import train
import uuid


def make_model_save_path(dataset, rank, full_reconstruct):

    uid = str(uuid.uuid4())

    if full_reconstruct:
        return f"EE_model_LSM_{dataset}_{rank}_FR_{uid}.pt"
    else:
        return f"EE_model_LSM_{dataset}_{rank}_{uid}.pt"


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

    # !TODO make exp_id and pass to train function for logging to wandb
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


    print(f'Training FIRST model with rank {upper_bound}')
    model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=upper_bound, device=device)
    is_fully_reconstructed = train(model, N1, N2, edges, exp_id=exp_id)
    save_path = make_model_save_path(dataset, upper_bound, is_fully_reconstructed)
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
        model, N1, N2, edges  = create_model(dataset=dataset, latent_dim=current_rank, device=device)
        model.latent_z = torch.nn.Parameter(X@V, requires_grad=True)
        model.latent_w = torch.nn.Parameter(Y@V, requires_grad=True)

        is_fully_reconstructed = train(model, N1, N2, edges, exp_id=exp_id)
        save_path = make_model_save_path(dataset, current_rank, is_fully_reconstructed)
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
    device = 'cuda'
    dataset_relpath = "datasets"
    find_optimal_rank(1, 2, dataset=f"{dataset_relpath}/Cora", device=device)
