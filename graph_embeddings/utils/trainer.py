import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

from torch_geometric.utils import to_dense_adj

from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.utils.load_data import load_adj
from graph_embeddings.utils.logger import JSONLogger

from graph_embeddings.utils.nearest_neighbours_reconstruction_check import get_edge_index_embeddings, compare_edge_indices
from graph_embeddings.utils.set_ops import equals_set

class Trainer:
    def __init__(self, 
                 dataloader, 
                 model_class, 
                 loss_fn, 
                 threshold, 
                 num_epochs, 
                 save_ckpt, 
                 load_ckpt=None, 
                 model_init='random',
                 device='cpu', 
                 loggers=[JSONLogger], 
                 project_name='GraphEmbeddings',
                 reconstruction_check="frob",
                 exp_id='not specified'):
        """Initialize the trainer."""   
        
        self.dataloader = dataloader
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.save_ckpt = save_ckpt
        self.load_ckpt = load_ckpt
        self.model_init = model_init
        self.device = device
        self.loggers = loggers
        self.project_name = project_name
        self.exp_id = exp_id

        assert reconstruction_check in ["frob", "neigh", "both"]
        self.reconstruction_check = reconstruction_check



    def calc_frob_error_norm(self, logits, A):
        """Compute the Frobenius error norm between the logits and the adjacency matrix."""
        logits[logits >= self.thresh] = 1.
        logits[logits < self.thresh] = 0.
        frob_err = torch.linalg.norm(logits - A) / torch.linalg.norm(A)
        return frob_err


    def init_model(self,
                   rank: int|None = None):
            if self.model_init == 'random':
                assert rank is not None
                model = self.model_class.init_random(self.dataloader.num_total_nodes, 
                                                     self.dataloader.num_total_nodes, 
                                                     rank).to(self.device)
            elif self.model_init == 'load':
                model_params = torch.load(self.load_ckpt,map_location=self.device)
                model = self.model_class(*model_params)
            else:
                raise Exception(f"selected model initialization ({self.model_init}) is not currently implemented")
                
            return model

    def train(self, 
              rank: int, 
              model: L2Model|PCAModel|None = None,
              lr: float = 0.01, 
              adjust_lr_patience: float = None, 
              eval_recon_freq: int = 100, # ! evaluate full reconstruction every {x}'th epoch
              save_path: str = None):
        """ Train the model using the given optimizer and loss function.
        
        Args:
            rank (int): The rank of the model
            print_loss_interval (int): The interval at which the loss is printed. Default is 10.
        
        Returns:
            U (torch.Tensor): The left singular vectors
            V (torch.Tensor): The right singular vectors
        """
        
        model = model or self.init_model(rank)

        loss_fn = self.loss_fn
        num_epochs = self.num_epochs
        full_reconstruction = False
        batch_size = self.dataloader.batch_size
        last_recons_check_epoch = None
        perc_edges_reconstructed = None
        frob_error_norm = None
        is_fully_reconstructed = False
        recons_report_str = None

        # ----------- Initialize logging -----------
        # get loss_fn function name
        loss_fn_name = loss_fn.__class__.__name__
        # get self.model_class function name
        model_class_name = self.model_class.__name__
        dataloader_class_name = self.dataloader.__class__.__name__

        # initialize logging to all loggers
        for logger in self.loggers:
            logger.init(project=self.project_name, 
                        config={'rank': rank, 
                                'num_epochs': num_epochs, 
                                'learning_rate': lr,
                                'loss_fn': loss_fn_name, 
                                'model_class': model_class_name,
                                'data': self.dataloader.dataset_name,
                                'adjust_lr_patience': adjust_lr_patience,
                                'batch_size': batch_size,
                                'batch_type': dataloader_class_name,
                                'exp_id': self.exp_id,
                                'reconstruction_check': self.reconstruction_check,
                                })         

        if self.reconstruction_check == "frob" or self.reconstruction_check == "both":
            print("[WARNING]: Loading dense Adjacency Matrix!")
            self.adj = self.dataloader.full_adj.to(self.device) # ! used for small graphs for FROB
   
        # ----------- Optimizer ----------- 
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # ----------- Scheduler -----------
        step_size = 1
        gamma = .5  # multiplicative decay factor
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # ----------- Loss and Reconstruction measures -----------
        if loss_fn_name == 'PoissonLoss': self.thresh = torch.log(torch.tensor(.5))  # for the Poisson objective, we threshold our logits at log(1/2), corresponding to 0.5 under Poisson formulation
        else: self.thresh = 0 # for the Bernoulli objectives, we threshold logits at 0, corresponding to 0.5 under Bernoulli formulation
                
        # ----------- Training loop -----------
        with tqdm(range(num_epochs)) as pbar:
            best_loss = float('inf')  # Keep track of best loss -> track improvement to check if we want to decrease learning rate
            epochs_no_improve = 0  # Counter to keep track of epochs with no improvement -> track when to decrease learning rate
            for epoch in pbar:
                losses = []
                for b_idx, batch in enumerate(self.dataloader):
                    # Forward pass
                    optimizer.zero_grad(set_to_none=True)
                    
                    if self.dataloader.__class__.__name__ == "CaseControlDataLoader": 
                        # for CC, we only reconstruct specific indices
                        links,nonlinks,coeffs = batch.links, batch.nonlinks, batch.coeffs
                        # pdb.set_trace()
                        preds = model.reconstruct_subset(links, nonlinks) 
                        loss = loss_fn(preds, links.shape[1], coeffs)
                    else:
                        # ? specific to "full" adjacency matrix reconstruction
                        batch.to(self.device)
                        if loss_fn_name == 'PoissonLoss': A = batch.adj # use normal adjacency matrix (indices in {0,1})
                        else: A = batch.adj_s                           # use shifted adjacency matrix (indices in {-1,1})
                        A_hat = model.reconstruct(batch.indices) 
                        loss = loss_fn(A_hat, A)

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                epoch_loss = sum(losses) / len(losses)


                if (epoch % eval_recon_freq == 0) and (epoch != 0): # ! only check every {x}'th epoch
                    last_recons_check_epoch = epoch
                    recons_report_str = ""

                    metrics = {'epoch': epoch, 'loss': epoch_loss}
                    if self.reconstruction_check in {"frob", "both"}:
                        last_recons_check_epoch = epoch
                        # Compute Frobenius error for diagnostics
                        with torch.no_grad():  # Ensure no gradients are computed in this block
                            A_hat = model.reconstruct()
                            frob_error_norm = self.calc_frob_error_norm(A_hat, self.adj)

                        # Log metrics to all loggers'
                        metrics['frob_error_norm'] = frob_error_norm.item()
                       

                        is_fully_reconstructed = frob_error_norm <= self.threshold

                        recons_report_str += f" frob-err={frob_error_norm or .0:.4f}" # for progress bar
                            
                    if self.reconstruction_check in {"neigh", "both"}:
                        assert loss_fn_name != 'PoissonLoss', "Nearest neighbors reconstruction check not implemented for PoissonLoss"
                        # Compute Frobenius error for diagnostics
                        if model.beta >= 0: # ! ensure beta is nonnegative, as we use it for radius when computing nearest neighbors
                            with torch.no_grad():
                                edge_index_from_neighbors = get_edge_index_embeddings(model.X, model.Y, model.beta)
                                is_fully_reconstructed, frac_correct = equals_set(edge_index_from_neighbors, 
                                                                                #   self.dataloader.data.edge_index,         # ? normal edge_index
                                                                                  self.dataloader.edge_index_with_selfloops, # ? edge_index augmented with selfloops
                                                                                  return_frac=True)
                        
                            # TODO testing
                            # recon_dense_adj = to_dense_adj(edge_index_from_neighbors).squeeze()
                            # print()

                        recons_report_str += f" knn-reconstruct={frac_correct*100 or .0:.2f}%" # for progress bar

                        metrics['knn_reconstruction'] = frac_correct

                    # Log metrics to all loggers
                    for logger in self.loggers:
                        logger.log(metrics)
                

                # update progress bar
                model_report_str = f"{model_class_name}" + (f"[beta={model.beta.item():.1f}]" if model_class_name == "L2Model" else "")
                pbar.set_description(f"{self.dataloader.dataset_name} {loss_fn_name} {model_report_str} lr={scheduler.get_last_lr()[0]} rank={rank}, loss={epoch_loss:.1f} [@{last_recons_check_epoch or 0}:{recons_report_str or 'Ø'}]")

                # Break if fully reconstructed
                if is_fully_reconstructed:
                    pbar.close()
                    for logger in self.loggers:
                        logger.config.update({'full_reconstruction': True})
                    full_reconstruction = True
                    print(f'Full reconstruction at epoch {epoch} with rank {rank}')
                    break

                if adjust_lr_patience is not None:
                    # Early stopping condition based on loss improvement
                    if loss < best_loss:
                        best_loss = loss  # Update best loss
                        epochs_no_improve = 0  # Reset the counter as we have improvement
                    else:
                        epochs_no_improve += 1  # Increment counter if no improvement

                    # Check if early stopping is triggered
                    if epochs_no_improve >= adjust_lr_patience:
                        for logger in self.loggers:
                            logger.config.update({'early_stop_triggered': True})
                        print(f"Early stopping triggered at epoch {epoch}. No improvement in loss for {adjust_lr_patience} consecutive epochs.")
                        scheduler.step()

                        if scheduler.get_last_lr()[0] <= 1e-5:
                            print("Learning rate is too small. Stopping training.")
                            break

                        epochs_no_improve = 0
                        # break

        # After training, retrieve parameters
        with torch.no_grad():  # Ensure no gradients are computed in this block
            # Save model to file
            if save_path:
                # Add _FR to the file name if full reconstruction is achieved
                save_path = save_path.replace('.pt', '_FR.pt') if full_reconstruction else save_path

                self._save_model(model, save_path)
                for logger in self.loggers:
                    logger.config.update({"model_path": save_path})

        # Finish logging
        for logger in self.loggers:
            logger.finish()

        # return final_outputs
        return is_fully_reconstructed

    def _save_model(self, model, path):
        """Save the model to a file."""
        # check if folder exists
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # save model
        torch.save(model(), path)


    def _make_model_save_path(self, experiment_name, results_folder='results', rank=None, model_type=None):
        """Create a save path for the model."""
        if not experiment_name:
            raise ValueError('experiment_name cannot be None')

        models_folder = os.path.join(results_folder, 'models')

        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        if model_type:
            experiment_name = f'{experiment_name}_{model_type}'
        if rank:
            experiment_name = f'{experiment_name}_{rank}'
        
        experiment_name = f'{experiment_name}.pt'
        return os.path.join(models_folder, experiment_name)
    
    def find_optimal_rank(self,
                          min_rank: int, 
                          max_rank: int,
                          lr: float = 0.01,
                          early_stop_patience=None, 
                          experiment_name=None, 
                          results_folder='results'):
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
        lower_bound = min_rank
        upper_bound = max_rank
        optimal_rank = upper_bound  # Assume the worst case initially

        print(f'{"="*50}\nFinding optimal rank between {min_rank} and {max_rank}\n{"="*50}')

        # 1. Train initial high guess
        save_path = None # ! just ignore initial model
        model = self.init_model(rank=max_rank)
        if self.model_init != 'load':
            self.train(max_rank, model=model, lr=lr, adjust_lr_patience=early_stop_patience, save_path=save_path)
        
        def compute_svd_target(model):
            """
            Computes SVD on the the centered concatenation of our X and Y embedding matrices.
            The SVD will thus correspond to PCA.
            """
            X,Y,*_ = model.forward() # get X and Y, disregard other parameters (i.e. beta)
            svd_target = torch.concatenate([X,Y], dim=0)
            return svd_target - svd_target.mean(dim=0).unsqueeze(0) # center svd_target -> PCA
        svd_target = compute_svd_target(model)

        while lower_bound <= upper_bound:
            current_rank = (lower_bound + upper_bound) // 2
            print(f'Training model with rank {current_rank}')

            # Create a save path for the model
            save_path = self._make_model_save_path(experiment_name, 
                                                    results_folder=results_folder, 
                                                    rank=current_rank, 
                                                    model_type=self.model_class.__name__) if experiment_name else None

            # 2. Perform SVD estimate from higher into lower rank approx.
            _,_,V = torch.svd_lowrank(svd_target, q=current_rank)
            X,Y = torch.chunk(svd_target, 2, dim=0)
            model = model.__class__(X@V, Y@V).to(self.device) # create new model instance with the PCA projected embedding matrices

            # 3. Train new model
            is_fully_reconstructed = self.train(current_rank, 
                               model=model, 
                               lr=lr, 
                               adjust_lr_patience=early_stop_patience, 
                               save_path=save_path)

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