import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.utils.load_data import load_adj
from graph_embeddings.utils.logger import JSONLogger

from graph_embeddings.utils.nearest_neighbours_reconstruction_check import get_edge_index_embeddings, compare_edge_indices

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
                 dataset_path='not specified',
                 reconstruction_check="frob"):
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
        self.dataset_path = dataset_path

        assert reconstruction_check in ["frob", "neigh"]
        self.reconstruction_check = reconstruction_check



    def calc_frob_error_norm(self, logits, adj):
        """Compute the Frobenius error norm between the logits and the adjacency matrix."""
        clipped_logits = torch.clip(logits, min=0, max=1)
        return torch.linalg.norm(clipped_logits - adj) / torch.linalg.norm(adj)

    def init_model(self,
                   rank: int|None = None):
            if self.model_init == 'random':
                assert rank is not None
                model = self.model_class.init_random(self.dataloader.num_total_nodes, self.dataloader.num_total_nodes, rank).to(self.device)
            elif self.model_init == 'load':
                model_params = torch.load(self.load_ckpt,map_location=self.device)
                model = self.model_class(*model_params)
            # elif self.model_init == 'pre-svd':
            #     assert rank is not None
            #     U,_,V = torch.svd_lowrank(self.adj, q=rank)
            #     model = self.model_class.init_pre_svd(U, V).to(self.device)
            # elif self.model_init == 'post-svd':
            #     model_params = torch.load(self.load_ckpt,map_location=self.device)
            #     model = self.model_class.init_post_svd(*model_params)
            else:
                raise Exception(f"selected model initialization ({self.model_init}) is not currently implemented")
                
            return model

    def train(self, 
              rank: int, 
              model: L2Model|PCAModel|None = None,
              lr: float = 0.01, 
              early_stop_patience: float = None, 
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
        dataset_path = self.dataset_path
        full_reconstruction = False
        batch_size = self.dataloader.batch_size

        # ----------- Initialize logging -----------
        # get loss_fn function name
        loss_fn_name = loss_fn.__class__.__name__
        # get self.model_class function name
        model_class_name = self.model_class.__name__

        # initialize logging to all loggers
        for logger in self.loggers:
            logger.init(project=self.project_name, 
                        config={'rank': rank, 
                                'num_epochs': num_epochs, 
                                'learning_rate': lr,
                                'loss_fn': loss_fn_name, 
                                'model_class': model_class_name,
                                'dataset_path': dataset_path,
                                'early_stop_patience': early_stop_patience,
                                'batch_size': batch_size
                                })


        """
        # ----------- Shift adjacency matrix -----------
        # shift adj matrix to -1's and +1's
        if loss_fn_name == 'PoissonLoss':
            print("using adjacency")
            A = self.adj # adjacency matrix used in loss function
        else:
            print("using shifted adjacency")
            A = self.adj*2 - 1 # shifted adjacency matrix used in loss function
        # # ----------- Shift adjacency matrix -----------
        # # shift adj matrix to -1's and +1's
        # adj_s = self.adj*2 - 1
        """
            
        if self.reconstruction_check == "frob":
            self.adj = self.dataloader.full_adj.to(self.device) # ! used for small graphs for FROB

        # ----------- Optimizer ----------- 
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # ----------- Scheduler -----------
        step_size = 1
        gamma = .5  # Decay factor
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # ----------- Training loop -----------
        with tqdm(range(num_epochs)) as pbar:

            best_loss = float('inf')  # Initialize best_loss to a very high value
            epochs_no_improve = 0  # Counter to keep track of epochs with no improvement

            for epoch in pbar:
                """
                # Forward pass
                optimizer.zero_grad()
                A_hat = model.reconstruct()
                loss = loss_fn(A_hat, A)
                loss.backward()
                optimizer.step()
                """

                losses = []

                for b_idx, batch in enumerate(self.dataloader):
                    batch.to(self.device)
                    # Forward pass
                    optimizer.zero_grad()
                    A_hat = model.reconstruct(batch.indices) 
                    loss = loss_fn(A_hat, batch.adj_s)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                epoch_loss = sum(losses) / len(losses)


                if self.reconstruction_check == "frob":
                    if epoch % 100 == 0: # ! only check every {x}'th epoch

                        last_frob_epoch = epoch
                        # Compute Frobenius error for diagnostics
                        with torch.no_grad():  # Ensure no gradients are computed in this block
                            A_hat = model.reconstruct()
                            frob_error_norm = self.calc_frob_error_norm(A_hat, self.adj)

                        # Log metrics to all loggers
                        metrics = {'epoch': epoch, 'loss': epoch_loss, 'frob_error_norm': frob_error_norm.item()}
                        for logger in self.loggers:
                            logger.log(metrics)

                        fully_reconstructed = frob_error_norm <= self.threshold

                    # update progress bar
                    pbar.set_description(f"{model.__class__.__name__} rank={rank}, loss={epoch_loss:.1f} frob_err@{last_frob_epoch}={frob_error_norm or .0:.4f}")
                        
                elif self.reconstruction_check == "neigh":
                    if epoch % 100 == 0: # ! only check every {x}'th epoch

                        last_recons_check_epoch = epoch

                        # Compute Frobenius error for diagnostics
                        with torch.no_grad():
                            edge_index_from_neighbors = get_edge_index_embeddings(model.X, model.Y, model.beta)
                            common_edges = compare_edge_indices(self.dataloader.data.edge_index, edge_index_from_neighbors)
                            perc_edges_reconstructed = len(common_edges) / self.dataloader.data.edge_index.size(1) * 100
                        
                        fully_reconstructed = len(common_edges) == self.dataloader.data.num_edges

                    # update progress bar
                    pbar.set_description(f"{model.__class__.__name__} rank={rank}, loss={epoch_loss:.1f} edges_reconstructed@{last_recons_check_epoch}={perc_edges_reconstructed:.2f}%")
                        

                
                # Break if fully reconstructed
                if fully_reconstructed:
                    pbar.close()
                    for logger in self.loggers:
                        logger.config.update({'full_reconstruction': True})
                    full_reconstruction = True
                    print(f'Full reconstruction at epoch {epoch} with rank {rank}')
                    break

                if early_stop_patience is not None:
                    # Early stopping condition based on loss improvement
                    if loss < best_loss:
                        best_loss = loss  # Update best loss
                        epochs_no_improve = 0  # Reset the counter as we have improvement
                    else:
                        epochs_no_improve += 1  # Increment counter if no improvement

                    # Check if early stopping is triggered
                    if epochs_no_improve >= early_stop_patience:
                        for logger in self.loggers:
                            logger.config.update({'early_stop_triggered': True})
                        print(f"Early stopping triggered at epoch {epoch}. No improvement in loss for {early_stop_patience} consecutive epochs.")
                        scheduler.step()
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
        return model

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
            model = self.train(max_rank, model=model, lr=lr, early_stop_patience=early_stop_patience, save_path=save_path)
        
        def compute_svd_target(model):
            X,Y,*_ = model.forward()
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
            model = model.__class__(X@V, Y@V).to(self.device)

            # 3. Train new model
            model = self.train(current_rank, 
                               model=model, 
                               lr=lr, 
                               early_stop_patience=early_stop_patience, 
                               save_path=save_path)


            if self.reconstruction_check == "frob":
                # Calculate the Frobenius error
                logits = model.reconstruct()
                frob_error = self.calc_frob_error_norm(logits, self.adj)
                fully_reconstructed = frob_error <= self.threshold
            
            elif self.reconstruction_check == "neigh":
                # Calculate the Frobenius error
                edge_index_from_neighbors = get_edge_index_embeddings(model.X, model.Y, model.beta)
                common_edges = compare_edge_indices(self.dataloader.data.edge_index, edge_index_from_neighbors)
                fully_reconstructed = len(common_edges) == self.dataloader.data.num_edges

            # Check if the reconstruction is within the threshold
            if fully_reconstructed:
                print(f'Full reconstruction at rank {current_rank}\n')
                optimal_rank = current_rank  # Update the optimal rank if reconstruction is within the threshold
                upper_bound = current_rank - 1  # Try to find a smaller rank
                
                # compute new svd_target for next iteration
                svd_target = compute_svd_target(model)
            else:
                print(f'Full reconstruction NOT found for rank {current_rank}\n')
                lower_bound = current_rank + 1
        print()
        return optimal_rank