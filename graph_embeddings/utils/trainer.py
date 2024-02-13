import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from graph_embeddings.utils.logger import JSONLogger

class Trainer:
    def __init__(self, adj, model_class, loss_fn, threshold, num_epochs, optim_type='lbfgs', max_eval=25, device='cpu', loggers=[JSONLogger], project_name='GraphEmbeddings'):
        self.adj = adj.to(device)
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.optim_type = optim_type
        self.max_eval = max_eval
        self.device = device
        self.loggers = loggers
        self.project_name = project_name

    def calc_frob_error_norm(self, logits, adj):
        """Compute the Frobenius error norm between the logits and the adjacency matrix."""
        clipped_logits = torch.clip(logits, min=0, max=1)
        return torch.linalg.norm(clipped_logits - adj) / torch.linalg.norm(adj)

    def train(self, rank, lr=0.01):
        """ Train the model using the given optimizer and loss function.
        
        Args:
            rank (int): The rank of the model
            print_loss_interval (int): The interval at which the loss is printed. Default is 10.
        
        Returns:
            U (np.ndarray): The left singular vectors
            V (np.ndarray): The right singular vectors
        """

        adj = self.adj.to(self.device)
        model = self.model_class(adj.size(0), adj.size(1), rank).to(self.device)
        loss_fn = self.loss_fn
        optim_type = self.optim_type
        num_epochs = self.num_epochs


        # ----------- Initialize logging -----------
        # get loss_fn function name
        loss_fn_name = loss_fn.__name__
        # get self.model_class function name
        model_class_name = self.model_class.__name__

        # initialize logging to all loggers
        for logger in self.loggers:
            logger.init(project=self.project_name, 
                        config={'rank': rank, 
                                'num_epochs': num_epochs, 
                                'learning_rate': lr,
                                'optim_type': optim_type,
                                'loss_fn': loss_fn_name, 
                                'model_class': model_class_name})


        # ----------- Shift adjacency matrix -----------
        # shift adj matrix to -1's and +1's
        adj_s = adj*2 - 1

        # ----------- Closure function (LBFGS) -----------
        def closure():
            """Closure function for LBFGS optimizer. This function is called internally by the optimizer."""
            optimizer.zero_grad()
            A_hat = model.reconstruct()
            loss = loss_fn(A_hat, adj_s) 
            loss.backward()
            return loss

        # ----------- Optimizer ----------- 
        if optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optim_type == 'lbfgs':
            optimizer = optim.LBFGS(model.parameters(), lr=lr, max_eval=self.max_eval)
        else:
            raise ValueError(f'Optimizer {optim_type} not supported')


        # ----------- Training loop -----------
        with tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                
                if optim_type == 'lbfgs':
                    # LBFGS optimizer step takes the closure function and internally calls it multiple times
                    loss = optimizer.step(closure)
                else: 
                    # Forward pass
                    optimizer.zero_grad()
                    A_hat = model.reconstruct() 
                    loss = loss_fn(A_hat, adj_s)  # Ensure lpca_loss is compatible with PyTorch and returns a scalar tensor
                    loss.backward()
                    optimizer.step()

                # Compute and print the Frobenius norm for diagnostics
                with torch.no_grad():  # Ensure no gradients are computed in this block
                    A_hat = model.reconstruct()
                    frob_error_norm = self.calc_frob_error_norm(A_hat, adj)
                    pbar.set_description(f"epoch={epoch}, loss={loss:.1f} Frobenius error: {frob_error_norm}")

                # Log metrics to all loggers
                metrics = {'epoch': epoch, 'loss': loss.item(), 'frob_error_norm': frob_error_norm.item()}
                for logger in self.loggers:
                    logger.log(metrics)

                # Break if Froebenius error is less than 1e-7
                if frob_error_norm < self.threshold:
                    pbar.close()
                    for logger in self.loggers:
                        logger.config.update({'full_reconstruction': True})
                    print(f'Full reconstruction at epoch {epoch} with rank {rank}')
                    break
        
        # After training, retrieve parameters
        with torch.no_grad():  # Ensure no gradients are computed in this block
            final_outputs = model.forward()

            # return final_outputs
            return final_outputs

    def find_optimal_rank(self, min_rank, max_rank):
        """Find the optimal rank for the model using binary search. 

        Args:
            min_rank (int): The minimum rank
            max_rank (int): The maximum rank
        
        Returns:
            optimal_rank (int): The optimal rank
        
        """

        lower_bound = min_rank
        upper_bound = max_rank
        optimal_rank = upper_bound  # Assume the worst case initially
        thr = self.threshold

        print('='*50)
        print(f'Finding optimal rank between {min_rank} and {max_rank}')
        print('='*50)

        while lower_bound <= upper_bound:
            current_rank = (lower_bound + upper_bound) // 2
            print(f'Training model with rank {current_rank}')

            # Set the threshold for the Frobenius error
            thr = 10e-5

            # Train the model
            U, V = self.train(current_rank)

            # Calculate the Frobenius error
            logits = U @ V
            frob_error = self.calc_frob_error_norm(logits, self.adj)

            # Check if the reconstruction is within the threshold
            if frob_error <= thr:
                print(f'Full reconstruction at rank {current_rank}\n')
                optimal_rank = current_rank  # Update the optimal rank if reconstruction is within the threshold
                upper_bound = current_rank - 1  # Try to find a smaller rank
            else:
                print(f'Full reconstruction NOT found for rank {current_rank}\n')
                lower_bound = current_rank + 1
        print()
        return optimal_rank