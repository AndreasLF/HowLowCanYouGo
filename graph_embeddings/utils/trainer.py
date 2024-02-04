import torch
import torch.nn as nn
import torch.optim as optim

from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss


# Ensure CUDA is available and select device, if not check for Macbook Pro support (MPS) and finally use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class Trainer:
    def __init__(self, adj, model_class, loss_funct, threshold, num_epochs, optim_type='lbfgs', device='cpu'):
        self.adj = adj.to(device)
        self.model_class = model_class
        self.loss_funct = loss_funct
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.optim_type = optim_type
        self.device = device

    def calc_frob_error_norm(self, logits, adj):
        """Compute the Frobenius error norm between the logits and the adjacency matrix."""
        clipped_logits = torch.clip(logits, min=0, max=1)
        return torch.linalg.norm(clipped_logits - adj) / torch.linalg.norm(adj)

    def train(self, rank, print_loss_interval=10):
        """ Train the model using the given optimizer and loss function.
        
        Args:
            rank (int): The rank of the model
            print_loss_interval (int): The interval at which the loss is printed. Default is 10.
        
        Returns:
            U (np.ndarray): The left singular vectors
            V (np.ndarray): The right singular vectors
        
        """
        adj = self.adj.to(device)
        model = self.model_class(adj.size(0), adj.size(1), rank).to(device)
        loss_funct = self.loss_funct
        optim_type = self.optim_type
        num_epochs = self.num_epochs


        # shift adj matrix to -1's and +1's
        adj_s = adj*2 - 1

        def closure():
            """Closure function for LBFGS optimizer. This function is called internally by the optimizer."""
            optimizer.zero_grad()
            logits = model.forward() 
            loss = loss_funct(logits, adj_s) 
            loss.backward()
            return loss

        if optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=0.1)
        elif optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        elif optim_type == 'lbfgs':
            optimizer = optim.LBFGS(model.parameters(), lr=0.01)
        else:
            raise ValueError(f'Optimizer {optim_type} not supported')

        for epoch in range(num_epochs):

            if optim_type == 'lbfgs':
                # LBFGS optimizer step takes the closure function and internally calls it multiple times
                loss = optimizer.step(closure)
            else: 
                # Forward pass
                optimizer.zero_grad()
                logits = model.forward()  # Or simply model() if you have defined the forward method
                loss = loss_funct(logits, adj_s)  # Ensure lpca_loss is compatible with PyTorch and returns a scalar tensor
                loss.backward()
                optimizer.step()


            # evaluate the loss
            with torch.no_grad():  # Ensure no gradients are computed in this block
                logits = model.forward()
                frob_error_norm = self.calc_frob_error_norm(logits, adj)

            # Break if Froebenius error is less than 1e-7
            if frob_error_norm < self.threshold:
                print(f'Epoch {epoch}, Loss: {loss.item()}, Frobenius error: {frob_error_norm}')
                break
                
            if epoch % print_loss_interval == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}, Frobenius error: {frob_error_norm}')


        # After training, retrieve parameters
        with torch.no_grad():  # Ensure no gradients are computed in this block
            U, V = model.U, model.V

        return U, V

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