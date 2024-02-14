import torch
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss, L2_loss
from graph_embeddings.utils.trainer import Trainer
import argparse

# import loggers
import wandb
from graph_embeddings.utils.logger import JSONLogger

from utils.config import Config

def run_experiment(config: Config, device: str = 'cpu', results_folder: str = 'results', experiment_name: str = 'experiment'):
    # Load and prepare your data
    adj = torch.load(config.get("dataset_path")).to(config.get('device'))
    
    model_types = config.get('model_types')

    for model_type in model_types:
        print(f"# Training {model_type} model...")

        # Determine the model and loss function based on config
        model = LPCAModel if model_type == 'LPCA' else L2Model
        loss_fn = lpca_loss if model_type == 'LPCA' else L2_loss
        
        # Initialize the trainer
        trainer = Trainer(adj=adj, model_class=model, loss_fn=loss_fn, 
                        threshold=1e-7, num_epochs=config.get("num_epochs"), optim_type=config.get('optim_type'), 
                        device=device, max_eval=config.get('max_eval'), loggers=[JSONLogger, wandb])
        
        # If rank_range is specified, search for the optimal rank
        rank_range = config.get('rank_range')
        if rank_range:
            trainer.find_optimal_rank(rank_range['min'], rank_range['max'], experiment_name=experiment_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    device = args.device

    main_config_path = 'configs/config.yaml'
    main_config = Config(main_config_path)

    print(f"Running experiments from config {main_config_path}...")
    print(f"Using device: {device}")
    
    for experiment in main_config.get('experiments'):
        print(f"Running experiment {experiment['name']} from config {experiment['config_path']}...")
        exp_config = Config(experiment['config_path'])

        print("="*50)
        # print the whole config
        print(exp_config)
        print("="*50)

        run_experiment(exp_config, device, experiment['name'])

if __name__ == '__main__':
    main()