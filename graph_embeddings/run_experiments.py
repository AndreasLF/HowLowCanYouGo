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

def run_experiment(config: Config, device: str = 'cpu', results_folder: str = 'results', experiment_name: str = 'experiment', dev=False):
    # Load and prepare your data
    dataset_path = config.get("dataset_path")
    adj = torch.load(dataset_path).to(config.get('device'))
    
    model_types = config.get('model_types')

    for model_type in model_types:
        print(f"# Training {model_type} model...")

        # Determine the model and loss function based on config
        model = LPCAModel if model_type == 'LPCA' else L2Model
        loss_fn = lpca_loss if model_type == 'LPCA' else L2_loss
        

        loggers = [JSONLogger] if dev else [JSONLogger, wandb]
        # Initialize the trainer
        trainer = Trainer(adj=adj, model_class=model, loss_fn=loss_fn, 
                        threshold=1e-7, num_epochs=config.get("num_epochs"), optim_type=config.get('optim_type'), 
                        device=device, max_eval=config.get('max_eval'), loggers=loggers, dataset_path=dataset_path, save_ckpt=results_folder)
        
        # If rank_range is specified, search for the optimal rank
        rank_range = config.get('rank_range')
        if rank_range:
            trainer.find_optimal_rank(rank_range['min'], rank_range['max'], lr=config.get('lr'), early_stop_patience=config.get('early_stop_patience'), experiment_name=experiment_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--experiment', type=str, default=None, help='Run a specific experiment')
    parser.add_argument('--dev', action='store_true', help='Run in development mode, i.e. without WANDB logging')
    
    args = parser.parse_args()
    device = args.device

    main_config_path = 'configs/config.yaml'
    main_config = Config(main_config_path)

    print(f"Running experiments from config {main_config_path}...")
    print(f"Using device: {device}")
    
    if args.all:
        for experiment in main_config.get('experiments'):
            print(f"Running experiment {experiment['name']} from config {experiment['config_path']}...")
            exp_config = Config(experiment['config_path'])

            print("="*50)
            # print the whole config
            print(exp_config)
            print("="*50)

            run_experiment(config=exp_config, device=device, experiment_name=experiment['name'], dev=args.dev)
    
    else:
        # check if experiment is specified
        if args.experiment is None:
            raise ValueError("Please specify an experiment to run")

        exp_name = args.experiment
        # get experiment with name args.experiment from main_config
        config_path = [exp["config_path"] if exp["name"] == exp_name else None for exp in main_config.get('experiments')]
        config_path = [path for path in config_path if path is not None][0]

        print(f"Running experiment {exp_name} from config {config_path}...")

        if config_path is None:
            raise ValueError(f"Experiment {exp_name} not found in main config")

        exp_config = Config(config_path)

        print("="*50)
        # print the whole config
        print(exp_config)
        print("="*50)

        run_experiment(config=exp_config, device=device, experiment_name=exp_name, dev=args.dev)

if __name__ == '__main__':
    main()