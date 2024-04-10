import pdb
import torch
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.utils.loss import LogisticLoss, HingeLoss
from graph_embeddings.utils.trainer import Trainer
import argparse

# import loggers
import wandb
from graph_embeddings.utils.logger import JSONLogger
from graph_embeddings.utils.dataloader import CustomGraphDataLoader
from graph_embeddings.data.make_datasets import get_data_from_torch_geometric

from utils.config import Config

def run_experiment(config: Config, 
                   device: str = 'cpu', 
                   results_folder: str = 'results', 
                   experiment_name: str = 'experiment', 
                   loglevel=2):
    # Load and prepare your data
    dataset_path = config.get("dataset_path")
    # adj = load_adj(dataset_path).to(config.get('device'))


    cfg = Config("./configs/config.yaml")
    raw_path = cfg.get("data", "raw_path")

    dat = config.get("dataset_ref")
    if not dat:
        raise ValueError("Please specify a dataset_ref in the experiment config file")
    if "pytorch-geometric" in dat.lower(): 
        src_split = dat.split("/")
        dataset = get_data_from_torch_geometric(src_split[1], src_split[2], raw_path)
        data = dataset[0]
    elif "snapdataset" in dat.lower():
        src_split = dat.split("/")
        dataset = get_data_from_torch_geometric(src_split[0], src_split[1], raw_path)
        data = dataset[0]
    # Get first graph in dataset
    data = dataset[0]

    # Either use the batch size from the config or set it to the number of nodes i.e. the whole graph
    batch_size = config.get('batch_size') or int(data.num_nodes)
    dataloader = CustomGraphDataLoader(data, batch_size=batch_size)
    
    model_types = config.get('model_types')
    loss_types = config.get('loss_types')

    for model_type in model_types:
        for loss_type in loss_types:
            print(f"# Training {model_type} model with {loss_type}...")

            # Determine the model and loss function based on config
            model_class = {'PCA': PCAModel, 'L2': L2Model}[model_type]
            loss_fn = {'logistic': LogisticLoss, 'hinge': HingeLoss}[loss_type]()
            
            load_ckpt = config.get('load_ckpt')
            model_init = config.get('model_init') or 'random'
            
            loggers = {0: [], 1: [JSONLogger], 2: [JSONLogger, wandb], 3: [wandb]}[loglevel]
            
            # Initialize the trainer
            trainer = Trainer(dataloader=dataloader, model_class=model_class, 
                              loss_fn=loss_fn, model_init=model_init,
                              threshold=1e-10, num_epochs=config.get("num_epochs"),
                              device=device, loggers=loggers, dataset_path=dataset_path, 
                              save_ckpt=results_folder, load_ckpt=load_ckpt)
            
            # If rank_range is specified, search for the optimal rank
            rank_range = config.get('rank_range')
            if rank_range:
                trainer.find_optimal_rank(rank_range['min'], 
                                        rank_range['max'], 
                                        lr=config.get('lr'), 
                                        early_stop_patience=config.get('early_stop_patience'), 
                                        experiment_name=experiment_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--experiment', type=str, default=None, help='Run a specific experiment')
    parser.add_argument('--loglevel', default="2", choices=["0","1","2"], help='Log level [0: nothing, 1: logs to JSON, 2: logs to JSON and WANDB, 3: logs to WANDB only]')
    parser.add_argument('--loss', type=str, default=None, help='Loss function to use (logistic, hinge, poisson). Default uses config file, only pass this argument to overwrite it.')
    parser.add_argument('--model', type=str, default=None, help='Model to use (PCA, L2). Default uses config file, only pass this argument to overwrite it.')

    args = parser.parse_args()
    device = args.device
    args.loglevel = int(args.loglevel)

    main_config_path = 'configs/config.yaml'
    main_config = Config(main_config_path)

    print(f"Running experiments from config {main_config_path}...")
    print(f"Using device: {device}")
    
    if args.all:
        for experiment in main_config.get('experiments'):
            print(f"Running experiment {experiment['name']} from config {experiment['config_path']}...")
            exp_config = Config(experiment['config_path'])

            # check if loss and model are specified
            if args.loss:
                exp_config.set('loss_types', [args.loss])
            if args.model:
                exp_config.set('model_types', [args.model])

            # print the whole config
            print(f"{'='*50}\n{exp_config}\n{'='*50}")

            run_experiment(config=exp_config, device=device, experiment_name=experiment['name'], loglevel=args.loglevel)
    
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

        run_experiment(config=exp_config, device=device, experiment_name=exp_name, loglevel=args.loglevel)

if __name__ == '__main__':
    main()