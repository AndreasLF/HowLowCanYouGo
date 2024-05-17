import torch
import torch.optim as optim
import argparse

from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.LatentEigenModel import LatentEigenModel
from graph_embeddings.utils.load_data import load_adj
from graph_embeddings.utils.loss import CaseControlLogisticLoss, LogisticLoss, HingeLoss, PoissonLoss
from graph_embeddings.utils.trainer import Trainer

import pdb

# import loggers
import wandb
from graph_embeddings.utils.logger import JSONLogger

from graph_embeddings.utils.dataloader import CaseControlDataLoader, RandomNodeDataLoader
from graph_embeddings.data.make_datasets import get_data_from_torch_geometric
from graph_embeddings.utils.config import Config

import uuid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='PCA', choices=['PCA','L2','LatentEigen'], help='Type of reconstruction model to use {LPCA, L2} (default: %(default)s)')
    parser.add_argument('--loss-type', type=str, default='logistic', choices=['logistic','hinge', 'poisson'], help='Type of loss to use {logistic, hinge, poisson} (default: %(default)s)')
    parser.add_argument('--rank', type=int, default=32, metavar='R', help='dimension of the embedding space (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--load-ckpt', type=str, default='none', help='path to load model checkpoint (ckpt) from (default: %(default)s)')
    parser.add_argument('--save-ckpt', type=str, default='results/model.pt', help='path to save model checkpoint (ckpt) to (default: %(default)s)')
    parser.add_argument('--experiment', type=str, default="PubMed1", help="Experiment config file to use")
    parser.add_argument('--loglevel', type=int, default=0, help="Experiment config file to use")

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load and prepare your data
    # adj = load_adj(f'./data/adj_matrices/{args.data}.pt').to(device)
    
    cfg = Config("./configs/config.yaml")

    exp_name = args.experiment
    # get experiment with name args.experiment from main_config
    exp_config_path = [exp["config_path"] if exp["name"] == exp_name else None for exp in cfg.get('experiments')]
    exp_config_path = [path for path in exp_config_path if path is not None][0]
    print(f"Running single run with parameters from experiment {exp_name} from config {exp_config_path}...")
    if exp_config_path is None:
        raise ValueError(f"Experiment {exp_name} not found in main config")
    exp_cfg = Config(exp_config_path)

    # Dataset
    dataset_ref = exp_cfg.get("dataset_ref")
    dataset_split = dataset_ref.split("/") # e.g. Planetoid/Cora
    raw_path = cfg.get("data", "raw_path")
    dataset = get_data_from_torch_geometric(dataset_split[-2], dataset_split[-1], raw_path)
    # Get first graph in dataset
    data = dataset[0]

    batch_type = exp_cfg.get("batching_type") or "random"
    batchsize_percentage = exp_cfg.get("batch_size_percentage") or 1.0
    if batch_type == 'random':
        print("Using random batching with batch size: ", int(batchsize_percentage*data.num_nodes))
        dataloader = RandomNodeDataLoader(data, batch_size=int(batchsize_percentage*data.num_nodes), dataset_name=dataset.name, shuffle=True)
    elif batch_type == 'casecontrol': 
        print("Using case control batching with batch size: ", int(batchsize_percentage*data.num_nodes), " and negative sampling ratio: ", args.negatives_ratio)
        dataloader = CaseControlDataLoader(data, batch_size=int(batchsize_percentage*data.num_nodes), dataset_name=dataset.name, negative_sampling_ratio=args.negatives_ratio)
        assert args.loss_type == 'logistic', "Case Control batching only implemented for logistic loss!"

    model_init = exp_cfg.get("model_init") or "random"

    model = {"PCA": PCAModel, 
             "L2": L2Model, 
             "LatentEigen": LatentEigenModel}[args.model_type]
    loss_fn = {"logistic": CaseControlLogisticLoss if batch_type == 'casecontrol' else LogisticLoss,
               "hinge": HingeLoss, 
               "poisson": PoissonLoss}[args.loss_type]()

    loggers = {0: [], 1: [JSONLogger], 2: [JSONLogger, wandb], 3: [wandb]}[args.loglevel]

    exp_id = str(uuid.uuid4())
    # Initialize the trainer
    trainer = Trainer(dataloader=dataloader, model_class=model, loss_fn=loss_fn, model_init=model_init,
                      threshold=1e-10, num_epochs=exp_cfg.get("num_epochs") or 10_000, save_ckpt=args.save_ckpt,
                      load_ckpt=args.load_ckpt, device=args.device, 
                      loggers=loggers, reconstruction_check=exp_cfg.get("recons_check") or "frob",
                      exp_id = exp_id)
    
    # Train one model model
    model = trainer.init_model(args.rank)
    trainer.train(args.rank, 
                  model=model,
                  lr=exp_cfg.get("lr"), 
                  eval_recon_freq=exp_cfg.get("eval_recon_freq"),
                    adjust_lr_patience=exp_cfg.get("early_stop_patience"),
                  save_path=args.save_ckpt)

    if args.loglevel == 3 or args.loglevel == 2:
        from graph_embeddings.utils.wandb_api_utils import WANDbAPIUtils

        project_name = "GraphEmbeddings"  # format: "username/projectname"
        wandb_api = WANDbAPIUtils(project_name)

        runs = wandb_api.get_exp_runs(exp_id)

        assert not (len(runs) > 1), "There are multiple runs with this exp_id"
        assert not (len(runs) == 0), "There are no runs with this exp_id"

        wandb_api.tag_run(runs[0], "cold_start")

    