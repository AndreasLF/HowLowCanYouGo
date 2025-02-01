# GraphEmbeddings


## Setting up the project
First create a virtual environment and install the requirements and the project itself.

```bash
# Create virtual environment
python -m venv venv
# Activate 
source venv/bin/activate # Linux
venv\Scripts\activate # Windows
# Install requirements make
make requirements # make
```

> [!NOTE]
> Alternatively you can use [Conda](https://docs.conda.io/en/latest/) to create a virtual environment.


## Logging and W&B Integration
This project supports two types of logging:
- JSON Logger – Stores logs locally in a structured JSON format.
 - W&B (Weights & Biases) – A powerful tool for experiment tracking, visualization, and logging.

[Weights & Biases (W&B)](https://docs.wandb.ai/quickstart/) is used to track experiments, store results, and some plots are made based on these experiment logs.

> [!NOTE]
> If W&B is not used, many plots and visualizations will not work, as they retrieve experiment data directly from W&B through the API.

## Get Data

The datasets are obtained from multiple sources and converted into adjacency matrices for processing. The sources are defined in the `configs/config.yaml` file. The data is automatically downloaded and preprocessed using the `make datasets` command, which calls the `make_datasets.py` script.

### 📥 Download and Preprocess Data

To download and preprocess all required datasets, simply run:

```bash
make datasets
```

### 📊 Generate Dataset Statistics Table (Table 1)

To generate **dataset statistics** such as the number of nodes, edges, degree distribution, clustering coefficient, and other structural properties, run:

```bash
make get_stats
```

This command will:
- Compute key graph statistics for each dataset.
- Store the results in a CSV file: ``<results_directory>/adj_matrix_stats.csv``
 You can configure the exact path in ``configs/config.yaml`` under:
```yaml
results:
  stats_path: "./results/stats"
```
- Format the output into a LaTeX table


## Run Experiments to Find Lowest Rank Representation

Experiments can be run using the following command:

```bash
make run_experiments ARGS="--device <DEVICE> --experiment <EXPERIMENT_NAME> [OPTIONS]"
```


The following arguments are available. Some will overide the experiment config parameters.

| Argument        | Description                                                                                          |
|---------------|--------------------------------------------------------------------------------------------------|
| `--device`     | Specify the computation device (`cpu` or `cuda`).                                                 |
| `--all`        | Run all experiments in `configs/config.yaml`.                                                      |
| `--experiment` | Run a specific experiment by name.                                                                |
| `--loglevel`   | Logging level: `0` (nothing), `1` (JSON logs), `2` (JSON + WANDB), `3` (WANDB only).               |
| `--recons-check` | Method for checking full reconstruction: `frob` (Frobenius error) or `neigh` (nearest neighbors in embedding space). |
| `--loss`       | Override the loss function (options: `logistic`, `hinge`, `poisson`).                              |
| `--model`      | Override the model type (options: `PCA`, `L2`, `Hyperbolic`, `LatentEigen`).                       |




### Example: Running a Specific Experiment 

To run an experiment, specify its name as defined in ``configs/config.yaml``. 

```bash
make run_experiments ARGS="--device cpu --experiment Cora1"
```

This will run an experiment according to the specific experiment config in ``configs/experiments``

### Example: Running All Experiments

To run all experiments defined in `configs/config.yaml`, use:

```bash
make run_experiments ARGS="--all"
```

This will execute all configured experiments sequentially.




### ⚙️ Experiment Configuration

Each experiment is defined in a YAML configuration file located in `configs/experiments/`and referenced in `configs/config.yaml`. These configurations specify which dataset, models, and hyperparameters to use for training.
### 📁 Structure of an Experiment Config File

A typical experiment configuration file (.yaml) includes:

```yaml
dataset_path: './data/adj_matrices/Cora.pt'  # Path to the dataset
dataset_ref: 'PyTorch-Geometric/Planetoid/Cora'  # Reference to dataset source
model_types: ['L2', 'PCA']  # Models to train
loss_types: ['poisson', 'hinge', 'logistic']  # Loss functions to apply
num_epochs: 30_000  # Total number of epochs
model_init: 'random'  # Model initialization ('random' or 'svd')
lr: 1.0  # Learning rate
early_stop_patience: 200  # Stop training if no improvement for this many epochs
rank_range:  # Rank search space for finding the lowest embedding dimension
  min: 1  # Minimum rank
  max: 64  # Maximum rank
```

All the config parameters are described below

| Parameter            | Description |
|----------------------|------------|
| `dataset_path`      | Path to the dataset adjacency matrix file. |
| `dataset_ref`       | Reference to the dataset source (e.g., PyTorch Geometric). |
| `model_types`       | List of models to train (`L2`, `PCA`, `Hyperbolic` , `LatentEigen`, etc.). |
| `loss_types`        | List of loss functions (`poisson`, `hinge`, `logistic`). |
| `num_epochs`        | Number of training epochs. |
| `model_init`        | Model initialization method (`random` or `svd`). |
| `lr`               | Learning rate for optimization. |
| `early_stop_patience` | Stop training if loss doesn't improve for this many epochs. |
| `rank_range`        | Search range for rank selection (used in binary search). |
