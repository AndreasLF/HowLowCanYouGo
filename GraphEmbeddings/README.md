# GraphEmbeddings


This directory contains the code for learning **low-dimensional metric embeddings** of complex networks. It provides implementations of various embedding models, experiment configurations, and tools for evaluation and visualization. Specifically, this part of the repo covers:
- **Graph embedding models**: Implementations of L2, Hyperbolic, PCA-based, and Latent Eigen embeddings.
- **Loss functions**: Implementations of logistic Loss, hinge Loss, and poisson Loss for optimizing embeddings.
- **Efficient dimension search**: An algorithm to search for the lowest exact embedding dimension (EED).
- **Experiment tracking & visualization**: Tools for logging experiments and generating plots.

For details on running experiments and reproducing results, see the sections below.


##  🚀 Setting up the project
Get started quickly by setting up your environment and installing dependencies.

Using **venv**:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate      # On Windows
```

Or with **Conda**:
```bash
conda create --name graph-embeddings python=3.9
conda activate graph-embeddings
```

Run the following to install required packages:
```bash
make requirements
```

For development purposes, install the package in editable mode:


```bash
pip install -e .
````


## 📂 File Structure
Below is an overview of the key files and folders:
```bash
GraphEmbeddings/ 
├── configs/ # Experiment configurations 
│ ├── config.yaml # Global configuration file 
│ └── experiments/ # Dataset-specific experiment configs 
├── graph_embeddings/ # Core implementation 
│ ├── models/ # Graph embedding models (L2, Hyperbolic, PCA, etc.) 
│ ├── plotting/ # Scripts for visualization & plotting 
│ ├── utils/ # Helper functions (logging, data loading, etc.) 
│ ├── train.py # Training script for embeddings 
│ ├── run_experiments.py # Main script to execute experiments 
├── notebooks/ # Jupyter notebooks for exploration 
├── jobscripts/ # Job submission scripts (see README inside) 
├── requirements.txt # Dependencies list 
├── Makefile # Automation commands 
├── README.md # Documentation 
└── results/ # Stores experiment outputs
```

## 🧩   Datasets

The datasets are obtained from multiple sources and converted into adjacency matrices for processing. The sources are defined in the `configs/config.yaml` file. The data is automatically downloaded and preprocessed using the `make datasets` command, which calls the `make_datasets.py` script.

### Download and Preprocess Data

To download and preprocess all required datasets, simply run:

```bash
make datasets
```

This downloads and prepares the datasets defined in `configs/config.yaml`. 

### Generate Dataset Statistics Table (Table 1)

To generate **dataset statistics** such as the number of nodes, edges, degree distribution, clustering coefficient, and other structural properties, run:

```bash
make get_stats
```

This will:
- Compute graph statistics for each dataset.
- Save the results in a CSV file: <results_directory>/adj_matrix_stats.csv.
- Format the output as a LaTeX table for easy inclusion in reports.


## 🔬 Models & Implementations

This project implements multiple **graph embedding models** to learn low-dimensional representations of complex networks. The models are designed to efficiently preserve graph structure and enable exact network reconstruction.

### Implemented Models  
The following embedding models are available in **`graph_embeddings/models/`**:

- **L2 Model (`L2Model.py`)** 
- **Hyperbolic Model (`HyperbolicModel.py`)** 
- **PCA-based Model (`PCAModel.py`)** 
- **Latent Eigenmodel (`LatentEigenModel.py`)**

### Loss Functions  
Each model optimizes embeddings using different loss functions, implemented in **`graph_embeddings/utils/loss.py`**:
- **Logistic Loss**
- **Hinge Loss** 
- **Poisson Loss**

Modify the configuration file to select different models, loss functions, and hyperparameters.
For more details on defining custom models or modifying training behavior, explore `graph_embeddings/models/`

### Running Models  
To train a single model and generate embeddings, use `graph_embeddings/train.py`, e.g.:

```bash
python graph_embeddings/train.py --model-type L2 --loss-type hinge
```

For help with available parameters such as learning rate, epochs, etc. run:
```bash
python graph_embeddings/train.py --help
```





## 🧪 Run Experiments to Find Lowest Rank Representation

This section explains how to define experiments to find the **lowest exact embedding dimension (EED)** for different graph datasets in the config and run them.



### Experiment Configuration

Each experiment is defined in a YAML configuration file located in `configs/experiments/`and referenced in `configs/config.yaml`. These configurations specify which dataset, models, and hyperparameters to use for training.

#### Reference an experiment file in `configs/config.yaml`
In `configs/config.yaml` each experiment has a name and an experiment config path:

```yaml
experiments:
  - name: Cora1
    config_path: './configs/experiments/exp1_cora.yaml'
  - name: Cora2
    config_path: './configs/experiments/exp2_cora.yaml'
  ...
```

#### Structure of an Experiment Config File in `configs/experiments/`

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

#### Config parameter overview:

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


When the configs are set up you are ready to run experiments.



### Running experiments

Experiments can be executed using the following command:

```bash
make run_experiments ARGS="--device <DEVICE> --experiment <EXPERIMENT_NAME> [OPTIONS]"
```


Some arguments override the experiment configuration parameters:

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

## 📊 Visualization & Plotting

Located in **`graph_embeddings/plotting/`**, the following scripts generate key visualizations:

- **`batching_plotCora.py`** – Plot that compares the frobenius reconstruction error (y) per epoch (x) for sampling methods random node sampling and case control sampling with HBDM ([HBDM implementation](https://github.com/AndreasLF/HowLowCanYouGo/tree/master/HBDM-for-EED-search)) while experiments with case control and random node sampling are controlled in the experiment configs and with arguments.
- **`batching_plots.py`** & **`batching_plot2.py`** – Displaying some other batching experiments to getting an understanding (not presented in the paper).
- **`plot_frob_errors.py`** – Plots the frobenius reconstruction error for different batch sizes to compare convergence statistics.  
- **`plot_hot_n_cold.py`** – Plots the frobenisus reconstruction error throughout training 
using the SVD hot start approach (part of algorithm 1) and a random initialization cold start.
- **`hbdm_plots.py`** – Plots missclassified dyads for each of the ranks in the HBDM seach for EED ([HBDM implementation](https://github.com/AndreasLF/HowLowCanYouGo/tree/master/HBDM-for-EED-search)).
- **`plotter.py`** – Just a wrapper to apply some constistent plotting styles. Only some elements of the class is used in the plotting functions.

> [!IMPORTANT]  
> Many of the plotting scripts fetches data from Weights & Biases (W&B) with the experiment IDs hardcoded in the plotting scripts. Make sure you run experiments and add the experiment IDs to the code before running the plotting scripts.

> [!NOTE]  
> Many visualizations can also be done directly in W&B. These plotting scripts are therefore for the most part used to make prettier paper ready plots.

> [!NOTE]  
> Some of the visualizations used data from the HBDM implementation (see [HBDM implementation](https://github.com/AndreasLF/HowLowCanYouGo/tree/master/HBDM-for-EED-search)).


## Logging and W&B Integration
This project supports two types of logging:
- JSON Logger – Stores logs locally in a structured JSON format.
 - W&B (Weights & Biases) – A powerful tool for experiment tracking, visualization, and logging.

[Weights & Biases (W&B)](https://docs.wandb.ai/quickstart/) is used to track experiments, store results, and some plots are made based on these experiment logs.


When running experiments you can set the loglevel: 

```bash
make run_experiments ARGS="--device cuda --experiment Cora1 --loglevel 2"
```
This runs the experiment logging both to a json and wandb.


| Log Level | Description                        |
|-----------|------------------------------------|
| `0`       | No logging.                       |
| `1`       | JSON logs only (local storage).   |
| `2`       | JSON + W&B logging.               |
| `3`       | W&B logging only.                 |


> [!NOTE]  
> Since we implemented plotting functions retrieving data from W&B, we have not used the JSON logging. Recommendation is to use loglevel 3, or 0 for quick testing. 

##  🏗️ Running experiments on HPC cluster
We are using the HPC cluster at the [Techincal University of Denmark](https://www.dtu.dk/) to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).
For running experiments on compute clusters, job submission templates and instructions are provided in the **`jobscripts/`** folder.
Most experiments are run on the NVIDIA Tesla A100 and A10 GPUs.