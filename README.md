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




## Get Data
The data is downloaded from different sources and stored in the `data` folder.
The data sources are defined in the `configs/config.yaml` file. Here there are either urls to the data or a specification of the [Pytorch Geometric dataset](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html) to be used.

To get the data you can either us

### Get data with DVC
If you have access to our data version control folder on Google Drive you can use the following command to get the data:
```bash
dvc pull
```
This will download the data from the Google Drive folder and place it in the `data` folder. The graph adjacency matrices are stored in the folder specified in the `configs/config.yaml` file.

### Get data without DVC
If you do not have access to our data version control folder on Google Drive you have to download the data from the sources and preprocess it yourself. We have tried to make the process as seamless as possible. 
Just run the following command:
```bash
make datasets
```
This should download the data from the sources and preprocess it into adjacency matrices with edge weights of either 1 or 0.


## Run experiments to find lowest rank representation for LPCA and L2
To run an experiment you can use:
```bash
make run_experiments ARGS="--device cpu --experiment Cora1"
```
The following arguments can be used:
<!-- Create table with args -->
| Argument | Description | Default |
| --- | --- | --- |
| `--device` | Device to run the experiments on. | `cpu` |
| `--all` | Run all experiments. | `False` |
| `--experiment` | Run a specific experiment. This should be defined in the main config. | `None` |
| `--dev` | Run in development mode, i.e. without WANDB logging. | `False` |

This will run the experiments defined in the `configs/config.yaml` file.

### Experiment definition (main config)
The experiments are defined in the `configs/config.yaml` file. Here you can define the experiments with names and an experiment configuration. An example can be seen below:
```yaml
experiments:
  - name: Cora1 # Name of the experiment
    config_path: './configs/experiments/exp1_cora.yaml' # Path to the experiment configuration file
```

### Experiment configuration file
The experiment configuration file is a yaml file that defines the experiment. An example can be seen below:
```yaml
dataset_path: './data/adj_matrices/Cora.pt' # Path to the dataset
model_types: ['L2', 'PCA']
loss_types: ['logistic', 'hinge'] # Which models to run
num_epochs: 50_000 # Number of epochs to run
model_init: 'random' # Model initialization, random or svd
lr: 0.1 # Learning rate
early_stop_patience: 500 # Early stopping patience, if no improvement in loss after this number of consecutive epochs, stop
rank_range: # Range of ranks to make binary search over
  min: 1 # Minimum rank
  max: 50 # Maximum rank
```