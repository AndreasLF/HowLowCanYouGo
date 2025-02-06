# HBDM for EED Search

This directory contains the implementation of the Hierarchical Block Distance Model (HBDM) adapted from [Nikolaos Nakis' HBDM framework](https://github.com/Nicknakis/HBDM) for Exact Embedding Dimension (EED) search.


## 🚀 Installation & Setup
You can set up the environment using either **virtualenv** or **conda**.

Using **venv**:
```bash
python3.8 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate      # On Windows
```

Or with **Conda**:
```bash
conda create --name graph-embeddings python=3.8
conda activate graph-embeddings
```

Run the following to install required packages:
```bash
pip install -r requirements.txt
```





## 📂 File Structure

Below is an overview of the key files and folders in **HBDM-for-EED-search**:

```bash
HBDM-for-EED-search/ 
├── README.md # Documentation 
├── find_optimal_rank.py # Script for finding the optimal embedding rank, used to run experiments
├── fractal_kmeans_bip.py # Fractal-based clustering implementation 
├── fractal_main_bip.py 
├── jobscripts/ # Job submission scripts (see README inside) 
│ ├── README.md 
│ ├── continue_job.sh 
│ ├── jobscript_template.sh 
│ ├── jobscript_template_resubmit.sh 
│ ├── submit_job.sh 
│ └── submit_job_resubmit.sh 
├── main.py # Main script
├── main_with_rns.py 
├── make_data.py # Prepares dataset for experiments 
├── missing_data.py  
├── spectral_clustering.py # Spectral clustering utilities 
└── wandb_api_utils.py # Weights & Biases tracking utilities
```


## 🧩 Datasets & Format

The datasets used in this project are pre-processed versions of popular graph datasets from **[SNAP](https://snap.stanford.edu/data/)** and **[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/)**. All datasets are stored in the **`datasets/`** folder in a sparse adjacency format. They were downloaded with `make_datasets.py` from the [GraphEmbeddings](https://github.com/AndreasLF/HowLowCanYouGo/blob/master/GraphEmbeddings/graph_embeddings/data/make_datasets.py) part of the repo and then processed with `make_data.py`.

Each dataset folder contains two files representing the adjacency matrix of the graph in a sparse format:
- **`sparse_i.pt`**: A tensor storing the **source nodes** of all edges.
- **`sparse_j.pt`**: A tensor storing the **target nodes** of all edges.

## 🧪 Running Experiments
To search for the **Exact Embedding Dimension (EED)** using the HBDM-based search algorithm, run the script **`find_optimal_rank.py`**.

```bash
python find_optimal_rank.py [OPTIONS]
```

Below are all the arguments that can be passed to `find_optimal_rank.py`:
| Argument      | Description                                                            |
|---------------|------------------------------------------------------------------------|
| `--device`    | Specify the computation device (`cuda` or `cpu`). Default: `cuda`.     |
| `--dataset`   | The dataset (graph) to run the search on. Default: `Cora`.             |
| `--load-ckpt` | Path to a model checkpoint to initialize the search.                   |
| `--phase1`    | Number of steps to run Phase 1. Default: `1,000`.                      |
| `--phase2`    | Number of steps to run Phase 2. Default: `0`.                          |
| `--phase3`    | Number of steps to run Phase 3. Default: `10,000`.                     |
| `--max`       | Maximum rank (upper bound of search range). Default: `100`.            |
| `--min`       | Minimum rank (lower bound of search range). Default: `1`.              |
| `--wandb`     | Flag to enable logging to Weights & Biases (W&B).                      |
| `--cexp`      | Continue an existing experiment by specifying its ID.                  |



### Examples
#### Run an Experiment on a Specific Dataset

To find the EED for the Cora dataset:
```python
python find_optimal_rank.py --dataset Cora --device cuda
```

#### Specify Search Range

To search for ranks between 5 and 50:
```python
python find_optimal_rank.py --dataset Cora --min 5 --max 50
```

#### Enable W&B Logging

To log the experiment to Weights & Biases (W&B):
```python
python find_optimal_rank.py --dataset Cora --wandb
```

#### Load a Checkpoint

To initialize the search from a checkpoint:
```python
python find_optimal_rank.py --dataset Cora --load-ckpt ./checkpoints/model.pt
```

#### Continue an Existing Experiment

To continue logging to a previously interrupted experiment, just use the experiment id and it will continue logging to that experiment. This is useful for the very big datasets where many hours are needed and jobs might be interrupted.
```python
python find_optimal_rank.py --dataset Cora --cexp experiment_id
```



## ✨ Logging with W&B
[Weights & Biases (W&B)](https://docs.wandb.ai/quickstart/) is used to track experiments, store results, and some plots can be made based on these experiment logs.
Just pass `--wandb` flag to `find_optimal_rank.py` and make sure you are logged in beforehand.


##  🏗️ Running experiments on HPC cluster
We are using the HPC cluster at the [Techincal University of Denmark](https://www.dtu.dk/) to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).
For running experiments on compute clusters, job submission templates and instructions are provided in the **`jobscripts/`** folder.
Most experiments are run on the NVIDIA Tesla A100 and A10 GPUs.

> [!NOTE]  
> As the larger graphs require many hours for embedding and uses a lot of memory, multiple jobs working on the same embedding are often submitted following each other.
