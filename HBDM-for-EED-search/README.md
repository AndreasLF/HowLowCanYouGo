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

## 🔬 HBDM Algorithm & Implementation
TBD

## 🧪 Running Experiments
TBD

## ✨ Logging with W&B
TBD

## 🏗️ Job Submission
TBD
