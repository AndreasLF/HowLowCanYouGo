# How Low Can You Go?
[![ICLR2025](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/virtual/2025/poster/29429)
[![arXiv](https://img.shields.io/badge/arXiv-2503.01723-b31b1b.svg)](https://arxiv.org/abs/2503.01723) 
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper%20&%20Reviews-green)](https://openreview.net/forum?id=V71ITh2w40)

This repository contains code presented in the [ICLR 2025](https://iclr.cc/virtual/2025/poster/29429) paper: **How Low Can You Go? Searching for the Intrinsic Dimensionality of Complex Networks Using Metric Node Embeddings** 

### Authors
- Nikolaos Nakis <a href="https://github.com/Nicknakis"><img src="https://skillicons.dev/icons?i=github" width="16"></a>  <a href="https://www.linkedin.com/in/nikolaos-nakis-67a07a147/"><img src="https://skillicons.dev/icons?i=linkedin" width="16"></a>   
- Niels Raunkjær Holm  <a href="https://github.com/nrholm1"><img src="https://skillicons.dev/icons?i=github" width="16"></a>  <a href="https://www.linkedin.com/in/nielsrh/"><img src="https://skillicons.dev/icons?i=linkedin" width="16"></a>  
- Andreas Lyhne Fiehn  <a href="https://github.com/AndreasLF"><img src="https://skillicons.dev/icons?i=github" width="16"></a>  <a href="https://www.linkedin.com/in/andreas-fiehn/"><img src="https://skillicons.dev/icons?i=linkedin" width="16"></a>  
- Morten Mørup  <a href="https://www.linkedin.com/in/morten-moerup-b86a90/"><img src="https://skillicons.dev/icons?i=linkedin" width="16"></a>  

## Repository Structure  

The code is split into two main folders as it was originally developed in two repositories due to different dependencies and Python versions.  

### 📁 [GraphEmbeddings/](GraphEmbeddings)  
Use exact adjacency matrices to check for reconstruction. *Not* scalable to larger graphs.

- **Logarithmic search for EED** (Algorithm 1).  
- **Metric-based node embeddings** (L2, Hyperbolic, PCA, Latent Eigen).  
- **Custom loss functions**: Logistic, Hinge, and Poisson loss.  
- **Experiments on [SNAP](https://snap.stanford.edu/data/) and [PyG](https://www.pyg.org/) datasets** (Cora, Citeseer, PubMed, etc.).  
- **Batch sampling experiments**: Random node sampling vs. case-control sampling.  
- **Results presented in Table 2**.  
- **W&B experiment tracking and logging** for reproducibility.  
- **HPC job submission scripts** for large-scale training.  


### 📁 [HBDM-for-EED-search/](HBDM-for-EED-search)  
Implementation of the HBDM framework leveraging metric embeddings.  
Uses KD-Trees to verify exact reconstruction in a sparse manner. Scalable to larger graphs.

- **HBDM implementation** for embedding large graphs.
- **Results presented in Table 3.**  
- **Adapted from** the [HBDM framework by Nikolaos Nakis](https://github.com/Nicknakis/HBDM).  

## Abstract  

Low-dimensional embeddings are essential for machine learning tasks involving graphs, such as node classification, link prediction, community detection, network visualization, and network compression. Although recent studies have identified exact low-dimensional embeddings, the limits of the required embedding dimensions remain unclear. 

We presently prove that lower dimensional embeddings are possible when using Euclidean metric embeddings as opposed to vector-based Logistic PCA (LPCA) embeddings. In particular, we provide an efficient logarithmic search procedure for identifying the exact embedding dimension and demonstrate how metric embeddings enable inference of the exact embedding dimensions of large-scale networks by exploiting that the metric properties can be used to provide linearithmic scaling.

Empirically, we show that our approach extracts substantially lower dimensional representations of networks than previously reported for small-sized networks.
For the first time, we demonstrate that even large-scale networks can be effectively embedded in very low-dimensional spaces, and provide examples of scalable, exact reconstruction for graphs with up to a million nodes.

Our approach highlights that the intrinsic dimensionality of networks is substantially lower than previously reported and provides a computationally efficient assessment of the exact embedding dimension also of large-scale networks. The surprisingly low dimensional representations achieved demonstrate that networks in general can be losslessly represented using very low dimensional feature spaces, which can be used to guide existing network analysis tasks from community detection and node classification to structure revealing exact network visualizations.

📖 **Full Paper:** [Link to Paper](https://openreview.net/forum?id=V71ITh2w40)


## Cite
```
@inproceedings{nakis2025low,
  author       = {Nikolaos Nakis and
                  Niels Raunkj{\ae}r Holm and
                  Andreas Lyhne Fiehn and
                  Morten M{\o}rup},
  title        = {How Low Can You Go? Searching for the Intrinsic Dimensionality of
                  Complex Networks using Metric Node Embeddings},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  year         = {2025},
  url          = {https://openreview.net/forum?id=V71ITh2w40}
}
```



