# GraphEmbeddings

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
