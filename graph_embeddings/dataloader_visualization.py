# %%
from graph_embeddings.utils.dataloader import CaseControlDataLoader, RandomNodeDataLoader
from torch_geometric.data import Data
from matplotlib import pyplot as plt
import numpy as np
import imageio
import os
from tqdm import tqdm
import torch
# %% 

class BatchGifCreator:
    def __init__(self, vmin=0, vmax=100, node_val=20, temp_dir="temp"):
        self.vmin = vmin
        self.vmax = vmax
        self.node_val = node_val
        self.temp_dir = temp_dir

    def make_temp_dir(self, temp_dir):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    def create_gif(self, dataloader, gif_name="movie", epochs = 10):

        vmin, vmax = self.vmin, self.vmax
        node_val = self.node_val
        N = dataloader.data.num_nodes

        temp_dir = f"RN_{self.temp_dir}"
        self.make_temp_dir(temp_dir)

        print("Looping through epochs...")
        overlays_sum = np.zeros((N,N))
        for i in tqdm(range(epochs)):
            for bidx, batch in enumerate(dataloader):
                overlay = np.zeros((N,N))


                if dataloader.__class__ == RandomNodeDataLoader:
                    indices = batch.indices
                    srcs = indices
                    tgts = indices
                elif dataloader.__class__ == CaseControlDataLoader:
                    links = batch.links
                    nonlinks = batch.nonlinks

                    srcs = torch.cat([links[0], nonlinks[0]])
                    tgts = torch.cat([links[1], nonlinks[1]])
                else:
                    raise ValueError("DataLoader not supported.")
                
                overlay[srcs.unsqueeze(1), tgts.unsqueeze(0)] = node_val    

            overlays_sum += overlay

            plt.imshow(overlays_sum, vmin=vmin, vmax=vmax, cmap='Blues')
            plt.title(f"Epoch {i}")    
            save_img = f"{temp_dir}/epoch_{i}.png"
            plt.savefig(save_img)

        print("Creating gif...")
        self.__make_gif(gif_name=gif_name, temp_dir=temp_dir)

    
    def __make_gif(self, gif_name, temp_dir="temp"):
        # check if temp dir exists
        if not os.path.exists(temp_dir):
            raise FileNotFoundError(f"Directory {temp_dir} not found. Run random_node_loader_gif() first.")

        # get length of files in temp dir
        epochs = len(os.listdir(temp_dir))

        images = []
        for i in range(epochs):
            filename = f"{temp_dir}/epoch_{i}.png"
            images.append(imageio.imread(filename))
            os.remove(filename)

        # remove test dir 
        os.rmdir(temp_dir)

        imageio.mimsave(f'{gif_name}.gif', images)

# %%

if __name__ == "__main__":
    # Make dummy adj 
    N = 100
    adj = (torch.rand((N,N)) <= .02).long()
    # adj = torch.zeros((N,N))
    adj.fill_diagonal_(1)
    edge_index = torch.nonzero(adj).t()
    data = Data(edge_index=edge_index)

    # Create RandomNodeDataLoader
    dataloader = RandomNodeDataLoader(data, batch_size=5, shuffle=True)
    cc_dataloader = CaseControlDataLoader(data, batch_size=5, negative_sampling_ratio=5, shuffle=True)

    gif_creator = BatchGifCreator(vmin=0, vmax=100, node_val=10, temp_dir="temp")
    gif_creator.create_gif(dataloader,epochs=100, gif_name="rn_movie")
    gif_creator.create_gif(cc_dataloader,epochs=100, gif_name="cc_movie")
