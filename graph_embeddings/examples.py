from collections import defaultdict
import pdb

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from tqdm import tqdm
import imageio
import os
from seaborn import set_style
set_style('white')

from graph_embeddings.models.PCAModel import PCAModel
from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.utils.loss import LogisticLoss

def create_1D_example(N:int = 20, num_blocks:int = 5):
    block_size = N // num_blocks
    remainder = N % num_blocks
    block_size_last = block_size + remainder
    
    block = torch.ones((block_size, block_size))
    block_last = torch.ones((block_size_last, block_size_last))
    adj = torch.block_diag(*([block]*(num_blocks-1) + [block_last]))
    adj.fill_diagonal_(0)
    return adj


def create_2D_example(N: int = 20, num_blocks: int = 5):
    adj_1D = create_1D_example(N=N, num_blocks=num_blocks)
    adj_2D = 1-adj_1D
    adj_2D.fill_diagonal_(0)
    # adj_2D.fill_diagonal_(1)
    return adj_2D 


def create_toygraph_ch2020(num_blocks: int = 34, ring_structure=False):
    N = num_blocks * 3
    block1 = torch.ones((3,3))

    adj = torch.block_diag(*([block1] * num_blocks))
    def diag_indices(k: int): 
        # k: offset, i.e. -1 => diag-1, +1 => diag+1
        _range = torch.arange(0,N)
        if k > 0: return torch.vstack([_range[:N-k], _range[k:]])
        if k < 0: k = -k; return torch.vstack([_range[k:], _range[:N-k]]) 
        if k == 0: return torch.vstack([_range, _range])

    super_diag = diag_indices(+1)
    sub_diag = diag_indices(-1)
    adj[super_diag[0],super_diag[1]] = 1
    adj[sub_diag[0],sub_diag[1]] = 1

    if ring_structure: # make the graph 'loop', i.e. add connections between the end and force it to be a ring
        adj[N-1, 0] = 1
        adj[0, N-1] = 1

    adj.fill_diagonal_(0)
    return adj



def shift_adj(adj: torch.Tensor):
    adj_s = 2*adj - 1 # ! shift adjacency matrix {0,1}^(NxN) => {-1,1}^(NxN)
    return adj_s
    
def plot_adj(adj: torch.Tensor):
    # plt.spy(adj, markersize=1)
    plt.spy(adj, markersize=3, color='black')
    # plt.tight_layout()
    # plt.title(f"[adj] Sparsity plot, {adj.shape[0]}x{adj.shape[1]}")

@torch.no_grad
def plot_latent(model: PCAModel|L2Model, step: int|str, plot_beta: bool = False):
    plt.figure(figsize=(5, 5))
    X_plot = model.X.detach().squeeze()
    Y_plot = model.Y.detach().squeeze()
    if model.X.shape[1] == 1:
        O = torch.zeros(model.X.shape[0])
        plt.scatter(X_plot, O, color='black', label="$X_i$", s=55)
        plt.scatter(X_plot, O, color='dodgerblue', label="$X_i$")
        plt.scatter(Y_plot, O, color='darkorange', marker='*', label='$Y_j$')
        plotmin = torch.min(torch.min(X_plot), torch.min(Y_plot))
        plotmax = torch.max(torch.max(X_plot), torch.max(Y_plot))
        if model.__class__.__name__ == 'L2Model' and plot_beta:
            radius = model.beta.item()
            beta_u = X_plot + radius
            beta_l = X_plot - radius
            plt.scatter(torch.hstack([beta_u,beta_l]), torch.hstack([O,O]), color='green', marker="|", label="beta-radius")
        plotmin -= radius
        plotmax += radius
        plt.yticks([]) # disable yticks
    elif model.X.shape[1] == 2:
        plt.scatter(X_plot[:,0], X_plot[:,1], 
                    color='dodgerblue',
                    label="$X_i$", zorder=1)
        plt.scatter(X_plot[:,0], X_plot[:,1], 
                    color='black', zorder=0, s=55)
        plt.scatter(Y_plot[:,0], Y_plot[:,1], 
                    color='darkorange',
                    marker='*', label='$Y_j$',zorder=2)
        xmin = torch.min(torch.min(X_plot[:,0]), torch.min(Y_plot[:,0])) 
        xmax = torch.max(torch.max(X_plot[:,0]), torch.max(Y_plot[:,0]))
        ymin = torch.min(torch.min(X_plot[:,1]), torch.min(Y_plot[:,1])) 
        ymax = torch.max(torch.max(X_plot[:,1]), torch.max(Y_plot[:,1])) 
        plotmin, plotmax = torch.min(xmin,ymin), torch.max(xmax,ymax)
        if model.__class__.__name__ == 'L2Model' and plot_beta:
            # Define the radius of the circles, you can adjust this value
            radius = model.beta.item()
            # Plot circles centered in X_plot with dashed line style
            for idx, val in enumerate(X_plot):
                x_center, y_center = val
                if idx == 0: label='beta-radius'
                else: label = None
                circle_out = Circle((x_center, y_center), radius, color='green', fill=False, alpha=0.25, linestyle='--', label=label, zorder=-1)
                plt.gca().add_patch(circle_out)
                circle_fill = Circle((x_center, y_center), radius, edgecolor='none', facecolor='green', alpha=0.05, fill=True, zorder=-1)
                plt.gca().add_patch(circle_fill)
            plotmin -= radius
            plotmax += radius
    plt.legend(loc='upper right')
    plt.xlim(left=plotmin - 3, right=plotmax + 3)
    plt.ylim(bottom=plotmin - 3, top=plotmax + 3)
    # plt.title(f"Step {step}")
    file_format = 'pdf' if step == 'final' else "png"
    image_path = f'plots/step_{step}.{file_format}'
    plt.savefig(image_path)
    plt.close()
    return image_path


def simple_frob_check(adj_hat:torch.Tensor, adj:torch.Tensor):
    adj_hat[adj_hat >= 0.] = 1.
    adj_hat[adj_hat < 0.] = 0.
    adj_hat.fill_diagonal_(0)
    frob_err = torch.linalg.norm(adj_hat - adj) / torch.linalg.norm(adj)
    return frob_err

def simple_train(adj: torch.Tensor, model: "PCAModel|L2Model", steps:int=1_000, record: bool = False, plot_beta: bool = False) -> bool: 
    """
    Returns: boolean, indicating convergence.
    """
    adj_s = shift_adj(adj)

    loss_fn = LogisticLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=250, mode='min', factor=0.1, threshold=1e-3, verbose=True)
    
    frob_recon = 1.0
    pbar = tqdm(range(steps))

    # Directory to save plot images
    os.makedirs('plots', exist_ok=True)
    image_files = []

    for i in pbar:
        optimizer.zero_grad()
        adj_hat = model.reconstruct()
        loss = loss_fn(adj_hat, adj_s)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if i % 10 == 0: 
            if record and (model.X.shape[1] == 1 or model.X.shape[1] == 2):
                image_files.append(plot_latent(model, i, plot_beta=plot_beta))

            frob_recon = simple_frob_check(adj_hat, adj)
            if frob_recon == 0.:
                pbar.set_description(f"Loss={loss.item():2f} [@{i} frob-recon={frob_recon:.4f}]")
                # print("Converged!")
                if model.X.shape[1] <= 2: plot_latent(model, step="final", plot_beta=plot_beta)
                # return True

        pbar.set_description(f"Loss={loss.item():2f} [@{i} frob-recon={frob_recon:.4f}]")
    
    if frob_recon > 0:
        print("Not converged yet!")

    if record:
        # Generate GIF
        with imageio.get_writer('training_progress.gif', mode='I') as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
                # Optional: Remove files after adding them to the GIF
                os.remove(filename)
    
    return False




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='single', choices=['single', 'multiple'], help='single=easy problem, multiple=run problem multiple times in hopes of reachin exact reconstruction, for hard problems (default: %(default)s)')
    parser.add_argument('--example', type=str, default='1', choices=['1','2','ch2020'], help='which example graph to run on (default: %(default)s)')
    parser.add_argument('--l2-rank', type=int, default=2, help='rank of the L2 model - choose 1 or 2 for visualization (default: %(default)s)')
    parser.add_argument('--N', type=int, default=50, help='number of nodes in graph (default: %(default)s)')
    parser.add_argument('--num-blocks', type=int, help='number of blocks (default: %(default)s)')
    parser.add_argument('--plot-beta-radius', type=bool, default=False, help='plot beta radius or not (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)


    l2_rank = args.l2_rank
    N = args.N
    num_blocks = args.num_blocks

    if args.example == '1':
        # ! simple 1D example
        adj = create_1D_example(N=N, num_blocks=num_blocks)
        plot_adj(adj)
        plt.tight_layout()
        plt.savefig("ex1.pdf")
    elif args.example == '2':
        # ! simple 2D example
        adj = create_2D_example(N=N, num_blocks=num_blocks)
        plot_adj(adj)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig("ex2.pdf")
    elif args.example == 'ch2020':
        # ! toy graph with triangles example [Chanpuriya et al. 2020]
        adj = create_toygraph_ch2020(num_blocks=num_blocks, ring_structure=True)
        plot_adj(adj)
        plt.tight_layout()
        plt.savefig("ex_toygraph_ch2020.pdf")

    # exit()

    if args.mode == 'single':
        print("Training L2 model")
        l2model = L2Model.init_random(N,N, rank=l2_rank)
        simple_train(adj=adj, model=l2model, steps=1_000, plot_beta=args.plot_beta_radius)

        print("Training PCA model")
        pcamodel = PCAModel.init_random(N,N, rank=2)
        simple_train(adj=adj, model=pcamodel, steps=1_000)


    elif args.mode == 'multiple':
        print("Training models")
        num_runs = 500
        counts = {"PCAModel": defaultdict(float), "L2Model": defaultdict(float)} # number of converged runs (value) for each rank (key)
        for _model in [L2Model, PCAModel]:
            model_name = _model.__name__
            plot_beta = args.plot_beta_radius and (model_name == "L2Model")
            for _rank in range(l2_rank, l2_rank + 4):
                if counts[model_name][_rank - 1] > 0: break
                print()
                print(f"rank = {_rank}")
                print()
                for _ in range(num_runs):
                    model = _model.init_random(N,N, rank=_rank)
                    converged = simple_train(adj=adj, model=model, steps=1_000, plot_beta=plot_beta)
                    if converged: counts[model_name][_rank] += 1

        print(counts)