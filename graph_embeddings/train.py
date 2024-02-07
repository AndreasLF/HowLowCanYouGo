import torch
import torch.optim as optim
from tqdm import tqdm
import argparse

from graph_embeddings.models.L2Model import L2Model
from graph_embeddings.models.LPCAModel import LPCAModel
from graph_embeddings.utils.loss import lpca_loss, L2_loss


parser = argparse.ArgumentParser()
parser.add_argument('--model-type', type=str, default='LPCA', choices=['LPCA','L2'], help='Type of model and loss to use {LPCA, L2} (default: %(default)s)')
parser.add_argument('--num-epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
parser.add_argument('--max-eval', type=int, default=25, metavar='me', help='max_eval parameter of LBFGS optimizer [set to 1 for L2-loss] (default: %(default)s)')
parser.add_argument('--rank', type=int, default=32, metavar='R', help='dimension of the embedding space (default: %(default)s)')
parser.add_argument('--optim-type', type=str, default='lbfgs', choices=['lbfgs', 'adam', 'sgd'], help='which optimizer to use (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='V', help='learning rate for training (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
# parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
# parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
# parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
# parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
# parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')

args = parser.parse_args()
print('# Options')
for key, value in sorted(vars(args).items()):
    print(key, '=', value)


# Ensure CUDA is available and select device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(args.device) # ? why can it not train on mps??

# Load and prepare your data
adj = torch.load('./data/adj_matrices/Cora.pt').to(device)

# shift adj matrix, i.e. map (0,1) to (-1,1)
adj_s = adj*2 - 1

n_row, n_col = adj.size()

# Initialize model and move it to GPU
model = LPCAModel(n_row, n_col, args.rank).to(device) if args.model_type == 'LPCA' \
    else L2Model(n_row, n_col, args.rank).to(device)

# Set loss function according to model type
if args.model_type == 'LPCA':
    loss_fn = lpca_loss
elif args.model_type == 'L2': 
    loss_fn = L2_loss

def closure():
    optimizer.zero_grad()
    A_hat = model.reconstruct()  # Or simply model() if you have defined the forward method
    loss = loss_fn(A_hat, adj_s)  # Ensure loss function is compatible with PyTorch and returns a scalar tensor
    # test = L2_loss_naive(logits, adj_s) # ! remove - test to compare naïve w/ optimized
    loss.backward()
    return loss

if args.optim_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optim_type == 'lbfgs':
    optimizer = optim.LBFGS(model.parameters(), lr=args.lr, 
                            max_eval=args.max_eval ) # ! L2 needs 1, but converges way faster (for LPCA) with default (=25)

with tqdm(range(args.num_epochs)) as pbar:
    for epoch in pbar:
        if args.optim_type == 'lbfgs':
            # LBFGS optimizer step takes the closure function and internally calls it multiple times
            loss = optimizer.step(closure)
        else: 
            # Forward pass
            optimizer.zero_grad()
            A_hat = model.reconstruct() 
            loss = loss_fn(A_hat, adj_s)
            loss.backward()
            optimizer.step()
    
        # Compute and print the Frobenius norm for diagnostics
        with torch.no_grad():  # Ensure no gradients are computed in this block
            A_hat = model.reconstruct()
            frobenius_loss = torch.norm(adj - A_hat, p='fro').item()
            clipped_recon = torch.clip(A_hat, min=0, max=1)
            frob_error_norm = torch.linalg.norm(clipped_recon - adj) / torch.linalg.norm(adj)
            pbar.set_description(f"epoch={epoch}, loss={loss:.1f} Frobenius error: {frob_error_norm}")


# After training, retrieve parameters
with torch.no_grad():  # Ensure no gradients are computed in this block
    X, Y = model.X.cpu(), model.Y.cpu()
    torch.save((X,Y),'./models/metric-emb/X_Y.pt')
    beta = model.beta.cpu()
    torch.save(beta,'./models/metric-emb/beta.pt')
    