import argparse
import os
import numpy as np
import torch
from models import Generator

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize generator without seq_len
    G = Generator(latent_dim=args.latent_dim, out_channels=args.channels).to(device)
    G.load_state_dict(torch.load(args.model, map_location=device))
    G.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(args.n):
            z = torch.randn(1, args.latent_dim).to(device)
            sample = G(z).cpu().numpy()
            out_path = os.path.join(args.out_dir, f'sample_{i}.npy')
            np.save(out_path, sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='generated')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--channels', type=int, default=22)

    args = parser.parse_args()
    generate(args)
