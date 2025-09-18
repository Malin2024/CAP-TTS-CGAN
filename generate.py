import argparse
import os
import torch
import numpy as np
from models import Generator

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(latent_dim=args.latent_dim, out_channels=args.channels, seq_len=args.seq_len).to(device)
    G.load_state_dict(torch.load(args.model, map_location=device))
    G.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    n = args.n
    batch = 64
    generated = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, args.latent_dim).to(device)
        with torch.no_grad():
            fake = G(z)
        fake = fake.cpu().numpy()
        for j in range(fake.shape[0]):
            out = fake[j]
            np.save(os.path.join(args.out_dir, f"sample_{i+j}.npy"), out)
            generated.append(out)
    print(f"Saved {len(generated)} samples to {args.out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default='generated')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=3000)
    args = parser.parse_args()
    generate(args)
