import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CAPDataset
from models import Generator, Discriminator

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CAPDataset(args.data_dir, window_sec=args.window_sec, fs=args.fs, cache_npy=args.cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    sample_x, _ = ds[0]
    in_channels, seq_len = sample_x.shape

    G = Generator(latent_dim=args.latent_dim, out_channels=in_channels, seq_len=seq_len).to(device)
    D = Discriminator(in_channels=in_channels, seq_len=seq_len).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        pbar = tqdm(loader)
        for real, _ in pbar:
            real = torch.tensor(real).to(device)
            batch_size = real.size(0)

            z = torch.randn(batch_size, args.latent_dim).to(device)
            fake = G(z)

            d_real = D(real)
            d_fake = D(fake.detach())
            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)

            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            d_fake_for_g = D(fake)
            g_loss = criterion(d_fake_for_g, real_labels)
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

            pbar.set_description(f"E{epoch} D:{d_loss.item():.4f} G:{g_loss.item():.4f}")

        torch.save(G.state_dict(), f"checkpoints/generator_epoch{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoints/discriminator_epoch{epoch}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--cache', type=str, default='data/processed/cap_windows.npz')
    parser.add_argument('--window_sec', type=int, default=30)
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    train(args)
