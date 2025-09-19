import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import CAPDataset
from models import Generator, Discriminator
from tqdm import tqdm

def train(args):
    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Dataset ---
    ds = CAPDataset(args.data_dir, window_sec=args.window_sec, fs=args.fs, cache_npy=args.cache)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # --- Models ---
    G = Generator()
    D = Discriminator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, D = G.to(device), D.to(device)

    # --- Optimizers ---
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # --- Training loop ---
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for real_x, _ in pbar:
            real_x = real_x.to(device)

            # Train discriminator
            z = torch.randn(real_x.size(0), 100, device=device)
            fake_x = G(z)

            d_real = D(real_x)
            d_fake = D(fake_x.detach())
            d_loss = -(torch.mean(d_real) - torch.mean(d_fake))

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train generator
            z = torch.randn(real_x.size(0), 100, device=device)
            fake_x = G(z)
            g_loss = -torch.mean(D(fake_x))

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix({"d_loss": d_loss.item(), "g_loss": g_loss.item()})

        # --- Save checkpoints ---
        torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, f"generator_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_epoch{epoch+1}.pth"))

        # Save latest model separately
        torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, "generator_latest.pth"))
        torch.save(D.state_dict(), os.path.join(args.checkpoint_dir, "discriminator_latest.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to EDF dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--window_sec", type=int, default=30)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    train(args)
