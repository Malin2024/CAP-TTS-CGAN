import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import CAPNPZDataset
from models import Generator, Discriminator
from tqdm import tqdm
import torch.nn as nn

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and dataloader
    ds = CAPNPZDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Models with updated embedding_dim if you want (optional)
    embedding_dim = 50  # match updated model definition
    G = Generator(latent_dim=100, signal_len=640, num_classes=3, embedding_dim=embedding_dim).to(device)
    D = Discriminator(signal_len=640, num_classes=3, embedding_dim=embedding_dim).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for real_x, labels in pbar:
            real_x = real_x.to(device)
            labels = labels.to(device)
            batch_size = real_x.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            z = torch.randn(batch_size, 100, device=device)
            gen_labels = torch.randint(0, 3, (batch_size,), device=device)

            fake_x = G(z, gen_labels)

            real_pred = D(real_x, labels)
            fake_pred = D(fake_x.detach(), gen_labels)

            d_real_loss = criterion(real_pred, valid)
            d_fake_loss = criterion(fake_pred, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train Generator
            z = torch.randn(batch_size, 100, device=device)
            gen_labels = torch.randint(0, 3, (batch_size,), device=device)

            fake_x = G(z, gen_labels)
            fake_pred = D(fake_x, gen_labels)
            g_loss = criterion(fake_pred, valid)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix({"d_loss": d_loss.item(), "g_loss": g_loss.item()})

        # Save checkpoints
        torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, f"generator_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_epoch{epoch+1}.pth"))
        torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, "generator_latest.pth"))
        torch.save(D.state_dict(), os.path.join(args.checkpoint_dir, "discriminator_latest.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args)
