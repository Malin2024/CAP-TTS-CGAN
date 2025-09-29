import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import CAPNPZDataset
from models import Generator, Discriminator  # or ConvGenerator if you switched

# ---------------------------
# Load dataset
# ---------------------------
npz_path = "/content/drive/MyDrive/CAPSLPDB/CapMiniDb/processed/CAP_windows.npz"
dataset = CAPNPZDataset(npz_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

print(f"✅ Loaded dataset: {len(dataset)} CAP windows")

# ---------------------------
# Initialize models
# ---------------------------
latent_dim = 100
signal_len = 640
num_classes = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose your Generator here:
generator = Generator(latent_dim=latent_dim, signal_len=signal_len, num_classes=num_classes).to(device)
# OR, for conv generator:
# from models import ConvGenerator
# generator = ConvGenerator(latent_dim=latent_dim, signal_len=signal_len, num_classes=num_classes).to(device)

discriminator = Discriminator(signal_len=signal_len, num_classes=num_classes).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ---------------------------
# Training loop
# ---------------------------
num_epochs = 50
D_losses, G_losses = [], []

for epoch in range(1, num_epochs + 1):
    for real_data, labels in dataloader:
        real_data, labels = real_data.to(device), labels.to(device)
        batch_size = real_data.size(0)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # --- Train Generator ---
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        gen_data = generator(z, gen_labels)  # Shape: (batch, 1, signal_len)
        g_loss = criterion(discriminator(gen_data, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_data, labels), valid)
        fake_loss = criterion(discriminator(gen_data.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    D_losses.append(d_loss.item())
    G_losses.append(g_loss.item())

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{num_epochs}] | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

print("✅ TTS-CGAN training complete")

# ---------------------------
# Plot losses
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(D_losses, label="Discriminator Loss")
plt.plot(G_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("TTS-CGAN Training Losses")
plt.legend()
plt.grid(True)
plt.show()
