import torch
import torch.nn as nn

class ConditionalVector(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(n_classes, latent_dim)
    def forward(self, labels):
        return self.embed(labels)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, seq_len=3000, cond=False, n_classes=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond = cond
        input_dim = latent_dim
        if cond:
            self.cond_vec = ConditionalVector(n_classes, latent_dim)

        self.project = nn.Sequential(
            nn.Linear(input_dim, 256 * (seq_len // 64)),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if self.cond and labels is not None:
            cv = self.cond_vec(labels)
            z = z + cv
        x = self.project(z)
        batch = x.shape[0]
        x = x.view(batch, 256, -1)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, seq_len=3000, cond=False, n_classes=3):
        super().__init__()
        self.cond = cond
        if cond:
            self.label_proj = nn.Embedding(n_classes, in_channels)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x, labels=None):
        if self.cond and labels is not None:
            emb = self.label_proj(labels)
            emb = emb.unsqueeze(-1)
            x = x + emb
        out = self.net(x)
        return out
