import torch
import torch.nn as nn

# ------------------------------
# Generator
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=22, hidden_channels=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(hidden_channels, hidden_channels//2, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(hidden_channels//2),
            nn.ReLU(True),

            nn.ConvTranspose1d(hidden_channels//2, out_channels, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # z: [batch_size, latent_dim]
        z = z.unsqueeze(2)  # add time dimension
        return self.net(z)


# ------------------------------
# Discriminator
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=22, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_channels*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_channels*4, 1, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        out = self.net(x)
        # global average pooling to get single value per sample
        return out.mean(dim=2)
