import torch
import torch.nn as nn

# ------------------------------
# Generator
# ------------------------------
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, signal_len=640, num_classes=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.signal_len = signal_len
        self.num_classes = num_classes

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = latent_dim + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, signal_len),
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)            # (batch, num_classes)
        x = torch.cat([z, c], dim=1)          # (batch, latent_dim + num_classes)
        out = self.model(x)                   # (batch, 640)
        return out.unsqueeze(1)               # (batch, 1, 640)


# ------------------------------
# Discriminator
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, signal_len=640, num_classes=3):
        super().__init__()
        self.signal_len = signal_len
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = signal_len + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)             # flatten (batch, 640)
        c = self.label_emb(labels)            # (batch, num_classes)
        d_in = torch.cat([x, c], dim=1)       # (batch, 640 + num_classes)
        return self.model(d_in)               # (batch,1)
