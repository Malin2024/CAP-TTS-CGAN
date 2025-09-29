import torch
import torch.nn as nn

# ------------------------------
# Generator
# ------------------------------
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, signal_len=640, num_classes=3, embedding_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.signal_len = signal_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Learnable label embedding
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        input_dim = latent_dim + embedding_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, signal_len),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)           # (batch, embedding_dim)
        x = torch.cat([z, c], dim=1)         # (batch, latent_dim + embedding_dim)
        out = self.model(x)                  # (batch, signal_len)
        return out.unsqueeze(1)              # (batch, 1, signal_len)


# ------------------------------
# Discriminator
# ------------------------------
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, signal_len=640, num_classes=3, embedding_dim=50):
        super().__init__()
        self.signal_len = signal_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Learnable label embedding
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        input_dim = signal_len + embedding_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)            # flatten (batch, signal_len)
        c = self.label_emb(labels)           # (batch, embedding_dim)
        d_in = torch.cat([x, c], dim=1)      # (batch, signal_len + embedding_dim)
        return self.model(d_in)              # (batch, 1)
