import torch
import torch.nn as nn

# ------------------------------
# Conv1D Generator
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, signal_len=640, num_classes=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.signal_len = signal_len
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)  # One-hot style

        input_dim = latent_dim + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, signal_len),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (batch_size, latent_dim)
        # labels: (batch_size,)
        label_embedding = self.label_emb(labels)  # (batch_size, num_classes)
        x = torch.cat([z, label_embedding], dim=1)  # (batch_size, latent_dim + num_classes)
        out = self.model(x)  # (batch_size, signal_len)
        return out.unsqueeze(1)  # (batch_size, 1, signal_len)


# ------------------------------
# Conv1D Discriminator
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, signal_len=640, num_classes=3):
        super().__init__()
        self.signal_len = signal_len
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, signal_len)

        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),  # (B,16,320)
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),  # (B,32,160)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),  # (B,64,80)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # (B, 64*80)
            nn.Linear((signal_len // 8) * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (batch_size, 1, signal_len)
        # labels: (batch_size,)
        label_embedding = self.label_emb(labels).unsqueeze(1)  # (batch_size, 1, signal_len)
        x = x + label_embedding  # simple conditional input
        out = self.model(x)  # (batch_size, 1)
        return out
