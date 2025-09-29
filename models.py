import torch
import torch.nn as nn

# ------------------------------
# Generator with BatchNorm (MLP)
# ------------------------------
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
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, signal_len),
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)            # (batch, num_classes)
        x = torch.cat([z, c], dim=1)          # (batch, latent_dim + num_classes)
        out = self.model(x)                   # (batch, signal_len)
        return out.unsqueeze(1)               # (batch, 1, signal_len)


# ------------------------------
# Conv1d-based Generator (optional)
# ------------------------------
class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=100, signal_len=640, num_classes=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.signal_len = signal_len
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Linear(latent_dim + num_classes, 256 * 5)  # Project to (batch, 256*5)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 5 -> 10
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 10 -> 20
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, kernel_size=16, stride=32, padding=7),  # 20 -> 640
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 5)   # Reshape to (batch, channels, length)
        out = self.conv_blocks(x)
        return out


# ------------------------------
# Discriminator (unchanged)
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
        x = x.view(x.size(0), -1)            # flatten (batch, signal_len)
        c = self.label_emb(labels)            # (batch, num_classes)
        d_in = torch.cat([x, c], dim=1)      # (batch, signal_len + num_classes)
        return self.model(d_in)               # (batch,1)
