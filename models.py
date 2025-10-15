import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#  Generator (Transformer-based Conditional TTS-CGAN Generator)
# ============================================================
class Generator(nn.Module):
    def __init__(self, seq_len=640, num_classes=3, latent_dim=100, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # --- Conditional embedding ---
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        # --- Latent projection ---
        self.project = nn.Linear(latent_dim, embed_dim)

        # --- Transformer encoder stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output projection to EEG sequence ---
        self.to_signal = nn.Sequential(
            nn.Linear(embed_dim, seq_len),
            nn.Tanh()  # normalize output between -1 and 1
        )

    def forward(self, z, labels):
        """
        z: (B, latent_dim)
        labels: (B,)
        """
        cond = self.label_emb(labels)  # (B, latent_dim)
        x = z + cond                   # conditional latent fusion
        x = self.project(x).unsqueeze(1)  # (B, 1, embed_dim)
        x = self.transformer(x)           # (B, 1, embed_dim)
        out = self.to_signal(x.squeeze(1))  # (B, seq_len)
        return out


# ============================================================
#  Discriminator (1D CNN with Label Conditioning)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, seq_len=640, num_classes=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        # --- Label embedding projected into signal space ---
        self.label_emb = nn.Embedding(num_classes, seq_len)

        # --- Convolutional layers for signal discrimination ---
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, 7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, 7, padding=3)
        self.pool = nn.AvgPool1d(2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # --- Output layers ---
        self.flatten_dim = (seq_len // 8) * 64  # adjust if pool changes
        self.fc_adv = nn.Linear(self.flatten_dim, 1)        # Real/Fake
        self.fc_cls = nn.Linear(self.flatten_dim, num_classes)  # Class prediction (optional)

    def forward(self, x, labels):
        """
        x: (B, 1, seq_len)
        labels: (B,)
        """
        # Project label embedding to signal shape and add as conditioning
        label_signal = self.label_emb(labels).unsqueeze(1)  # (B, 1, seq_len)
        x = x + label_signal

        # Feature extraction
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Outputs
        adv_out = self.fc_adv(x)
        cls_out = self.fc_cls(x)
        return adv_out, cls_out


# ============================================================
#  Utility: Simple spectral loss (optional)
# ============================================================
def spectral_loss(real, fake, eps=1e-8):
    """
    Compare real and synthetic signals in frequency domain.
    """
    real_fft = torch.fft.rfft(real, dim=-1)
    fake_fft = torch.fft.rfft(fake, dim=-1)
    mag_real = torch.abs(real_fft)
    mag_fake = torch.abs(fake_fft)
    return F.l1_loss(torch.log(mag_real + eps), torch.log(mag_fake + eps))
