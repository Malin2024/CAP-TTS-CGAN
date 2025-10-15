import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Generator: Transformer-based Conditional TTS-CGAN
# ============================================================
class Generator(nn.Module):
    def __init__(self, seq_len=640, num_classes=3, latent_dim=100, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding + latent projection
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.project = nn.Linear(latent_dim, embed_dim)

        # Transformer Encoder backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to 1D EEG/ECG waveform
        self.to_signal = nn.Sequential(
            nn.Linear(embed_dim, seq_len),
            nn.Tanh()
        )

    def forward(self, z, labels):
        """
        z: (B, latent_dim)
        labels: (B,)
        output: (B, 1, seq_len)
        """
        cond = self.label_emb(labels)
        x = z + cond
        x = self.project(x).unsqueeze(1)  # (B, 1, embed_dim)
        x = self.transformer(x)
        out = self.to_signal(x.squeeze(1))  # (B, seq_len)
        return out.unsqueeze(1)  # ensure (B, 1, seq_len)


# ============================================================
# Discriminator: 1D CNN + Label Conditioning
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, seq_len=640, num_classes=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Label embedding projected to signal length
        self.label_emb = nn.Embedding(num_classes, seq_len)

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.pool = nn.AvgPool1d(2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Fully connected layers
        self.flatten_dim = (seq_len // 8) * 64
        self.fc_adv = nn.Linear(self.flatten_dim, 1)         # Real/Fake
        self.fc_cls = nn.Linear(self.flatten_dim, num_classes)  # Auxiliary label classification

    def forward(self, x, labels):
        """
        x: (B, 1, seq_len)
        labels: (B,)
        """
        # Ensure input shape consistency
        if x.ndim == 2:   # (B, seq_len)
            x = x.unsqueeze(1)
        elif x.ndim == 4: # (B, C, ?, seq_len)
            x = x.squeeze(2)
        elif x.shape[1] != 1:
            # collapse extra channels if somehow misaligned
            x = x.mean(dim=1, keepdim=True)

        # Label conditioning (add embedding)
        label_signal = self.label_emb(labels).unsqueeze(1)  # (B, 1, seq_len)
        if label_signal.shape[-1] != x.shape[-1]:
            label_signal = F.interpolate(label_signal, size=x.shape[-1], mode="linear")
        x = x + label_signal

        # Convolutional processing
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)

        # Flatten & output
        x = x.view(x.size(0), -1)
        adv_out = self.fc_adv(x)
        cls_out = self.fc_cls(x)
        return adv_out, cls_out


# ============================================================
# Optional Spectral Loss for Training
# ============================================================
def spectral_loss(real, fake, eps=1e-8):
    """
    Compare real and synthetic signals in the frequency domain.
    Encourages spectral similarity (important for EEG/ECG-like data).
    """
    real_fft = torch.fft.rfft(real, dim=-1)
    fake_fft = torch.fft.rfft(fake, dim=-1)
    mag_real = torch.abs(real_fft)
    mag_fake = torch.abs(fake_fft)
    return F.l1_loss(torch.log(mag_real + eps), torch.log(mag_fake + eps))
