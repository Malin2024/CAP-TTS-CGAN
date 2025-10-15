import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Patch Embedding (1D)
# ---------------------------
class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len, embed_dim, patch_size=16):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = seq_len // patch_size

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, num_patches)
        return x.permute(0, 2, 1)  # (B, num_patches, embed_dim)

# ---------------------------
# Multi-Head Self-Attention Block
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------
# Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, seq_len=640, num_classes=3, latent_dim=100, embed_dim=128, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.class_embed = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim, seq_len)
        self.patch_embed = PatchEmbed1D(seq_len, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(3)]
        )
        self.deconv = nn.ConvTranspose1d(embed_dim, 1, kernel_size=16, stride=16)
        self.out_norm = nn.Tanh()

    def forward(self, z, labels):
        cond = self.class_embed(labels)
        x = z + cond
        x = self.fc(x).unsqueeze(1)  # (B,1,seq_len)
        patches = self.patch_embed(x)
        trans_out = self.transformer(patches)
        out = self.deconv(trans_out.permute(0, 2, 1))
        return self.out_norm(out.squeeze(1))

# ---------------------------
# Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, seq_len=640, num_classes=3):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear((seq_len // 8) * 64, 1)

    def forward(self, x, labels):
        cond = self.label_embed(labels).unsqueeze(1)
        x = x + cond
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return self.fc(x)
