# models.py
# Transformer-style Generator & Discriminator adapted from imics-lab/tts-cgan (TransCGAN_model.py)
# Adapted for single-channel CAP windows of length 640 and 3 CAP classes (A1,A2,A3).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# -----------------------
# Basic transformer blocks
# -----------------------
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return x + res

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads=4, dropout=0.0):
        super().__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x: (batch, seq_len, emb_size)
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # (b, h, q, k)
        scaling = (self.emb_size // self.num_heads) ** 0.5
        att = torch.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, attn_drop=0.0, forward_expansion=4, forward_drop=0.0):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads=num_heads, dropout=attn_drop),
                nn.Dropout(attn_drop)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop),
                nn.Dropout(forward_drop)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=3, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

# -----------------------
# Patch embedding (for discriminator)
# -----------------------
class PatchEmbedding(nn.Module):
    """
    Convert 1D signal (batch, channels, seq_len) into patch tokens.
    We'll reshape into patches of length patch_size and linearly embed each patch.
    """
    def __init__(self, in_channels=1, patch_size=16, emb_size=64, seq_length=640):
        super().__init__()
        assert seq_length % patch_size == 0, "seq_length must be divisible by patch_size"
        self.patch_size = patch_size
        n_patches = seq_length // patch_size
        self.projection = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),  # (batch, n_patches, patch_size * channels)
            nn.Linear(patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # positions length = n_patches + 1 (cls)
        self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_size))

    def forward(self, x: Tensor):
        # x: (batch, channels, seq_len)
        b = x.shape[0]
        x = self.projection(x)               # (b, n_patches, emb_size)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # prepend cls token
        x = x + self.positions
        return x  # (b, n_patches+1, emb_size)

# -----------------------
# Generator (Transformer-based)
# -----------------------
class Generator(nn.Module):
    def __init__(self,
                 seq_len=640,
                 channels=1,
                 num_classes=3,
                 latent_dim=100,
                 data_embed_dim=64,
                 label_embed_dim=16,
                 depth=4,
                 num_heads=4,
                 attn_drop=0.0,
                 forward_drop=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim

        # linear projection from z+label to sequence tokens
        self.l1 = nn.Linear(self.latent_dim + self.label_embed_dim, self.seq_len * self.data_embed_dim)
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embed_dim)

        # transformer blocks over (seq_len, data_embed_dim)
        self.blocks = TransformerEncoder(depth=depth,
                                         emb_size=self.data_embed_dim,
                                         num_heads=num_heads,
                                         attn_drop=attn_drop,
                                         forward_expansion=4,
                                         forward_drop=forward_drop)

        # final conv to produce channels
        # We'll reshape to (batch, seq_len, data_embed_dim) -> (batch, channels, seq_len)
        self.deconv = nn.Sequential(
            Rearrange('b n e -> b e n'),
            nn.Conv1d(self.data_embed_dim, self.channels, kernel_size=1),
            nn.Tanh()   # output in [-1,1], match data normalization
        )

    def forward(self, z: Tensor, labels: Tensor):
        # z: (batch, latent_dim), labels: (batch,)
        c = self.label_embedding(labels)
        x = torch.cat([z, c], dim=1)
        x = self.l1(x)
        x = x.view(-1, self.seq_len, self.data_embed_dim)  # (batch, seq_len, emb)
        x = self.blocks(x)  # (batch, seq_len, emb)
        out = self.deconv(x)  # (batch, channels, seq_len)
        return out

# -----------------------
# Discriminator (patch + transformer + classification head)
# -----------------------
class ClassificationHead(nn.Module):
    def __init__(self, emb_size=64, adv_classes=1, cls_classes=3):
        super().__init__()
        # adv head outputs real/fake logit; cls head outputs class logits (optional)
        self.adv_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, adv_classes)
        )
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, cls_classes)
        )

    def forward(self, x):
        out_adv = self.adv_head(x)   # (batch, adv_classes)
        out_cls = self.cls_head(x)   # (batch, cls_classes)
        return out_adv, out_cls

class Discriminator(nn.Module):
    def __init__(self,
                 in_channels=1,
                 patch_size=16,
                 data_emb_size=64,
                 label_emb_size=16,
                 seq_length=640,
                 depth=4,
                 num_heads=4,
                 n_classes=3):
        super().__init__()
        self.patch = PatchEmbedding(in_channels=in_channels,
                                    patch_size=patch_size,
                                    emb_size=data_emb_size,
                                    seq_length=seq_length)
        # transformer encoder over tokens (cls + patches)
        self.encoder = TransformerEncoder(depth=depth,
                                          emb_size=data_emb_size,
                                          num_heads=num_heads,
                                          attn_drop=0.0,
                                          forward_expansion=4,
                                          forward_drop=0.0)
        # classification head
        self.head = ClassificationHead(emb_size=data_emb_size, adv_classes=1, cls_classes=n_classes)

    def forward(self, x: Tensor):
        # x: (batch, channels, seq_len)
        x = self.patch(x)   # (batch, n_patches+1, emb)
        x = self.encoder(x) # (batch, n_patches+1, emb)
        adv_out, cls_out = self.head(x)  # adv_out: (batch,1)
        # adv_out is raw logits; if you want probabilities apply sigmoid in loss
        return adv_out, cls_out
