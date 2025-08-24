import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import ArcMarginProduct

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 32x32
            nn.Conv2d(64, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),   # 16x16
            nn.Conv2d(96, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),   # 8x8
        )
        self.proj = nn.Linear(96, 192)

    def forward(self, x):
        f = self.stem(x)  # B,96,8,8
        B,C,H,W = f.shape
        t = f.permute(0,2,3,1).reshape(B, H*W, C)  # B,64,96
        return self.proj(t)  # B,64,192

class TransformerBlock(nn.Module):
    def __init__(self, dim=192, num_heads=3, mlp_ratio=3, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio*dim)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(mlp_ratio*dim), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = x
        x = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + h
        h = x
        x = self.mlp(self.norm2(x))
        x = x + h
        return x

class TinyViT(nn.Module):
    def __init__(self, depth=6, dim=192, heads=3):
        super().__init__()
        self.patch = PatchEmbed(1, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio=3, drop=0.1) for _ in range(depth)])
        self.head = nn.Linear(dim, 128)

    def forward(self, x):
        x = self.patch(x)      # B,64,dim
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)      # token mean pooling
        z = self.head(x)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)  # 128-d
        return z

class WriterIDNet(nn.Module):
    """Backbone + ArcFace head"""
    def __init__(self, num_writers: int, arc_s: float = 30.0, arc_m: float = 0.40):
        super().__init__()
        self.backbone = TinyViT(depth=6, dim=192, heads=3)
        self.arcface  = ArcMarginProduct(in_features=128, out_features=num_writers, s=arc_s, m=arc_m)

    def forward(self, x, labels=None):
        z = self.backbone(x)
        if labels is None:
            return z, None
        logits = self.arcface(z, labels)
        return z, logits
