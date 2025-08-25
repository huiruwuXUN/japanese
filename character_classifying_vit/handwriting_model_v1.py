# -*- coding: utf-8 -*-


# Adopted from https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/, https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


path_prefix = './data'

"""2. Import Libraries"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from PIL import Image
import os
import dataset_forming

# Seed every thing for reproduct
SEED = 8539
torch.manual_seed(SEED) # Setting the seed
# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""Prepare Dataset"""
# see below

"""
Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

def pair(t):
    """Helper to ensure tuple format for sizes."""
    return t if isinstance(t, tuple) else (t, t)

# 2d RoPE
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """
    返回 cos, sin: [seq_len, head_dim]；要求 head_dim 为偶数
    """
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [head_dim/2]
    t = torch.arange(seq_len, dtype=torch.float32)                                 # [seq_len]
    freqs = torch.outer(t, inv_freq)                                              # [seq_len, head_dim/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)                           # [seq_len, head_dim]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)                           # [seq_len, head_dim]
    return cos, sin

def apply_rope(x, cos, sin):
    """
    x:   [B, heads, N, d_sub]
    cos: [N, d_sub] 或可广播到 [1,1,N,d_sub]
    sin: [N, d_sub] 或可广播到 [1,1,N,d_sub]
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)
    return x * cos + x_rot * sin

"""# Patch Embedding Layer

Transformers accept 1D sequences. To handle 2D images, we reshape an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ into $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(H, W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the resolution of each image patch, and $N = HW / P^2$ is the resulting number of patches. The 1D patch sequences ($\mathbf{x}_p$) will be map to $D$ dimensions with a trainable linear projection.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    """Helper to ensure tuple format for sizes."""
    return t if isinstance(t, tuple) else (t, t)

# ----------------------
# Patch Embedding Layer
# ----------------------
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, **kwargs):
        super().__init__()
        self.num_channels = kwargs.get('num_channels', 1)   # 单通道

        image_size = pair(image_size)
        patch_size = pair(patch_size)
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.num_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        # 将图像分割为 patch 并 flatten，再线性投影到 dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        return self.to_patch_embedding(x)

"""# Embedding Layer (Patch + Positional + CLS)

ViT use a learnable embedding $\mathbf{x}_\text{class}$ to serve as the representation of the image, which will be prepend to the embedded image patches($\mathbf{Z}_{0:N+1} = [\mathbf{x}_\text{class} \quad , \mathbf{x}_p]$). ViT use standard learnable 1D position embeddings $\mathbf{P}_{0:N+1} = [\mathbf{P}_0, \mathbf{P}_1, \ldots, \mathbf{P}_{N+1}]$. The input of the transformer layers is the summation of $\mathbf{Z}_{0:N+1}$ and $\mathbf{P}_{0:N+1}$.
"""


class Embeddings(nn.Module):
    def __init__(self, image_size, patch_size, dim, **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, **kwargs)
        num_patches = self.patch_embedding.num_patches

        # CLS token + Position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(kwargs.get('emb_dropout', 0.1))

    def forward(self, x):
        x = self.patch_embedding(x)                     # [B, N, D]
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)           # [B, N+1, D]

        # x += self.pos_embedding[:, :n+1, :]             # 加位置编码
        return self.dropout(x)

"""Multi-Head Self-Attention

![](https://cdn.jsdelivr.net/gh//Charlemagnescl/image-host/202508031017288.png)

Zhang, A., et al, Dive into Deep Learning. Cambridge University Press, 2023.

In multi-head self attention, we have the input $\mathbf{x}$ serves as the queries, keys, values ($\mathbf{q,k,v}$) simultaneouly. The attention output of the $i$-th head is
$$\mathbf{h}_i = f(\mathbf{W}^{(q)}_i \mathbf{q}, \mathbf{W}^{(k)}_i \mathbf{k}, \mathbf{W}^{(v)}_i \mathbf{v}) \in \mathbb{R}^{D / N_\text{head}}$$
where $\mathbf{W}^{(q)}_i, \mathbf{W}^{(k)}_i, \mathbf{W}^{(v)}_i \in \mathbb{R}^{D / N_\text{head} \  \times D}$ are learnable parameters, and $f$ is scaled dot product attention
$$f(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times d}$$
where $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d}$.

The multi-head attention output is another linear transformation via learnable parameters $\mathbf{W}_o \in \mathbb{R}^{D \times D}$ of the concatenation of $N_\text{head} \ $ heads
$$\mathbf{W}_o \begin{bmatrix}\mathbf{h}_1 \\ \vdots \\ \mathbf{h}_{N_\text{head}}\end{bmatrix} \in \mathbb{R}^{D}$$
"""

# ---- 对称 2D RoPE 的 MHA（写死 head_dim/seq_h/seq_w）----
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=8, image_size=64,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        seq_h   = image_size // patch_size
        seq_w   = image_size // patch_size
        d_half  = head_dim // 2
        assert d_half % 2 == 0

        self.scale = qk_scale or head_dim ** -0.5
        self.norm  = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj  = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        cos_row, sin_row = build_rope_cache(seq_h, d_half)
        cos_col, sin_col = build_rope_cache(seq_w, d_half)
        self.register_buffer("cos_row", cos_row)
        self.register_buffer("sin_row", sin_row)
        self.register_buffer("cos_col", cos_col)
        self.register_buffer("sin_col", sin_col)

        self._seq_h = seq_h
        self._seq_w = seq_w
        self._dhalf = d_half        

    def forward(self, x):
        # x: [B, N, D]，N=65（1 CLS + 64 patches）
        x = self.norm(x)
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]                 # [B, H, N, 32]

        # ---- 线性索引 → (row, col)，index 0 为 CLS → 设为 (0,0)
        patch_lin = torch.arange(N - 1, device=x.device)                  # 0..63
        rows = torch.div(patch_lin, self._seq_w, rounding_mode='floor')   # 0..7
        cols = patch_lin % self._seq_w                                    # 0..7
        row_ids = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), rows], dim=0)  # [N]
        col_ids = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cols], dim=0)  # [N]

        # ---- 选择对应 cos/sin（维度 [N, d_half]）
        cos_r = self.cos_row[row_ids, :]
        sin_r = self.sin_row[row_ids, :]
        cos_c = self.cos_col[col_ids, :]
        sin_c = self.sin_col[col_ids, :]

        # ---- 对称 2D RoPE：Q/K 同步按 row/col 各占一半维度旋转
        d2 = self._dhalf
        q_row, q_col = q[..., :d2], q[..., d2:]   # [B,H,N,16], [B,H,N,16]
        k_row, k_col = k[..., :d2], k[..., d2:]

        q_row = apply_rope(q_row, cos_r, sin_r)  # 行
        q_col = apply_rope(q_col, cos_c, sin_c)  # 列
        k_row = apply_rope(k_row, cos_r, sin_r)
        k_col = apply_rope(k_col, cos_c, sin_c)

        q = torch.cat([q_row, q_col], dim=-1)    # [B,H,N,32]
        k = torch.cat([k_row, k_col], dim=-1)

        # ---- 标准注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return self.proj_drop(out)

"""# Feedforward MLP

The MLP contains two layers with a GELU non-linearity.
"""


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

"""# Transformer Encoder Block """

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=6, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)   # 残差
            x = x + ff(x)     # 残差
        return self.norm(x)

"""# Vision Transformer Classifier

$$\mathbf{z}^{0} = [\mathbf{x}_\text{class} \ \ , \mathbf{x}_p] + \mathbf{P},$$
$$\tilde{\mathbf{z}}^{l} = \operatorname{MSA}(\operatorname{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1}, \quad l = 1, \ldots, L$$
$$\mathbf{z}^{l} = \operatorname{MLP}(\operatorname{LN}(\tilde{\mathbf{z}}^{l})) + \tilde{\mathbf{z}}^{l}, \quad l = 1, \ldots, L$$
$$\mathbf{y} = \operatorname{LN}(\mathbf{z}^L_0)$$



class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super().__init__()
        self.embeddings = Embeddings(image_size, patch_size, dim, emb_dropout=emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.embeddings(img)
        x = self.encoder(x)
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)
"""
class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.embeddings = Embeddings(image_size, patch_size, dim, emb_dropout=emb_dropout)
        self.encoder    = TransformerEncoder(dim, depth, heads, mlp_dim, dropout)

    def forward(self, img):           # img: [B, 3, H, W]
        x = self.embeddings(img)      # [B, N+1, D]
        x = self.encoder(x)           # [B, N+1, D]
        return x[:, 0]                # CLS: [B, D]

# 可选：注意力池化（不想用就删掉，默认用 mean）
class AttnPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, tokens):        # [B, T, D]
        w = self.scorer(tokens).squeeze(-1)   # [B, T]
        a = torch.softmax(w, dim=1)           # [B, T]
        return torch.sum(tokens * a.unsqueeze(-1), dim=1)  # [B, D]

# 2) 主模型：20 张 => 前10与后10两组 => 聚合 => 对比特征 => 二分类
class WriterMatcher(nn.Module):
    def __init__(self, image_size=128, patch_size=16,
                 dim=384, depth=6, heads=6, mlp_dim=768,
                 dropout=0.1, emb_dropout=0.1, pool='mean', head_hidden=512):
        super().__init__()
        self.encoder = ViTEncoder(image_size, patch_size, dim, depth, heads, mlp_dim, dropout, emb_dropout)
        self.pool    = pool
        if pool == 'attn':
            self.pool_a = AttnPool(dim)
            self.pool_b = AttnPool(dim)
        # 对比特征： [ea, eb, |ea-eb|, ea*eb, cos]
        in_feats = dim*4 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)    # 输出 logit
        )

    def forward(self, x20):                   # [B, 20, 3, 128, 128]
        B, T, C, H, W = x20.shape
        assert T == 20, f"Expect 20 images, got {T}"
        x = x20.view(B*T, C, H, W)           # 合并到 batch 维
        z = self.encoder(x)                  # [B*T, D]
        D = z.shape[-1]
        z = z.view(B, T, D)                  # [B, 20, D]

        a, b = z[:, :10, :], z[:, 10:, :]    # 两组各10张

        if self.pool == 'attn':
            ea = self.pool_a(a)              # [B, D]
            eb = self.pool_b(b)              # [B, D]
        else:
            ea = a.mean(dim=1)               # [B, D]
            eb = b.mean(dim=1)               # [B, D]

        diff = torch.abs(ea - eb)            # [B, D]
        prod = ea * eb                       # [B, D]
        cos  = nn.functional.cosine_similarity(ea, eb, dim=-1, eps=1e-8).unsqueeze(-1)  # [B, 1]

        feats = torch.cat([ea, eb, diff, prod, cos], dim=-1)  # [B, 4D+1]
        logit = self.head(feats).squeeze(-1)                  # [B]
        return logit

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        # Move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Update metrics
            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = loss_sum / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

class WriterMatcherCE(WriterMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.encoder.encoder.layers[0][0].norm.normalized_shape[0]  # 等于 self.encoder 输出维度
        in_feats = dim*4 + 1
        head_hidden = 512
        self.head = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1 if 'dropout' not in kwargs else kwargs['dropout']),
            nn.Linear(head_hidden, 2)   # <-- 2 类 logits
        )

    def forward(self, x20):   # [B, 20, 3, 128, 128]
        logits_1d = super().forward(x20)     # 这里原版返回的是 [B] 的 logit；我们重写：
        # 实际上我们刚刚重写了 head，super().forward 会直接返回 [B] 不再适用
        # 所以复制一份 super().forward 的主体，返回 2 维 logits：

        B, T, C, H, W = x20.shape
        assert T == 20
        x = x20.view(B*T, C, H, W)
        z = self.encoder(x)                  # [B*T, D]
        D = z.shape[-1]
        z = z.view(B, T, D)

        a, b = z[:, :10, :], z[:, 10:, :]
        ea = a.mean(dim=1)                   # 也可以切到 attn 池化：self.pool == 'attn' 时用 self.pool_a/b
        eb = b.mean(dim=1)

        diff = torch.abs(ea - eb)
        prod = ea * eb
        cos  = nn.functional.cosine_similarity(ea, eb, dim=-1, eps=1e-8).unsqueeze(-1)
        feats = torch.cat([ea, eb, diff, prod, cos], dim=-1)  # [B, 4D+1]

        logits = self.head(feats)            # [B, 2]
        return logits


# ==== 2) Initialize model / optimizer / scheduler ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== 3) Start training ====
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset_forming import ETLWriterDataset

    dataset = ETLWriterDataset("data/ETL8B2_index/ETL8B2_index.csv", p_pos=0.5)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    xb, yb = next(iter(train_loader))
    print(xb.shape, yb)

    # ===== 训练循环 =====
    model = WriterMatcherCE(
        image_size=64, patch_size=8,
        dim=256, depth=4, heads=4, mlp_dim=512,
        dropout=0.1, emb_dropout=0.1, pool='mean'
    ).to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(3):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device="cuda")
        print(f"[Epoch {epoch}] loss={train_loss:.4f} acc={train_acc*100:.2f}%")

        # 保存最好模型
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), "best_writer_matcher.pth")

    print("Training done. Best acc:", best_acc)
