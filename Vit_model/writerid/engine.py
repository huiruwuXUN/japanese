# writerid/engine.py
import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from .utils import get_device

def train_one_epoch(
    model,
    loader,
    optimizer,
    ce_criterion,
    supcon_criterion: Optional[nn.Module],
    supcon_weight: float,
    device=None,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
):
    """
    训练一个 epoch（带 AMP 与进度条）。
    - loss 使用 ArcFace（带 margin）的 logits
    - acc 使用“无 margin 的 cosine”做 argmax（更真实反映分类能力）
    """
    if device is None:
        device = get_device(force_cuda=True)
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                z, logits = model(imgs, labels)              # logits: 带 margin（给 loss 用）
                loss = ce_criterion(logits, labels)
                if supcon_criterion is not None and supcon_weight > 0:
                    loss = loss + supcon_weight * supcon_criterion(z, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            z, logits = model(imgs, labels)
            loss = ce_criterion(logits, labels)
            if supcon_criterion is not None and supcon_weight > 0:
                loss = loss + supcon_weight * supcon_criterion(z, labels)
            loss.backward()
            optimizer.step()

        # ====== 用“无 margin 的 cosine”算准确率 ======
        with torch.no_grad():
            # z 已 L2-normalized；ArcFace 的权重做 L2-normalize
            W = torch.nn.functional.normalize(model.arcface.weight, p=2, dim=1)
            cosine_nomargin = torch.nn.functional.linear(z, W)      # (B, C)
            pred = torch.argmax(cosine_nomargin, dim=1)
            bs = imgs.size(0)
            acc = (pred == labels).float().sum().item()

            loss_sum += loss.item() * bs
            acc_sum  += acc
            n        += bs
            pbar.set_postfix(loss=f"{loss_sum/n:.4f}", acc=f"{acc_sum/n:.4f}")

    return loss_sum / n, acc_sum / n


@torch.no_grad()
def evaluate(
    model,
    loader,
    ce_criterion,
    device=None,
    epoch: int = 0,
    split: str = "Val",
):
    """
    验证/测试循环（进度条）。
    - loss 使用 ArcFace（带 margin）的 logits
    - acc 使用“无 margin 的 cosine”做 argmax
    """
    if device is None:
        device = get_device(force_cuda=True)
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"{split} Epoch {epoch}", ncols=100)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        z, logits = model(imgs, labels)                         # logits: 带 margin（给 loss 用）
        loss = ce_criterion(logits, labels)

        # ====== 用“无 margin 的 cosine”算准确率 ======
        W = torch.nn.functional.normalize(model.arcface.weight, p=2, dim=1)
        cosine_nomargin = torch.nn.functional.linear(z, W)       # (B, C)
        pred = torch.argmax(cosine_nomargin, dim=1)

        bs = imgs.size(0)
        acc = (pred == labels).float().sum().item()

        loss_sum += loss.item() * bs
        acc_sum  += acc
        n        += bs
        pbar.set_postfix(loss=f"{loss_sum/n:.4f}", acc=f"{acc_sum/n:.4f}")

    return loss_sum / n, acc_sum / n
