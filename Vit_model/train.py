import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from writerid.utils import set_seed, ensure_dir, get_device
from writerid.dataset import HandwritingDataset, split_sample_level, split_LOCO
from writerid.model import WriterIDNet
from writerid.losses import SupConLoss
from writerid.engine import train_one_epoch, evaluate

def main():
    # 取得设备：强制 GPU
    device = get_device(force_cuda=True)

    # AMP 混合精度
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    parser = argparse.ArgumentParser(description="Writer-ID Training")
    parser.add_argument("--csv", type=str, default="ETL8B2_index.csv",
                        help="CSV 文件路径")
    parser.add_argument("--img_dirs", nargs="+", default=["output_etl8b3"],
                        help="图片根目录，可以只传一个，例如 output_etl8b3")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supcon", action="store_true")
    parser.add_argument("--supcon_w", type=float, default=0.5)
    parser.add_argument("--loco", action="store_true")
    parser.add_argument("--loco_uni", type=str, default="")
    parser.add_argument("--save", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save)

    # Split
    if args.loco:
        hold = [u for u in args.loco_uni.split(",") if u.strip() != ""]
        train_df, val_df, test_df, le = split_LOCO(args.csv, holdout_unicodes=hold, seed=args.seed)
        print(f"[LOCO] Holdout {hold}")
    else:
        train_df, val_df, test_df, le = split_sample_level(args.csv, test_size=0.1, val_size_from_full=0.1, seed=args.seed)

    num_writers = len(le.classes_)
    print(f"Writers: {num_writers} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Datasets & loaders
    train_ds = HandwritingDataset(train_df, args.img_dirs, augment=True)
    val_ds = HandwritingDataset(val_df, args.img_dirs, augment=False)
    test_ds = HandwritingDataset(test_df, args.img_dirs, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # Model, loss, optim
    device = get_device()
    model = WriterIDNet(num_writers=num_writers, arc_s=args.arc_s, arc_m=args.arc_m).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=0.07) if args.supcon else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_path = os.path.join(args.save, "best_model.pth")

    # ===== 在这里放你的训练循环 =====
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, ce_criterion,
            supcon_criterion, args.supcon_w,
            device=device, use_amp=use_amp, scaler=scaler, epoch=epoch
        )
        val_loss, val_acc = evaluate(
            model, val_loader, ce_criterion, device=device, epoch=epoch, split="Val"
        )
        scheduler.step()
        print(f"Epoch {epoch:03d} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "label_encoder": le.classes_.tolist(),
                "args": vars(args),
            }, best_path)
    print(f"[Saved best] {best_path} (Val Acc={best_val:.4f})")

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, ce_criterion, device)
    print(f"[Test] CE/ArcFace Head -> Loss {test_loss:.4f} | Acc {test_acc:.4f}")

if __name__ == "__main__":
    main()
