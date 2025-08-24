import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from writerid.utils import get_device
from writerid.dataset import HandwritingDataset, split_sample_level
from writerid.model import WriterIDNet
from writerid.retrieval import build_writer_prototypes, retrieve_topk

def main():
    parser = argparse.ArgumentParser(description="Eval / Retrieval")
    parser.add_argument("--csv", type=str, default="labels.csv")
    parser.add_argument("--img_dirs", nargs="+", required=True,
                        help="一个或多个图片根目录（并列文件夹）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--query", type=str, default="", help="可选：单张图片路径做检索示例")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_writers = len(ckpt["label_encoder"])
    model = WriterIDNet(num_writers=num_writers,
                        arc_s=ckpt["args"]["arc_s"], arc_m=ckpt["args"]["arc_m"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 用训练集构建原型（你也可以选择 train+val 一起）
    train_df, _, _, _ = split_sample_level(args.csv, test_size=0.1, val_size_from_full=0.1, seed=ckpt["args"]["seed"])
    train_ds = HandwritingDataset(train_df, args.img_dirs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)

    prototypes = build_writer_prototypes(model, train_loader, num_writers, device)
    torch.save({"prototypes": prototypes, "label_encoder": ckpt["label_encoder"]},
               os.path.join(os.path.dirname(args.checkpoint), "writer_prototypes.pt"))
    print("[Saved] writer_prototypes.pt")

    # 可选：对单张图片做 Top-k 检索
    if args.query:
        tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(0,0,1,0), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        im = tf(Image.open(args.query).convert("L")).unsqueeze(0)  # 1x1x64x64
        idxs, sims = retrieve_topk(model, im, prototypes, topk=args.topk, device=device)
        id2writer = {i:w for i,w in enumerate(ckpt["label_encoder"])}
        print("Top-{} nearest writers:".format(args.topk))
        for r,(i,s) in enumerate(zip(idxs, sims), start=1):
            print(f"  #{r}: writer_id={id2writer[i]} (enc={i}) | sim={s:.4f}")

if __name__ == "__main__":
    main()
