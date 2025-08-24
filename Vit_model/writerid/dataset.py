import os
from typing import List
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
class HandwritingDataset(Dataset):
    """
    支持两种目录结构：
    1) images/UNK_9250_0.png
    2) images/ETL8B2C1/UNK_9250_0.png   （当 CSV 含有 source_file 列时）
    同时把 64x63 右侧 pad 到 64x64，并做轻量增广（可选）。
    """

    def __init__(self, df: pd.DataFrame, img_dirs, augment: bool = False):
        self.df = df.reset_index(drop=True)
        # 支持一个或多个根目录
        if isinstance(img_dirs, (list, tuple)):
            self.img_dirs = [Path(p) for p in img_dirs]
        else:
            self.img_dirs = [Path(img_dirs)]
        self.has_source = "source_file" in self.df.columns

        base_tf = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(0, 0, 1, 0), fill=0),  # 右侧补1列 -> 64x64
        ]
        aug_tf = [
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ] if augment else []
        tail_tf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        self.tf = transforms.Compose(base_tf + aug_tf + tail_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        candidates = []

        # 逐个根目录尝试（优先带 source_file，再尝试平铺）
        for root in self.img_dirs:
            if self.has_source:
                candidates.append(root / str(row["source_file"]) / row["image_filename"])
            candidates.append(root / row["image_filename"])

        path = None
        for p in candidates:
            if p.exists():
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"Image not found in any roots: {candidates}")

        img = Image.open(path).convert("L")
        img = self.tf(img)
        label = int(row["writer_id_enc"])
        return img, label


def _encode_writers(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    assert "writer_id" in df.columns and "image_filename" in df.columns, \
        "CSV 必须至少包含 writer_id 与 image_filename"
    le = LabelEncoder()
    df["writer_id_enc"] = le.fit_transform(df["writer_id"].astype(str))
    return df, le

def split_sample_level(csv_path: str, test_size: float = 0.1, val_size_from_full: float = 0.1, seed: int = 42):
    """
    按样本分层（writer_id）进行 训练/验证/测试 划分。
    返回 train_df, val_df, test_df, LabelEncoder
    """
    df = pd.read_csv(csv_path)
    df, le = _encode_writers(df)

    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size_from_full),
        stratify=df["writer_id_enc"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["writer_id_enc"], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            le)

def split_LOCO(csv_path: str, holdout_unicodes: List[str], val_size_from_rest: float = 0.1, seed: int = 42):
    """
    LOCO：把指定 unicode 的字符全部留出（不进训练），
    其中一半进验证，一半进测试；其余样本按 writer_id 分层切出 val。
    """
    df = pd.read_csv(csv_path)
    assert "unicode" in df.columns, "使用 LOCO 需要 CSV 含 unicode 列"
    df, le = _encode_writers(df)

    mask = df["unicode"].astype(str).isin(set(holdout_unicodes))
    df_loco = df[mask].copy()
    df_rest = df[~mask].copy()
    assert len(df_loco) > 0, "LOCO 列表未匹配到样本，请检查 unicode 值。"

    loco_val, loco_test = train_test_split(
        df_loco, test_size=0.5,
        stratify=df_loco["writer_id_enc"], random_state=seed
    )
    train_df, small_val = train_test_split(
        df_rest, test_size=val_size_from_rest,
        stratify=df_rest["writer_id_enc"], random_state=seed
    )
    val_df = pd.concat([small_val, loco_val], ignore_index=True).reset_index(drop=True)
    test_df = loco_test.reset_index(drop=True)
    return train_df, val_df, test_df, le
