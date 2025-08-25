# dataset_forming.py
import os
import random
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ETLWriterDataset(Dataset):
    """
    生成 20 张图片的样本:
      - 正样本: 20 张来自同一 writer_id, 且 jis_code 不重复
      - 负样本: 前 10 张来自一个 writer_id, 后 10 张来自另一个 writer_id, 且 jis_code 不重复
    """

    def __init__(self, csv_path="data/ETL8B2_index/ETL8B2_index.csv",
                 transform=None, p_pos=0.5, min_chars_per_writer=30):
        super().__init__()
        self.csv_path = csv_path
        self.root_dir = os.path.dirname(csv_path)   # 图片就在这个目录下的 ETL8B2C?/...
        self.p_pos = p_pos

        # 默认 transform: 转 tensor 并归一化到 [-1,1]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),                      # [H,W] -> [1,H,W]
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 读 csv
        df = pd.read_csv(csv_path)
        self.df = df

        # 建 writer_id -> [(path, jis_code), ...] 映射
        self.writer_dict = {}
        for _, row in df.iterrows():
            writer = int(row["writer_id"])
            jis = row["jis_code"]
            folder = row["source_file"]        # ETL8B2C1 / ETL8B2C2 / ...
            fname = row["image_filename"]
            fpath = os.path.join(self.root_dir, folder, fname)
            if writer not in self.writer_dict:
                self.writer_dict[writer] = []
            self.writer_dict[writer].append((fpath, jis))

        # 过滤掉样本太少的 writer
        self.writer_dict = {w: imgs for w, imgs in self.writer_dict.items() if len(imgs) >= min_chars_per_writer}
        self.writers = list(self.writer_dict.keys())
        print(f"[ETLWriterDataset] usable writers: {len(self.writers)}")

        # 为了近似无限采样
        self.N = 100000

    def __len__(self):
        return self.N

    def _sample_unique_jis(self, img_list, k):
        """从 (fpath, jis_code) 列表里采 k 个, 且 jis_code 不重复"""
        by_jis = {}
        for f, j in img_list:
            by_jis.setdefault(j, []).append(f)
        if len(by_jis) < k:
            return None
        jis_choices = random.sample(list(by_jis.keys()), k)
        paths = []
        for j in jis_choices:
            paths.append(random.choice(by_jis[j]))
        return paths

    def _load_image(self, path):
        """读图，灰度，补齐到 64x64"""
        img = Image.open(path).convert("L")
        arr = np.array(img)
        if arr.shape == (63, 64):  # 高=63, 宽=64
            pad = np.zeros((1, 64), dtype=arr.dtype)
            arr = np.vstack([arr, pad])       # -> (64,64)
        elif arr.shape != (64, 64):
            raise ValueError(f"Unexpected shape {arr.shape} for {path}")
        img = Image.fromarray(arr)
        return img

    def __getitem__(self, idx):
        is_pos = random.random() < self.p_pos
        if is_pos:
            # 正样本
            w = random.choice(self.writers)
            paths = self._sample_unique_jis(self.writer_dict[w], 20)
            while paths is None:
                w = random.choice(self.writers)
                paths = self._sample_unique_jis(self.writer_dict[w], 20)
            label = 1
        else:
            # 负样本
            w1, w2 = random.sample(self.writers, 2)
            paths1 = self._sample_unique_jis(self.writer_dict[w1], 10)
            paths2 = self._sample_unique_jis(self.writer_dict[w2], 10)
            while paths1 is None or paths2 is None:
                w1, w2 = random.sample(self.writers, 2)
                paths1 = self._sample_unique_jis(self.writer_dict[w1], 10)
                paths2 = self._sample_unique_jis(self.writer_dict[w2], 10)
            paths = paths1 + paths2
            label = 0

        imgs = []
        for p in paths:
            img = self._load_image(p)
            img = self.transform(img)
            imgs.append(img)
        x = torch.stack(imgs, dim=0)  # [20, 1, 64, 64]
        y = torch.tensor(label, dtype=torch.long)
        return x, y
