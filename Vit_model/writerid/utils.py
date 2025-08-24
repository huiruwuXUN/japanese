import torch
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(force_cuda: bool = True):
    if force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用的 CUDA GPU，但已要求强制使用 GPU。请安装/启用 CUDA 后再试。")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True   # 根据输入尺寸自动选择最快算法
        return device
    # 非强制：回退到 CPU
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)