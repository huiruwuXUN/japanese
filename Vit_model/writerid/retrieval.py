import torch
import torch.nn.functional as F
from .utils import get_device

@torch.no_grad()
def build_writer_prototypes(model, loader, num_writers, device=None):
    if device is None:
        device = get_device()
    model.eval()
    buckets = [[] for _ in range(num_writers)]
    for imgs, labels in loader:
        imgs = imgs.to(device)
        z, _ = model(imgs, labels=None)  # (B,128)
        for i, lab in enumerate(labels):
            buckets[int(lab)].append(z[i].cpu())
    prot = []
    for k in range(num_writers):
        if len(buckets[k]) == 0:
            prot.append(torch.zeros(128))
        else:
            prot.append(torch.stack(buckets[k]).mean(dim=0))
    prot = torch.stack(prot)          # (C,128)
    prot = F.normalize(prot, p=2, dim=1)
    return prot  # CPU tensor

@torch.no_grad()
def retrieve_topk(model, img_tensor, prototypes, topk=5, device=None):
    if device is None:
        device = get_device()
    model.eval()
    f, _ = model(img_tensor.to(device), labels=None)  # (1,128)
    sims = F.cosine_similarity(f, prototypes.to(device))  # (C,)
    vals, idxs = torch.topk(sims, k=topk, largest=True, sorted=True)
    return idxs.cpu().tolist(), vals.cpu().tolist()
