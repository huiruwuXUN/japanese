import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    """
    ArcFace: logits = s * cos(theta + m)
    需要特征和权重均 L2 归一化。
    """
    def __init__(self, in_features=128, out_features=1000, s=30.0, m=0.40, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label):
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(input, W)  # (B, out_features)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss（假设输入特征已 L2 归一化）"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        B = features.size(0)
        sim = torch.div(torch.matmul(features, features.t()), self.temperature)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(torch.clamp(exp_sim.sum(dim=1, keepdim=True), min=1e-9))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        loss = - mean_log_prob_pos.mean()
        return loss
