"""Loss multi-label untuk perbandingan strategi imbalance.

Semua loss menerima logits (B, C) dan target (B, C) lalu mengembalikan rata-rata
per elemen, supaya skalanya sebanding antar strategi.

Nama strategi:
  bce        BCE tanpa penanganan                     (B0)
  pos_weight BCE dengan pos_weight = (neg/pos)^power   (B1)
  focal      focal loss, gamma=2                       (B2)
  asl        asymmetric loss, gamma-=4 gamma+=1 clip=0.05  (B3)
  cb         class-balanced (effective number), beta=0.999 (B4)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        return (((1 - pt) ** self.gamma) * bce).mean()


class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021. Negatif diberi down-weighting kuat dan margin (clip)."""

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = y * ((1 - p) ** self.gp) * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - y) * (p_neg ** self.gn) * torch.log((1 - p_neg).clamp(min=self.eps))
        return -(loss_pos + loss_neg).mean()


class ClassBalancedBCE(nn.Module):
    """Bobot per kelas dari effective number of samples (Cui et al. 2019)."""

    def __init__(self, counts, beta=0.999):
        super().__init__()
        counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.sum() * len(w)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (bce * self.w).mean()


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pw", torch.as_tensor(pos_weight, dtype=torch.float32))

    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)


class PlainBCE(nn.Module):
    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)


def make_loss(name, y_train, pw_power=1.0, gamma=2.0, beta=0.999):
    """y_train: array (N, C) biner dari fold latih. Hitungan kelas hanya dari data latih."""
    pos = y_train.sum(0).astype(np.float64)
    if name == "bce":
        return PlainBCE()
    if name == "pos_weight":
        pw = ((len(y_train) - pos) / np.maximum(pos, 1.0)) ** pw_power
        return WeightedBCE(pw)
    if name == "focal":
        return FocalLoss(gamma)
    if name == "asl":
        return AsymmetricLoss()
    if name == "cb":
        return ClassBalancedBCE(pos, beta)
    raise ValueError(f"loss tidak dikenal: {name}")
