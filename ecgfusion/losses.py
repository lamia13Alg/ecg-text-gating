"""Pertes multi-label avec gestion du déséquilibre et des labels indisponibles."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al., ICCV 2021) + pondération par classe + masque de labels.

    gamma_neg > gamma_pos : les négatifs faciles (très nombreux pour les
    classes rares) sont fortement atténués ; ``clip`` décale la probabilité
    des négatifs pour ignorer les négatifs très faciles / mal étiquetés.
    """

    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, class_weights=None, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps
        w = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("w", w)

    def forward(self, logits, y, label_mask=None, sample_mask=None):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = y * torch.log(p.clamp(min=self.eps)) * (1 - p) ** self.gp
        loss_neg = (1 - y) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg ** self.gn
        loss = -(loss_pos + loss_neg)
        if self.w is not None:
            loss = loss * (y * self.w + (1 - y))  # pondère les positifs des classes rares
        m = torch.ones_like(loss)
        if label_mask is not None:
            m = m * label_mask
        if sample_mask is not None:
            m = m * sample_mask[:, None]
        return (loss * m).sum() / m.sum().clamp(min=1.0)


class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        w = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("w", w)

    def forward(self, logits, y, label_mask=None, sample_mask=None):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = y * p + (1 - y) * (1 - p)
        loss = (1 - pt) ** self.gamma * bce
        if self.w is not None:
            loss = loss * (y * self.w + (1 - y))
        m = torch.ones_like(loss)
        if label_mask is not None:
            m = m * label_mask
        if sample_mask is not None:
            m = m * sample_mask[:, None]
        return (loss * m).sum() / m.sum().clamp(min=1.0)


def class_weights_from_prevalence(labels: np.ndarray, power: float = 0.5, max_w: float = 20.0):
    pos = labels.mean(0).clip(1e-4, 1 - 1e-4)
    return np.clip(((1 - pos) / pos) ** power, 1.0, max_w).astype(np.float32)


def build_loss(name: str, class_weights=None, **kw):
    if name == "asl":
        return AsymmetricLoss(class_weights=class_weights, **kw)
    if name == "focal":
        return FocalBCE(class_weights=class_weights, **kw)
    if name == "bce":
        return FocalBCE(gamma=0.0, class_weights=class_weights)
    raise ValueError(name)
