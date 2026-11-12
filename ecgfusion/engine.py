"""Fonctions partagées entre entraînement et évaluation."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import Collator, ECGMultimodalDataset, HashTokenizer
from .labels import NUM_LABELS
from .model import MultimodalECGNet
from .signal_processing import NUM_FEATURES


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """--set train.batch_size=64 data.train_cache=/kaggle/input/x/mimic ..."""
    import yaml

    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tokenizer(cfg):
    if cfg["model"]["text_encoder"] == "hf":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(cfg["model"]["text_model_name"])
    return HashTokenizer()


def build_model(cfg) -> MultimodalECGNet:
    m = cfg["model"]
    return MultimodalECGNet(
        n_labels=NUM_LABELS, n_features=NUM_FEATURES, d_model=m["d_model"],
        text_encoder=m["text_encoder"], text_model_name=m["text_model_name"],
        text_trainable_layers=m["text_trainable_layers"], signal_layers=m["signal_layers"],
        fusion_layers=m["fusion_layers"], n_heads=m["n_heads"], dropout=m["dropout"])


def make_loader(ds: ECGMultimodalDataset, tokenizer, cfg, shuffle: bool):
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["train"]["num_workers"], drop_last=shuffle,
                      collate_fn=Collator(tokenizer, cfg["model"]["max_text_len"]),
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg["train"]["num_workers"] > 0)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device, return_attention=False):
    model.eval()
    logits, labels, attn, idx = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(batch, return_attention=return_attention)
        logits.append(out["logits"].float().cpu())
        labels.append(batch["labels"].cpu())
        idx.append(batch["index"].cpu())
        if return_attention:
            attn.append(out["modality_attention"].float().cpu())
    res = {"logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy(),
           "index": torch.cat(idx).numpy()}
    if return_attention:
        res["modality_attention"] = torch.cat(attn).numpy()
    return res
