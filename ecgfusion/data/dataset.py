"""Dataset PyTorch multimodal avec masquage de modalités et de dérivations."""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..labels import LABEL_AVAILABILITY, mask_label_statements, mask_strict
from ..signal_processing import LEADS, normalize

# Sous-ensembles de dérivations utilisés en entraînement et en évaluation
LEAD_SUBSETS = {
    "12": LEADS,
    "limb6": ["I", "II", "III", "aVR", "aVL", "aVF"],
    "I+II": ["I", "II"],
    "I": ["I"],                      # type montre connectée
    "II+V1+V5": ["II", "V1", "V5"],  # monitoring
}


@dataclass
class Scenario:
    """Modalités disponibles au moment de l'inférence."""
    name: str = "full"
    use_text: bool = True
    use_features: bool = True
    leads: list[str] = field(default_factory=lambda: list(LEADS))


SCENARIOS = {
    "full": Scenario("full"),
    "no_text": Scenario("no_text", use_text=False),
    "no_features": Scenario("no_features", use_features=False),
    "signal_only": Scenario("signal_only", use_text=False, use_features=False),
    "no_signal": Scenario("no_signal", leads=[]),
    "text_only": Scenario("text_only", use_features=False, leads=[]),
    "features_only": Scenario("features_only", use_text=False, leads=[]),
    "limb6_no_text": Scenario("limb6_no_text", use_text=False, leads=LEAD_SUBSETS["limb6"]),
    "leadI_no_text": Scenario("leadI_no_text", use_text=False, leads=LEAD_SUBSETS["I"]),
    "leadsI+II_full": Scenario("leadsI+II_full", leads=LEAD_SUBSETS["I+II"]),
    # pour la référence « signal seul » entraînée séparément (pas de caractéristiques)
    "limb6_signal_only": Scenario("limb6_signal_only", use_text=False, use_features=False,
                                  leads=LEAD_SUBSETS["limb6"]),
    "leadI_signal_only": Scenario("leadI_signal_only", use_text=False, use_features=False,
                                  leads=LEAD_SUBSETS["I"]),
}


class FeatureNormalizer:
    """Standardisation robuste (médiane / IQR) apprise sur le train MIMIC uniquement."""

    def __init__(self, median=None, iqr=None):
        self.median, self.iqr = median, iqr

    def fit(self, x: np.ndarray):
        self.median = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        self.iqr = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
        self.median = np.nan_to_num(self.median)
        return self

    def __call__(self, x: np.ndarray):
        z = (x - self.median) / self.iqr
        z = np.clip(z, -10, 10)
        missing = np.isnan(z)
        return np.nan_to_num(z).astype(np.float32), missing.astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({"median": self.median.tolist(), "iqr": self.iqr.tolist()}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["median"], np.float32), np.array(d["iqr"], np.float32))


class ECGMultimodalDataset(Dataset):
    def __init__(self, cache_dir: str, split: str | None, feat_norm: FeatureNormalizer,
                 text_policy: str = "full", train: bool = False, scenario: Scenario | None = None,
                 p_drop_text: float = 0.3, p_drop_features: float = 0.2, p_drop_leads: float = 0.3,
                 max_records: int | None = None, signal_norm: str = "lead"):
        d = Path(cache_dir)
        self.signals = np.load(d / "signals.npy", mmap_mode="r")
        feats = np.load(d / "features.npy")
        labels = np.load(d / "labels.npy")
        meta = pd.read_csv(d / "meta.csv", keep_default_na=False)
        mask = meta["valid"].astype(str).str.lower().eq("true").to_numpy().copy()
        if split not in (None, "all"):
            mask &= meta["split"].values == split
        self.idx = np.where(mask)[0]
        if max_records:
            self.idx = self.idx[:max_records]

        self.feats, self.labels, self.meta = feats, labels, meta
        self.feat_norm, self.train = feat_norm, train
        self.scenario = scenario or SCENARIOS["full"]
        self.signal_norm = signal_norm
        self.p_text, self.p_feat, self.p_leads = p_drop_text, p_drop_features, p_drop_leads
        self.dataset_name = str(meta["dataset"].iloc[0]) if len(meta) else "unknown"
        self.label_mask = np.array(LABEL_AVAILABILITY.get(self.dataset_name,
                                   [1] * labels.shape[1]), np.float32)

        texts = meta["text"].astype(str).values
        if text_policy == "masked":
            texts = np.array([mask_label_statements(t) for t in texts], dtype=object)
        elif text_policy == "strict":
            texts = np.array([mask_strict(t) for t in texts], dtype=object)
        elif text_policy == "none":
            texts = np.array([""] * len(texts), dtype=object)
        self.texts = texts

    def __len__(self):
        return len(self.idx)

    def _lead_mask(self) -> np.ndarray:
        m = np.array([l in self.scenario.leads for l in LEADS], np.float32)
        if self.train and m.sum() > 0 and np.random.rand() < self.p_leads:
            choice = np.random.choice(["limb6", "I+II", "I", "II+V1+V5", "random"])
            if choice == "random":
                m = (np.random.rand(12) > 0.5).astype(np.float32)
                m[np.random.randint(12)] = 1.0
            else:
                m = np.array([l in LEAD_SUBSETS[choice] for l in LEADS], np.float32)
        return m

    def __getitem__(self, k):
        i = self.idx[k]
        sig = normalize(np.asarray(self.signals[i], dtype=np.float32), mode=self.signal_norm)
        if self.train:  # augmentations légères
            sig = sig * np.random.uniform(0.8, 1.2, (12, 1)).astype(np.float32)
            sig = sig + np.random.normal(0, 0.03, sig.shape).astype(np.float32)
        lead_mask = self._lead_mask()
        sig = sig * lead_mask[:, None]
        has_signal = float(lead_mask.sum() > 0)

        feat, feat_missing = self.feat_norm(self.feats[i])
        has_feat = float(self.scenario.use_features and feat_missing.mean() < 1.0)
        text = self.texts[i] if self.scenario.use_text else ""
        has_text = float(len(text.strip()) > 0)

        if self.train:
            if np.random.rand() < self.p_text:
                has_text = 0.0
            if np.random.rand() < self.p_feat:
                has_feat = 0.0
            if has_signal + has_text + has_feat == 0:
                has_signal, lead_mask = 1.0, np.ones(12, np.float32)
                sig = normalize(np.asarray(self.signals[i], dtype=np.float32), mode=self.signal_norm)
        if not has_text:
            text = ""

        return {
            "signal": torch.from_numpy(sig),
            "lead_mask": torch.from_numpy(lead_mask),
            "features": torch.from_numpy(feat * has_feat),
            "feat_missing": torch.from_numpy(np.maximum(feat_missing, 1 - has_feat)),
            "text": text,
            "present": torch.tensor([has_signal, has_text, has_feat], dtype=torch.float32),
            "labels": torch.from_numpy(self.labels[i]),
            "label_mask": torch.from_numpy(self.label_mask),
            "index": int(i),
        }


# --------------------------------------------------------------------------
class HashTokenizer:
    """Tokeniseur sans dépendance (tests hors-ligne / encodeur texte léger)."""

    def __init__(self, vocab_size: int = 2 ** 15):
        self.vocab_size = vocab_size

    def __call__(self, texts, max_length=128, **_):
        ids = np.zeros((len(texts), max_length), np.int64)
        att = np.zeros((len(texts), max_length), np.int64)
        for r, t in enumerate(texts):
            toks = re.findall(r"[a-zA-Zäöüß]+|\d+", t.lower())[:max_length - 1]
            seq = [1] + [2 + (zlib.crc32(w.encode()) % (self.vocab_size - 2)) for w in toks]
            ids[r, :len(seq)], att[r, :len(seq)] = seq, 1
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(att)}


class Collator:
    def __init__(self, tokenizer, max_length: int = 128):
        self.tok, self.max_length = tokenizer, max_length

    def __call__(self, batch):
        out = {k: torch.stack([b[k] for b in batch]) for k in
               ["signal", "lead_mask", "features", "feat_missing", "present", "labels", "label_mask"]}
        out["index"] = torch.tensor([b["index"] for b in batch])
        texts = [b["text"] if b["text"] else "[none]" for b in batch]
        tok = self.tok(texts, max_length=self.max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
        out["input_ids"], out["attention_mask"] = tok["input_ids"], tok["attention_mask"]
        return out
