"""Métriques par pathologie, calibration et seuils."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ece_score(y, p, n_bins=10):
    """Expected Calibration Error (classe binaire, bins de largeur égale)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        s = idx == b
        if s.any():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def per_label_metrics(y, prob, thresholds, label_names, label_mask, min_pos=5):
    """Retourne un tableau (une ligne par label). Labels indisponibles ou avec
    moins de ``min_pos`` positifs : métriques NaN, exclues des moyennes macro."""
    rows = []
    for c, name in enumerate(label_names):
        yc, pc = y[:, c], prob[:, c]
        n_pos = int(yc.sum())
        r = {"label": name, "n_pos": n_pos, "prevalence": float(yc.mean()) if len(yc) else np.nan}
        ok = label_mask[c] > 0 and n_pos >= min_pos and n_pos < len(yc)
        if ok:
            pred = pc >= thresholds[c]
            tp, fp = (pred & (yc == 1)).sum(), (pred & (yc == 0)).sum()
            fn, tn = (~pred & (yc == 1)).sum(), (~pred & (yc == 0)).sum()
            sens = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            r.update(auroc=roc_auc_score(yc, pc), auprc=average_precision_score(yc, pc),
                     f1=2 * prec * sens / max(prec + sens, 1e-9), sensitivity=sens,
                     specificity=tn / max(tn + fp, 1), ppv=prec,
                     ece=ece_score(yc, pc), brier=float(np.mean((pc - yc) ** 2)))
        else:
            r.update({k: np.nan for k in ["auroc", "auprc", "f1", "sensitivity", "specificity",
                                          "ppv", "ece", "brier"]})
        rows.append(r)
    return pd.DataFrame(rows)


def macro_summary(df: pd.DataFrame, rare_quantile: float = 0.34) -> dict:
    d = df.dropna(subset=["auroc"])
    out = {f"macro_{k}": float(d[k].mean()) for k in
           ["auroc", "auprc", "f1", "sensitivity", "specificity", "ece", "brier"]}
    out["n_labels_evaluated"] = len(d)
    if len(d) >= 3:  # classes rares = tiers le moins prévalent
        rare = d[d["prevalence"] <= d["prevalence"].quantile(rare_quantile)]
        out["rare_macro_auroc"] = float(rare["auroc"].mean())
        out["rare_macro_f1"] = float(rare["f1"].mean())
    return out


def bootstrap_macro_auroc(y, prob, label_mask, n_boot=200, min_pos=5, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        aucs = [roc_auc_score(y[s, c], prob[s, c]) for c in range(y.shape[1])
                if label_mask[c] and min_pos <= y[s, c].sum() < n]
        if aucs:
            vals.append(np.mean(aucs))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------
# Calibration et seuils (appris sur la validation interne MIMIC uniquement)
# --------------------------------------------------------------------------
def fit_temperatures(logits, y, label_mask=None):
    """Température par classe minimisant la NLL de validation."""
    T = np.ones(y.shape[1])
    for c in range(y.shape[1]):
        if (label_mask is not None and not label_mask[c]) or y[:, c].sum() == 0:
            continue
        z, t = logits[:, c], y[:, c]

        def nll(logT):
            p = sigmoid(z / np.exp(logT)).clip(1e-7, 1 - 1e-7)
            return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

        T[c] = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    return T


def fit_thresholds(y, prob, label_mask=None, grid=np.linspace(0.02, 0.98, 97)):
    """Seuil par classe maximisant le F1 en validation."""
    th = np.full(y.shape[1], 0.5)
    for c in range(y.shape[1]):
        if (label_mask is not None and not label_mask[c]) or y[:, c].sum() == 0:
            continue
        best = -1
        for t in grid:
            pred = prob[:, c] >= t
            tp = (pred & (y[:, c] == 1)).sum()
            f1 = 2 * tp / max(pred.sum() + y[:, c].sum(), 1)
            if f1 > best:
                best, th[c] = f1, t
    return th
