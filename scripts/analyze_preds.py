"""Analyse a posteriori des prédictions (CPU, sans réentraînement).

À partir de <run>/eval/preds_<jeu>_<scénario>.npz et <run>/calibration.json :

 1. Adaptation aux prévalences (label shift) par EM — Saerens et al., Neural Comput. 2002 —
    sans aucun label externe ; comparée au seuil MIMIC brut, à une recalibration
    « few-shot » (n ECG étiquetés du centre externe) et au seuil oracle (borne haute).
 2. Calibration : ECE avant / après EM et courbes de fiabilité par pathologie (PNG).
 3. Gating par accord : on utilise le texte seulement si les prédictions avec et sans texte
    concordent ; seuil de désaccord fixé sur le jeu interne (aucune donnée externe utilisée).
 4. Tests de DeLong appariés entre scénarios (par pathologie) + IC bootstrap de Δ AUROC macro.

  python scripts/analyze_preds.py --run runs/fusion --out runs/fusion/analysis
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.labels import LABEL_AVAILABILITY, LABELS  # noqa: E402
from ecgfusion.metrics import ece_score, fit_thresholds  # noqa: E402

EPS = 1e-6


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def load_preds(eval_dir: Path) -> dict:
    out = {}
    for f in sorted(eval_dir.glob("preds_*.npz")):
        m = re.match(r"preds_(.+?_(?:internal|external|val))_(.+)\.npz", f.name)
        if not m:
            continue
        d = np.load(f)
        out[(m.group(1), m.group(2))] = {"prob": d["prob"].astype(np.float64),
                                         "y": d["labels"].astype(int), "index": d["index"]}
    return out


def label_mask(set_name: str) -> np.ndarray:
    key = next((k for k in LABEL_AVAILABILITY if set_name.startswith(k)), None)
    return np.array(LABEL_AVAILABILITY.get(key, [1] * len(LABELS)), bool)


def pick_threshold(calib: dict, scenario: str) -> np.ndarray:
    """Même logique que evaluate.py : seuils appris sur le scénario de validation le plus proche."""
    if scenario in calib:
        return np.array(calib[scenario]["threshold"])
    no_text = "no_text" in scenario or scenario in ("signal_only", "features_only", "gated")
    if scenario in ("signal_only",) and "signal_only" in calib:
        return np.array(calib["signal_only"]["threshold"])
    if no_text and "no_text" in calib:
        return np.array(calib["no_text"]["threshold"])
    return np.array(calib.get("full", next(iter(calib.values())))["threshold"])


# --------------------------------------------------------------------------
# 0. Recalibration de Platt (a·logit(p)+b par pathologie), apprise sur le jeu interne
# --------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y, p):
    from sklearn.linear_model import LogisticRegression

    ab = []
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < len(y):
            lr = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p[:, c])[:, None], y[:, c])
            ab.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        else:
            ab.append((1.0, 0.0))
    return np.array(ab)


def apply_platt(p, ab):
    return 1 / (1 + np.exp(-(ab[:, 0] * _logit(p) + ab[:, 1])))


# --------------------------------------------------------------------------
# 1. Label shift (EM)
# --------------------------------------------------------------------------
def em_prior_adjust(p: np.ndarray, prior_src: float, n_iter: int = 200, tol: float = 1e-7):
    """Saerens et al. 2002, cas binaire. Retourne (proba ajustées, prévalence estimée)."""
    p = np.clip(p, EPS, 1 - EPS)
    ps = float(np.clip(prior_src, EPS, 1 - EPS))
    pt = ps
    for _ in range(n_iter):
        w1, w0 = pt / ps, (1 - pt) / (1 - ps)
        q = w1 * p / (w1 * p + w0 * (1 - p))
        new = float(q.mean())
        if abs(new - pt) < tol:
            pt = new
            break
        pt = new
    w1, w0 = pt / ps, (1 - pt) / (1 - ps)
    return w1 * p / (w1 * p + w0 * (1 - p)), pt


def bbse_prior(p_tgt, p_ref, y_ref, th):
    """BBSE binaire (Lipton et al., ICML 2018) : π_t = (q_t − FPR) / (TPR − FPR),
    TPR / FPR mesurés sur le jeu interne au seuil ``th``. Plus robuste qu'EM
    quand les probabilités sont imparfaitement calibrées."""
    pred_ref = p_ref >= th
    tpr = pred_ref[y_ref == 1].mean() if (y_ref == 1).any() else 1.0
    fpr = pred_ref[y_ref == 0].mean() if (y_ref == 0).any() else 0.0
    q = (p_tgt >= th).mean()
    if tpr - fpr < 0.05:
        return None
    return float(np.clip((q - fpr) / (tpr - fpr), 1e-4, 1 - 1e-4))


def prior_adjust(p, prior_src, prior_tgt):
    p = np.clip(p, EPS, 1 - EPS)
    w1, w0 = prior_tgt / prior_src, (1 - prior_tgt) / (1 - prior_src)
    return w1 * p / (w1 * p + w0 * (1 - p))


def binary_metrics(y, p, th):
    pred = p >= th
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    return sens, spec, ppv, f1


def oracle_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.005, 0.995, 199)))
    best, bt = -1, 0.5
    for t in grid:
        f = binary_metrics(y, p, t)[3]
        if f > best:
            best, bt = f, t
    return bt


def fewshot(y, p, label_ok, n_shot=200, repeats=20, seed=0):
    """Seuils réappris sur n_shot ECG étiquetés du centre externe, évalués sur le reste."""
    rng = np.random.default_rng(seed)
    res = []
    n_shot = min(n_shot, len(y) // 4)
    for _ in range(repeats):
        idx = rng.permutation(len(y))
        cal, test = idx[:n_shot], idx[n_shot:]
        th = fit_thresholds(y[cal].astype(float), p[cal], label_ok)
        res.append([binary_metrics(y[test, c], p[test, c], th[c]) for c in range(y.shape[1])])
    return np.array(res).mean(0)  # (C, 4)


def label_shift_table(preds, calib, src_set, ref_set, min_pos):
    """Compare, par pathologie : seuil MIMIC brut / Platt + EM / few-shot / oracle.

    Platt et les seuils associés sont appris sur ``ref_set`` (validation interne si
    disponible, sinon test interne) ; EM n'utilise aucun label externe.
    """
    rows = []
    src_prior = {sc: v["y"].mean(0) for (s, sc), v in preds.items() if s == src_set}
    platt, th_platt = {}, {}
    for (s, sc), v in preds.items():
        if s == ref_set:
            platt[sc] = fit_platt(v["y"], v["prob"])
            th_platt[sc] = fit_thresholds(v["y"].astype(float), apply_platt(v["prob"], platt[sc]))
    for (s, sc), v in preds.items():
        if sc not in src_prior or sc not in platt:
            continue
        y, p = v["y"], v["prob"]
        pc = apply_platt(p, platt[sc])
        ok = label_mask(s)
        th = pick_threshold(calib, sc)
        fs = fewshot(y, p, ok) if s != src_set else None
        for c, lab in enumerate(LABELS):
            npos = int(y[:, c].sum())
            if not ok[c] or npos < min_pos or npos == len(y):
                continue
            prior_c = float(np.mean(apply_platt(preds[(src_set, sc)]["prob"], platt[sc])[:, c]))
            q, pt = em_prior_adjust(pc[:, c], prior_c)
            ref_p = apply_platt(preds[(ref_set, sc)]["prob"], platt[sc])[:, c]
            pb = bbse_prior(pc[:, c], ref_p, preds[(ref_set, sc)]["y"][:, c], th_platt[sc][c])
            qb = prior_adjust(pc[:, c], prior_c, pb) if pb is not None else pc[:, c]
            b = binary_metrics(y[:, c], p[:, c], th[c])
            a = binary_metrics(y[:, c], q, th_platt[sc][c])
            bb = binary_metrics(y[:, c], qb, th_platt[sc][c])
            o = binary_metrics(y[:, c], p[:, c], oracle_threshold(y[:, c], p[:, c]))
            r = {"eval_set": s, "scenario": sc, "label": lab, "n_pos": npos,
                 "prev_true": y[:, c].mean(), "prev_source": src_prior[sc][c],
                 "prev_mean_prob": pc[:, c].mean(), "prev_em": pt,
                 "prev_bbse": pb if pb is not None else np.nan,
                 "auroc": roc_auc_score(y[:, c], p[:, c]),
                 "sens_raw": b[0], "spec_raw": b[1], "ppv_raw": b[2], "f1_raw": b[3],
                 "sens_em": a[0], "spec_em": a[1], "ppv_em": a[2], "f1_em": a[3],
                 "sens_bbse": bb[0], "spec_bbse": bb[1], "ppv_bbse": bb[2], "f1_bbse": bb[3],
                 "sens_oracle": o[0], "spec_oracle": o[1], "ppv_oracle": o[2], "f1_oracle": o[3],
                 "ece_raw": ece_score(y[:, c], p[:, c]), "ece_platt": ece_score(y[:, c], pc[:, c]),
                 "ece_em": ece_score(y[:, c], q), "ece_bbse": ece_score(y[:, c], qb)}
            if fs is not None:
                r.update(sens_fewshot=fs[c, 0], spec_fewshot=fs[c, 1], ppv_fewshot=fs[c, 2], f1_fewshot=fs[c, 3])
            rows.append(r)
    return pd.DataFrame(rows), platt


# --------------------------------------------------------------------------
# 2. Courbes de fiabilité
# --------------------------------------------------------------------------
PAPER_LABELS = ["NORM", "AFIB", "MI", "LVH", "SBRAD", "STT_ISCH"]
SHORT = {"raw": "raw", "Platt (MIMIC)": "MIMIC", "Platt (200 local ECGs)": "local"}
COLORS = {"raw": "#1f6feb", "Platt (MIMIC)": "#2f9e44", "Platt (200 local ECGs)": "#d9480f"}


def _calibration_variants(p, y, platt_ab, n_local=200, seed=0):
    """Retourne {nom: (proba, masque d'évaluation)} pour les trois variantes."""
    out = {"raw": (p, np.ones(len(y), bool))}
    if platt_ab is not None:
        out["Platt (MIMIC)"] = (apply_platt(p, platt_ab), np.ones(len(y), bool))
        if len(y) >= 4 * n_local:
            idx = np.random.default_rng(seed).permutation(len(y))
            cal = np.zeros(len(y), bool); cal[idx[:n_local]] = True
            ab_local = fit_platt(y[cal], p[cal])
            out["Platt (200 local ECGs)"] = (apply_platt(p, ab_local), ~cal)
    return out


def _reliability_axis(ax, y, variants, c, n_bins=10, min_count=20, show_hist=True, legend=True):
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    if show_hist:  # effectifs (probabilités brutes), échelle logarithmique à droite
        ax2 = ax.twinx()
        ax2.hist(variants["raw"][0][:, c], bins=n_bins, range=(0, 1), color="#ced4da", alpha=0.6, log=True)
        ax2.yaxis.set_visible(False); ax2.tick_params(which="both", right=False, labelright=False)
        ax2.set_zorder(0); ax.set_zorder(1); ax.patch.set_visible(False)
    for name, (pp, m) in variants.items():
        yy, q = y[m, c], pp[m, c]
        bins = np.clip((q * n_bins).astype(int), 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            sel = bins == b
            if sel.sum() >= min_count:
                xs.append(q[sel].mean()); ys.append(yy[sel].mean())
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=COLORS[name],
                label=f"{name} — ECE {ece_score(yy, q):.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if legend:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)


def paper_calibration_figure(y, variants, ok, title, stem, n_bins=10):
    """Figure article : 6 pathologies (2 × 3), PNG + PDF ; renvoie les chemins écrits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = [l for l in PAPER_LABELS if ok[LABELS.index(l)] and 0 < y[:, LABELS.index(l)].sum() < len(y)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6))
    for ax, lab in zip(axes.flat, labs):
        c = LABELS.index(lab)
        _reliability_axis(ax, y, variants, c, n_bins, legend=False)
        eces = "   ".join(f"{SHORT[n]} {ece_score(y[m, c], pp[m, c]):.3f}" for n, (pp, m) in variants.items())
        ax.set_title(f"{lab} (prevalence {y[:, c].mean():.1%})\nECE: {eces}", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(labs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Predicted probability", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency", fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[n], marker="o", ms=3.5, lw=1.4, label=n) for n in variants]
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    paths = []
    for ext in ("png", "pdf"):
        path = Path(str(stem) + f".{ext}")
        fig.savefig(path, dpi=200); paths.append(path)
    plt.close(fig)
    return paths


def reliability_plots(preds, src_set, out_dir, platt, scenarios=("full", "no_text", "signal_only"), n_bins=10):
    """Une figure complète par (jeu, scénario) + une figure « article » (6 pathologies,
    PNG + PDF) pour chaque jeu externe. Points : intervalles avec ≥ 20 ECG ; barres grises :
    effectifs des probabilités brutes (échelle log)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for (s, sc), v in preds.items():
        if sc not in scenarios:
            continue
        y, p, ok = v["y"], v["prob"], label_mask(s)
        variants = _calibration_variants(p, y, platt.get(sc) if s != src_set else None)
        fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
        for c, lab in enumerate(LABELS):
            ax = axes.flat[c]
            if ok[c] and 0 < y[:, c].sum() < len(y):
                _reliability_axis(ax, y, variants, c, n_bins)
            else:
                ax.text(0.5, 0.5, "not evaluated", ha="center", va="center", color="#999")
            ax.set_title(f"{lab} (prev. {y[:, c].mean():.1%})", fontsize=10); ax.tick_params(labelsize=7)
        axes.flat[-1].axis("off")
        fig.suptitle(f"Reliability — {s} / {sc}   (x: predicted probability, y: observed frequency)")
        fig.tight_layout()
        path = out_dir / f"reliability_{s}_{sc}.png"
        fig.savefig(path, dpi=110); plt.close(fig); paths.append(path)

        # figure article : jeux externes, scénario principal
        if s != src_set and "external" in s and sc in ("no_text", "signal_only"):
            title = f"Calibration at external validation — {s.replace('_external', '').upper()} ({sc})"
            paths += paper_calibration_figure(y, variants, ok, title, out_dir / f"fig_calibration_{s}_{sc}")
    return paths


# --------------------------------------------------------------------------
# 3. Gating par accord texte / sans texte
# --------------------------------------------------------------------------
def disagreement(p_full, p_notext):
    a = np.clip(p_full, EPS, 1 - EPS); b = np.clip(p_notext, EPS, 1 - EPS)
    kl = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    kl2 = b * np.log(b / a) + (1 - b) * np.log((1 - b) / (1 - a))
    return 0.5 * (kl + kl2).mean(1)


def gating(preds, ref_set, min_pos=10, percentiles=(90, 95, 99), text_avail=None):
    """Seuil de désaccord = percentile sur le jeu de référence interne."""
    if (ref_set, "full") not in preds or (ref_set, "no_text") not in preds:
        return pd.DataFrame(), {}
    ref = preds[(ref_set, "full")], preds[(ref_set, "no_text")]
    assert np.array_equal(ref[0]["index"], ref[1]["index"])
    d_ref = disagreement(ref[0]["prob"], ref[1]["prob"])
    rows, gated = [], {}
    for pc in percentiles:
        tau = float(np.percentile(d_ref, pc))
        for (s, sc), v in preds.items():
            if sc != "full" or (s, "no_text") not in preds:
                continue
            nt = preds[(s, "no_text")]
            assert np.array_equal(v["index"], nt["index"]), f"ordre différent pour {s}"
            d = disagreement(v["prob"], nt["prob"])
            use_text = d <= tau
            pg = np.where(use_text[:, None], v["prob"], nt["prob"])
            ok = label_mask(s)
            y = v["y"]

            def macro(pp):
                return np.mean([roc_auc_score(y[:, c], pp[:, c]) for c in range(y.shape[1])
                                if ok[c] and min_pos <= y[:, c].sum() < len(y)])
            # Gating au niveau du centre : si la proportion d'ECG en désaccord dépasse nettement
            # la proportion attendue (1 − pc/100, par construction sur le jeu interne), le texte
            # est jugé hors domaine et désactivé pour tout le centre. Aucun label utilisé.
            expected = 1 - pc / 100
            k, n = int((d > tau).sum()), len(d)
            p_site = float(stats.binomtest(k, n, expected, alternative="greater").pvalue)
            # centre sans aucun compte rendu (ex. LSAS) : rien à désactiver
            no_text_at_site = (text_avail or {}).get(s, 1.0) < 0.01
            drop_text = (not no_text_at_site) and (p_site < 0.01) and (k / n > 1.5 * expected)
            p_site_prob = nt["prob"] if drop_text else v["prob"]
            rows.append({"eval_set": s, "percentile": pc, "tau": tau, "text_used_rate": use_text.mean(),
                         "disagree_rate": k / n, "disagree_expected": expected, "p_site_shift": p_site,
                         "site_decision": ("pas de texte" if no_text_at_site else
                                           "sans texte" if drop_text else "avec texte"),
                         "macro_auroc_full": macro(v["prob"]), "macro_auroc_no_text": macro(nt["prob"]),
                         "macro_auroc_gated": macro(pg), "macro_auroc_site_gated": macro(p_site_prob)})
            if pc == 95:
                gated[s] = {"prob": pg, "y": y, "index": v["index"]}
                gated[s + "|site"] = {"prob": p_site_prob, "y": y, "index": v["index"]}
    return pd.DataFrame(rows), gated


# --------------------------------------------------------------------------
# 4. DeLong apparié (Sun & Xu 2014, version rapide)
# --------------------------------------------------------------------------
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong_paired(y, p1, p2):
    """Retourne (auc1, auc2, p-valeur bilatérale) pour deux scores appariés."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2 / n)
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.array(v10)), np.cov(np.array(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 0:
        return aucs[0], aucs[1], 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(z)))


def compare_scenarios(preds, pairs, min_pos, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    rows, macro_rows = [], []
    sets = sorted({s for s, _ in preds})
    for s in sets:
        ok = label_mask(s)
        for a, b in pairs:
            if (s, a) not in preds or (s, b) not in preds:
                continue
            A, B = preds[(s, a)], preds[(s, b)]
            assert np.array_equal(A["index"], B["index"])
            y = A["y"]
            labs = [c for c in range(y.shape[1]) if ok[c] and min_pos <= y[:, c].sum() < len(y)]
            for c in labs:
                a1, a2, pv = delong_paired(y[:, c], A["prob"][:, c], B["prob"][:, c])
                rows.append({"eval_set": s, "A": a, "B": b, "label": LABELS[c],
                             "auroc_A": a1, "auroc_B": a2, "delta": a1 - a2, "p_delong": pv})
            # IC bootstrap apparié de Δ AUROC macro
            deltas = []
            for _ in range(n_boot):
                i = rng.integers(0, len(y), len(y))
                da = [roc_auc_score(y[i, c], A["prob"][i, c]) - roc_auc_score(y[i, c], B["prob"][i, c])
                      for c in labs if 0 < y[i, c].sum() < len(i)]
                deltas.append(np.mean(da))
            d0 = np.mean([roc_auc_score(y[:, c], A["prob"][:, c]) - roc_auc_score(y[:, c], B["prob"][:, c])
                          for c in labs])
            macro_rows.append({"eval_set": s, "A": a, "B": b, "delta_macro_auroc": d0,
                               "ci_low": np.percentile(deltas, 2.5), "ci_high": np.percentile(deltas, 97.5)})
    df = pd.DataFrame(rows)
    if len(df):  # correction de Holm sur toutes les comparaisons par pathologie
        order = np.argsort(df["p_delong"].values)
        m = len(df); adj = np.empty(m); run = 0.0
        for rank, k in enumerate(order):
            run = max(run, min(1.0, (m - rank) * df["p_delong"].values[k])); adj[k] = run
        df["p_holm"] = adj
    return df, pd.DataFrame(macro_rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dossier du run (contient eval/ et calibration.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=200)
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    preds = load_preds(run / "eval")
    calib = json.load(open(run / "calibration.json"))
    sets = sorted({s for s, _ in preds})
    src = next((s for s in sets if "internal" in s), sets[0])
    # référence : validation interne si disponible (aucune fuite du test), sinon test interne
    ref = next((s for s in sets if s.endswith("_val")), src)
    print(f"[analyse] {len(preds)} fichiers de prédictions ; source = {src} ; "
          f"référence (Platt, seuils, gating) = {ref}"
          + ("" if ref.endswith("_val") else "  [pas de validation sauvegardée : test interne utilisé]"))

    ls, platt = label_shift_table(preds, calib, src, ref, a.min_pos)
    ls.to_csv(out / "label_shift_per_label.csv", index=False)
    cols = [c for c in ls.columns if c.split("_")[0] in ("sens", "ppv", "f1", "ece") or c.startswith("prev_")]
    summ = ls.groupby(["eval_set", "scenario"])[cols].mean().reset_index()
    summ.to_csv(out / "label_shift_summary.csv", index=False)

    text_avail = {}
    if (run / "eval" / "summary.csv").exists():
        sm = pd.read_csv(run / "eval" / "summary.csv")
        text_avail = sm[sm.scenario == "full"].set_index("eval_set")["text_available"].to_dict()
    gate_df, gated = gating(preds, ref, a.min_pos, text_avail=text_avail)
    gate_df.to_csv(out / "gating.csv", index=False)
    for s, v in gated.items():
        if s.endswith("|site"):
            preds[(s[:-5], "site_gated")] = v
        else:
            preds[(s, "gated")] = v

    pairs = [("full", "no_text"), ("no_text", "signal_only"), ("gated", "no_text"), ("site_gated", "full"),
             ("no_text", "leadI_no_text"), ("no_text", "limb6_no_text")]
    dl, dl_macro = compare_scenarios(preds, pairs, a.min_pos, a.n_boot)
    dl.to_csv(out / "delong_per_label.csv", index=False)
    dl_macro.to_csv(out / "delta_macro_auroc_bootstrap.csv", index=False)

    figs = reliability_plots(preds, src, out, platt)

    # ---- résumé lisible
    pd.set_option("display.width", 200)
    print("\n=== 1. Seuils et prévalences (moyenne sur les pathologies) ===")
    show = ["eval_set", "scenario", "prev_true", "prev_mean_prob", "prev_em", "prev_bbse",
            "sens_raw", "sens_em", "sens_bbse", "sens_fewshot", "sens_oracle",
            "f1_raw", "f1_em", "f1_bbse", "f1_fewshot", "f1_oracle",
            "ece_raw", "ece_platt", "ece_em", "ece_bbse"]
    print(summ[[c for c in show if c in summ.columns]][summ.scenario.isin(["full", "no_text"])]
          .round(3).to_string(index=False))
    print("\n=== 2. Gating par accord : par ECG (gated) et par centre (site_gated) ===")
    print(gate_df.round(3).to_string(index=False) if len(gate_df) else "  (scénarios full/no_text absents)")
    print("\n=== 3. Δ AUROC macro (A − B), IC 95 % bootstrap apparié ===")
    print(dl_macro.round(4).to_string(index=False))
    if len(dl):
        sig = dl[dl.p_holm < 0.05]
        print(f"\n   {len(sig)}/{len(dl)} comparaisons par pathologie significatives (DeLong, Holm < 0,05)")
    print(f"\nFigures : {len(figs)} courbes de fiabilité ; fichiers -> {out}")


if __name__ == "__main__":
    main()
