"""Pourquoi STT_ISCH s'améliore-t-il avec la dérivation I seule sur PTB-XL ? (CPU, quelques minutes)

Hypothèse : sur 12 dérivations, le modèle (entraîné sur MIMIC) signale des anomalies ST-T
« secondaires » (IDM, HVG, blocs de branche) que PTB-XL n'annote pas en STTC. Avec la dérivation I seule,
ces faux positifs diminuent, ce qui augmente l'AUROC sans que le modèle « voie » mieux l'ischémie.

Trois contrôles, à partir des prédictions déjà sauvegardées (no_text vs leadI_no_text) :
  A. le gain existe-t-il aussi sur MIMIC et LSAS, ou seulement sur PTB-XL ?
  B. AUROC de STT_ISCH contre chaque groupe de négatifs (IDM, HVG, bloc de branche, autre anomalie, normal) ;
  C. AUROC par type d'annotation STT de PTB-XL (NDT, NST_, STD_, ISC*, INJ*) et par probabilité
     (likelihood) — nécessite ptbxl_database.csv (téléchargé automatiquement si absent).

  python scripts/stt_lead_check.py --runs runs/v2_amplitude runs/v2_amp_seed1 runs/v2_amp_seed2 \
      --ptbxl_cache /kaggle/working/cache/ptbxl --out stt_check/
"""
import argparse
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_preds import delong_paired  # noqa: E402
from ecgfusion.labels import LABELS  # noqa: E402

TARGET = "STT_ISCH"
A, B = "no_text", "leadI_no_text"
SITES = {"mimic_internal": "MIMIC test", "ptbxl_external": "PTB-XL", "lsas_external": "LSAS"}
PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
INK, MUTED, GRID, NEG, POS = "#1f2933", "#616e7c", "#e4e7eb", "#d9480f", "#1f6feb"
SEED_MARK = ["o", "s", "^", "D", "v"]


def load(run: Path, site: str, sc: str):
    f = run / "eval" / f"preds_{site}_{sc}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {"p": d["prob"].astype(float), "y": d["labels"].astype(int), "index": d["index"]}


def boot_delta(y, p1, p2, mask_pos, mask_neg, n_boot=300, seed=0):
    """Δ AUROC (p2 − p1) restreint à des positifs / négatifs donnés, IC 95 % par bootstrap apparié."""
    ip, ineg = np.where(mask_pos)[0], np.where(mask_neg)[0]
    if len(ip) < 10 or len(ineg) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    idx = np.concatenate([ip, ineg]); yy = y[idx]
    a1, a2 = roc_auc_score(yy, p1[idx]), roc_auc_score(yy, p2[idx])
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        ds.append(roc_auc_score(y[b], p2[b]) - roc_auc_score(y[b], p1[b]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return a1, a2, a2 - a1, lo, hi


def negative_strata(y):
    """Négatifs STT_ISCH répartis en groupes exclusifs (priorité dans cet ordre)."""
    L = {l: y[:, LABELS.index(l)] == 1 for l in LABELS}
    neg = ~L[TARGET]
    groups, taken = {}, np.zeros(len(y), bool)
    for name, m in [("MI", L["MI"]), ("LVH", L["LVH"]), ("bundle branch block", L["LBBB"] | L["RBBB"]),
                    ("other abnormality", np.any(np.stack([L[l] for l in LABELS if l not in (TARGET, "NORM")]), 0)),
                    ("normal / no label", np.ones(len(y), bool))]:
        g = neg & m & ~taken
        groups[name] = g; taken |= g
    return groups


def ptbxl_codes(cache: Path, db_path: str | None):
    """Codes SCP de chaque prédiction PTB-XL (alignés sur meta.csv du cache) ; None si indisponible."""
    meta_f = cache / "meta.csv"
    if not meta_f.exists():
        print(f"[C] {meta_f} introuvable : analyse par code SCP ignorée"); return None
    meta = pd.read_csv(meta_f, keep_default_na=False)
    src = db_path or PTBXL_URL
    try:
        db = pd.read_csv(src, usecols=["ecg_id", "scp_codes"])
    except Exception as e:  # pas d'accès internet, par exemple
        print(f"[C] ptbxl_database.csv indisponible ({e}) : analyse par code SCP ignorée"); return None
    db["record_id"] = "ptbxl_" + db["ecg_id"].astype(str)
    db["scp"] = db["scp_codes"].apply(ast.literal_eval)
    return meta[["record_id"]].merge(db[["record_id", "scp"]], on="record_id", how="left")["scp"]


def positive_strata(y, scp_by_index, index):
    pos = y[:, LABELS.index(TARGET)] == 1
    out = {}
    L = {l: y[:, LABELS.index(l)] == 1 for l in LABELS}
    secondary = L["MI"] | L["LVH"] | L["LBBB"] | L["RBBB"]
    out["STT with MI/LVH/BBB"] = pos & secondary
    out["STT isolated"] = pos & ~secondary
    if scp_by_index is None:
        return out
    scp = [scp_by_index.iloc[i] if i < len(scp_by_index) and isinstance(scp_by_index.iloc[i], dict) else {}
           for i in index]
    fams = {"NDT (non-diagnostic T)": lambda c: c == "NDT", "NST_ (non-specific ST)": lambda c: c == "NST_",
            "STD_ (ST depression)": lambda c: c == "STD_", "ISC* (ischaemia)": lambda c: c.startswith("ISC"),
            "INJ* (injury)": lambda c: c.startswith("INJ")}
    for name, f in fams.items():
        out[name] = pos & np.array([any(f(c) for c in d) for d in scp])
    lik = np.array([max([v for c, v in d.items() if any(g(c) for g in fams.values())] or [np.nan]) for d in scp])
    out["likelihood < 50"] = pos & (lik < 50)
    out["likelihood ≥ 50"] = pos & (lik >= 50)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ptbxl_cache", default=None, help="dossier du cache PTB-XL (contient meta.csv)")
    ap.add_argument("--ptbxl_db", default=None, help="chemin de ptbxl_database.csv (sinon téléchargé)")
    ap.add_argument("--out", default="stt_check")
    ap.add_argument("--n_boot", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in a.runs if (Path(r) / "eval").exists()]
    print("Runs :", [r.name for r in runs])
    c = LABELS.index(TARGET)

    # A. par site
    rows = []
    for run in runs:
        for site, nice in SITES.items():
            p1, p2 = load(run, site, A), load(run, site, B)
            if p1 is None or p2 is None or p1["y"][:, c].sum() < 10:
                continue
            y = p1["y"][:, c]
            a1, a2, pv = delong_paired(y, p1["p"][:, c], p2["p"][:, c])
            rows.append({"run": run.name, "site": nice, "n_pos": int(y.sum()), "auroc_12": a1, "auroc_leadI": a2,
                         "delta": a2 - a1, "p_delong": pv})
    A_df = pd.DataFrame(rows); A_df.to_csv(out / "A_by_site.csv", index=False)
    print("\n=== A. Δ AUROC STT_ISCH (dérivation I − 12 dérivations), par site ===")
    print(A_df.round(4).to_string(index=False) if len(A_df) else "aucune prédiction leadI_no_text trouvée")

    # B et C. PTB-XL
    scp = ptbxl_codes(Path(a.ptbxl_cache), a.ptbxl_db) if a.ptbxl_cache else None
    if scp is None and a.ptbxl_cache is None:
        print("[C] --ptbxl_cache non fourni : analyse par code SCP ignorée")
    Brows, Crows = [], []
    for run in runs:
        p1, p2 = load(run, "ptbxl_external", A), load(run, "ptbxl_external", B)
        if p1 is None or p2 is None:
            continue
        y, q1, q2 = p1["y"], p1["p"][:, c], p2["p"][:, c]
        yt, pos = y[:, c], y[:, c] == 1
        for name, g in negative_strata(y).items():
            a1, a2, d, lo, hi = boot_delta(yt, q1, q2, pos, g, a.n_boot)
            Brows.append({"run": run.name, "negatives": name, "n_neg": int(g.sum()), "auroc_12": a1,
                          "auroc_leadI": a2, "delta": d, "ci_lo": lo, "ci_hi": hi,
                          "mean_p_12": q1[g].mean(), "mean_p_leadI": q2[g].mean()})
        for name, g in positive_strata(y, scp, p1["index"]).items():
            a1, a2, d, lo, hi = boot_delta(yt, q1, q2, g, ~pos, a.n_boot)
            Crows.append({"run": run.name, "positives": name, "n_pos": int(g.sum()), "auroc_12": a1,
                          "auroc_leadI": a2, "delta": d, "ci_lo": lo, "ci_hi": hi})
    B_df, C_df = pd.DataFrame(Brows), pd.DataFrame(Crows)
    B_df.to_csv(out / "B_ptbxl_by_negative_group.csv", index=False)
    C_df.to_csv(out / "C_ptbxl_by_positive_group.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== B. PTB-XL : tous les positifs STT contre chaque groupe de négatifs ===")
    print(B_df.round(3).to_string(index=False) if len(B_df) else "prédictions PTB-XL absentes")
    print("\n=== C. PTB-XL : chaque groupe de positifs STT contre tous les négatifs ===")
    print(C_df.round(3).to_string(index=False) if len(C_df) else "prédictions PTB-XL absentes")
    if B_df.empty:
        return

    # figure supplémentaire
    def panel(ax, df, key, title):
        g = df.groupby(key, sort=False)
        names = list(g.groups)[::-1]
        for yi, nm in enumerate(names):
            d = g.get_group(nm)
            first = d.iloc[0]
            if not np.isnan(first["ci_lo"]):
                ax.plot([first["ci_lo"], first["ci_hi"]], [yi, yi], color=MUTED, lw=2.2, alpha=0.5)
            for k, (_, r) in enumerate(d.iterrows()):
                ax.plot(r["delta"], yi, SEED_MARK[k % 5], color=POS if r["delta"] >= 0 else NEG, ms=5)
        n_col = "n_neg" if "n_neg" in df else "n_pos"
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{nm}  (n={int(g.get_group(nm)[n_col].iloc[0])})" for nm in names], fontsize=7.5)
        ax.margins(x=0.06, y=0.6 / max(len(names), 1))
        ax.axvline(0, color=INK, lw=0.8); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("Δ AUROC STT_ISCH (lead I − 12 leads)", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5)

    sizes = ([A_df.site.nunique()] if len(A_df) else []) + [B_df.negatives.nunique(), C_df.positives.nunique()]
    fig, axes = plt.subplots(len(sizes), 1, figsize=(7.2, 0.9 * len(sizes) + 0.28 * sum(sizes)),
                             gridspec_kw={"height_ratios": [n + 1.5 for n in sizes]})
    k = 0
    if len(A_df):
        A_tmp = A_df.assign(ci_lo=np.nan, ci_hi=np.nan, n_neg=A_df.n_pos)
        panel(axes[0], A_tmp.rename(columns={"site": "grp"}), "grp", "a  By site (all STT positives vs all negatives)")
        k = 1
    panel(axes[k], B_df.rename(columns={"negatives": "grp"}), "grp",
          f"{'abc'[k]}  PTB-XL: all STT positives vs one group of negatives")
    panel(axes[k + 1], C_df.rename(columns={"positives": "grp"}), "grp",
          f"{'abc'[k + 1]}  PTB-XL: one group of STT positives vs all negatives")
    fig.text(0.01, 0.005, "Markers: one per seed. Grey bar: 95 % bootstrap CI (first seed).", fontsize=7,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS5_stt_leadI_check.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    # lecture automatique (à confirmer en regardant les tableaux)
    m = B_df.groupby("negatives")["delta"].mean()
    clean = m.get("normal / no label", np.nan)
    sec = B_df[B_df.negatives.isin(["MI", "LVH", "bundle branch block"])].groupby("negatives")["delta"].mean()
    print("\n=== Lecture ===")
    print(f"Δ moyen contre négatifs normaux seulement : {clean:+.3f}")
    print("Δ moyen contre négatifs IDM / HVG / bloc :", {k: round(v, 3) for k, v in sec.items()})
    if not np.isnan(clean) and len(sec) and sec.max() > clean + 0.02:
        print("→ Le gain vient surtout des négatifs avec IDM/HVG/bloc : le modèle 12 dérivations y signale des "
              "anomalies ST-T secondaires que PTB-XL n'annote pas en STTC. C'est un effet de convention "
              "d'annotation, pas une meilleure détection de l'ischémie avec une seule dérivation.")
    elif not np.isnan(clean) and clean > 0.01:
        print("→ Le gain persiste contre les négatifs normaux : regarder le tableau C (types d'annotation).")
    else:
        print("→ Pas de gain net une fois les négatifs « secondaires » retirés.")
    print("Figure :", out / "figS5_stt_leadI_check.pdf")


if __name__ == "__main__":
    main()
