"""Figures 2 et 3 de l'article à partir d'un run évalué et analysé (CPU, quelques secondes).

  Figure 2 : Δ AUROC par pathologie sur PTB-XL — effet du texte et effet de la dérivation I seule
             (DeLong apparié, correction de Holm).
  Figure 4 : calibration PTB-XL (S4 : LSAS), brut / Platt MIMIC / Platt 200 ECG locaux.
  Figure 3 : courbe de survie du désaccord d (avec / sans texte) sur la validation MIMIC, le test MIMIC
             et PTB-XL, avec les seuils τ90 et τ95 appris sur la validation MIMIC.

  python scripts/make_figures.py --run runs/v2_amplitude --out figures/
  (le dossier d'analyse est cherché dans <run>/analysis, sinon passer --analysis)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_preds import (LABELS, _calibration_variants, apply_platt, disagreement, fit_platt,  # noqa: E402
                           label_mask, load_preds, paper_calibration_figure)

INK, MUTED, GRID = "#1f2933", "#616e7c", "#e4e7eb"
NEG, POS = "#d9480f", "#1f6feb"
SITE_COL = {"mimic_val": "#868e96", "mimic_internal": "#1f6feb", "ptbxl_external": "#d9480f"}
SITE_NAME = {"mimic_val": "MIMIC validation (reference)", "mimic_internal": "MIMIC test",
             "ptbxl_external": "PTB-XL (external)"}
ORDER = ["NORM", "AFIB", "AVB", "LVH", "STT_ISCH", "MI", "RBBB", "LBBB", "FASC", "STACH", "SBRAD"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def fig_per_label(delong: pd.DataFrame, out: Path):
    """Deux panneaux, mêmes pathologies en ordonnée, barres signées autour de zéro."""
    text = delong[(delong.eval_set == "ptbxl_external") & (delong.A == "full") & (delong.B == "no_text")]
    lead = delong[(delong.eval_set == "ptbxl_external") & (delong.A == "no_text") & (delong.B == "leadI_no_text")]
    if text.empty or lead.empty:
        print("[fig2] comparaisons absentes, figure ignorée"); return
    text = text.set_index("label"); lead = lead.set_index("label")
    labs = [l for l in ORDER if l in text.index and l in lead.index]
    panels = [("Effect of report text\n(all modalities − no text)", text.loc[labs, "delta"], text.loc[labs, "p_holm"]),
              ("Effect of lead I alone\n(lead I − 12 leads, no text)", -lead.loc[labs, "delta"], lead.loc[labs, "p_holm"])]
    lim = max(0.02, max(float(np.abs(v).max()) for _, v, _ in panels) * 1.18)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    y = np.arange(len(labs))[::-1]
    for ax, (title, vals, p) in zip(axes, panels):
        cols = [NEG if v < 0 else POS for v in vals]
        vals = vals.where(vals.abs() >= 5e-4, 0.0)  # évite « -0.000 »
        cols = [NEG if v < 0 else POS for v in vals]
        for yi, v, q, c in zip(y, vals.values, p.values, cols):
            ax.barh(yi, v, color=c, height=0.62, alpha=1.0 if q < 0.05 else 0.3)
            ax.text(v + (0.004 if v >= 0 else -0.004), yi, (f"{v:+.3f}" if v != 0 else "0.000") + ("" if q < 0.05 else " (n.s.)"),
                    va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-lim * 1.25, lim * 0.75)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Δ AUROC on PTB-XL", fontsize=8.5, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        style(ax)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labs, fontsize=8.5)
    fig.text(0.01, 0.01, "Faded bars: not significant (paired DeLong test, Holm-corrected p ≥ 0.05).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_per_label_effects.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("[fig2] écrit :", out / "fig2_per_label_effects.pdf")


def fig_disagreement(eval_dir: Path, out: Path):
    d = {}
    for s in ("mimic_val", "mimic_internal", "ptbxl_external"):
        f, n = eval_dir / f"preds_{s}_full.npz", eval_dir / f"preds_{s}_no_text.npz"
        if f.exists() and n.exists():
            a, b = np.load(f), np.load(n)
            assert np.array_equal(a["index"], b["index"])
            d[s] = disagreement(a["prob"].astype(float), b["prob"].astype(float))
    if "mimic_val" not in d or "ptbxl_external" not in d:
        print("[fig3] prédictions absentes, figure ignorée"); return
    t90, t95 = np.percentile(d["mimic_val"], [90, 95])
    # courbe de survie P(d > x) : la zone où agit le test (queue droite) devient lisible
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    lo = max(min(np.percentile(v, 1) for v in d.values()), 1e-5)
    hi = max(v.max() for v in d.values())
    xs = np.logspace(np.log10(lo), np.log10(hi), 400)
    for s, v in d.items():
        sv = np.sort(v)
        surv = 1.0 - np.searchsorted(sv, xs, side="right") / len(sv)
        ax.plot(xs, np.maximum(surv, 0.5 / len(sv)), lw=1.8, color=SITE_COL[s],
                ls="--" if s == "mimic_val" else "-",
                label=f"{SITE_NAME[s]}: {100 * (v > t90).mean():.1f} % > τ90, {100 * (v > t95).mean():.1f} % > τ95")
    for t, name, exp, ha in ((t90, "τ90 ", 0.10, "right"), (t95, " τ95", 0.05, "left")):
        ax.axvline(t, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.text(t, 1.02, name, transform=ax.get_xaxis_transform(), fontsize=8, color=INK, ha=ha, va="bottom")
        ax.plot([t], [exp], marker="o", ms=5, mfc="white", mec=INK, zorder=5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(max(0.5 / max(len(v) for v in d.values()), 1e-3), 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{100 * y:g} %"))
    ax.set_xlabel("Disagreement d between predictions with and without text (mean symmetric KL, log scale)",
                  fontsize=8.5, color=INK)
    ax.set_ylabel("Share of ECGs with disagreement > d", fontsize=8.5, color=INK)
    ax.grid(color=GRID, lw=0.6, which="major"); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK,
                    label="Expected in-domain share above τ90 / τ95 (10 % / 5 %)"))
    ax.legend(handles=h, fontsize=7.5, frameon=False, loc="lower left")
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_site_gate_disagreement.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("[fig3] écrit :", out / "fig3_site_gate_disagreement.pdf",
          "| au-dessus de τ95 :", {k: round(float((v > t95).mean()), 3) for k, v in d.items()})


def fig_calibration(eval_dir: Path, out: Path, scenario="no_text"):
    """Figure 4 (PTB-XL) et S4 (LSAS) : courbes de fiabilité brutes / Platt appris sur la validation
    MIMIC / Platt appris sur 200 ECG locaux (évalué sur les ECG restants)."""
    preds = load_preds(eval_dir)
    ref = ("mimic_val", scenario) if ("mimic_val", scenario) in preds else ("mimic_internal", scenario)
    if ref not in preds:
        print(f"[fig4] prédictions {scenario} de référence absentes, figure ignorée"); return
    ab = fit_platt(preds[ref]["y"], preds[ref]["prob"])
    names = {"ptbxl_external": ("fig4_calibration_ptbxl", "PTB-XL"),
             "lsas_external": ("figS4_calibration_lsas", "LSAS-ECG")}
    for s, (stem, nice) in names.items():
        if (s, scenario) not in preds:
            print(f"[fig4] {s}/{scenario} absent"); continue
        v = preds[(s, scenario)]
        variants = _calibration_variants(v["prob"], v["y"], ab)
        what = "ECG signal + derived features, no text" if scenario == "no_text" else scenario
        paper_calibration_figure(v["y"], variants, label_mask(s),
                                 f"Calibration at external validation — {nice} ({what})", out / stem)
        print("[fig4] écrit :", out / f"{stem}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ana = Path(a.analysis) if a.analysis else run / "analysis"
    if (ana / "delong_per_label.csv").exists():
        fig_per_label(pd.read_csv(ana / "delong_per_label.csv"), out)
    else:
        print(f"[fig2] {ana / 'delong_per_label.csv'} introuvable")
    fig_disagreement(run / "eval", out)
    fig_calibration(run / "eval", out)


if __name__ == "__main__":
    main()
