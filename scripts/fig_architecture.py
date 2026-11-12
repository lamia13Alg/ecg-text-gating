"""Figure 1 : architecture du modèle et gating par centre (PDF vectoriel + PNG).

  python scripts/fig_architecture.py --out figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
MUTED = "#616e7c"
SIG, TXT, FEA = "#1f6feb", "#d9480f", "#2f9e44"   # signal, texte, mesures
FUS = "#5f3dc4"
LIGHT = {SIG: "#e7f1fb", TXT: "#fbeee6", FEA: "#e6f4e8", FUS: "#efedff", INK: "#f1f3f5"}


def box(ax, x, y, w, h, title, sub=None, color=INK, fill=None, lw=1.2, fs=8.5, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                       linewidth=lw, edgecolor=color, facecolor=fill or LIGHT.get(color, "white"),
                       linestyle=(0, (4, 3)) if dashed else "solid")
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h - 0.024, title, ha="center", va="top", fontsize=fs, color=INK, weight="bold")
        ax.text(x + w / 2, y + h - 0.066, sub, ha="center", va="top", fontsize=fs - 1.6, color=MUTED,
                linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=INK, weight="bold")


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.1, ls="solid", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw,
                                 color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(-0.012, 1.0); ax.set_ylim(0, 1.045); ax.axis("off")

    # ---- (a) modèle -------------------------------------------------------------
    ax.text(-0.005, 1.04, "a", fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.02, 1.038, "Fusion model (training and inference)", fontsize=9.5, color=INK, va="top")

    rows = [  # y, couleur, entrée, encodeur, jetons
        (0.80, SIG, ("12-lead ECG", "10 s, 250 Hz, mV\n+ lead-availability mask"),
         ("Signal encoder", "1D ResNet (4 stages)\n+ 4-layer Transformer"), "~78 time tokens"),
        (0.585, TXT, ("Report text", "diagnoses and verdicts\nremoved (strict masking)"),
         ("Text encoder", "Bio_ClinicalBERT\n(last 2 layers tuned)"), "word tokens"),
        (0.37, FEA, ("24 ECG measurements", "intervals, axis, voltages,\nSokolow-Lyon, Cornell"),
         ("Measurement encoder", "feature tokenizer\n+ 2-layer Transformer"), "24 tokens"),
    ]
    h = 0.15
    for y, c, inp, enc, tok in rows:
        box(ax, 0.00, y, 0.17, h, inp[0], inp[1], color=c)
        arrow(ax, 0.17, y + h / 2, 0.205, y + h / 2, color=c)
        box(ax, 0.205, y, 0.17, h, enc[0], enc[1], color=c)
        arrow(ax, 0.375, y + h / 2, 0.425, y + h / 2, color=c)
        # pile de jetons + jeton ABSENT
        for k in range(4):
            ax.add_patch(FancyBboxPatch((0.43 + k * 0.016, y + 0.035), 0.011, 0.08,
                                        boxstyle="round,pad=0.002", fc=c, ec="none", alpha=0.75))
        ax.add_patch(FancyBboxPatch((0.43 + 4 * 0.016, y + 0.035), 0.011, 0.08,
                                    boxstyle="round,pad=0.002", fc="white", ec=c, lw=1.0, ls=(0, (2, 1.5))))
        ax.text(0.468, y + 0.012, tok, ha="center", va="center", fontsize=6.6, color=MUTED)

    ax.text(0.468, 0.335, "dashed token = learned [ABSENT]\n(used when the modality is missing)",
            ha="center", va="top", fontsize=6.6, color=MUTED, style="italic")

    # modality dropout (entraînement) : trait pointillé entre entrées et encodeurs, légende en bas
    ax.plot([0.1875, 0.1875], [0.37, 0.95], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.1875, 0.335, "training only — modality dropout:\ntext 0.5 · measurements 0.2 · lead subsets 0.3",
            ha="center", va="top", fontsize=6.6, color=MUTED, style="italic")

    # fusion
    fx, fy, fw, fh = 0.54, 0.37, 0.17, 0.58
    box(ax, fx, fy, fw, fh, "", color=FUS)
    ax.text(fx + fw / 2, fy + fh - 0.045, "Label-query fusion", ha="center", fontsize=8.5, weight="bold", color=INK)
    ax.text(fx + fw / 2, fy + fh - 0.09, "11 learned queries (one per label)\ncross-attend to all tokens\n"
            "+ type embeddings · 2 layers", ha="center", va="top", fontsize=6.9, color=MUTED, linespacing=1.3)
    for k in range(11):
        ax.add_patch(FancyBboxPatch((fx + 0.017 + k * 0.0125, fy + 0.07), 0.009, 0.16,
                                    boxstyle="round,pad=0.002", fc=FUS, ec="none", alpha=0.7))
    ax.text(fx + fw / 2, fy + 0.035, "label queries", ha="center", fontsize=6.6, color=MUTED)
    for y, c, *_ in rows:
        arrow(ax, 0.51, y + h / 2, fx, min(max(y + h / 2, fy + 0.05), fy + fh - 0.05), color=c)

    # tête + sorties
    box(ax, 0.745, 0.60, 0.11, 0.12, "Shared head", "1 logit per label", color=INK)
    arrow(ax, fx + fw, 0.66, 0.745, 0.66)
    box(ax, 0.885, 0.53, 0.105, 0.26, "11 labels", "NORM · AFIB\nAVB · LVH\nSTT_ISCH · MI\nRBBB · LBBB\nFASC · STACH\nSBRAD",
        color=INK, fs=8)
    arrow(ax, 0.855, 0.66, 0.885, 0.66)
    box(ax, 0.745, 0.37, 0.245, 0.11, "Auxiliary heads (training)",
        "one linear head per encoder on its pooled tokens,\nloss weight 0.3",
        color=MUTED, fill="white", dashed=True, fs=7.8)

    # ---- (b) gating par centre ---------------------------------------------------
    ax.plot([0, 1], [0.255, 0.255], color="#d9dde1", lw=0.8)
    ax.text(-0.005, 0.235, "b", fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.025, 0.233, "Label-free site-level text gate (deployment)", fontsize=9.5, color=INK, va="top")
    gy, gh = 0.02, 0.15
    box(ax, 0.00, gy, 0.16, gh, "New site, unlabelled", "ECGs with reports", color=INK)
    arrow(ax, 0.16, gy + gh / 2, 0.19, gy + gh / 2)
    box(ax, 0.19, gy, 0.19, gh, "Predict twice", "with text  p_full\nwithout text  p_no-text", color=FUS)
    arrow(ax, 0.38, gy + gh / 2, 0.41, gy + gh / 2)
    box(ax, 0.41, gy, 0.20, gh, "Disagreement per ECG", "d = mean symmetric KL\n(p_full, p_no-text) over labels",
        color=FUS)
    arrow(ax, 0.61, gy + gh / 2, 0.64, gy + gh / 2)
    box(ax, 0.64, gy, 0.19, gh, "Binomial test", "share of ECGs with d > τ vs 1 − q\nτ: q-th percentile on MIMIC val.",
        color=FUS)
    arrow(ax, 0.83, gy + gh * 0.72, 0.86, gy + gh * 0.86, color=SIG)
    arrow(ax, 0.83, gy + gh * 0.28, 0.86, gy + gh * 0.14, color=TXT)
    ax.text(0.865, gy + gh * 0.86, "no excess → keep text", va="center", fontsize=7.2, color=SIG)
    ax.text(0.865, gy + gh * 0.14, "excess (p < 0.01, > 1.5×)\n→ disable text for the site",
            va="center", fontsize=7.2, color=TXT)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_architecture.{ext}", dpi=300 if ext == "png" else None)
    print("écrit :", out / "fig1_architecture.pdf")


if __name__ == "__main__":
    main()
