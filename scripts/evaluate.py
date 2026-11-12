"""Évaluation interne + externe, par scénario de modalités disponibles.

  python scripts/evaluate.py --run runs/fusion_full

Sorties dans <run>/eval/ :
  per_label.csv              métriques par jeu × scénario × pathologie
  summary.csv                moyennes macro (+ IC bootstrap 95 % de l'AUROC macro)
  degradation.csv            AUROC par pathologie : interne vs externe (Δ)
  modality_contribution.csv  Δ AUROC quand on retire chaque modalité, par pathologie
  modality_attention.csv     part d'attention des requêtes-label sur chaque modalité
  preds_<jeu>_<scénario>.npz probabilités calibrées (tests statistiques ultérieurs)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion import metrics as M  # noqa: E402
from ecgfusion.data.dataset import SCENARIOS, ECGMultimodalDataset, FeatureNormalizer  # noqa: E402
from ecgfusion.engine import apply_overrides, build_model, build_tokenizer, make_loader, predict  # noqa: E402
from ecgfusion.labels import LABELS  # noqa: E402


def pick_calibration(calib, scenario):
    """Calibration apprise sur le scénario de validation le plus proche."""
    if scenario.name in calib:
        return calib[scenario.name]
    if not scenario.use_text and not scenario.use_features and "signal_only" in calib:
        return calib["signal_only"]
    if not scenario.use_text and "no_text" in calib:
        return calib["no_text"]
    return calib.get("full", next(iter(calib.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="surcharges, ex. data.eval_sets.lsas_external.cache=...")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : <run>/eval)")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = apply_overrides(yaml.safe_load(open(run / "config.yaml")), args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) if args.out else run / "eval"
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    tok = build_tokenizer(cfg)
    feat_norm = FeatureNormalizer.load(run / "feature_norm.json")
    calib = json.load(open(run / "calibration.json"))
    scen_names = args.scenarios or cfg["eval"]["scenarios"]
    if "no_signal" not in scen_names:
        scen_names = list(scen_names) + ["no_signal"]

    per_label, summary, attn_rows = [], [], []
    for set_name, spec in cfg["data"]["eval_sets"].items():
        if not (Path(spec["cache"]) / "meta.csv").exists():
            print(f"[skip] {set_name} : cache absent ({spec['cache']})")
            continue
        for sname in scen_names:
            sc = SCENARIOS[sname]
            ds = ECGMultimodalDataset(spec["cache"], spec["split"], feat_norm,
                                      text_policy=cfg["data"]["text_policy"], scenario=sc,
                                      signal_norm=cfg["data"].get("signal_norm", "lead"))
            if len(ds) == 0:
                continue
            r = predict(model, make_loader(ds, tok, cfg, shuffle=False), device, return_attention=True)
            cal = pick_calibration(calib, sc)
            prob = M.sigmoid(r["logits"] / np.array(cal["temperature"]))
            th = np.array(cal["threshold"])
            np.savez_compressed(out / f"preds_{set_name}_{sname}.npz", prob=prob,
                                labels=r["labels"], index=r["index"])

            df = M.per_label_metrics(r["labels"], prob, th, LABELS, ds.label_mask, cfg["eval"]["min_pos"])
            df.insert(0, "scenario", sname)
            df.insert(0, "eval_set", set_name)
            per_label.append(df)

            s = M.macro_summary(df)
            lo, hi = M.bootstrap_macro_auroc(r["labels"], prob, ds.label_mask,
                                             cfg["eval"]["n_bootstrap"], cfg["eval"]["min_pos"])
            text_rate = float(np.mean([len(t.strip()) > 0 for t in ds.texts[ds.idx]])) if sc.use_text else 0.0
            summary.append({"eval_set": set_name, "scenario": sname, "n": len(ds),
                            "text_available": round(text_rate, 3), **s,
                            "macro_auroc_ci_low": lo, "macro_auroc_ci_high": hi})

            share = r["modality_attention"].mean(0)  # (C, 3)
            for c, lab in enumerate(LABELS):
                attn_rows.append({"eval_set": set_name, "scenario": sname, "label": lab,
                                  "attn_signal": share[c, 0], "attn_text": share[c, 1],
                                  "attn_features": share[c, 2]})
            print(f"[{set_name:>15s} | {sname:>15s}] n={len(ds):6d}  "
                  f"AUROC={s['macro_auroc']:.3f} [{lo:.3f}-{hi:.3f}]  F1={s['macro_f1']:.3f}  "
                  f"ECE={s['macro_ece']:.3f}")

    pl = pd.concat(per_label)
    pl.to_csv(out / "per_label.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(attn_rows).to_csv(out / "modality_attention.csv", index=False)

    # dégradation interne -> externe, à scénario identique
    piv = pl.pivot_table(index=["scenario", "label"], columns="eval_set", values="auroc")
    internal = next((k for k in cfg["data"]["eval_sets"] if "internal" in k), None)
    if internal in piv.columns:
        for col in [c for c in piv.columns if c != internal]:
            piv[f"delta_{col}"] = piv[col] - piv[internal]
    piv.reset_index().to_csv(out / "degradation.csv", index=False)

    # contribution de chaque modalité = AUROC(full) - AUROC(sans la modalité)
    full = pl[pl.scenario == "full"].set_index(["eval_set", "label"])["auroc"]
    contrib = {}
    for mod, sc in [("text", "no_text"), ("features", "no_features"), ("signal", "no_signal")]:
        if sc in pl.scenario.values:
            contrib[f"delta_auroc_{mod}"] = full - pl[pl.scenario == sc].set_index(["eval_set", "label"])["auroc"]
    pd.DataFrame(contrib).reset_index().to_csv(out / "modality_contribution.csv", index=False)
    print(f"[done] résultats -> {out}")


if __name__ == "__main__":
    main()
