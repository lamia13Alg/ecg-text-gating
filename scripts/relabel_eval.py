"""Replace the labels of one evaluation set in an existing run, from saved predictions (no GPU).

Predictions do not depend on labels: preds_<set>_<scenario>.npz keep the probabilities and the
cache row index of every ECG, so new labels can be swapped in and all metrics recomputed.
Used when a run was evaluated before the LSAS cache was relabelled (labels v2).

  python scripts/relabel_eval.py --run runs/L2_final_s42 --set lsas_external --cache /tmp/cache_labels/lsas
Then rerun scripts/analyze_preds.py --run <run> (the analysis folder is deleted here).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion import metrics as M  # noqa: E402
from ecgfusion.data.dataset import SCENARIOS  # noqa: E402
from ecgfusion.labels import LABEL_AVAILABILITY, LABELS  # noqa: E402


def pick_calibration(calib, scenario):
    """Same rule as scripts/evaluate.py."""
    if scenario.name in calib:
        return calib[scenario.name]
    if not scenario.use_text and not scenario.use_features and "signal_only" in calib:
        return calib["signal_only"]
    if not scenario.use_text and "no_text" in calib:
        return calib["no_text"]
    return calib.get("full", next(iter(calib.values())))

MARK = "labels_v2_{}.done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", required=True, help="ex. lsas_external")
    ap.add_argument("--cache", required=True, help="cache relabelled (labels.npy, info.json)")
    a = ap.parse_args()
    run, ev = Path(a.run), Path(a.run) / "eval"
    if (ev / MARK.format(a.set)).exists():
        print(f"[relabel_eval] {run.name}/{a.set} : déjà fait")
        return
    cfg = yaml.safe_load(open(run / "config.yaml"))
    new_labels = np.load(Path(a.cache) / "labels.npy")
    ds_name = str(pd.read_csv(Path(a.cache) / "meta.csv", usecols=["dataset"], nrows=1)["dataset"].iloc[0])
    label_mask = np.array(LABEL_AVAILABILITY.get(ds_name, [1] * len(LABELS)), dtype=np.float32)
    calib = json.load(open(run / "calibration.json"))

    pl = pd.read_csv(ev / "per_label.csv")
    sm = pd.read_csv(ev / "summary.csv")
    pl_new, sm_new = [], []
    files = sorted(ev.glob(f"preds_{a.set}_*.npz"))
    assert files, f"aucune prédiction preds_{a.set}_*.npz dans {ev}"
    for f in files:
        sname = f.stem[len(f"preds_{a.set}_"):]
        z = dict(np.load(f))
        y = new_labels[z["index"]].astype(np.float32)
        z["labels"] = y
        np.savez_compressed(f, **z)
        # same thresholds as at evaluation time (calibration of the matching scenario)
        th = np.array(pick_calibration(calib, SCENARIOS[sname])["threshold"])
        df = M.per_label_metrics(y, z["prob"], th, LABELS, label_mask, cfg["eval"]["min_pos"])
        df.insert(0, "scenario", sname)
        df.insert(0, "eval_set", a.set)
        pl_new.append(df)
        s = M.macro_summary(df)
        lo, hi = M.bootstrap_macro_auroc(y, z["prob"], label_mask, cfg["eval"]["n_bootstrap"], cfg["eval"]["min_pos"])
        old = sm[(sm.eval_set == a.set) & (sm.scenario == sname)]
        row = {"eval_set": a.set, "scenario": sname, "n": len(y),
               "text_available": float(old["text_available"].iloc[0]) if len(old) else 0.0,
               **s, "macro_auroc_ci_low": lo, "macro_auroc_ci_high": hi}
        sm_new.append(row)
        print(f"[{a.set:>15s} | {sname:>15s}] n={len(y):6d}  AUROC={s['macro_auroc']:.3f} "
              f"[{lo:.3f}-{hi:.3f}]  F1={s['macro_f1']:.3f}  ECE={s['macro_ece']:.3f}")

    pl = pd.concat([pl[pl.eval_set != a.set]] + pl_new, ignore_index=True)
    sm = pd.concat([sm[sm.eval_set != a.set], pd.DataFrame(sm_new)], ignore_index=True)
    pl.to_csv(ev / "per_label.csv", index=False)
    sm.to_csv(ev / "summary.csv", index=False)

    piv = pl.pivot_table(index=["scenario", "label"], columns="eval_set", values="auroc")
    internal = next((k for k in cfg["data"]["eval_sets"] if "internal" in k), None)
    if internal in piv.columns:
        for col in [c for c in piv.columns if c != internal]:
            piv[f"delta_{col}"] = piv[col] - piv[internal]
    piv.reset_index().to_csv(ev / "degradation.csv", index=False)
    full = pl[pl.scenario == "full"].set_index(["eval_set", "label"])["auroc"]
    contrib = {}
    for mod, sc in [("text", "no_text"), ("features", "no_features"), ("signal", "no_signal")]:
        if sc in pl.scenario.values:
            contrib[f"delta_auroc_{mod}"] = full - pl[pl.scenario == sc].set_index(["eval_set", "label"])["auroc"]
    pd.DataFrame(contrib).reset_index().to_csv(ev / "modality_contribution.csv", index=False)

    shutil.rmtree(run / "analysis", ignore_errors=True)  # to be recomputed with the new labels
    (ev / MARK.format(a.set)).write_text("ok\n")
    print(f"[relabel_eval] {run.name}/{a.set} : labels remplacés, métriques recalculées")


if __name__ == "__main__":
    main()
