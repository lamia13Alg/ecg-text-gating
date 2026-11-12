"""Audit des labels et des caractéristiques d'un cache, et ré-étiquetage MIMIC.

  # voir les phrases qui déclenchent chaque label (MIMIC)
  python scripts/audit_labels.py --cache cache/mimic

  # recalculer les labels MIMIC à partir du texte stocké (aucun signal relu)
  python scripts/audit_labels.py --cache /kaggle/input/ecg-fusion-cache/cache/mimic \
         --relabel --out /kaggle/working/cache/mimic
"""
import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.labels import LABELS, mimic_matches, mimic_text_to_labels  # noqa: E402
from ecgfusion.signal_processing import FEATURE_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--out", default=None, help="cache de sortie (copie) ; défaut : modifie --cache")
    ap.add_argument("--link", action="store_true",
                    help="avec --out : liens symboliques vers les gros fichiers au lieu de copies")
    ap.add_argument("--quiet", action="store_true", help="n'affiche que les prévalences")
    a = ap.parse_args()
    src = Path(a.cache)
    meta = pd.read_csv(src / "meta.csv", keep_default_na=False)
    valid = meta["valid"].astype(str).str.lower().eq("true").to_numpy()
    y = np.load(src / "labels.npy")
    feats = np.load(src / "features.npy")
    is_mimic = str(meta["dataset"].iloc[0]) == "mimic"

    print(f"== {src}  ({valid.sum()} ECG valides)")
    print("\nPrévalences actuelles :")
    print("  " + "  ".join(f"{l}={v:.3f}" for l, v in zip(LABELS, y[valid].mean(0))))

    if is_mimic:
        y_new = np.stack([mimic_text_to_labels(t) for t in meta["text"].astype(str)])
        print("\nPrévalences avec les règles actuelles du code :")
        print("  " + "  ".join(f"{l}={v:.3f}" for l, v in zip(LABELS, y_new[valid].mean(0))))
        counts = {l: collections.Counter() for l in LABELS}
        for t in meta["text"].astype(str)[valid]:
            for lab, s in mimic_matches(t):
                counts[lab][s] += 1
        if a.quiet:
            counts = {}
        else:
            print(f"\nPhrases les plus fréquentes par label (top {a.top}) :")
        for l in (LABELS if counts else []):
            print(f"  [{l}]")
            for s, n in counts[l].most_common(a.top):
                print(f"     {n:6d}  {s}")

        if a.relabel:
            dst = Path(a.out) if a.out else src
            if dst != src:
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    if f.name in ("labels.npy", "info.json"):
                        if f.name == "info.json":
                            shutil.copy(f, dst / f.name)
                        continue
                    target = dst / f.name
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    if a.link:
                        target.symlink_to(f.resolve())
                    else:
                        shutil.copy(f, target)
            np.save(dst / "labels.npy", y_new.astype(np.float32))
            info = json.loads((dst / "info.json").read_text())
            info["label_prevalence"] = dict(zip(LABELS, y_new[valid].mean(0).round(4).tolist()))
            (dst / "info.json").write_text(json.dumps(info, indent=2))
            print(f"\n[relabel] labels.npy réécrit -> {dst}")

    if a.quiet:
        return
    nan = np.isnan(feats[valid]).mean(0)
    print("\nCaractéristiques non mesurables (% NaN) :")
    print("  " + "  ".join(f"{n}={100 * v:.0f}%" for n, v in zip(FEATURE_NAMES, nan)))
    med = np.nanmedian(feats[valid], 0)
    print("Médianes :", "  ".join(f"{n}={v:.1f}" for n, v in zip(FEATURE_NAMES[:10], med[:10])))


if __name__ == "__main__":
    main()
