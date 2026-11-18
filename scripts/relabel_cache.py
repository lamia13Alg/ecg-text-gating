"""Recompute the labels of an existing cache with the current rules (labels v2, AHA/ACCF/HRS).

Signals and features are NOT recomputed: the new cache directory links to the original files
and only rewrites labels.npy (and the labels column of manifest.csv). Useful on Kaggle, where
the original cache is a read-only input dataset.

  python scripts/relabel_cache.py --dataset mimic --cache /kaggle/input/.../mimic --out cache_v2/mimic
  python scripts/relabel_cache.py --dataset lsas  --cache .../lsas --out cache_v2/lsas --codes lsas_codes.csv
  python scripts/relabel_cache.py --dataset ptbxl --cache .../ptbxl --out cache_v2/ptbxl   # mapping unchanged

LSAS: --codes is the lsas_codes.csv written by notebook 8 (columns stem, codes). Without it the
headers are read again from PhysioNet (slower).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion import __version__  # noqa: E402
from ecgfusion import labels as L  # noqa: E402


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic", "ptbxl", "lsas"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codes", default=None, help="LSAS: lsas_codes.csv (stem, codes)")
    a = ap.parse_args()
    src, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(src / "meta.csv", keep_default_na=False)
    old = np.load(src / "labels.npy")
    assert len(meta) == len(old), "meta.csv and labels.npy differ in length"

    if a.dataset == "mimic":
        new = np.stack([L.mimic_text_to_labels(t) for t in meta["text"].astype(str)])
    elif a.dataset == "lsas":
        if a.codes:
            codes = pd.read_csv(a.codes, dtype=str, keep_default_na=False)
        else:
            from lsas_code_check import fetch_codes  # noqa: E402
            codes = fetch_codes()
        cmap = dict(zip(codes["stem"], codes["codes"]))
        stems = meta["record_id"].str.replace("lsas_", "", regex=False)
        missing = [s for s in stems if s not in cmap or not cmap[s]]
        new = np.stack([L.lsas_codes_to_labels((cmap.get(s) or "").split(",")) if s in cmap and cmap[s]
                        else old[i] for i, s in enumerate(stems)])
        if missing:
            print(f"[warn] {len(missing)} LSAS records without readable codes: previous labels kept")
    else:
        new = old.copy()  # PTB-XL SCP mapping unchanged in v2
    new = new.astype(np.float32)

    for f in src.iterdir():
        if f.name not in {"labels.npy", "manifest.csv", "info.json"}:
            link_or_copy(f, out / f.name)
    np.save(out / "labels.npy", new)
    if (src / "manifest.csv").exists():
        man = pd.read_csv(src / "manifest.csv", keep_default_na=False)
        if "labels" in man.columns and len(man) == len(new):
            man["labels"] = ["|".join(L.LABELS[j] for j in np.where(r > 0)[0]) for r in new]
        man.to_csv(out / "manifest.csv", index=False)
    info = json.loads((src / "info.json").read_text()) if (src / "info.json").exists() else {}
    info.update({"labels_version": "v2-aha", "relabelled_with": __version__})
    (out / "info.json").write_text(json.dumps(info, indent=2))

    valid = meta["valid"].astype(str).str.lower().eq("true").to_numpy() if "valid" in meta else np.ones(len(meta), bool)
    rows = []
    for j, l in enumerate(L.LABELS):
        o, n = old[valid, j], new[valid, j]
        rows.append({"label": l, "prev_v1_%": 100 * o.mean(), "prev_v2_%": 100 * n.mean(),
                     "gained": int(((n > 0) & (o == 0)).sum()), "lost": int(((n == 0) & (o > 0)).sum())})
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "relabel_report.csv", index=False)
    changed = (new[valid] != old[valid]).any(1).mean()
    print(rep.round(2).to_string(index=False))
    print(f"\n{a.dataset}: {100 * changed:.2f} % of valid records changed at least one label -> {out}")


if __name__ == "__main__":
    main()
