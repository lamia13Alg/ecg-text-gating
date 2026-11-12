"""Construit le manifeste puis le cache prétraité d'une base (reprenable).

Exemples :
  python scripts/prepare_data.py --dataset mimic --root /data/mimic-iv-ecg/1.0 --out cache/mimic
  python scripts/prepare_data.py --dataset mimic --remote --root csv_dir --max_records 100000 \
         --threads 16 --max_hours 9 --resume_from <cache précédent> --out cache/mimic
  python scripts/prepare_data.py --dataset ptbxl --root /data/ptb-xl/1.0.3 --out cache/ptbxl
  python scripts/prepare_data.py --dataset lsas  --root /data/ecg-arrhythmia/1.0.0 --out cache/lsas

Code de sortie : 0 = cache complet, 3 = incomplet (à reprendre).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.data.builders import build_lsas, build_mimic, build_ptbxl  # noqa: E402
from ecgfusion.data.cache import build_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic", "ptbxl", "lsas"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label_source", default="report", choices=["report", "icd"])
    ap.add_argument("--hosp_root", default=None)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--target_fs", type=int, default=250)
    ap.add_argument("--feature_source", default="computed", choices=["computed", "machine"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1,
                    help="téléchargements simultanés par processus (mode --remote)")
    ap.add_argument("--remote", action="store_true",
                    help="MIMIC : lire les signaux directement sur PhysioNet (pas de téléchargement des 90 Go)")
    ap.add_argument("--max_hours", type=float, default=None, help="arrêt propre après ce délai")
    ap.add_argument("--resume_from", default=None, help="cache incomplet d'une session précédente")
    a = ap.parse_args()
    deadline = time.time() + a.max_hours * 3600 if a.max_hours is not None else None

    prev_manifest = Path(a.resume_from) / "manifest.csv" if a.resume_from else None
    if prev_manifest and prev_manifest.exists():
        df = pd.read_csv(prev_manifest, keep_default_na=False, low_memory=False)
        print(f"[manifest] repris de {prev_manifest} ({len(df)} enregistrements)")
    else:
        if a.root is None:
            ap.error("--root est requis (sauf en reprise)")
        if a.dataset == "mimic":
            df = build_mimic(a.root, a.label_source, a.hosp_root, a.max_records, remote=a.remote)
        elif a.dataset == "ptbxl":
            hits = sorted(Path(a.root).rglob("ptbxl_database.csv"))
            if not hits:
                raise FileNotFoundError(f"ptbxl_database.csv introuvable sous {a.root}")
            df = build_ptbxl(str(hits[0].parent))
        else:
            df = build_lsas(a.root)
        if a.max_records and a.dataset != "mimic":
            df = df.sample(n=min(a.max_records, len(df)), random_state=0).reset_index(drop=True)
        print(f"[manifest] {len(df)} enregistrements")

    complete = build_cache(df, a.out, target_fs=a.target_fs, feature_source=a.feature_source,
                           workers=a.workers, threads=a.threads, deadline=deadline,
                           resume_from=a.resume_from)
    sys.exit(0 if complete else 3)


if __name__ == "__main__":
    main()
