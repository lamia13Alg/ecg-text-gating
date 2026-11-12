"""Construction des manifestes (une ligne par ECG) pour chaque base.

Chaque manifeste a les colonnes :
    record_id, patient_id, wfdb_path, text, labels (str "0,1,0,..."), split
et éventuellement des mesures machine (MIMIC).

Arborescences attendues (téléchargements PhysioNet) :
  MIMIC-IV-ECG  : <root>/record_list.csv, <root>/machine_measurements.csv, <root>/files/...
  MIMIC-IV hosp : <hosp>/admissions.csv(.gz), <hosp>/diagnoses_icd.csv(.gz)   [option ICD]
  PTB-XL        : <root>/ptbxl_database.csv, <root>/records500/...
  LSAS-ECG      : <root>/WFDB*/**/*.hea   (base "A large scale 12-lead ECG database
                  for arrhythmia study", PhysioNet ecg-arrhythmia)
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import labels as L


def _labels_to_str(y: np.ndarray) -> str:
    return ",".join(str(int(v)) for v in y)


def _patient_split(pid, val_frac=0.1, test_frac=0.1) -> str:
    """Split déterministe par patient (aucun patient dans deux splits)."""
    h = int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


# --------------------------------------------------------------------------
def build_mimic(root: str, label_source: str = "report", hosp_root: str | None = None,
                max_records: int | None = None, remote: bool = False,
                remote_base: str = "pn://mimic-iv-ecg/1.0") -> pd.DataFrame:
    """remote=True : ``root`` ne contient que record_list.csv et
    machine_measurements.csv ; les signaux sont lus à la volée sur PhysioNet."""
    root = Path(root)
    rec = pd.read_csv(root / "record_list.csv")
    mm = pd.read_csv(root / "machine_measurements.csv", low_memory=False)
    report_cols = [c for c in mm.columns if c.startswith("report_")]
    mm["text"] = (mm[report_cols].fillna("").astype(str)
                  .agg(lambda r: ". ".join(s.strip() for s in r if s.strip()), axis=1))
    keep = ["subject_id", "study_id", "text", "rr_interval", "p_onset", "p_end",
            "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"]
    df = rec.merge(mm[[c for c in keep if c in mm.columns]], on=["subject_id", "study_id"], how="inner")
    if max_records:
        df = df.sample(n=min(max_records, len(df)), random_state=0)

    df["record_id"] = "mimic_" + df["study_id"].astype(str)
    df["patient_id"] = df["subject_id"]
    if remote:
        df["wfdb_path"] = df["path"].apply(lambda p: f"{remote_base.rstrip('/')}/{p}")
    else:
        df["wfdb_path"] = df["path"].apply(lambda p: str(root / p))

    if label_source == "report":
        df["labels"] = df["text"].apply(lambda t: _labels_to_str(L.mimic_text_to_labels(t)))
    elif label_source == "icd":
        if hosp_root is None:
            raise ValueError("label_source=icd nécessite --hosp_root (MIMIC-IV hosp)")
        df = _attach_icd_labels(df, Path(hosp_root))
    else:
        raise ValueError(label_source)

    df["split"] = df["patient_id"].apply(_patient_split)
    df["dataset"] = "mimic"
    return df.drop(columns=["path"], errors="ignore").reset_index(drop=True)


def _read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p if p.exists() else Path(str(p) + ".gz"))


def _attach_icd_labels(df: pd.DataFrame, hosp: Path) -> pd.DataFrame:
    """Labels = codes ICD-10 du séjour hospitalier contenant l'ECG.

    Les ECG hors hospitalisation sont exclus. Ces labels sont indépendants
    du compte rendu ECG -> pas de fuite texte->label.
    """
    adm = _read(hosp / "admissions.csv")[["subject_id", "hadm_id", "admittime", "dischtime"]]
    dx = _read(hosp / "diagnoses_icd.csv")
    dx = dx[dx["icd_version"] == 10]
    codes = dx.groupby("hadm_id")["icd_code"].apply(list)

    df = df.copy()
    df["ecg_time"] = pd.to_datetime(df["ecg_time"])
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    m = df.merge(adm, on="subject_id", how="inner")
    m = m[(m["ecg_time"] >= m["admittime"]) & (m["ecg_time"] <= m["dischtime"])]
    m = m[m["hadm_id"].isin(codes.index)].drop_duplicates("study_id")
    m["labels"] = m["hadm_id"].map(lambda h: _labels_to_str(L.icd10_to_labels(codes[h])))
    return m.drop(columns=["admittime", "dischtime"])


# --------------------------------------------------------------------------
def build_ptbxl(root: str, min_likelihood: float = 0.0) -> pd.DataFrame:
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv")
    db["scp"] = db["scp_codes"].apply(ast.literal_eval)
    df = pd.DataFrame({
        "record_id": "ptbxl_" + db["ecg_id"].astype(str),
        "patient_id": db["patient_id"].astype("Int64").astype(str),
        "wfdb_path": db["filename_hr"].apply(lambda p: str(root / p)),
        # NB : une partie des comptes rendus PTB-XL est en allemand
        "text": db["report"].fillna("").astype(str),
        "labels": db["scp"].apply(lambda d: _labels_to_str(L.ptbxl_codes_to_labels(d, min_likelihood))),
        # folds officiels : 1-8 train, 9 val, 10 test
        "split": db["strat_fold"].map(lambda k: "test" if k == 10 else "val" if k == 9 else "train"),
    })
    df["dataset"] = "ptbxl"
    return df


# --------------------------------------------------------------------------
def build_lsas(root: str) -> pd.DataFrame:
    import wfdb

    root = Path(root)
    rows = []
    for hea in tqdm(sorted(root.rglob("*.hea")), desc="LSAS headers"):
        try:
            h = wfdb.rdheader(str(hea.with_suffix("")))
        except Exception:
            continue
        dx = []
        for c in (h.comments or []):
            if c.lower().startswith("dx"):
                dx = c.split(":", 1)[1].split(",")
        if not dx:
            continue
        rid = hea.stem
        rows.append({
            "record_id": "lsas_" + rid,
            "patient_id": rid,  # un ECG par patient dans cette base
            "wfdb_path": str(hea.with_suffix("")),
            "text": "",  # pas de compte rendu -> modalité texte absente
            "labels": _labels_to_str(L.lsas_codes_to_labels(dx)),
            "split": "test",
        })
    df = pd.DataFrame(rows)
    df["dataset"] = "lsas"
    return df


BUILDERS = {"mimic": build_mimic, "ptbxl": build_ptbxl, "lsas": build_lsas}
