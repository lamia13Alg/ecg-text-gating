"""Génère de fausses bases au FORMAT EXACT de MIMIC-IV-ECG, PTB-XL et LSAS-ECG.

Sert uniquement à vérifier que tout le pipeline tourne (prétraitement ->
entraînement -> évaluation) avant d'utiliser les vraies données PhysioNet.
Les ECG sont simulés : battements gaussiens modifiés selon les labels.

  python scripts/make_synthetic.py --out synthetic --n_mimic 600 --n_ptbxl 200 --n_lsas 200
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# projection (axe frontal, contribution précordiale) grossière par dérivation
GAIN = np.array([0.8, 1.0, 0.4, -0.9, 0.3, 0.7, -0.5, 0.2, 0.7, 1.1, 1.2, 1.0])
PRIORS = {"AFIB": .12, "AVB": .06, "LVH": .08, "STT_ISCH": .15, "MI": .08,
          "RBBB": .07, "LBBB": .04, "FASC": .05, "STACH": .08, "SBRAD": .08}
REPORT = {"AFIB": "Atrial fibrillation", "AVB": "1st degree A-V block",
          "LVH": "Left ventricular hypertrophy", "STT_ISCH": "ST-T changes, possible ischemia",
          "MI": "Inferior infarct - age undetermined", "RBBB": "Right bundle branch block",
          "LBBB": "Left bundle branch block", "FASC": "Left anterior fascicular block",
          "STACH": "Sinus tachycardia", "SBRAD": "Sinus bradycardia"}
SCP = {"AFIB": "AFIB", "AVB": "1AVB", "LVH": "LVH", "STT_ISCH": "ISC_", "MI": "IMI",
       "RBBB": "CRBBB", "LBBB": "CLBBB", "FASC": "LAFB", "STACH": "STACH", "SBRAD": "SBRAD"}
SNOMED = {"AFIB": "164889003", "AVB": "270492004", "LVH": "164873001", "STT_ISCH": "164934002",
          "MI": "164865005", "RBBB": "59118001", "LBBB": "164909002", "FASC": "445118002",
          "STACH": "427084000", "SBRAD": "426177001"}


def sample_labels(rng, shift=1.0):
    labs = {k for k, p in PRIORS.items() if rng.random() < p * shift}
    if "AFIB" in labs:
        labs -= {"AVB", "STACH", "SBRAD"}
    if "STACH" in labs and "SBRAD" in labs:
        labs.discard("SBRAD")
    if "LBBB" in labs:
        labs.discard("RBBB")
    return labs


def g(t, mu, sd, a):
    return a * np.exp(-((t - mu) ** 2) / (2 * sd ** 2))


def simulate(labs, rng, fs=500, dur=10.0, noise=0.03):
    n = int(fs * dur)
    t = np.arange(n) / fs
    hr = 115 if "STACH" in labs else 50 if "SBRAD" in labs else rng.uniform(60, 90)
    rr = 60 / hr
    beats, cur = [], rng.uniform(0.1, 0.5)
    while cur < dur:
        beats.append(cur)
        cur += rr * (rng.uniform(0.6, 1.4) if "AFIB" in labs else rng.uniform(0.97, 1.03))
    wide = "LBBB" in labs or "RBBB" in labs
    qrs_sd = 0.022 if wide else 0.011
    pr = 0.28 if "AVB" in labs else 0.16
    sig = np.zeros((n, 12))
    for li, lead in enumerate(LEADS):
        gain = GAIN[li]
        if "FASC" in labs and lead in ("II", "III", "aVF"):
            gain = -abs(gain) * 0.6
        if "LVH" in labs and lead in ("V5", "V6", "aVL"):
            gain *= 2.2
        if "LVH" in labs and lead in ("V1", "V2", "V3"):
            gain -= 1.2
        x = np.zeros(n)
        for b in beats:
            if "AFIB" not in labs:
                x += g(t, b - pr, 0.025, 0.12 * np.sign(gain + 1e-3))
            if "MI" in labs and lead in ("II", "III", "aVF"):
                x += g(t, b - 0.03, 0.012, -0.4)
            x += g(t, b, qrs_sd, 1.2 * gain)
            if "RBBB" in labs and lead in ("V1", "V2"):
                x += g(t, b + 0.06, 0.015, 0.8)
            st = -0.15 if "STT_ISCH" in labs and lead in ("V4", "V5", "V6", "II") else 0.0
            x += g(t, b + 0.12, 0.04, st)
            tamp = -0.25 if "STT_ISCH" in labs and lead in ("V5", "V6") else 0.3 * np.sign(gain + 1e-3)
            x += g(t, b + 0.3, 0.05, tamp)
        if "AFIB" in labs:
            x += 0.05 * np.sin(2 * np.pi * rng.uniform(5, 7) * t + rng.uniform(0, 6))
        x += rng.normal(0, noise, n) + 0.1 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))
        sig[:, li] = x
    return sig


def write_record(path: Path, sig, fs, comments=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(path.name, fs=fs, units=["mV"] * 12, sig_name=LEADS, p_signal=sig,
                fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12,
                write_dir=str(path.parent), comments=comments or [])


def make_mimic(out: Path, n, rng):
    rec, mm = [], []
    for i in range(n):
        sid, stid = 10_000_000 + i // 2, 40_000_000 + i  # 2 ECG par patient
        labs = sample_labels(rng)
        rel = f"files/p{str(sid)[:4]}/p{sid}/s{stid}/{stid}"
        write_record(out / rel, simulate(labs, rng), 500)
        stmts = ["Sinus rhythm"] if "AFIB" not in labs else []
        stmts += [REPORT[k] for k in sorted(labs)]
        stmts += ["Normal ECG"] if not labs else [rng.choice(["Abnormal ECG", "Borderline ECG"])]
        rec.append({"subject_id": sid, "study_id": stid, "file_name": stid,
                    "ecg_time": "2150-01-01 10:00:00", "path": rel})
        row = {"subject_id": sid, "study_id": stid, "cart_id": 1, "ecg_time": "2150-01-01 10:00:00",
               "rr_interval": 800, "p_onset": 40, "p_end": 120, "qrs_onset": 200, "qrs_end": 290,
               "t_end": 600, "p_axis": 50, "qrs_axis": 30, "t_axis": 40}
        row.update({f"report_{k}": s for k, s in enumerate(stmts)})
        mm.append(row)
    pd.DataFrame(rec).to_csv(out / "record_list.csv", index=False)
    pd.DataFrame(mm).to_csv(out / "machine_measurements.csv", index=False)


def make_ptbxl(out: Path, n, rng):
    rows = []
    for i in range(1, n + 1):
        labs = sample_labels(rng, shift=1.3)  # population différente
        rel = f"records500/{(i // 1000) * 1000:05d}/{i:05d}_hr"
        write_record(out / rel, simulate(labs, rng, noise=0.05), 500)
        scp = {SCP[k]: 100.0 for k in labs} or {"NORM": 100.0}
        rows.append({"ecg_id": i, "patient_id": float(20000 + i), "scp_codes": str(scp),
                     "report": "sinusrhythmus " + " ".join(SCP[k].lower() for k in labs),
                     "filename_lr": rel.replace("500", "100").replace("_hr", "_lr"),
                     "filename_hr": rel, "strat_fold": rng.integers(1, 11)})
    pd.DataFrame(rows).to_csv(out / "ptbxl_database.csv", index=False)


def make_lsas(out: Path, n, rng):
    for i in range(1, n + 1):
        labs = sample_labels(rng, shift=1.5)
        codes = [SNOMED[k] for k in labs] or ["426783006"]
        sig = simulate(labs, rng, noise=0.06) * 1.1  # appareil différent
        write_record(out / "WFDB" / "01" / "010" / f"JS{i:05d}", sig, 500,
                     comments=["Age: 60", "Sex: Male", "Dx: " + ",".join(codes), "Rx: Unknown"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--n_mimic", type=int, default=600)
    ap.add_argument("--n_ptbxl", type=int, default=200)
    ap.add_argument("--n_lsas", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    make_mimic(out / "mimic", a.n_mimic, rng)
    make_ptbxl(out / "ptbxl", a.n_ptbxl, rng)
    make_lsas(out / "lsas", a.n_lsas, rng)
    (out / "README.json").write_text(json.dumps({"note": "données SIMULÉES, tests uniquement"}))
    print(f"[synthetic] bases écrites dans {out}/")


if __name__ == "__main__":
    main()
