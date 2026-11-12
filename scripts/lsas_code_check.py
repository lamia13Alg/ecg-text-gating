"""Vérification de la correspondance SNOMED-CT -> 11 étiquettes pour LSAS (ecg-arrhythmia 1.0.0).

Comparé à la liste officielle (ConditionNames_SNOMED-CT.csv), la correspondance v1 :
  * oublie « ST-T Change » 428750005 et « ST tilt up » 164931005 (STT_ISCH) ;
  * oublie Mobitz I 54016002, Mobitz II 28189009 et « AV block » 233917008 (AVB) ;
  * ne peut pas séparer LBBB des hémiblocs : LSAS code LBBB, « left front bundle branch block »
    et « left back bundle branch block » avec le MÊME code 164909002.
Le script lit les codes de tous les en-têtes, compare v1 et v2 (prévalences) et recalcule les AUROC
LSAS à partir des prédictions sauvegardées (aucun réentraînement).

  python scripts/lsas_code_check.py --lsas_cache .../cache/lsas --runs runs/v2_amplitude ... --out lsas_check/
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.data.remote import _get  # noqa: E402
from ecgfusion.labels import LABELS, LSAS_SINUS_RHYTHM, LSAS_SNOMED_MAP  # noqa: E402

BASES = ["https://physionet.org/files/ecg-arrhythmia/1.0.0/",
         "https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/"]
# liste officielle (ConditionNames_SNOMED-CT.csv) : code -> acronymes
OFFICIAL = {
    "270492004": "1AVB", "195042002": "2AVB", "54016002": "2AVB1", "28189009": "2AVB2", "27885002": "3AVB",
    "251173003": "ABI", "39732003": "ALS", "284470004": "APB", "164917005": "AQW", "47665007": "ARS",
    "233917008": "AVB", "251199005": "CCR", "251198002": "CR", "428417006": "ERV", "164942001": "FQRS",
    "698252002": "IDC/IVB", "426995002": "JEB", "251164006": "JPT", "164909002": "LBBB/LBBBB/LFBBB",
    "164873001": "LVH", "251146004": "LVQRSAL", "251148003": "LVQRSCL", "251147008": "LVQRSLL",
    "164865005": "MI(+BW/FW/LW/SW)", "164947007": "PRIE", "164912004": "PWC", "111975006": "QTIE",
    "446358003": "RAH", "59118001": "RBBB", "89792004": "RVH", "429622005": "STDD", "164930006": "STE",
    "428750005": "STTC", "164931005": "STTU", "164934002": "TWC", "59931005": "TWO", "164937009": "UW",
    "11157007": "VB", "75532003": "VEB", "13640000": "VFW", "17338001": "VPB", "195060002": "VPE",
    "251180001": "VET", "195101003": "WAVN/SAAWR", "74390002": "WPW", "426177001": "SB", "426783006": "SR",
    "164889003": "AFIB", "427084000": "ST", "164890007": "AF", "427393009": "SA", "426761007": "SVT",
    "713422000": "AT", "233896004": "AVNRT", "233897008": "AVRT",
}
V2_EXTRA = {"428750005": "STT_ISCH", "164931005": "STT_ISCH",
            "54016002": "AVB", "28189009": "AVB", "233917008": "AVB"}
# v3 : codes présents dans les en-têtes mais absents de la liste officielle (codes SNOMED des Challenges
# PhysioNet). 55827005 = « Left ventricular hypertrophy (disorder) », 55930002 = « S-T changes »,
# 251120003 = « incomplete left bundle branch block », 57054005 = « acute myocardial infarction ».
V3_EXTRA = {**V2_EXTRA, "55930002": "STT_ISCH", "55827005": "LVH", "251120003": "LBBB", "57054005": "MI"}
AMBIGUOUS_LBBB = "164909002"


def fetch_codes(workers=32):
    for base in BASES:
        try:
            recs = _get(base + "RECORDS", timeout=60).decode().split()
            break
        except Exception as e:  # noqa: BLE001
            print("RECORDS indisponible sur", base, type(e).__name__)
    else:
        sys.exit("PhysioNet injoignable : activez Internet dans les réglages du notebook.")
    dirs = [r for r in recs if r.endswith("/")]
    if dirs:  # RECORDS liste des dossiers -> RECORDS de chaque dossier
        def sub(d):
            try:
                return [d + x for x in _get(base + d + "RECORDS", timeout=60).decode().split()]
            except Exception:  # noqa: BLE001
                return []
        with ThreadPoolExecutor(workers) as ex:
            recs = [r for lst in ex.map(sub, dirs) for r in lst] + [r for r in recs if not r.endswith("/")]
    print(f"{len(recs)} enregistrements LSAS à lire")

    def one(rel):
        for _ in range(3):
            try:
                txt = _get(base + rel + ".hea", timeout=30).decode(errors="ignore")
                dx = next((l.split(":", 1)[1] for l in txt.splitlines() if l.lstrip("#").strip().lower().startswith("dx")), "")
                return rel.rsplit("/", 1)[-1], ",".join(c.strip() for c in dx.split(",") if c.strip())
            except Exception:  # noqa: BLE001
                continue
        return rel.rsplit("/", 1)[-1], None
    out = []
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, recs)):
            out.append(r)
            if (k + 1) % 5000 == 0:
                print(f"  {k + 1}/{len(recs)}", flush=True)
    df = pd.DataFrame(out, columns=["stem", "codes"])
    print(f"échecs de lecture : {df['codes'].isna().sum()}")
    return df.dropna()


def to_labels(codes, mapping, ambiguous_to=None):
    y = np.zeros(len(LABELS), int)
    cs = [c for c in codes if c]
    for c in cs:
        if c in mapping:
            y[LABELS.index(mapping[c])] = 1
        if ambiguous_to and c == AMBIGUOUS_LBBB:
            y[LABELS.index(ambiguous_to)] = 1
    if cs and all(c == LSAS_SINUS_RHYTHM for c in cs):
        y[LABELS.index("NORM")] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsas_cache", required=True, help="cache LSAS (meta.csv)")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--codes", default=None, help="lsas_codes.csv déjà extrait (évite le téléchargement)")
    ap.add_argument("--out", default="lsas_check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    codes = pd.read_csv(a.codes, dtype=str) if a.codes else fetch_codes()
    codes.to_csv(out / "lsas_codes.csv", index=False)
    lists = {s: (c or "").split(",") for s, c in zip(codes["stem"], codes["codes"])}

    # 1. fréquences de tous les codes
    from collections import Counter
    cnt = Counter(c for v in lists.values() for c in v if c)
    n = len(lists)
    rows = []
    for c, k in cnt.most_common():
        rows.append({"code": c, "acronym": OFFICIAL.get(c, "?? (absent de la liste officielle)"), "n": k,
                     "pct": 100 * k / n, "v1_label": LSAS_SNOMED_MAP.get(c, ""),
                     "v2_label": {**LSAS_SNOMED_MAP, **V2_EXTRA}.get(c, "")
                     + (" (ambigu : LBBB ou hémibloc)" if c == AMBIGUOUS_LBBB else "")})
    freq = pd.DataFrame(rows); freq.to_csv(out / "lsas_code_frequencies.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== Codes SNOMED présents dans {n} en-têtes LSAS ===")
    print(freq.round(2).to_string(index=False))
    unk = freq[freq["acronym"].str.startswith("??")]
    if len(unk):
        print("\nCodes absents de la liste officielle :", ", ".join(unk["code"]))

    # 2. prévalences v1 / v2
    v1 = {s: to_labels(v, LSAS_SNOMED_MAP) for s, v in lists.items()}
    v2 = {s: to_labels(v, {**LSAS_SNOMED_MAP, **V2_EXTRA}) for s, v in lists.items()}
    v3 = {s: to_labels(v, {**LSAS_SNOMED_MAP, **V3_EXTRA}) for s, v in lists.items()}
    P1, P2, P3 = (np.stack(list(v.values())).mean(0) for v in (v1, v2, v3))
    prev = pd.DataFrame({"label": LABELS, "prev_v1_%": 100 * P1, "prev_v2_%": 100 * P2, "prev_v3_%": 100 * P3})
    print("\n=== Prévalences LSAS : v1 (utilisée), v2 (+ liste officielle), v3 (+ codes Challenge) ===")
    print(prev.round(2).to_string(index=False))
    lb = sum(AMBIGUOUS_LBBB in v for v in lists.values())
    print(f"\nCode {AMBIGUOUS_LBBB} (LBBB OU hémibloc antérieur/postérieur, indissociables) : {lb} ECG "
          f"({100 * lb / n:.2f} %). Codes FASC 445118002/445211001 présents : "
          f"{cnt.get('445118002', 0) + cnt.get('445211001', 0)}")

    both = sum(("55827005" in v) and ("164873001" in v) for v in lists.values())
    print(f"55827005 et 164873001 ensemble : {both} ECG (55827005 seul : {cnt.get('55827005', 0) - both})")
    # 3. AUROC LSAS recalculées
    meta = pd.read_csv(Path(a.lsas_cache) / "meta.csv", keep_default_na=False)
    stems = meta["record_id"].str.replace("lsas_", "", regex=False).to_numpy()
    res = []
    for r in a.runs:
        f = Path(r) / "eval" / "preds_lsas_external_no_text.npz"
        if not f.exists():
            print("absent :", f); continue
        d = np.load(f)
        st = stems[d["index"]]
        ok = np.array([s in v2 for s in st])
        p, y1 = d["prob"][ok], d["labels"][ok].astype(int)
        y2 = np.stack([v2[s] for s in st[ok]])
        y3 = np.stack([v3[s] for s in st[ok]])
        y1chk = np.stack([v1[s] for s in st[ok]])
        agree = (y1chk == y1).all(1).mean()
        for j, l in enumerate(LABELS):
            def auc(y):
                return roc_auc_score(y, p[:, j]) if 0 < y.sum() < len(y) else np.nan
            res.append({"run": Path(r).name, "label": l, "n_pos_v1": int(y1[:, j].sum()),
                        "n_pos_v2": int(y2[:, j].sum()), "n_pos_v3": int(y3[:, j].sum()),
                        "auroc_v1": auc(y1[:, j]), "auroc_v2": auc(y2[:, j]), "auroc_v3": auc(y3[:, j])})
        print(f"[{Path(r).name}] {ok.sum()} ECG appariés ; étiquettes v1 recalculées = étiquettes du cache "
              f"pour {100 * agree:.1f} % des ECG")
    if res:
        R = pd.DataFrame(res); R.to_csv(out / "lsas_auroc_v1_v2.csv", index=False)
        m = R.groupby("label", sort=False)[["n_pos_v1", "n_pos_v2", "n_pos_v3", "auroc_v1", "auroc_v2",
                                             "auroc_v3"]].mean()
        m["d_v2"] = m["auroc_v2"] - m["auroc_v1"]; m["d_v3"] = m["auroc_v3"] - m["auroc_v1"]
        print("\n=== AUROC LSAS (sans texte, moyenne des runs) : v1 / v2 / v3 ===")
        print(m.round(3).to_string())
        keep = [l for l in LABELS if l not in ("MI", "LBBB", "FASC")]
        noMI = [l for l in LABELS if l != "MI"]
        for name, labs in (("sans MI", noMI), ("sans MI, LBBB, FASC", keep)):
            print(f"Macro-AUROC LSAS {name} : " + " | ".join(f"{v} {m.loc[labs, f'auroc_{v}'].mean():.3f}"
                                                             for v in ("v1", "v2", "v3")))
        per_run = R[R.label.isin(noMI)].groupby("run")[["auroc_v1", "auroc_v2", "auroc_v3"]].mean()
        print("\nMacro-AUROC LSAS sans MI par graine :\n" + per_run.round(4).to_string())
        print(f"moyenne ± écart-type v3 : {per_run['auroc_v3'].mean():.3f} ± {per_run['auroc_v3'].std():.3f}")


if __name__ == "__main__":
    main()
