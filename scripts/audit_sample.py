"""Audit manuel des étiquettes MIMIC — étape 1 : échantillon stratifié + grille Excel à remplir.

Strates (chaque compte rendu appartient à une seule strate) :
  * « label:<X> » : les règles donnent X (X = l'étiquette la plus rare du compte rendu) ;
  * « excluded »  : aucun label retenu pour un terme pourtant présent (négation, « possible »,
                    repolarisation secondaire…) — c'est là que se cachent les faux négatifs ;
  * « none »      : aucun label et aucun terme diagnostique.
Chaque compte rendu reçoit un poids = (taille de sa strate) / (nombre tiré dans la strate), ce qui
permet d'estimer précision et rappel pour toute la base malgré le sur-échantillonnage des pathologies rares.

L'annotateur ne voit PAS les étiquettes des règles (évaluation en aveugle) : elles sont dans audit_key.csv.

  python scripts/audit_sample.py --cache /kaggle/input/.../cache/mimic --out audit/
Sorties : audit/grille_audit_mimic.xlsx (à remplir), audit/audit_key.csv (à garder, ne pas modifier).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.labels import LABELS, MIMIC_TEXT_RULES, mimic_text_to_labels  # noqa: E402

RAW = {k: [re.compile(p) for p in v] for k, v in MIMIC_TEXT_RULES.items()}

GUIDE = [
    ("But", "Pour chaque compte rendu, cochez (1) les pathologies que le compte rendu AFFIRME, selon les "
            "conventions ci-dessous. Vous jugez le texte, pas le tracé. Laissez vide si absent. "
            "Les étiquettes produites par les règles sont cachées : ne cherchez pas à les deviner."),
    ("Colonne 'fait'", "Mettez x quand la ligne est terminée (même si aucune pathologie n'est présente). "
                       "Les lignes sans x ne sont pas comptées."),
    ("Colonne 'incertain'", "Mettez x si le texte est ambigu pour vous ; expliquez dans 'commentaire'."),
    ("Incertitude", "« possible », « consider », « borderline », « questionable », « suggests » devant le "
                    "diagnostic → 0. « probable » → 1. « cannot rule out », « rule out » → 0."),
    ("Négation", "« no longer », « not », « no evidence of », « resolved » … → 0."),
    ("NORM", "1 seulement si le compte rendu dit « normal ECG » ET ne mentionne aucune autre des 10 "
             "pathologies. « otherwise normal ECG » → 0. « Borderline ECG » / « Abnormal ECG » ne sont pas des pathologies."),
    ("AFIB", "Fibrillation ou flutter atrial."),
    ("Référence", "Définitions AHA/ACCF/HRS « Standardization and Interpretation of the ECG » (Mason 2007 ; "
                  "parties III-VI 2009), définition universelle de l'infarctus (Thygesen 2018), recommandation "
                  "bradycardie/conduction (Kusumoto 2019)."),
    ("AVB", "BAV du 1er (PR > 200 ms, y compris « prolonged PR interval »), 2e (Mobitz I/II, Wenckebach) ou "
            "3e degré. « borderline prolonged PR » → 0. Le « 2:1 » d'un flutter n'est pas un BAV."),
    ("LVH", "Hypertrophie ventriculaire gauche (« LVH », y compris « LVH with secondary repolarization »)."),
    ("STT_ISCH", "Anomalies ST-T PRIMAIRES : non spécifiques ou évocatrices d'ischémie/lésion (sus/sous-décalage "
                 "ST, inversion ou anomalie de T, ondes T amples). NE PAS compter les anomalies SECONDAIRES "
                 "(« due to » HVG, bloc de branche, trouble de conduction, stimulation ; « LVH with ST-T "
                 "changes »), la repolarisation précoce, les variantes normales, les ondes T juvéniles. "
                 "« may be due to hypertrophy and/or ischemia » → 1 (ischémie non exclue)."),
    ("MI", "Infarctus, quel que soit l'âge (« age undetermined », « old », « acute ») ; STEMI."),
    ("RBBB / LBBB", "Bloc de branche droit / gauche, complet ou incomplet."),
    ("FASC", "Hémibloc antérieur ou postérieur gauche (« fascicular block », « hemiblock »)."),
    ("STACH / SBRAD", "Tachycardie / bradycardie SINUSALE uniquement (pas une tachycardie atriale, etc.)."),
]


def raw_mentions(text: str) -> set:
    t = text.lower()
    return {lab for lab, pats in RAW.items() if any(p.search(t) for p in pats)}


def write_grid(df: pd.DataFrame, path: Path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active; ws.title = "grille"
    cols = ["audit_id", "compte_rendu"] + LABELS + ["fait", "incertain", "commentaire"]
    ws.append(cols)
    for _, r in df.iterrows():
        ws.append([r["audit_id"], r["text"]] + [None] * (len(LABELS) + 3))
    head = PatternFill("solid", fgColor="DDE5F0")
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = head
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 70
    from openpyxl.utils import get_column_letter
    for j in range(3, 3 + len(LABELS)):
        ws.column_dimensions[get_column_letter(j)].width = 8.5
    nL = 2 + len(LABELS)
    ws.column_dimensions[get_column_letter(nL + 1)].width = 6
    ws.column_dimensions[get_column_letter(nL + 2)].width = 9
    ws.column_dimensions[get_column_letter(nL + 3)].width = 40
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[2:]:
            c.alignment = Alignment(horizontal="center", vertical="top")
    n = len(df) + 1
    dv1 = DataValidation(type="list", formula1='"1"', allow_blank=True)
    dv1.error = "Mettez 1 (présent) ou laissez vide (absent)"
    dvx = DataValidation(type="list", formula1='"x"', allow_blank=True)
    ws.add_data_validation(dv1); ws.add_data_validation(dvx)
    dv1.add(f"C2:{get_column_letter(nL)}{n}")
    dvx.add(f"{get_column_letter(nL + 1)}2:{get_column_letter(nL + 2)}{n}")
    ws.freeze_panes = "C2"

    g = wb.create_sheet("guide")
    g.append(["Règle", "Consigne"])
    for k, v in GUIDE:
        g.append([k, v])
    g.column_dimensions["A"].width = 20; g.column_dimensions["B"].width = 110
    for row in g.iter_rows():
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    wb.active = 1  # ouvre sur le guide
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dossier du cache MIMIC (meta.csv)")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--per_label", type=int, default=13)
    ap.add_argument("--n_excluded", type=int, default=35)
    ap.add_argument("--n_total", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--exclude", default=None,
                    help="audit_key.csv d'un audit précédent : ses comptes rendus ne sont pas retirés")
    ap.add_argument("--prefix", default="A", help="préfixe des identifiants (ex. B pour un 2e audit)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(a.cache) / "meta.csv", keep_default_na=False)
    meta = meta[meta["valid"].astype(str).str.lower().eq("true") & meta["text"].str.strip().ne("")]
    if a.exclude:
        prev_ids = set(pd.read_csv(a.exclude)["record_id"].astype(str))
        meta = meta[~meta["record_id"].astype(str).isin(prev_ids)]
        print(f"{len(prev_ids)} comptes rendus de l'audit précédent exclus")
    meta = meta.reset_index(drop=True)
    print(f"{len(meta)} comptes rendus MIMIC valides")

    Y = np.stack([mimic_text_to_labels(t) for t in meta["text"]]).astype(int)
    prev = Y.mean(0)
    order = np.argsort(prev)  # du plus rare au plus fréquent
    strata = []
    for i, t in enumerate(meta["text"]):
        pos = [LABELS[j] for j in order if Y[i, j]]
        if pos:
            strata.append("label:" + pos[0])
        elif raw_mentions(t):
            strata.append("excluded")
        else:
            strata.append("none")
    meta["stratum"] = strata
    sizes = meta["stratum"].value_counts()

    rng = np.random.default_rng(a.seed)
    quota = {f"label:{l}": a.per_label for l in LABELS}
    quota["excluded"] = a.n_excluded
    quota["none"] = max(a.n_total - a.per_label * len(LABELS) - a.n_excluded, 0)
    picks = []
    for s, q in quota.items():
        idx = np.where(meta["stratum"].to_numpy() == s)[0]
        if len(idx) == 0 or q == 0:
            continue
        picks.append(rng.choice(idx, min(q, len(idx)), replace=False))
    sel = np.concatenate(picks)
    rng.shuffle(sel)  # ordre aléatoire : l'annotateur ne voit pas les strates

    df = meta.iloc[sel].copy().reset_index(drop=True)
    df["audit_id"] = [f"{a.prefix}{k + 1:03d}" for k in range(len(df))]
    n_in = df["stratum"].value_counts()
    df["weight"] = df["stratum"].map(lambda s: sizes[s] / n_in[s])
    key = df[["audit_id", "record_id", "stratum", "weight"]].copy()
    for j, l in enumerate(LABELS):
        key[f"rule_{l}"] = Y[sel, j]
    key.to_csv(out / "audit_key.csv", index=False)
    write_grid(df, out / "grille_audit_mimic.xlsx")

    summ = pd.DataFrame({"taille_strate": sizes, "tirés": n_in}).fillna(0).astype(int)
    summ["poids"] = (summ["taille_strate"] / summ["tirés"].replace(0, np.nan)).round(1)
    print(summ.sort_index().to_string())
    print(f"\nGrille : {out / 'grille_audit_mimic.xlsx'}  ({len(df)} comptes rendus)")
    print(f"Clé    : {out / 'audit_key.csv'}  (étiquettes des règles, à ne pas ouvrir avant d'avoir annoté)")


if __name__ == "__main__":
    main()
