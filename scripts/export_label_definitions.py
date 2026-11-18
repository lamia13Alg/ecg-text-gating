"""Export the label nomenclature table (label -> AHA/ACCF/HRS statement -> SCP-ECG -> SNOMED-CT).

  python scripts/export_label_definitions.py --out docs/
Writes label_definitions.csv and label_definitions.tex (booktabs table for the paper).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.labels import LABELS  # noqa: E402
from ecgfusion.nomenclature import LABEL_DEFINITIONS, codes_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for l in LABELS:
        d = LABEL_DEFINITIONS[l]
        scp, sn = codes_for(l)
        rows.append({"label": l, "aha_statement": d["statement"], "definition": d["definition"],
                     "scp_ecg_codes": " ".join(scp), "snomed_ct_codes": " ".join(sn),
                     "references": " ".join(d["refs"])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "label_definitions.csv", index=False)

    def tex(s):
        return s.replace("_", r"\_").replace("*", r"$^*$").replace(">=", r"$\geq$").replace(">", r"$>$")

    lines = [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.5cm}"
             r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{3.2cm}"
             r">{\raggedright\arraybackslash}p{3.4cm}}", r"\toprule",
             r"Label & Definition (AHA/ACCF/HRS statement) & SCP-ECG (PTB-XL) & SNOMED-CT (LSAS) \\",
             r"\midrule"]
    for r in rows:
        cite = r"~\cite{" + ",".join(r["references"].split()) + "}"
        lines.append(rf"\lbl{{{tex(r['label'])}}} & {tex(r['definition'])}{cite} & "
                     rf"{tex(r['scp_ecg_codes'])} & {tex(r['snomed_ct_codes'])} \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (out / "label_definitions.tex").write_text("\n".join(lines) + "\n")
    print(df[["label", "aha_statement"]].to_string(index=False))


if __name__ == "__main__":
    main()
