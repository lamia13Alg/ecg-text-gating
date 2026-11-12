"""Audit manuel des étiquettes MIMIC — étape 2 : précision / rappel des règles par pathologie.

  python scripts/audit_score.py --grid audit/grille_audit_mimic.xlsx --key audit/audit_key.csv --out audit/
  # accord inter-annotateurs (facultatif) : --grid2 grille_annotateur2.xlsx

Estimations pondérées par l'inverse de la probabilité de tirage (poids de strate), IC 95 % par
bootstrap stratifié. Sorties : audit_results.csv, audit_disagreements.csv (cas à relire pour corriger
les règles), audit_kappa.csv si --grid2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion.labels import LABELS  # noqa: E402


def read_grid(path):
    g = pd.read_excel(path, sheet_name="grille", dtype=str).fillna("")
    done = g["fait"].str.strip().str.lower().eq("x")
    out = pd.DataFrame({"audit_id": g["audit_id"], "text": g["compte_rendu"], "done": done,
                        "unsure": g["incertain"].str.strip().str.lower().eq("x"),
                        "comment": g["commentaire"]})
    for l in LABELS:
        out[f"true_{l}"] = g[l].str.strip().isin(["1", "1.0", "x", "X"]).astype(int)
    return out


def estimates(df, w):
    rows = {}
    for l in LABELS:
        r, t = df[f"rule_{l}"].to_numpy(), df[f"true_{l}"].to_numpy()
        tp, fp, fn = (w * (r & t)).sum(), (w * (r & (1 - t))).sum(), (w * ((1 - r) & t)).sum()
        p = tp / (tp + fp) if tp + fp > 0 else np.nan
        rc = tp / (tp + fn) if tp + fn > 0 else np.nan
        f1 = 2 * p * rc / (p + rc) if p + rc > 0 else np.nan
        rows[l] = (p, rc, f1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--grid2", default=None)
    ap.add_argument("--out", default="audit")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(a.key)
    key["N_stratum"] = key["weight"] * key.groupby("stratum")["audit_id"].transform("count")
    g = read_grid(a.grid)
    df = key.merge(g, on="audit_id")
    n_all = len(df)
    df = df[df["done"]].reset_index(drop=True)
    print(f"{len(df)} / {n_all} lignes marquées 'fait' ; {int(df['unsure'].sum())} marquées 'incertain'")
    if df.empty:
        sys.exit("Aucune ligne terminée : mettez x dans la colonne 'fait'.")
    # poids recalculés sur les lignes réellement annotées
    df["w"] = df["N_stratum"] / df.groupby("stratum")["audit_id"].transform("count")
    for l in LABELS:
        df[f"rule_{l}"] = df[f"rule_{l}"].astype(int)

    est = estimates(df, df["w"].to_numpy())
    rng = np.random.default_rng(0)
    groups = [np.where(df["stratum"].to_numpy() == s)[0] for s in df["stratum"].unique()]
    boots = {l: [] for l in LABELS}
    for _ in range(a.n_boot):
        idx = np.concatenate([rng.choice(gi, len(gi)) for gi in groups])
        b = df.iloc[idx]
        e = estimates(b, b["w"].to_numpy())
        for l in LABELS:
            boots[l].append(e[l])

    rows = []
    for l in LABELS:
        bt = np.array(boots[l], float)
        lo, hi = np.nanpercentile(bt, 2.5, axis=0), np.nanpercentile(bt, 97.5, axis=0)
        p, r, f = est[l]
        rr, tt = df[f"rule_{l}"].to_numpy(), df[f"true_{l}"].to_numpy()
        def cp_lo(x, n):  # borne basse exacte (Clopper-Pearson), non pondérée
            return np.nan if n == 0 else (0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1)))
        tp_n = int((rr & tt).sum())
        lo = lo.copy(); hi = hi.copy()
        if hi[0] - lo[0] < 1e-9:  # aucun désaccord : le bootstrap est dégénéré
            lo[0] = cp_lo(tp_n, int(rr.sum()))
        if hi[1] - lo[1] < 1e-9:
            lo[1] = cp_lo(tp_n, int(tt.sum()))
        rows.append({"label": l, "n_rule_pos": int(df[f"rule_{l}"].sum()), "n_true_pos": int(df[f"true_{l}"].sum()),
                     "precision": p, "prec_lo": lo[0], "prec_hi": hi[0],
                     "recall": r, "rec_lo": lo[1], "rec_hi": hi[1], "f1": f})
    res = pd.DataFrame(rows)
    res.to_csv(out / "audit_results.csv", index=False)
    show = res.copy()
    for m, lo, hi in [("precision", "prec_lo", "prec_hi"), ("recall", "rec_lo", "rec_hi")]:
        show[m] = [f"{v:.2f} [{x:.2f}; {y:.2f}]" if not np.isnan(v) else "—"
                   for v, x, y in zip(res[m], res[lo], res[hi])]
    show["f1"] = res["f1"].round(2)
    print("\n=== Précision et rappel des règles (pondérés, IC 95 % bootstrap ; borne basse exacte"
          " de Clopper-Pearson quand aucune erreur n'est observée) ===")
    print(show[["label", "n_rule_pos", "n_true_pos", "precision", "recall", "f1"]].to_string(index=False))
    print(f"\nMoyenne macro : précision {res['precision'].mean():.2f}, rappel {res['recall'].mean():.2f}, "
          f"F1 {res['f1'].mean():.2f}")
    exact = (df[[f"rule_{l}" for l in LABELS]].to_numpy() == df[[f"true_{l}" for l in LABELS]].to_numpy()).all(1)
    print(f"Comptes rendus entièrement corrects (pondéré) : {np.average(exact, weights=df['w']):.1%}")

    dis = []
    for _, r in df.iterrows():
        for l in LABELS:
            if r[f"rule_{l}"] != r[f"true_{l}"]:
                dis.append({"audit_id": r["audit_id"], "label": l,
                            "type": "faux positif" if r[f"rule_{l}"] else "faux négatif",
                            "stratum": r["stratum"], "text": r["text"], "comment": r["comment"]})
    dis = pd.DataFrame(dis)
    dis.to_csv(out / "audit_disagreements.csv", index=False)
    print(f"\n{len(dis)} désaccords → {out / 'audit_disagreements.csv'}")
    if len(dis):
        print(dis.groupby(["label", "type"]).size().unstack(fill_value=0).to_string())

    if a.grid2:
        g2 = read_grid(a.grid2)
        m = g.merge(g2, on="audit_id", suffixes=("_1", "_2"))
        m = m[m["done_1"] & m["done_2"]]
        ks = []
        for l in LABELS:
            x, y = m[f"true_{l}_1"].to_numpy(), m[f"true_{l}_2"].to_numpy()
            po = (x == y).mean()
            pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
            ks.append({"label": l, "n": len(m), "agreement": po,
                       "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan})
        kd = pd.DataFrame(ks); kd.to_csv(out / "audit_kappa.csv", index=False)
        print(f"\n=== Accord inter-annotateurs ({len(m)} comptes rendus communs) ===")
        print(kd.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
