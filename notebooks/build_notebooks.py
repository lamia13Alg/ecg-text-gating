"""Génère les deux notebooks Kaggle (exécuté une fois, côté développement)."""
import json
from pathlib import Path


def nb(cells):
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n"),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, s in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = r'''
import glob, os, re, shutil
def _version(init):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', open(init).read())
    return m.group(1) if m else "0"
# plusieurs copies du code peuvent être présentes (dataset du code, sorties d'anciens notebooks) :
# on prend la plus récente
hits = sorted(glob.glob("/kaggle/input/**/ecgfusion/__init__.py", recursive=True), key=_version)
assert hits, "Code introuvable : ajoutez le dataset contenant ecg_fusion.zip via « Add Input »."
src = os.path.dirname(os.path.dirname(hits[-1]))
shutil.rmtree("/tmp/ecg_fusion", ignore_errors=True)
shutil.copytree(src, "/tmp/ecg_fusion")   # hors /kaggle/working : le code n'encombre pas la sortie
os.chdir("/tmp/ecg_fusion")
!pip install -q wfdb
print(f"Code version {_version(hits[-1])} prêt (source : {src})")
'''

PREP = [
    ("markdown", r'''
# Notebook 1 — Préparation des données (MIMIC-IV-ECG, PTB-XL, LSAS-ECG)

**Réglages Kaggle (panneau de droite) :** *Accelerator* **None**, *Internet* **On**,
*Add Input* : le dataset du code.

Lancez avec **Save Version → Save & Run All**. Le notebook s'arrête de lui-même avant la
limite des 12 h et **sauvegarde ce qui est fait**. S'il affiche **« À REPRENDRE »** :
1. *Edit* → *Add Input* → *Your Work* → ajoutez la sortie de **ce même notebook** (dernière version) ;
2. *Save Version → Save & Run All*. Ce qui est déjà fait est recopié, le reste est complété.

Répétez jusqu'à **« PRÉPARATION TERMINÉE »**, puis *Output → New Dataset* (`ecg-fusion-cache`).
À chaque reprise, ne gardez en Input que la sortie de la version la plus récente.
'''),
    ("code", SETUP),
    ("code", r'''
import glob, json, shutil, time
T0 = time.time()
SESSION_HOURS = 11.0    # limite Kaggle = 12 h ; 1 h de marge pour sauvegarder la sortie
N_MIMIC = 100_000       # fixé lors de la 1re session (les reprises gardent le même échantillon)
WORKERS, THREADS = 4, 16
RAW, CACHE = "/tmp/raw", "/kaggle/working/cache"
os.makedirs(RAW, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

def hours_left():
    return SESSION_HOURS - (time.time() - T0) / 3600

def previous(name):
    """Cache de ce nom dans les Inputs (sortie d'une version précédente)."""
    hits = sorted(glob.glob(f"/kaggle/input/**/cache/{name}/info.json", recursive=True))
    return os.path.dirname(hits[0]) if hits else None

def info(d):
    return json.load(open(f"{d}/info.json"))

for n in ["ptbxl", "lsas", "mimic"]:
    p = previous(n)
    print(f"{n:6s} :", f"trouvé dans les Inputs ({info(p).get('n_done', info(p)['n'])}/{info(p)['n']})" if p else "à construire")
'''),
    ("markdown", "## 1. PTB-XL (validation externe principale)"),
    ("code", r'''
prev = previous("ptbxl")
if prev and info(prev).get("complete", True):
    shutil.copytree(prev, f"{CACHE}/ptbxl", dirs_exist_ok=True); print("PTB-XL repris de la session précédente")
else:
    !wget -q -c -O {RAW}/ptbxl.zip https://physionet.org/content/ptb-xl/get-zip/1.0.3/
    !unzip -q -o {RAW}/ptbxl.zip -d {RAW}/ptbxl && rm {RAW}/ptbxl.zip
    !python scripts/prepare_data.py --dataset ptbxl --root {RAW}/ptbxl --workers 4 --out {CACHE}/ptbxl
    !rm -rf {RAW}/ptbxl
print(f"temps restant : {hours_left():.1f} h")
'''),
    ("markdown", "## 2. LSAS-ECG (validation externe secondaire)"),
    ("code", r'''
prev = previous("lsas")
if prev and info(prev).get("complete", True):
    shutil.copytree(prev, f"{CACHE}/lsas", dirs_exist_ok=True); print("LSAS repris de la session précédente")
else:
    !wget -q -c -O {RAW}/lsas.zip https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/
    !unzip -q -o {RAW}/lsas.zip -d {RAW}/lsas && rm {RAW}/lsas.zip
    !python scripts/prepare_data.py --dataset lsas --root {RAW}/lsas --workers 4 --out {CACHE}/lsas
    !rm -rf {RAW}/lsas
print(f"temps restant : {hours_left():.1f} h")
'''),
    ("markdown", r'''
## 3. MIMIC-IV-ECG (lecture à distance, reprenable)
Signaux lus sur le miroir AWS de PhysioNet (rapide, sans compte), sinon sur physionet.org.
Le débit (ECG/s) et le temps restant estimé s'affichent toutes les 2 minutes.
'''),
    ("code", r'''
prev = previous("mimic")
if prev and not os.path.exists(f"{prev}/done.npy"):
    prev = None   # cache d'une ancienne version du code : on repart de zéro
if prev and info(prev)["complete"]:
    shutil.copytree(prev, f"{CACHE}/mimic", dirs_exist_ok=True); print("MIMIC déjà complet : repris")
else:
    budget = max(hours_left() - 0.3, 0.2)   # 0,3 h pour recopier/sauvegarder
    if prev:
        print(f"REPRISE MIMIC depuis {prev} ({info(prev)['n_done']}/{info(prev)['n']}), budget {budget:.1f} h")
        cmd = (f"python scripts/prepare_data.py --dataset mimic --resume_from {prev} --workers {WORKERS} "
               f"--threads {THREADS} --max_hours {budget:.4f} --out {CACHE}/mimic")
        !{cmd}
    else:
        !mkdir -p {RAW}/mimic
        !wget -q -c -P {RAW}/mimic https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv
        !wget -q -c -P {RAW}/mimic https://physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv
        print(f"NOUVELLE préparation MIMIC ({N_MIMIC} ECG), budget {budget:.1f} h")
        cmd = (f"python scripts/prepare_data.py --dataset mimic --remote --root {RAW}/mimic --max_records {N_MIMIC} "
               f"--workers {WORKERS} --threads {THREADS} --max_hours {budget:.4f} --out {CACHE}/mimic")
        !{cmd}
'''),
    ("markdown", "## 4. Bilan"),
    ("code", r'''
all_done = True
for d in ["mimic", "ptbxl", "lsas"]:
    p = f"{CACHE}/{d}"
    if not os.path.exists(f"{p}/info.json"):
        print(f"{d:6s}  ABSENT"); all_done = False; continue
    i = info(p)
    complete = i.get("complete", True)
    all_done &= complete
    print(f"{d:6s}  {i.get('n_done', i['n'])}/{i['n']} traités, {i['n_valid']} valides  "
          f"{'complet' if complete else 'INCOMPLET'}")
    print("        prévalences :", {k: round(v, 3) for k, v in i["label_prevalence"].items()})
    if d == "mimic" and not complete:
        rate = i.get("last_session_rate_per_s", 0)
        left = i["n"] - i["n_done"]
        print(f"        débit {rate} ECG/s -> reste ~{left / max(rate, 1e-6) / 3600:.1f} h de calcul")
!du -sh {CACHE}/*
print("=" * 60)
print("PRÉPARATION TERMINÉE : créez le dataset depuis l'onglet Output." if all_done else
      "À REPRENDRE : ajoutez la sortie de cette version en Input puis relancez Save & Run All.")
print("=" * 60)
'''),
    ("markdown", "## 5. Audit des labels MIMIC\nPhrases des comptes rendus qui déclenchent chaque label : copiez cette sortie pour vérification."),
    ("code", r'''
if os.path.exists(f"{CACHE}/mimic/meta.csv"):
    !python scripts/audit_labels.py --cache {CACHE}/mimic --top 10
'''),
]

TRAIN = [
    ("markdown", r'''
# Notebook 2 — Entraînement et évaluation externe

**Réglages Kaggle :**
- *Accelerator* : **GPU T4** (évitez le P100 : les versions récentes de PyTorch ne le prennent plus en charge)
- *Internet* : **On** (téléchargement de ClinicalBERT)
- *Add Input* : le dataset du code **et** le dataset `ecg-fusion-cache` (sortie du notebook 1)

Une session Kaggle est limitée à 12 h. L'entraînement s'arrête proprement à `MAX_HOURS`
et sauvegarde `last.pt`. **Pour continuer, il n'y a rien à modifier dans ce notebook** :
1. Ouvrez ce notebook en mode édition (*Edit*).
2. *Add Input* → *Your Work* → choisissez **ce même notebook** (sa dernière version).
3. *Save Version → Save & Run All*.

Le notebook détecte tout seul `last.pt` et reprend à l'époque suivante. Répétez jusqu'à voir
**ENTRAÎNEMENT TERMINÉ** ; l'évaluation externe se lance alors automatiquement.
'''),
    ("code", SETUP),
    ("code", r'''
import json
meta = glob.glob("/kaggle/input/**/cache/mimic/meta.csv", recursive=True)
assert meta, "Cache introuvable : ajoutez la sortie du notebook 1 comme Input."
CACHE = os.path.dirname(os.path.dirname(meta[0]))
_i = json.load(open(f"{CACHE}/mimic/info.json"))
if not _i.get("complete", True):
    print(f"ATTENTION : cache MIMIC incomplet ({_i['n_done']}/{_i['n']}). Terminez le notebook 1 avant d'entraîner.")
CONFIG = "configs/kaggle_v2.yaml"   # v1 : configs/kaggle.yaml
RUN_NAME = "fusion_v2"              # changer de nom pour chaque nouvelle expérience
RUN = f"/kaggle/working/runs/{RUN_NAME}"
MAX_HOURS = 10      # marge pour la calibration et la sauvegarde avant la limite des 12 h
# Copie des caches sur le disque local de la machine. Lire des ECG au hasard directement dans
# /kaggle/input est extrêmement lent (le 1er test : ~25 min pour 12 000 ECG) ; une copie
# séquentielle prend quelques minutes, ensuite chaque époque est rapide.
import time
LOCAL = "/tmp/cache_local"          # hors /kaggle/working : non sauvegardé, recréé à chaque session
def copy_cache(name):
    src, dst = f"{CACHE}/{name}", f"{LOCAL}/{name}"
    os.makedirs(dst, exist_ok=True)
    t0, total = time.time(), 0
    for f in sorted(os.listdir(src)):
        s_, d_ = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(d_) and os.path.getsize(d_) == os.path.getsize(s_):
            continue
        shutil.copyfile(s_, d_); total += os.path.getsize(d_)
    dt = time.time() - t0
    print(f"  {name:6s} {total / 1e9:5.1f} Go copiés en {dt / 60:4.1f} min ({total / 1e6 / max(dt, 1e-6):.0f} Mo/s)")
need = sum(os.path.getsize(f"{CACHE}/{d}/signals.npy") for d in ["mimic", "ptbxl", "lsas"])
free = shutil.disk_usage("/tmp").free
print(f"Disque local libre : {free / 1e9:.0f} Go, besoin : {need / 1e9:.1f} Go")
if free > 1.15 * need:
    for d in ["mimic", "ptbxl", "lsas"]:
        copy_cache(d)
    SRC = LOCAL
else:
    print("ATTENTION : pas assez de place pour la copie locale ; lecture directe (lente) depuis l'Input")
    SRC = CACHE
# Labels MIMIC recalculés avec les règles actuelles du code (quelques secondes)
MIMIC = "/tmp/cache_labels/mimic"
!python scripts/audit_labels.py --cache {SRC}/mimic --relabel --link --quiet --out {MIMIC}
# LSAS : labels v2 (liste SNOMED-CT complète) recalculés depuis lsas_codes.csv (sortie du notebook 8)
_codes = sorted(glob.glob("/kaggle/input/**/lsas_codes.csv", recursive=True)
                + glob.glob("/kaggle/working/lsas_codes.csv"))
if not _codes:
    # pas de sortie du notebook 8 : lecture directe des en-têtes LSAS sur PhysioNet (10-15 min, Internet On).
    # Le fichier est gardé dans la sortie de ce notebook : les reprises suivantes le retrouvent.
    print("lsas_codes.csv absent des Inputs : lecture des en-têtes LSAS sur PhysioNet (10-15 min)...")
    !python -c "import sys; sys.path.insert(0, 'scripts'); from lsas_code_check import fetch_codes; fetch_codes().to_csv('/kaggle/working/lsas_codes.csv', index=False)"
    _codes = glob.glob("/kaggle/working/lsas_codes.csv")
if not _codes:
    raise SystemExit("ARRÊT : codes LSAS impossibles à obtenir. Vérifiez que Internet est sur On.")
print("Codes LSAS :", _codes[0])
LSAS = "/tmp/cache_labels/lsas"
!python scripts/relabel_cache.py --dataset lsas --cache {SRC}/lsas --out {LSAS} --codes {_codes[0]}
PATHS = (f"data.train_cache={MIMIC} "
         f"data.eval_sets.mimic_internal.cache={MIMIC} "
         f"data.eval_sets.mimic_val.cache={MIMIC} data.eval_sets.mimic_val.split=val "
         f"data.eval_sets.ptbxl_external.cache={SRC}/ptbxl "
         f"data.eval_sets.lsas_external.cache={LSAS}")

# Reprise automatique : cherche un last.pt dans les Inputs (sortie d'une version précédente)
prev = sorted(glob.glob(f"/kaggle/input/**/runs/{RUN_NAME}/last.pt", recursive=True))
PREV = os.path.dirname(prev[0]) if prev else None
print("Cache :", CACHE)
if PREV:
    st = json.load(open(f"{PREV}/status.json")) if os.path.exists(f"{PREV}/status.json") else {}
    print(f"REPRISE depuis {PREV} (dernière époque : {st.get('last_epoch', '?')}, "
          f"meilleur val : {st.get('best_val', float('nan')):.4f})")
else:
    print("NOUVEL ENTRAÎNEMENT (aucun last.pt trouvé dans les Inputs)")
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    ("markdown", "## Test rapide (désactivé par défaut : `RUN_TEST = True` pour le relancer)"),
    ("code", r'''
RUN_TEST = False   # mettre True pour refaire un essai de 1 époque sur 2 000 ECG
if PREV is None and RUN_TEST:
    !python scripts/train.py --config {CONFIG} --set {PATHS} output_dir=/kaggle/working/runs/test train.epochs=1 data.max_train_records=2000
else:
    print("Test ignoré")
'''),
    ("markdown", "## Entraînement (nouveau ou reprise, automatique)"),
    ("code", r'''
RESUME = f"--resume {PREV}" if PREV else ""
!python scripts/train.py --config {CONFIG} --max_hours {MAX_HOURS} {RESUME} --set {PATHS} output_dir={RUN}
STATUS = json.load(open(f"{RUN}/status.json"))
print("=" * 60)
if STATUS["finished"]:
    print(f"ENTRAÎNEMENT TERMINÉ ({STATUS['stop_reason']}) - époque {STATUS['last_epoch']}, "
          f"meilleur val = {STATUS['best_val']:.4f}")
else:
    print(f"À REPRENDRE : arrêt à l'époque {STATUS['last_epoch']} (budget de temps). "
          "Ajoutez la sortie de cette version en Input puis relancez.")
print("=" * 60)
'''),
    ("markdown", "## Évaluation interne + externe (seulement quand l'entraînement est terminé)"),
    ("code", r'''
if STATUS["finished"]:
    !python scripts/evaluate.py --run {RUN} --set {PATHS}
else:
    print("Entraînement non terminé : évaluation reportée à la prochaine version.")
'''),
    ("markdown", "## Analyse a posteriori (CPU, quelques minutes) : prévalences, calibration, gating, DeLong"),
    ("code", r'''
if STATUS["finished"]:
    !python scripts/analyze_preds.py --run {RUN}
'''),
    ("markdown", "## Résultats"),
    ("code", r'''
import pandas as pd
pd.set_option("display.width", 200)
if not STATUS["finished"]:
    print("Pas encore de résultats : entraînement à reprendre (voir l'en-tête du notebook).")
else:
    s = pd.read_csv(f"{RUN}/eval/summary.csv")
    display(s[["eval_set", "scenario", "n", "macro_auroc", "macro_auroc_ci_low", "macro_auroc_ci_high",
               "macro_f1", "macro_ece", "rare_macro_auroc"]].round(3))
    pl = pd.read_csv(f"{RUN}/eval/per_label.csv")
    display(pl[pl.scenario == "no_text"].pivot(index="label", columns="eval_set", values="auroc").round(3))
    display(pd.read_csv(f"{RUN}/eval/modality_contribution.csv").round(3))
'''),
    ("code", r'''
# courbe d'apprentissage (toutes les sessions cumulées)
h = pd.read_csv(f"{RUN}/history.csv")
ax = h.plot(x="epoch", y=[c for c in h.columns if c.startswith("val_auroc")], marker="o", figsize=(7, 4))
ax.set_ylabel("AUROC macro (validation MIMIC)"); ax.grid(alpha=.3)
'''),
]

ANALYSIS = [
    ("markdown", r'''
# Notebook 3 — Analyse des résultats (CPU, sans quota GPU)

**Réglages :** *Accelerator* **None** ; *Internet* non nécessaire.
*Add Input* : le dataset du code **et** la sortie du (des) notebook(s) d'entraînement.
Si plusieurs runs sont présents (ex. v1 et v2), ils sont analysés et comparés.

Pour chaque run : adaptation aux prévalences (EM et BBSE, sans label externe), recalibration
few-shot (200 ECG étiquetés), seuil oracle, courbes de fiabilité, gating par accord
texte / sans texte, tests de DeLong appariés et IC bootstrap.
'''),
    ("code", SETUP),
    ("code", r'''
import json, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

def read_or_empty(path):
    """Lit un CSV d'analyse ; vide ou absent (ex. run « signal seul » sans texte) -> tableau vide."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
runs = sorted({os.path.dirname(os.path.dirname(p))
               for p in glob.glob("/kaggle/input/**/runs/*/eval/summary.csv", recursive=True)})
runs = [r for r in runs if not r.endswith("/test")]
assert runs, "Aucun run trouvé : ajoutez la sortie du notebook d'entraînement en Input."
for r in runs:
    print(" -", r)
'''),
    ("code", r'''
OUT = "/kaggle/working/analysis"
for r in runs:
    name = os.path.basename(r)
    print("=" * 30, name, "=" * 30)
    !python scripts/analyze_preds.py --run {r} --out {OUT}/{name}
'''),
    ("markdown", "## Comparaison des runs (AUROC macro)"),
    ("code", r'''
tab = []
for r in runs:
    s = pd.read_csv(f"{r}/eval/summary.csv")
    s["run"] = os.path.basename(r)
    tab.append(s)
tab = pd.concat(tab)
keep = ["full", "no_text", "signal_only", "leadI_no_text", "text_only"]
display(tab[tab.scenario.isin(keep)].pivot_table(index=["eval_set", "scenario"], columns="run",
        values="macro_auroc").round(3))
'''),
    ("markdown", "## Seuils et prévalences : brut vs EM / BBSE vs few-shot vs oracle"),
    ("code", r'''
for r in runs:
    name = os.path.basename(r)
    d = read_or_empty(f"{OUT}/{name}/label_shift_summary.csv")
    print(name)
    if d.empty or "scenario" not in d.columns:
        print("  (non applicable pour ce run)"); continue
    cols = [c for c in ["eval_set", "scenario", "prev_true", "prev_em", "prev_bbse", "sens_raw", "sens_em",
                        "sens_bbse", "sens_fewshot", "sens_oracle", "f1_raw", "f1_em", "f1_bbse",
                        "f1_fewshot", "f1_oracle", "ece_raw", "ece_platt"] if c in d.columns]
    display(d[d.scenario.isin(["full", "no_text"])][cols].round(3))
'''),
    ("markdown", "## Gating par accord et tests statistiques"),
    ("code", r'''
for r in runs:
    name = os.path.basename(r)
    print(name)
    for f in ["gating.csv", "delta_macro_auroc_bootstrap.csv"]:
        t = read_or_empty(f"{OUT}/{name}/{f}")
        display(t.round(4)) if not t.empty else print(f"  {f} : non applicable pour ce run")
'''),
    ("markdown", "## Figures de calibration pour l'article (externes ; PNG et PDF dans analysis/<run>/)"),
    ("code", r'''
for r in runs:
    name = os.path.basename(r)
    for f in sorted(glob.glob(f"{OUT}/{name}/fig_calibration_*.png")):
        print(name, os.path.basename(f)); display(Image(f))
'''),
]

# ---------------------------------------------------------------------------------------
# Notebook 4 : file d'attente d'expériences (ablations, graines), reprenable
# ---------------------------------------------------------------------------------------
_setup_cell = next(c for t, c in TRAIN if "def copy_cache" in c)
_lines = _setup_cell.splitlines()
_end = next(i for i, l in enumerate(_lines) if 'lsas_external.cache={LSAS}")' in l)
CACHE_SETUP = "\n".join(l for l in _lines[:_end + 1]
                        if not l.startswith(("CONFIG =", "RUN_NAME =", "RUN =", "MAX_HOURS =")))

QUEUE = [
    ("markdown", r"""
# Notebook 4 — File d'attente d'expériences (ablations et graines)

**Réglages :** *Accelerator* **GPU T4** ; *Internet* **On** ;
*Input* : le dataset du code + la sortie du notebook 1 (cache).

Le notebook enchaîne les expériences de `EXPERIMENTS` (entraînement → évaluation → analyse),
s'arrête proprement avant la limite des 12 h et affiche **« À REPRENDRE »** s'il reste du travail.

**Pour continuer :** *Edit* → *Add Input* → *Your Work* → sortie de **ce même notebook**
(dernière version) → *Save Version → Save & Run All*. Les expériences terminées sont recopiées,
celle en cours reprend à son époque, les suivantes démarrent. Répétez jusqu'à
**« FILE D'ATTENTE TERMINÉE »**. Ne gardez en Input que la sortie de la version la plus récente.
"""),
    ("code", SETUP),
    ("code", CACHE_SETUP),
    ("markdown", r"""
## Expériences
Ordre = priorité. Chaque ligne : nom, configuration de base, surcharges (`--set`).
Le run de référence (`fusion_v2`, graine 42) vient du notebook 2 : il n'est pas refait ici.
"""),
    ("code", r"""
V2 = "configs/kaggle_v2.yaml"
EXPERIMENTS = [
    # nom                configuration   surcharges
    ("v2_amplitude",     V2, ["data.signal_norm=fixed"]),                 # amplitude en mV conservée
    ("v2_seed1",         V2, ["seed=1"]),
    ("v2_seed2",         V2, ["seed=2"]),
    ("signal_seul",      V2, ["train.train_scenario=signal_only", "train.val_scenarios=[signal_only]",
                              "model.text_encoder=light",
                              "eval.scenarios=[signal_only,limb6_signal_only,leadI_signal_only]"]),
    ("sans_abandon",     V2, ["train.p_drop_text=0", "train.p_drop_features=0", "train.p_drop_leads=0"]),
    ("bce",              V2, ["train.loss=bce"]),
]
SESSION_HOURS = 11.0   # limite Kaggle 12 h
EVAL_HOURS = 0.75      # évaluation + analyse d'une expérience
"""),
    ("markdown", "## Exécution de la file"),
    ("code", r"""
import json, shlex, time
T0 = time.time()
WORK = "/kaggle/working/runs"
os.makedirs(WORK, exist_ok=True)

def hours_left():
    return SESSION_HOURS - (time.time() - T0) / 3600

def status(d):
    p = f"{d}/status.json"
    return json.load(open(p)) if d and os.path.exists(p) else None

def previous(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/runs/{name}/config.yaml", recursive=True))
    return os.path.dirname(hits[0]) if hits else None

def copy_run(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("last.pt") if (status(src) or {}).get("finished") else None)

paths = shlex.split(PATHS)
report, stopped = [], False
for name, conf, extra in EXPERIMENTS:
    run, prev = f"{WORK}/{name}", previous(name)
    if prev and not os.path.exists(run):
        copy_run(prev, run)
        print(f"[{name}] repris depuis {prev}")
    st = status(run)
    done_eval = os.path.exists(f"{run}/eval/summary.csv")
    # run évalué avant le réétiquetage LSAS (labels v2) : on remplace les labels LSAS à partir
    # des prédictions sauvegardées (CPU, quelques minutes), puis l'analyse est refaite
    if (done_eval and globals().get("LSAS", "").startswith("/tmp/cache_labels")
            and not os.path.exists(f"{run}/eval/labels_v2_lsas_external.done")):
        !python scripts/relabel_eval.py --run {run} --set lsas_external --cache {LSAS}
    done_ana = os.path.exists(f"{run}/analysis/gating.csv")
    if st and st["finished"] and done_eval and done_ana:
        report.append((name, "terminée")); continue

    # 1) entraînement (nouveau ou reprise)
    if not (st and st["finished"]):
        budget = hours_left() - EVAL_HOURS
        if budget < 0.5:
            stopped = True; report.append((name, "pas commencée (temps)")); break
        resume = ["--resume", run] if os.path.exists(f"{run}/last.pt") else []
        cmd = shlex.join(["python", "scripts/train.py", "--config", conf, "--max_hours", f"{budget:.3f}",
                          *resume, "--set", *paths, f"output_dir={run}", *extra])
        print(f"\n===== {name} : entraînement ({'reprise' if resume else 'nouveau'}), budget {budget:.1f} h")
        !{cmd}
        st = status(run)
        if not (st and st["finished"]):
            stopped = True; report.append((name, f"entraînement en cours (époque {(st or {}).get('last_epoch', '?')})")); break

    # 2) évaluation
    if not os.path.exists(f"{run}/eval/summary.csv"):
        if hours_left() < EVAL_HOURS:
            stopped = True; report.append((name, "entraînée, évaluation à faire")); break
        cmd = shlex.join(["python", "scripts/evaluate.py", "--run", run, "--set", *paths, *extra])
        print(f"\n===== {name} : évaluation")
        !{cmd}
        if globals().get("LSAS", "").startswith("/tmp/cache_labels") and os.path.exists(f"{run}/eval/summary.csv"):
            open(f"{run}/eval/labels_v2_lsas_external.done", "w").write("ok\n")

    # 3) analyse (CPU)
    if not os.path.exists(f"{run}/analysis/gating.csv"):
        !python scripts/analyze_preds.py --run {run}
    if os.path.exists(f"{run}/last.pt"):
        os.remove(f"{run}/last.pt")   # inutile une fois terminé (place disque)
    report.append((name, "terminée"))

print("\n" + "=" * 60)
for name, s_ in report:
    print(f"  {name:15s} {s_}")
left = [n for n, _, _ in EXPERIMENTS if n not in {r for r, s_ in report if s_ == 'terminée'}]
print("=" * 60)
print("FILE D'ATTENTE TERMINÉE" if not left else
      f"À REPRENDRE ({len(left)} expérience(s) restante(s)) : ajoutez la sortie de cette version en Input et relancez.")
print("=" * 60)
"""),
    ("markdown", "## Récapitulatif des expériences terminées (AUROC macro)"),
    ("code", r"""
import pandas as pd
pd.set_option("display.width", 220)
rows = []
ref = sorted(glob.glob("/kaggle/input/**/runs/fusion_v2/eval/summary.csv", recursive=True))
sources = ([("fusion_v2 (graine 42)", ref[0])] if ref else []) + \
          [(n, f"{WORK}/{n}/eval/summary.csv") for n, _, _ in EXPERIMENTS]
for name, f in sources:
    if os.path.exists(f):
        s_ = pd.read_csv(f); s_["run"] = name; rows.append(s_)
if rows:
    tab = pd.concat(rows)
    keep = ["full", "no_text", "signal_only"]
    display(tab[tab.scenario.isin(keep)].pivot_table(index="run", columns=["eval_set", "scenario"],
                                                     values="macro_auroc").round(3))
    seeds = tab[tab.run.str.startswith(("fusion_v2", "v2_seed"))]
    if seeds.run.nunique() >= 2:
        print("Graines (moyenne ± écart-type) :")
        g = seeds[seeds.scenario.isin(keep)].groupby(["eval_set", "scenario"])["macro_auroc"]
        display((g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).unstack())
else:
    print("Aucune expérience terminée pour l'instant.")
"""),
]

# ---------------------------------------------------------------------------------------
# Notebook 5 : figures de l'article (CPU, quelques minutes)
# ---------------------------------------------------------------------------------------
FIGS = [
    ("markdown", r"""
# Notebook 5 — Figures de l'article (CPU)

**Réglages :** *Accelerator* **None**. *Input* : le dataset du code + la sortie du notebook 4
(ou du notebook 9 : run final `L2_final_s42`, son évaluation et son analyse).

Produit dans `/kaggle/working/figures/` (PDF vectoriel pour la revue + PNG) :
- `fig1_architecture` — architecture et gating par centre ;
- `fig2_per_label_effects` — Δ AUROC par pathologie sur PTB-XL (texte ; dérivation I seule) ;
- `fig3_site_gate_disagreement` — part des ECG au-dessus de chaque niveau de désaccord (τ90, τ95) ;
- `fig4_calibration_ptbxl` et `figS4_calibration_lsas` — courbes de fiabilité (brut / Platt MIMIC /
  Platt sur 200 ECG locaux), recalculées ici à partir des prédictions, en anglais.
"""),
    ("code", SETUP),
    ("code", r"""
RUN_NAME = "L2_final_s42"          # modèle final, labels v2 (graine 42) ; L2_final_s1 / L2_final_s2
hits = sorted(glob.glob(f"/kaggle/input/**/runs/{RUN_NAME}/eval/summary.csv", recursive=True))
assert hits, f"Run {RUN_NAME} introuvable : ajoutez la sortie du notebook 4 en Input."
RUN = os.path.dirname(os.path.dirname(hits[0]))
OUT = "/kaggle/working/figures"
print("Run :", RUN)
!python scripts/fig_architecture.py --out {OUT}
!python scripts/make_figures.py --run {RUN} --out {OUT}
print(sorted(os.listdir(OUT)))
"""),
    ("code", r"""
from IPython.display import Image, display
for f in sorted(glob.glob(f"{OUT}/*.png")):
    print(os.path.basename(f)); display(Image(f, width=900))
"""),
]

# ---------------------------------------------------------------------------------------
# Notebook 6 : contrôle STT_ISCH avec la dérivation I seule (CPU, quelques minutes)
# ---------------------------------------------------------------------------------------
STT = [
    ("markdown", r"""
# Notebook 6 — Pourquoi STT_ISCH s'améliore-t-il avec la dérivation I seule ? (CPU)

**Réglages :** *Accelerator* **None** ; *Internet* **On** (pour télécharger `ptbxl_database.csv`,
13 Mo, depuis PhysioNet — sans Internet, la partie C est simplement sautée).
*Inputs* : le dataset du code, la sortie du notebook 4 (runs) **et** la sortie du notebook 1 (cache, qui
contient `ptbxl/meta.csv`).

Le notebook n'entraîne rien : il relit les prédictions `no_text` et `leadI_no_text` déjà sauvegardées.
Résultats dans `/kaggle/working/stt_check/` : trois tableaux CSV et la figure `figS5_stt_leadI_check`.
"""),
    ("code", SETUP),
    ("code", r"""
RUN_NAMES = ["L2_final_s42", "L2_final_s1", "L2_final_s2"]   # les 3 graines du modèle final
runs = []
for n in RUN_NAMES:
    hits = sorted(glob.glob(f"/kaggle/input/**/runs/{n}/eval/summary.csv", recursive=True))
    if hits:
        runs.append(os.path.dirname(os.path.dirname(hits[0])))
    else:
        print("introuvable :", n)
assert runs, "Aucun run trouvé : ajoutez la sortie du notebook 4 en Input."
metas = sorted(glob.glob("/kaggle/input/**/ptbxl/meta.csv", recursive=True))
CACHE = os.path.dirname(metas[0]) if metas else ""
print("Runs :", runs); print("Cache PTB-XL :", CACHE or "introuvable (partie C sautée)")
cmd = f"python scripts/stt_lead_check.py --runs {' '.join(runs)} --out /kaggle/working/stt_check"
if CACHE:
    cmd += f" --ptbxl_cache {CACHE}"
!{cmd}
"""),
    ("code", r"""
from IPython.display import Image, display
display(Image("/kaggle/working/stt_check/figS5_stt_leadI_check.png", width=800))
"""),
]

# ---------------------------------------------------------------------------------------
# Notebook 7 : audit manuel de 200 comptes rendus MIMIC (CPU, 1 minute)
# ---------------------------------------------------------------------------------------
AUDIT = [
    ("markdown", r"""
# Notebook 7 — Audit manuel des étiquettes MIMIC (CPU)

**Réglages :** *Accelerator* **None**. *Inputs* : le dataset du code et la sortie du notebook 1 (cache).

1. La cellule 2 tire 200 comptes rendus (échantillon stratifié par pathologie) et écrit dans
   `/kaggle/working/audit/` :
   - `grille_audit_mimic.xlsx` : la grille à remplir (onglet **guide** = consignes, onglet **grille** = à annoter) ;
   - `audit_key.csv` : les étiquettes des règles, **à ne pas ouvrir avant d'avoir annoté** (évaluation en aveugle).
2. Téléchargez les deux fichiers (panneau *Output*), remplissez la grille dans Excel ou LibreOffice
   (environ 45 min), puis envoyez-moi la grille remplie **et** `audit_key.csv` : je calcule les résultats.
3. (Facultatif) Cellule 3 : calcul sur Kaggle si vous ajoutez la grille remplie comme dataset.
"""),
    ("code", SETUP),
    ("code", r"""
metas = sorted(glob.glob("/kaggle/input/**/mimic/meta.csv", recursive=True))
assert metas, "meta.csv MIMIC introuvable : ajoutez la sortie du notebook 1 en Input."
CACHE = os.path.dirname(metas[0]); print("Cache MIMIC :", CACHE)
!python scripts/audit_sample.py --cache {CACHE} --out /kaggle/working/audit
"""),
    ("code", r"""
# Facultatif : calcul des résultats sur Kaggle (grille remplie ajoutée en Input, nom contenant "rempli")
filled = sorted(glob.glob("/kaggle/input/**/*rempli*.xlsx", recursive=True))
if filled:
    !python scripts/audit_score.py --grid {filled[0]} --key /kaggle/working/audit/audit_key.csv --out /kaggle/working/audit
else:
    print("Pas de grille remplie en Input : envoyez-la-moi avec audit_key.csv.")
"""),
]

# ---------------------------------------------------------------------------------------
# Notebook 8 : vérification des codes SNOMED de LSAS (CPU + Internet, 10-15 min)
# ---------------------------------------------------------------------------------------
LSASCHK = [
    ("markdown", r"""
# Notebook 8 — Vérification de la correspondance SNOMED de LSAS (CPU, Internet)

**Réglages :** *Accelerator* **None**, *Internet* **On**. *Inputs* : le dataset du code, la sortie du
notebook 1 (cache, qui contient `lsas/meta.csv`) et la sortie du notebook 4 (runs).

Le script lit les codes SNOMED des 45 152 en-têtes LSAS sur PhysioNet (10 à 15 min), compare notre
correspondance à la liste officielle, puis recalcule les AUROC LSAS avec la correspondance corrigée à partir
des prédictions déjà sauvegardées (aucun réentraînement). Résultats dans `/kaggle/working/lsas_check/`.
"""),
    ("code", SETUP),
    ("code", r"""
runs = []
for n in ["L2_final_s42", "L2_final_s1", "L2_final_s2"]:
    hits = sorted(glob.glob(f"/kaggle/input/**/runs/{n}/eval/summary.csv", recursive=True))
    if hits:
        runs.append(os.path.dirname(os.path.dirname(hits[0])))
metas = sorted(glob.glob("/kaggle/input/**/lsas/meta.csv", recursive=True))
assert metas, "lsas/meta.csv introuvable : ajoutez la sortie du notebook 1 en Input."
print("Runs :", runs); print("Cache LSAS :", os.path.dirname(metas[0]))
prev = sorted(glob.glob("/kaggle/working/lsas_check/lsas_codes.csv") + glob.glob("/kaggle/input/**/lsas_codes.csv", recursive=True))
extra = f"--codes {prev[0]}" if prev else ""   # codes déjà extraits : pas de nouveau téléchargement
print("Codes :", prev[0] if prev else "téléchargement depuis PhysioNet")
!python scripts/lsas_code_check.py --lsas_cache {os.path.dirname(metas[0])} --runs {' '.join(runs)} --out /kaggle/working/lsas_check {extra}
"""),
]

out = Path(__file__).parent
(out / "kaggle_1_preparation_donnees.ipynb").write_text(json.dumps(nb(PREP), indent=1, ensure_ascii=False))
(out / "kaggle_2_entrainement_evaluation.ipynb").write_text(json.dumps(nb(TRAIN), indent=1, ensure_ascii=False))
(out / "kaggle_4_file_experiences.ipynb").write_text(json.dumps(nb(QUEUE), indent=1, ensure_ascii=False))
(out / "kaggle_8_codes_lsas.ipynb").write_text(json.dumps(nb(LSASCHK), indent=1, ensure_ascii=False))
(out / "kaggle_7_audit_mimic.ipynb").write_text(json.dumps(nb(AUDIT), indent=1, ensure_ascii=False))
(out / "kaggle_6_controle_stt.ipynb").write_text(json.dumps(nb(STT), indent=1, ensure_ascii=False))
(out / "kaggle_5_figures.ipynb").write_text(json.dumps(nb(FIGS), indent=1, ensure_ascii=False))
(out / "kaggle_3_analyse.ipynb").write_text(json.dumps(nb(ANALYSIS), indent=1, ensure_ascii=False))

# ---------------------------------------------------------------------------------------
# Notebook 9 : réentraînement complet avec les labels v2 (nomenclature AHA/ACCF/HRS)
# ---------------------------------------------------------------------------------------
_queue_src = next(c for t, c in QUEUE if "EXPERIMENTS = [" in c)
_exec_cells = [(t, c) for t, c in QUEUE if t == "code" and "def hours_left" in c]
_tail = QUEUE[QUEUE.index(_exec_cells[0]):]
RETRAIN = [
    ("markdown", r"""
# Notebook 9 — Réentraînement complet, labels v2 (nomenclature AHA/ACCF/HRS)

Les règles d'étiquetage suivent maintenant les recommandations AHA/ACCF/HRS (code ≥ 2026.11.20).
Tous les modèles de l'article sont réentraînés avec ces labels. Les noms commencent par `L2_`
pour ne jamais reprendre par erreur un ancien entraînement.

**Réglages :** *Accelerator* **GPU T4** ; *Internet* **On** ;
*Input* : (1) la **nouvelle version** du dataset du code, (2) la sortie du notebook 1 (cache),
(3) la sortie du notebook 8 (contient `lsas_codes.csv`).

**Pour continuer après 12 h :** *Add Input* → *Your Work* → sortie de **ce notebook** (dernière version),
puis *Save & Run All*. Répétez jusqu'à **« FILE D'ATTENTE TERMINÉE »**.
L'ordre de la liste est l'ordre de priorité : les 3 premiers suffisent pour les résultats principaux.
"""),
    ("code", SETUP),
    ("code", CACHE_SETUP),
    ("code", r"""
import ecgfusion; assert ecgfusion.__version__ >= "2026.11.20", "Ancienne version du code en Input"
V1, V2 = "configs/kaggle.yaml", "configs/kaggle_v2.yaml"
AMP = "data.signal_norm=fixed"
EXPERIMENTS = [
    # --- priorité 1 : modèle final, 3 graines (résultats principaux, gate, calibration, dérivations)
    ("L2_final_s42",   V2, [AMP]),
    ("L2_final_s1",    V2, [AMP, "seed=1"]),
    ("L2_final_s2",    V2, [AMP, "seed=2"]),
    # --- priorité 2 : ablations clés
    ("L2_sans_abandon", V2, [AMP, "train.p_drop_text=0", "train.p_drop_features=0", "train.p_drop_leads=0"]),
    ("L2_bce_s42",     V2, [AMP, "train.loss=bce"]),
    ("L2_bce_s1",      V2, [AMP, "train.loss=bce", "seed=1"]),
    ("L2_bce_s2",      V2, [AMP, "train.loss=bce", "seed=2"]),
    ("L2_signal_seul", V2, [AMP, "train.train_scenario=signal_only", "train.val_scenarios=[signal_only]",
                            "model.text_encoder=light",
                            "eval.scenarios=[signal_only,limb6_signal_only,leadI_signal_only]"]),
    # --- priorité 3 : normalisation par dérivation (z-score) et masquage partiel du texte (v1)
    ("L2_z_s42",       V2, []),
    ("L2_z_s1",        V2, ["seed=1"]),
    ("L2_z_s2",        V2, ["seed=2"]),
    ("L2_z_bce",       V2, ["train.loss=bce"]),
    ("L2_v1_masquage", V1, []),
]
SESSION_HOURS = 11.0
EVAL_HOURS = 0.75
"""),
] + _tail
(out / "kaggle_9_reentrainement_labels_v2.ipynb").write_text(json.dumps(nb(RETRAIN), indent=1, ensure_ascii=False))

# ---------------------------------------------------------------------------------------
# Notebook 10 : audit de validation B (200 nouveaux comptes rendus, labels v2)
# ---------------------------------------------------------------------------------------
AUDIT_B = [
    ("markdown", r"""
# Notebook 10 — Audit de validation B (CPU, 1 minute)

Les règles v2 ont été corrigées à partir de l'audit A : l'audit A ne peut donc plus servir de
validation indépendante. Ce notebook tire **200 nouveaux comptes rendus** (aucun de l'audit A),
avec les consignes alignées sur les recommandations AHA/ACCF/HRS.

**Réglages :** *Accelerator* **None**. *Inputs* : la nouvelle version du dataset du code et la sortie du notebook 1.

Téléchargez `grille_audit_B.xlsx` (à faire remplir par le médecin **sans** lui montrer les étiquettes)
et `audit_key_B.csv` (à garder pour moi, ne pas ouvrir).
"""),
    ("code", SETUP),
    ("code", r"""
import ecgfusion; assert ecgfusion.__version__ >= "2026.11.20", "Ancienne version du code en Input"
metas = sorted(glob.glob("/kaggle/input/**/mimic/meta.csv", recursive=True))
assert metas, "meta.csv MIMIC introuvable : ajoutez la sortie du notebook 1 en Input."
CACHE = os.path.dirname(metas[0]); print("Cache MIMIC :", CACHE)
!python scripts/audit_sample.py --cache {CACHE} --out /kaggle/working/audit_B --exclude audit/audit_key_A.csv --prefix B --seed 2027
os.rename("/kaggle/working/audit_B/grille_audit_mimic.xlsx", "/kaggle/working/audit_B/grille_audit_B.xlsx")
os.rename("/kaggle/working/audit_B/audit_key.csv", "/kaggle/working/audit_B/audit_key_B.csv")
from IPython.display import FileLink, display
os.chdir("/kaggle/working")   # FileLink attend un chemin relatif à /kaggle/working
for f in ["grille_audit_B.xlsx", "audit_key_B.csv"]:
    display(FileLink(f"audit_B/{f}"))
"""),
]
(out / "kaggle_10_audit_validation_B.ipynb").write_text(json.dumps(nb(AUDIT_B), indent=1, ensure_ascii=False))

# ---------------------------------------------------------------------------------------
# Notebook 11 : figures finales, contrôle STT et export des résultats (labels v2)
# ---------------------------------------------------------------------------------------
FINAL = [
    ("markdown", r"""
# Notebook 11 — Figures, contrôle STT et export des résultats (CPU, ~30 min)

Remplace les notebooks 3, 5 et 6 pour la version finale (labels v2).

**Réglages :** *Accelerator* **None** ; *Internet* **On** (téléchargement de `ptbxl_database.csv`).
*Inputs* : le dataset du code, la sortie du notebook 1 (cache) et la **sortie finale du notebook 9**
(celle qui affiche « FILE D'ATTENTE TERMINÉE »).

À la fin, téléchargez **un seul fichier** : `resultats_finaux.zip` (lien affiché en bas).
"""),
    ("code", SETUP),
    ("code", r"""
import ecgfusion, json, zipfile
assert ecgfusion.__version__ >= "2026.11.20", "Ancienne version du code en Input"
runs = {os.path.basename(os.path.dirname(os.path.dirname(p))): os.path.dirname(os.path.dirname(p))
        for p in glob.glob("/kaggle/input/**/runs/L2_*/eval/summary.csv", recursive=True)}
print(len(runs), "runs :", sorted(runs))
assert len(runs) == 13, "Il faut les 13 runs L2_* : ajoutez la sortie FINALE du notebook 9 en Input."
OUT = "/kaggle/working/final"
os.makedirs(OUT, exist_ok=True)
FINAL_RUNS = [runs[n] for n in ["L2_final_s42", "L2_final_s1", "L2_final_s2"]]
"""),
    ("code", r"""
# 1) Figures de l'article (modèle final, graine 42) : architecture, effets par pathologie, gate, calibration
!python scripts/fig_architecture.py --out {OUT}/figures
!python scripts/make_figures.py --run {FINAL_RUNS[0]} --out {OUT}/figures
"""),
    ("code", r"""
# 2) Contrôle STT_ISCH en dérivation I (3 graines du modèle final)
metas = sorted(glob.glob("/kaggle/input/**/cache/ptbxl/meta.csv", recursive=True))
PTB = os.path.dirname(metas[0]) if metas else None
cmd = ["python", "scripts/stt_lead_check.py", "--runs", *FINAL_RUNS, "--out", f"{OUT}/stt_check"]
if PTB:
    cmd += ["--ptbxl_cache", PTB]
!{shlex.join(cmd) if 'shlex' in globals() else ' '.join(cmd)}
""".replace("!{shlex.join(cmd) if 'shlex' in globals() else ' '.join(cmd)}", "import shlex\n!{shlex.join(cmd)}")),
    ("code", r"""
# 3) Export : tous les tableaux (CSV/JSON) des 13 runs + figures + contrôle STT, sans les prédictions brutes
keep_ext = (".csv", ".json", ".yaml", ".pdf", ".png", ".done")
zpath = "/kaggle/working/resultats_finaux.zip"
n = 0
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for name, r in sorted(runs.items()):
        for f in glob.glob(f"{r}/**/*", recursive=True):
            if os.path.isfile(f) and f.endswith(keep_ext) and "/reliability_" not in f:
                z.write(f, f"runs/{name}/" + os.path.relpath(f, r)); n += 1
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f):
            z.write(f, os.path.relpath(f, OUT)); n += 1
    for f in glob.glob("/kaggle/input/**/lsas_codes.csv", recursive=True)[:1]:
        z.write(f, "lsas_codes.csv"); n += 1
print(f"{n} fichiers -> {zpath} ({os.path.getsize(zpath) / 1e6:.1f} Mo)")
os.chdir("/kaggle/working")
from IPython.display import FileLink, display
display(FileLink("resultats_finaux.zip"))
"""),
]
(out / "kaggle_11_figures_export_final.ipynb").write_text(json.dumps(nb(FINAL), indent=1, ensure_ascii=False))



print("notebooks écrits dans", out)
