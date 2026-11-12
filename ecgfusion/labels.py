"""Espace de labels unifié et correspondances par jeu de données.

Les trois sources n'utilisent pas le même vocabulaire :
  * MIMIC-IV-ECG : énoncés textuels du compte rendu machine (anglais)
  * PTB-XL       : codes SCP-ECG (colonne ``scp_codes``)
  * LSAS-ECG     : codes SNOMED-CT dans l'en-tête WFDB (ligne ``#Dx:``)

On projette tout vers ``LABELS``. Lorsqu'une source n'annote pas
fiablement une classe, elle est marquée indisponible dans
``LABEL_AVAILABILITY`` et exclue des métriques pour ce jeu.

/!\\ Vérifiez les codes SNOMED contre ``ConditionNames_SNOMED-CT.csv``
fourni avec LSAS-ECG avant toute publication.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

LABELS: list[str] = [
    "NORM",      # ECG normal
    "AFIB",      # fibrillation / flutter atrial
    "AVB",       # bloc auriculo-ventriculaire (1er, 2e, 3e degré)
    "LVH",       # hypertrophie ventriculaire gauche
    "STT_ISCH",  # anomalies ST-T / ischémie
    "MI",        # infarctus (aigu ou ancien)
    "RBBB",      # bloc de branche droit (complet / incomplet)
    "LBBB",      # bloc de branche gauche (complet / incomplet)
    "FASC",      # hémibloc (antérieur / postérieur gauche)
    "STACH",     # tachycardie sinusale
    "SBRAD",     # bradycardie sinusale
]
NUM_LABELS = len(LABELS)
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}

# 1 = label fiable dans ce jeu, 0 = absent/peu fiable -> ignoré à l'évaluation
LABEL_AVAILABILITY: dict[str, list[int]] = {
    "mimic": [1] * NUM_LABELS,
    "ptbxl": [1] * NUM_LABELS,
    # LSAS : base orientée rythme ; ischémie et IDM mal représentés
    "lsas": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    "synthetic": [1] * NUM_LABELS,
}


def to_multi_hot(names: Iterable[str]) -> np.ndarray:
    y = np.zeros(NUM_LABELS, dtype=np.float32)
    for n in names:
        if n in LABEL_INDEX:
            y[LABEL_INDEX[n]] = 1.0
    return y


# --------------------------------------------------------------------------
# PTB-XL : codes SCP
# --------------------------------------------------------------------------
PTBXL_SCP_MAP: dict[str, str] = {
    "NORM": "NORM",
    "AFIB": "AFIB", "AFLT": "AFIB",
    "1AVB": "AVB", "2AVB": "AVB", "3AVB": "AVB",
    "LVH": "LVH",
    # ischémie / ST-T
    "ISC_": "STT_ISCH", "ISCAL": "STT_ISCH", "ISCIN": "STT_ISCH", "ISCIL": "STT_ISCH",
    "ISCAS": "STT_ISCH", "ISCLA": "STT_ISCH", "ISCAN": "STT_ISCH",
    "INJAS": "STT_ISCH", "INJAL": "STT_ISCH", "INJIN": "STT_ISCH", "INJLA": "STT_ISCH",
    "INJIL": "STT_ISCH", "STD_": "STT_ISCH", "NDT": "STT_ISCH", "NST_": "STT_ISCH",
    # infarctus
    "IMI": "MI", "AMI": "MI", "ASMI": "MI", "ALMI": "MI", "ILMI": "MI", "LMI": "MI",
    "PMI": "MI", "IPLMI": "MI", "IPMI": "MI",
    # conduction
    "CRBBB": "RBBB", "IRBBB": "RBBB",
    "CLBBB": "LBBB", "ILBBB": "LBBB",
    "LAFB": "FASC", "LPFB": "FASC",
    # rythme
    "STACH": "STACH", "SBRAD": "SBRAD",
}


def ptbxl_codes_to_labels(scp_codes: dict[str, float], min_likelihood: float = 0.0) -> np.ndarray:
    names = [PTBXL_SCP_MAP[c] for c, lik in scp_codes.items()
             if c in PTBXL_SCP_MAP and float(lik) >= min_likelihood]
    return to_multi_hot(names)


# --------------------------------------------------------------------------
# LSAS-ECG : codes SNOMED-CT  (À VÉRIFIER contre ConditionNames_SNOMED-CT.csv)
# --------------------------------------------------------------------------
LSAS_SNOMED_MAP: dict[str, str] = {
    "164889003": "AFIB",   # atrial fibrillation
    "164890007": "AFIB",   # atrial flutter
    "270492004": "AVB",    # 1st degree AV block
    "195042002": "AVB",    # 2nd degree AV block
    "27885002": "AVB",     # 3rd degree AV block
    "164873001": "LVH",    # left ventricular hypertrophy
    "164934002": "STT_ISCH",  # T wave abnormal
    "59931005": "STT_ISCH",   # T wave inversion
    "429622005": "STT_ISCH",  # ST depression
    "164930006": "STT_ISCH",  # ST elevation (à vérifier)
    "164865005": "MI",     # myocardial infarction
    "59118001": "RBBB", "713427006": "RBBB", "713426002": "RBBB",
    "164909002": "LBBB", "733534002": "LBBB",
    "445118002": "FASC",   # left anterior fascicular block
    "445211001": "FASC",   # left posterior fascicular block
    "427084000": "STACH",
    "426177001": "SBRAD",
}
LSAS_SINUS_RHYTHM = "426783006"


def lsas_codes_to_labels(codes: Iterable[str]) -> np.ndarray:
    codes = [c.strip() for c in codes if c.strip()]
    names = [LSAS_SNOMED_MAP[c] for c in codes if c in LSAS_SNOMED_MAP]
    # NORM = rythme sinusal sans autre diagnostic
    if codes and all(c == LSAS_SINUS_RHYTHM for c in codes):
        names.append("NORM")
    return to_multi_hot(names)


# --------------------------------------------------------------------------
# MIMIC-IV-ECG : énoncés textuels machine -> labels (règles regex)
# --------------------------------------------------------------------------
MIMIC_TEXT_RULES: dict[str, list[str]] = {
    "NORM": [r"\bnormal ecg\b", r"\bnormal electrocardiogram\b"],
    "AFIB": [r"atrial fibrillation", r"atrial flutter"],
    "AVB": [r"\b(1st|first|2nd|second|3rd|third)[- ]degree a-?v block",
            r"\ba-?v block", r"complete heart block", r"mobitz", r"wenckebach"],
    "LVH": [r"left ventricular hypertrophy", r"\blvh\b"],
    "STT_ISCH": [r"ischemi", r"st-?t (wave )?(changes|abnormalit)", r"st depression",
                 r"st elevation", r"t wave (changes|abnormalit|inversion)",
                 r"repolarization abnormalit", r"injury current"],
    "MI": [r"infarct", r"\bstemi\b", r"\bmi\b"],
    "RBBB": [r"right bundle branch block", r"\brbbb\b"],
    "LBBB": [r"left bundle branch block", r"\blbbb\b"],
    "FASC": [r"(left )?(anterior|posterior) fascicular block", r"hemiblock"],
    "STACH": [r"sinus tachycardia"],
    "SBRAD": [r"sinus bradycardia"],
}
# négation et formulations « à exclure » (fréquentes dans les comptes rendus GE/Philips :
# « Cannot rule out anterior infarct - age undetermined ») -> la phrase est ignorée
_NEGATION = re.compile(
    r"\b(no longer|not|no evidence of|without|resolved|absent|no significant)\b"
    r"|cannot rule out|can't rule out|cannot be ruled out|cannot be excluded|cannot exclude"
    r"|\brule out\b|\br/o\b")
# variantes bénignes / secondaires : ne doivent pas déclencher STT_ISCH
# (« LVH with secondary repolarization abnormality » = conséquence de l'HVG, pas une ischémie)
_BENIGN_STT = re.compile(r"early repol|normal variant|juvenile t|persistent juvenile|secondary repol")
# « otherwise normal ECG » = anomalie ailleurs, pas un ECG normal
_NOT_NORM = re.compile(r"otherwise normal|except")
# Diagnostic incertain : « possible / consider / borderline » placé juste avant le terme,
# dans la même proposition. « probable » est conservé (confiance plus élevée).
# Exemples exclus : « possible inferior infarct », « sinus rhythm with borderline 1st degree
# a-v block », « ivcd, consider atypical rbbb ». Exemple conservé : « possible LVH » -> exclu,
# mais « anterior q waves, possibly due to lvh » -> LVH conservé.
_HEDGE = re.compile(r"\b(possible|consider|borderline|questionable|suggests?)\b")
_CLAUSE_SEP = re.compile(r",| with | and | - |\*\*\*|:")
# « atrial flutter with 2:1 a-v block » : rapport de conduction du flutter, pas un BAV
_FLUTTER_RATIO = re.compile(r"\d\s*:\s*\d\s*a-?v (block|conduction)")
# Mode strict (défaut) : diagnostics incertains exclus. Mettre à False pour les inclure.
STRICT_MIMIC_LABELS = True
_COMPILED = {k: [re.compile(p) for p in v] for k, v in MIMIC_TEXT_RULES.items()}


def _hedged(s: str, start: int) -> bool:
    """Vrai si un mot d'incertitude précède la mention dans la même proposition."""
    before = s[:start]
    seps = list(_CLAUSE_SEP.finditer(before))
    clause = before[seps[-1].end():] if seps else before
    return bool(_HEDGE.search(clause))


def mimic_matches(report: str, strict: bool | None = None) -> list[tuple[str, str]]:
    """Liste (label, phrase) retenues — utile pour l'audit des règles."""
    strict = STRICT_MIMIC_LABELS if strict is None else strict
    out = []
    for sentence in re.split(r"[.;\n]", report.lower()):
        s = sentence.strip()
        if not s or _NEGATION.search(s):
            continue
        for lab, pats in _COMPILED.items():
            if lab == "STT_ISCH" and _BENIGN_STT.search(s):
                continue
            if lab == "NORM" and _NOT_NORM.search(s):
                continue
            for p in pats:
                m = p.search(s)
                if not m:
                    continue
                if lab == "AVB" and _FLUTTER_RATIO.search(s) and not re.search(r"degree|complete|mobitz|wenckebach", s):
                    continue
                if strict and _hedged(s, m.start()):
                    continue
                out.append((lab, s))
                break
    return out


def mimic_text_to_labels(report: str) -> np.ndarray:
    """Règles par phrase avec négation / incertitude. NORM uniquement si aucune
    autre anomalie n'est retenue dans le compte rendu."""
    names = {lab for lab, _ in mimic_matches(report)}
    if "NORM" in names and len(names) > 1:
        names.discard("NORM")
    return to_multi_hot(names)


def mask_label_statements(report: str) -> str:
    """Politique « masked » (v1) : supprime les phrases qui définissent directement un label.

    Indispensable quand les labels sont dérivés du texte (MIMIC) :
    sans ce masquage, l'encodeur texte « lit la réponse » (fuite).
    """
    kept = []
    for sentence in re.split(r"(?<=[.;\n])", report):
        low = sentence.lower()
        if any(p.search(low) for pats in _COMPILED.values() for p in pats):
            continue
        kept.append(sentence)
    return " ".join(s.strip() for s in kept if s.strip())


# Politique « strict » (v2) : retire aussi les conclusions globales et les diagnostics
# formulés autrement (y compris en allemand, langue d'une partie des comptes rendus PTB-XL).
# Le v1 laissait « Abnormal ECG » / « Borderline ECG », qui encodent indirectement NORM.
_SUMMARY_AND_DX = re.compile("|".join([
    r"\b(normal|abnormal|borderline)\s+(ecg|ekg|electrocardiogram)\b",
    r"otherwise normal", r"\bwithin normal limits\b", r"\bwnl\b",
    r"infarct", r"ischem", r"isch(a|ä)m", r"injury", r"st[- ]?(elev|depr|segment)",
    r"t[- ]?wave", r"repolari[sz]ation", r"st-t\b",
    r"hypertroph", r"\blvh\b", r"\brvh\b", r"high voltage",
    r"bundle branch", r"\b[rl]bbb\b", r"fascicular", r"hemiblock", r"\blafb\b", r"\blpfb\b",
    r"a-?v block", r"heart block", r"mobitz", r"wenckebach",
    r"fibrillation", r"flutter", r"tachycardia", r"bradycardia",
    r"age undetermined", r"undetermined age", r"\b(1st|2nd|3rd|first|second|third)[- ]degree",
    r"\b(grad|grades)\b", r"\bblock", r"blockierung", r"ventricular response", r"\bstemi\b", r"\bnstemi\b",
    # allemand (PTB-XL)
    r"\b(normales|pathologisches|grenzwertiges|unauff(ä|ae)lliges)\s+ekg\b",
    r"infarkt", r"vorhofflimmern", r"vorhofflattern", r"schenkelblock", r"hemiblock",
    r"av-?block", r"hypertrophie", r"st-?(senkung|hebung|strecken)", r"t-?(negativ|welle|abflachung)",
    r"erregungsr(ü|ue)ckbildung", r"tachykardie", r"bradykardie", r"isch(ä|ae)mie",
    r"linksanterior", r"linksposterior", r"linksschenkel", r"rechtsschenkel", r"faszikul",
]))


def mask_strict(report: str) -> str:
    """Ne garde que les phrases sans aucun contenu diagnostique ni conclusion."""
    kept = []
    # compte rendu sans ponctuation (style PTB-XL allemand) : retrait mot à mot ;
    # sinon : la phrase entière est retirée (plus sûr, aucun fragment révélateur)
    unpunctuated = not re.search(r"[.;,\n]", report)
    for sentence in re.split(r"(?<=[.;\n])|,\s+", report):
        low = sentence.lower()
        if not low.strip():
            continue
        hit = _SUMMARY_AND_DX.search(low) or any(p.search(low) for pats in _COMPILED.values() for p in pats)
        if not hit:
            kept.append(sentence.strip())
        elif unpunctuated and len(low.split()) > 3:
            # longue phrase sans ponctuation (fréquent dans PTB-XL) : on retire seulement
            # les mots diagnostiques (et le mot suivant pour « normales ekg », « st-senkung in »)
            cleaned = low
            for pat in [_SUMMARY_AND_DX] + [p for pats in _COMPILED.values() for p in pats]:
                cleaned = re.sub(r"\S*(?:" + pat.pattern + r")\S*(\s+\S+)?", " ", cleaned)
            cleaned = " ".join(cleaned.split())
            if cleaned:
                kept.append(cleaned)
    return " ".join(kept)


# --------------------------------------------------------------------------
# Labels indépendants pour MIMIC : codes ICD-10 du séjour (MIMIC-IV hosp)
# --------------------------------------------------------------------------
ICD10_PREFIX_MAP: dict[str, str] = {
    "I48": "AFIB",
    "I440": "AVB", "I441": "AVB", "I442": "AVB",
    "I517": "LVH",  # cardiomégalie : proxy imparfait de l'HVG
    "I20": "STT_ISCH", "I24": "STT_ISCH", "I25": "STT_ISCH",
    "I21": "MI", "I22": "MI", "I252": "MI",
    "I451": "RBBB", "I447": "LBBB",
    "I444": "FASC", "I445": "FASC",
}


def icd10_to_labels(codes: Iterable[str]) -> np.ndarray:
    names = []
    for c in codes:
        c = c.replace(".", "").upper()
        for prefix, lab in ICD10_PREFIX_MAP.items():
            if c.startswith(prefix):
                names.append(lab)
    return to_multi_hot(names)
