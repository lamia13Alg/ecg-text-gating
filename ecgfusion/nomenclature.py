"""Reference nomenclature for the 11 labels.

Each label is defined by the corresponding diagnostic statement(s) of the AHA/ACCF/HRS
"Recommendations for the Standardization and Interpretation of the Electrocardiogram"
(Part II statement list, Mason et al. 2007; criteria in Parts III-VI, 2009), complemented by
the Fourth Universal Definition of Myocardial Infarction (Thygesen et al. 2018) and the
2018 ACC/AHA/HRS bradycardia and conduction guideline (Kusumoto et al. 2019).

The same definition is projected onto each database's own coding system:
  * MIMIC-IV-ECG : machine statements, parsed by the rules in ``labels.py``
  * PTB-XL       : SCP-ECG codes (``labels.PTBXL_SCP_MAP``)
  * LSAS         : SNOMED-CT codes, as harmonised in the PhysioNet/CinC Challenges
                   (``labels.LSAS_SNOMED_MAP``)

``scripts/export_label_definitions.py`` writes this table to CSV/LaTeX.
"""
from __future__ import annotations

from .labels import LABELS, LSAS_SNOMED_MAP, PTBXL_SCP_MAP

# reference keys match the BibTeX keys of the paper
LABEL_DEFINITIONS: dict[str, dict] = {
    "NORM": {
        "statement": "Normal ECG",
        "definition": "Normal ECG with no other labelled abnormality",
        "refs": ["mason2007"],
    },
    "AFIB": {
        "statement": "Atrial fibrillation; atrial flutter",
        "definition": "Atrial fibrillation or atrial flutter (merged)",
        "refs": ["mason2007"],
    },
    "AVB": {
        "statement": "First-, second- (Mobitz I/II) and third-degree AV block",
        "definition": "Any degree of AV block; first degree = PR interval > 200 ms "
                      "(stated as first-degree AV block or prolonged PR interval)",
        "refs": ["mason2007", "kusumoto2019"],
    },
    "LVH": {
        "statement": "Left ventricular hypertrophy",
        "definition": "Left ventricular hypertrophy by voltage and/or non-voltage criteria",
        "refs": ["mason2007", "hancock2009"],
    },
    "STT_ISCH": {
        "statement": "ST deviation; T-wave abnormality (primary); suggestive of ischaemia/injury",
        "definition": "Primary ST-segment or T-wave abnormality, non-specific or suggestive of "
                      "ischaemia/injury; secondary changes (hypertrophy, bundle branch block, "
                      "conduction delay, pacing), early repolarisation and normal variants excluded",
        "refs": ["mason2007", "rautaharju2009", "wagner2009"],
    },
    "MI": {
        "statement": "Myocardial infarction (any age)",
        "definition": "ECG pattern of myocardial infarction (acute or prior, e.g. pathological "
                      "Q waves), as stated by the reader",
        "refs": ["mason2007", "wagner2009", "thygesen2018"],
    },
    "RBBB": {
        "statement": "Complete or incomplete right bundle branch block",
        "definition": "Complete (QRS >= 120 ms) or incomplete (110-119 ms) RBBB (merged)",
        "refs": ["mason2007", "surawicz2009"],
    },
    "LBBB": {
        "statement": "Complete or incomplete left bundle branch block",
        "definition": "Complete (QRS >= 120 ms) or incomplete (110-119 ms) LBBB (merged)",
        "refs": ["mason2007", "surawicz2009"],
    },
    "FASC": {
        "statement": "Left anterior or left posterior fascicular block",
        "definition": "Left anterior or left posterior fascicular block (merged)",
        "refs": ["mason2007", "surawicz2009"],
    },
    "STACH": {
        "statement": "Sinus tachycardia",
        "definition": "Sinus tachycardia, as stated by the reader",
        "refs": ["mason2007"],
    },
    "SBRAD": {
        "statement": "Sinus bradycardia",
        "definition": "Sinus bradycardia, as stated by the reader",
        "refs": ["mason2007"],
    },
}

assert list(LABEL_DEFINITIONS) == LABELS


def codes_for(label: str) -> tuple[list[str], list[str]]:
    """(SCP-ECG codes, SNOMED-CT codes) mapped to ``label``."""
    scp = sorted(c for c, l in PTBXL_SCP_MAP.items() if l == label)
    snomed = sorted(c for c, l in LSAS_SNOMED_MAP.items() if l == label)
    if label == "NORM":
        scp, snomed = ["NORM"], ["426783006 (sinus rhythm, no other code)"]
    return scp, snomed
