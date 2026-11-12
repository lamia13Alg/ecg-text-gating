"""Prétraitement du signal ECG et extraction de caractéristiques.

Toutes les bases sont ramenées au même format :
12 dérivations, ordre standard, ``target_fs`` Hz, 10 s, float32.

L'extraction de caractéristiques est volontairement la MÊME pour les trois
bases (même algorithme = pas de décalage introduit par l'outil de mesure).
C'est une implémentation légère (détection R type Pan-Tompkins + fenêtres
heuristiques) ; pour une publication, comparez avec NeuroKit2
(``ecg_delineate``) ou les mesures 12SL de PTB-XL+.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

FEATURE_NAMES = [
    "hr", "rr_mean", "rr_std", "rmssd", "rr_irregularity",
    "qrs_dur", "pr_int", "qt_int", "qtc_bazett",
    "qrs_axis",
    "r_amp_I", "r_amp_II", "r_amp_V1", "r_amp_V5", "r_amp_V6",
    "s_amp_V1", "s_amp_V2",
    "sokolow_lyon", "cornell",
    "st_level_II", "st_level_V2", "st_level_V5",
    "t_amp_II", "t_amp_V5",
]
NUM_FEATURES = len(FEATURE_NAMES)


def reorder_leads(sig: np.ndarray, names: list[str]) -> np.ndarray:
    """sig : (T, n_leads) -> (12, T) dans l'ordre LEADS ; dérivations absentes = 0."""
    names = [LEAD_ALIASES.get(n.upper(), n) for n in names]
    out = np.zeros((12, sig.shape[0]), dtype=np.float32)
    for i, lead in enumerate(LEADS):
        if lead in names:
            out[i] = sig[:, names.index(lead)]
    return out


def preprocess(sig: np.ndarray, fs: float, target_fs: int = 250,
               duration_s: float = 10.0) -> np.ndarray:
    """(12, T) -> (12, target_fs*duration) : NaN->0, passe-bande, rééchantillonnage."""
    sig = np.nan_to_num(sig.astype(np.float64))
    nyq = fs / 2
    b, a = sps.butter(3, [0.5 / nyq, min(45.0, 0.9 * nyq) / nyq], btype="band")
    sig = sps.filtfilt(b, a, sig, axis=1)
    n_target = int(round(target_fs * duration_s))
    n_src = int(round(fs * duration_s))
    sig = sig[:, :n_src]
    if sig.shape[1] < n_src:
        sig = np.pad(sig, ((0, 0), (0, n_src - sig.shape[1])))
    if fs != target_fs:
        sig = sps.resample_poly(sig, target_fs, int(fs), axis=1)
    sig = sig[:, :n_target]
    if sig.shape[1] < n_target:
        sig = np.pad(sig, ((0, 0), (0, n_target - sig.shape[1])))
    return sig.astype(np.float32)  # unités : mV


def normalize(sig: np.ndarray, eps: float = 1e-6, mode: str = "lead") -> np.ndarray:
    """Normalisation appliquée à la volée dans le Dataset.

    lead  : standardisation par dérivation (v1/v2). Efface l'amplitude absolue et les
            rapports d'amplitude entre dérivations -> pénalise l'HVG (critères de voltage).
    fixed : centrage par dérivation, échelle fixe en mV conservée (clip ±10 mV).
            Préserve les critères de voltage (Sokolow-Lyon, Cornell).
    """
    mu = np.median(sig, axis=-1, keepdims=True) if mode == "fixed" else sig.mean(axis=-1, keepdims=True)
    if mode == "fixed":
        return np.clip(sig - mu, -10.0, 10.0)
    sd = sig.std(axis=-1, keepdims=True)
    return (sig - mu) / (sd + eps)


# --------------------------------------------------------------------------
# Caractéristiques
# --------------------------------------------------------------------------
def detect_r_peaks(x: np.ndarray, fs: int) -> np.ndarray:
    b, a = sps.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    y = sps.filtfilt(b, a, x)
    e = np.diff(y, prepend=y[0]) ** 2
    win = max(1, int(0.12 * fs))
    e = np.convolve(e, np.ones(win) / win, mode="same")
    peaks, _ = sps.find_peaks(e, height=0.3 * np.percentile(e, 99),
                              distance=int(0.25 * fs))
    # recentrer sur le maximum absolu du signal brut
    half = int(0.06 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(x), p + half)
        refined.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    return np.unique(np.array(refined, dtype=int))


def _median_beat(x: np.ndarray, r: np.ndarray, fs: int, pre: float = 0.3, post: float = 0.5):
    a, b = int(pre * fs), int(post * fs)
    beats = [x[p - a:p + b] for p in r if p - a >= 0 and p + b <= len(x)]
    if not beats:
        return None
    return np.median(np.stack(beats), axis=0)


def extract_features(sig: np.ndarray, fs: int) -> np.ndarray:
    """sig : (12, T) en mV -> vecteur (NUM_FEATURES,) ; NaN si non mesurable."""
    f = dict.fromkeys(FEATURE_NAMES, np.nan)
    L = {n: i for i, n in enumerate(LEADS)}
    try:
        ref = sig[L["II"]] if np.abs(sig[L["II"]]).sum() > 0 else sig[L["I"]]
        r = detect_r_peaks(ref, fs)
        if len(r) >= 3:
            rr = np.diff(r) / fs * 1000.0  # ms
            f["rr_mean"] = rr.mean()
            f["rr_std"] = rr.std()
            f["rmssd"] = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 2 else np.nan
            f["rr_irregularity"] = f["rr_std"] / (f["rr_mean"] + 1e-6)
            f["hr"] = 60000.0 / f["rr_mean"]

        pre = int(0.3 * fs)
        beats = {n: _median_beat(sig[L[n]], r, fs) for n in LEADS} if len(r) >= 2 else {}
        beats = {k: v for k, v in beats.items() if v is not None}
        if beats:
            # isoélectrique ~ segment PR (de -80 à -40 ms avant R)
            def baseline(b):
                return np.median(b[pre - int(0.08 * fs):pre - int(0.04 * fs)])

            # amplitudes R / S (fenêtre ±60 ms)
            w = int(0.06 * fs)
            for lead in ["I", "II", "V1", "V5", "V6"]:
                if lead in beats:
                    b = beats[lead] - baseline(beats[lead])
                    f[f"r_amp_{lead}"] = b[pre - w:pre + w].max()
            for lead in ["V1", "V2"]:
                if lead in beats:
                    b = beats[lead] - baseline(beats[lead])
                    f[f"s_amp_{lead}"] = -b[pre - w:pre + w].min()
            # critères d'HVG
            if not np.isnan(f["s_amp_V1"]) and not np.isnan(f["r_amp_V5"]):
                f["sokolow_lyon"] = f["s_amp_V1"] + max(f["r_amp_V5"], np.nan_to_num(f["r_amp_V6"]))
            if "aVL" in beats and "V3" in beats:
                ravl = (beats["aVL"] - baseline(beats["aVL"]))[pre - w:pre + w].max()
                sv3 = -(beats["V3"] - baseline(beats["V3"]))[pre - w:pre + w].min()
                f["cornell"] = ravl + sv3

            # durée QRS : énergie de pente sommée sur les dérivations
            stack = np.stack([np.abs(np.diff(v - baseline(v))) for v in beats.values()]).sum(0)
            thr = 0.15 * stack[pre - w:pre + w].max()
            on = pre - w + np.argmax(stack[pre - w:pre] > thr) if (stack[pre - w:pre] > thr).any() else np.nan
            after = stack[pre:pre + int(0.12 * fs)] > thr
            off = pre + len(after) - np.argmax(after[::-1]) if after.any() else np.nan
            if not (np.isnan(on) or np.isnan(off)):
                f["qrs_dur"] = (off - on) / fs * 1000.0
                j = int(off)
                # niveau ST à J+60 ms
                for lead in ["II", "V2", "V5"]:
                    if lead in beats:
                        b = beats[lead] - baseline(beats[lead])
                        k = min(len(b) - 1, j + int(0.06 * fs))
                        f[f"st_level_{lead}"] = b[k]
                # onde T et intervalle QT (fin de T = retour sous 15 % du pic)
                if "II" in beats:
                    b = beats["II"] - baseline(beats["II"])
                    seg = b[j + int(0.08 * fs):]
                    if len(seg) > 5:
                        tp = int(np.argmax(np.abs(seg)))
                        f["t_amp_II"] = seg[tp]
                        tail = np.abs(seg[tp:]) < 0.15 * abs(seg[tp])
                        if tail.any():
                            t_end = j + int(0.08 * fs) + tp + int(np.argmax(tail))
                            f["qt_int"] = (t_end - on) / fs * 1000.0
                            if not np.isnan(f["rr_mean"]):
                                f["qtc_bazett"] = f["qt_int"] / np.sqrt(f["rr_mean"] / 1000.0)
                if "V5" in beats:
                    b = beats["V5"] - baseline(beats["V5"])
                    seg = b[j + int(0.08 * fs):]
                    if len(seg):
                        f["t_amp_V5"] = seg[int(np.argmax(np.abs(seg)))]
                # PR : début de P = premier dépassement avant QRS
                if "II" in beats:
                    b = beats["II"] - baseline(beats["II"])
                    pwin = b[pre - int(0.25 * fs):int(on)]
                    if len(pwin) > 5 and np.abs(pwin).max() > 0.03:
                        p_on = np.argmax(np.abs(pwin) > 0.3 * np.abs(pwin).max())
                        f["pr_int"] = (len(pwin) - p_on) / fs * 1000.0
            # axe QRS à partir de l'aire nette en I et aVF
            if "I" in beats and "aVF" in beats:
                ni = (beats["I"] - baseline(beats["I"]))[pre - w:pre + w].sum()
                nf = (beats["aVF"] - baseline(beats["aVF"]))[pre - w:pre + w].sum()
                f["qrs_axis"] = np.degrees(np.arctan2(nf, ni))
    except Exception:  # signal dégénéré : on garde les NaN
        pass
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


# Mesures machine de MIMIC-IV-ECG (machine_measurements.csv) -> FEATURE_NAMES
MIMIC_MACHINE_SENTINEL = 29999


def mimic_machine_features(row) -> np.ndarray:
    """Option ``feature_source: machine`` : n'utilise que les mesures disponibles."""
    def val(k):
        v = row.get(k, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        return np.nan if v >= MIMIC_MACHINE_SENTINEL else v

    f = dict.fromkeys(FEATURE_NAMES, np.nan)
    rr = val("rr_interval")
    f["rr_mean"] = rr
    f["hr"] = 60000.0 / rr if rr and not np.isnan(rr) else np.nan
    f["qrs_dur"] = val("qrs_end") - val("qrs_onset")
    f["pr_int"] = val("qrs_onset") - val("p_onset")
    f["qt_int"] = val("t_end") - val("qrs_onset")
    if not np.isnan(f["qt_int"]) and not np.isnan(rr):
        f["qtc_bazett"] = f["qt_int"] / np.sqrt(rr / 1000.0)
    f["qrs_axis"] = val("qrs_axis")
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
