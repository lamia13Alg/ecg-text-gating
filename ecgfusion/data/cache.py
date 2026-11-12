"""Lecture WFDB -> prétraitement -> cache sur disque (memmap), reprenable.

Sortie dans <out_dir> :
    signals.npy   (N, 12, T) float16, en mV, filtré et rééchantillonné
    features.npy  (N, F)     float32, NaN = non mesurable
    labels.npy    (N, C)     float32
    done.npy      (N,)       bool, enregistrement déjà traité
    meta.csv      record_id, patient_id, split, dataset, text, valid
    manifest.csv  manifeste complet (permet la reprise)
    info.json     dont "complete": true/false

Le traitement s'arrête proprement à ``deadline`` (horodatage) en sauvegardant
l'état ; relancer avec ``resume_from`` continue là où on s'était arrêté.
"""
from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .. import signal_processing as sp
from ..labels import LABELS, NUM_LABELS
from .remote import REMOTE_PREFIX, choose_mirror, read_remote, split_remote

_CFG: dict = {}
MACHINE_COLS = ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end",
                "t_end", "p_axis", "qrs_axis", "t_axis"]


def read_record(path: str):
    import wfdb

    if path.startswith(REMOTE_PREFIX):
        return read_remote(path, mirror=_CFG.get("mirror"))
    return wfdb.rdsamp(path)


def _init(cfg):
    _CFG.update(cfg)


def _process_one(args):
    i, path, machine = args
    fs_t, dur = _CFG["target_fs"], _CFG["duration_s"]
    try:
        sig, fields = read_record(path)
        sig = sp.reorder_leads(sig, fields["sig_name"])
        sig = sp.preprocess(sig, fields["fs"], fs_t, dur)
        if _CFG["feature_source"] == "machine" and machine is not None:
            feats = sp.mimic_machine_features(machine)
        else:
            feats = sp.extract_features(sig, fs_t)
        return i, sig.astype(np.float16), feats, True
    except Exception:
        n = int(fs_t * dur)
        return i, np.zeros((12, n), np.float16), np.full(sp.NUM_FEATURES, np.nan, np.float32), False


def _process_chunk(jobs):
    """Un paquet d'enregistrements : téléchargements en parallèle (threads)."""
    threads = _CFG.get("threads", 1)
    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            return list(ex.map(_process_one, jobs))
    return [_process_one(j) for j in jobs]


def _save_state(out, manifest, feats, labels, valid, done, cfg, t_start, n_session):
    np.save(out / "features.npy", feats)
    np.save(out / "labels.npy", labels)
    np.save(out / "done.npy", done)
    meta = manifest[["record_id", "patient_id", "split", "dataset", "text"]].copy()
    meta["valid"] = valid & done
    meta.to_csv(out / "meta.csv", index=False)
    ok = valid & done
    (out / "info.json").write_text(json.dumps({
        "n": int(len(done)), "n_done": int(done.sum()), "n_valid": int(ok.sum()),
        "complete": bool(done.all()), **cfg,
        "labels": LABELS, "features": sp.FEATURE_NAMES,
        "label_prevalence": dict(zip(LABELS, (labels[ok].mean(0) if ok.any() else
                                              np.zeros(NUM_LABELS)).round(4).tolist())),
        "last_session_records": n_session,
        "last_session_rate_per_s": round(n_session / max(time.time() - t_start, 1e-6), 2),
    }, indent=2))


def build_cache(manifest: pd.DataFrame, out_dir: str, target_fs: int = 250,
                duration_s: float = 10.0, feature_source: str = "computed",
                workers: int = 8, threads: int = 1, deadline: float | None = None,
                resume_from: str | None = None, checkpoint_every: int = 5000) -> bool:
    """Retourne True si le cache est complet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, T = len(manifest), int(target_fs * duration_s)
    labels = np.stack([np.array(s.split(","), dtype=np.float32) for s in manifest["labels"]])
    assert labels.shape[1] == NUM_LABELS
    manifest.to_csv(out / "manifest.csv", index=False)

    src = Path(resume_from) if resume_from else None
    if src and (src / "done.npy").exists():
        if src.resolve() != out.resolve():
            print(f"[cache] copie de l'état précédent depuis {src} ...", flush=True)
            for f in ["signals.npy", "features.npy", "done.npy", "meta.csv"]:
                shutil.copy(src / f, out / f)
        signals = np.load(out / "signals.npy", mmap_mode="r+")
        assert signals.shape == (n, 12, T), "Le cache précédent ne correspond pas au manifeste"
        feats = np.load(out / "features.npy")
        done = np.load(out / "done.npy")
        valid = pd.read_csv(out / "meta.csv", keep_default_na=False)["valid"].astype(str) \
            .str.lower().eq("true").to_numpy().copy()
        print(f"[cache] reprise : {done.sum()}/{n} déjà traités", flush=True)
    else:
        signals = np.lib.format.open_memmap(out / "signals.npy", mode="w+",
                                            dtype=np.float16, shape=(n, 12, T))
        feats = np.full((n, sp.NUM_FEATURES), np.nan, np.float32)
        done = np.zeros(n, bool)
        valid = np.zeros(n, bool)

    paths = manifest["wfdb_path"].tolist()
    has_machine = all(c in manifest.columns for c in MACHINE_COLS)
    todo = np.where(~done)[0]
    cfg = dict(target_fs=target_fs, duration_s=duration_s, feature_source=feature_source)
    run_cfg = dict(cfg, threads=threads)
    if todo.size and paths[todo[0]].startswith(REMOTE_PREFIX):
        base, rel = split_remote(paths[todo[0]])
        run_cfg["mirror"] = choose_mirror(base, rel)
        if "physionet.org" in run_cfg["mirror"] and threads > 4:
            # serveur principal : rester raisonnable pour ne pas être limité/bloqué
            run_cfg["threads"] = threads = 4
            print("[cache] physionet.org : 4 téléchargements par processus maximum", flush=True)

    def job(i):
        m = manifest.iloc[i][MACHINE_COLS].to_dict() if has_machine else None
        return (int(i), paths[i], m)

    chunk = max(8, threads * 4)
    chunks = [[job(i) for i in todo[k:k + chunk]] for k in range(0, len(todo), chunk)]
    t_start, n_sess, last_ckpt, last_log = time.time(), 0, 0, time.time()
    stopped = False

    def consume(results):
        nonlocal n_sess
        for i, s, f, ok in results:
            signals[i], feats[i], valid[i], done[i] = s, f, ok, True
        n_sess += len(results)

    def log_and_checkpoint(force=False):
        nonlocal last_ckpt, last_log
        if force or n_sess - last_ckpt >= checkpoint_every:
            signals.flush()
            _save_state(out, manifest, feats, labels, valid, done, cfg, t_start, n_sess)
            last_ckpt = n_sess
        if force or time.time() - last_log > 120:
            rate = n_sess / max(time.time() - t_start, 1e-6)
            left = (~done).sum()
            print(f"[cache {out.name}] {done.sum()}/{n} traités  {rate:.1f} ECG/s  "
                  f"reste ~{left / max(rate, 1e-6) / 3600:.1f} h", flush=True)
            last_log = time.time()

    print(f"[cache {out.name}] {len(todo)} à traiter, {workers} processus × {threads} threads", flush=True)
    if workers > 1:
        pool = Pool(workers, initializer=_init, initargs=(run_cfg,))
        try:
            for res in pool.imap_unordered(_process_chunk, chunks):
                consume(res)
                log_and_checkpoint()
                if deadline and time.time() > deadline:
                    stopped = True
                    break
        finally:
            pool.terminate()
            pool.join()
    else:
        _init(run_cfg)
        for c in chunks:
            consume(_process_chunk(c))
            log_and_checkpoint()
            if deadline and time.time() > deadline:
                stopped = True
                break

    log_and_checkpoint(force=True)
    complete = bool(done.all())
    msg = "COMPLET" if complete else ("INCOMPLET (budget de temps atteint, relancer pour reprendre)"
                                      if stopped else "INCOMPLET")
    print(f"[cache] {out.name} : {int((valid & done).sum())} valides / {int(done.sum())} traités / {n} -> {msg}",
          flush=True)
    return complete
