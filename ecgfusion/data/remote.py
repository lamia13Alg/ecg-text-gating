"""Téléchargement rapide d'enregistrements WFDB depuis PhysioNet.

Deux sources, essayées dans l'ordre :
  1. le miroir AWS Open Data  https://physionet-open.s3.amazonaws.com/<base>/   (rapide, sans compte)
  2. physionet.org            https://physionet.org/files/<base>/

Chaque enregistrement est téléchargé dans un dossier temporaire en mémoire
(/dev/shm si disponible), lu avec wfdb, puis supprimé : rien n'est stocké.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

REMOTE_PREFIX = "pn://"
MIRRORS = [
    "https://physionet-open.s3.amazonaws.com/{base}/",
    "https://physionet.org/files/{base}/",
]
# surcharge possible : ECGFUSION_MIRRORS="file:///data/{base}/,https://..."  (miroir local, tests)
if os.environ.get("ECGFUSION_MIRRORS"):
    MIRRORS = [m.strip() for m in os.environ["ECGFUSION_MIRRORS"].split(",") if m.strip()]

_local = threading.local()
_chosen: dict[str, str] = {}


def _session():
    if not hasattr(_local, "s"):
        import requests
        from requests.adapters import HTTPAdapter

        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=64, max_retries=2))
        s.headers["User-Agent"] = "ecgfusion-research/1.0"
        _local.s = s
    return _local.s


def _get(url: str, timeout: float = 30.0) -> bytes:
    if url.startswith("file://"):
        delay = float(os.environ.get("ECGFUSION_FAKE_LATENCY", "0"))
        if delay:
            time.sleep(delay)
        return Path(url[len("file://"):]).read_bytes()
    r = _session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def split_remote(path: str) -> tuple[str, str]:
    """pn://mimic-iv-ecg/1.0/files/p1000/p1/s2/2 -> ('mimic-iv-ecg/1.0', 'files/p1000/p1/s2/2')"""
    rest = path[len(REMOTE_PREFIX):]
    parts = rest.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])


def choose_mirror(base: str, example_rel: str) -> str:
    """Teste les miroirs sur un enregistrement et garde le plus rapide qui répond."""
    if base in _chosen:
        return _chosen[base]
    best, best_t = None, 1e9
    for tpl in MIRRORS:
        url = tpl.format(base=base)
        try:
            t0 = time.time()
            _get(url + example_rel + ".hea", timeout=20)
            dt = time.time() - t0
            print(f"[remote] {url}  OK ({dt * 1000:.0f} ms)", flush=True)
            if dt < best_t:
                best, best_t = url, dt
        except Exception as e:  # noqa: BLE001
            print(f"[remote] {url}  indisponible ({type(e).__name__})", flush=True)
    if best is None:
        raise RuntimeError("Aucun miroir PhysioNet joignable : vérifiez que Internet est activé.")
    _chosen[base] = best
    return best


def read_remote(path: str, mirror: str | None = None, retries: int = 4):
    """Retourne (signal, fields) comme wfdb.rdsamp."""
    import wfdb

    base, rel = split_remote(path)
    url = mirror or _chosen.get(base) or choose_mirror(base, rel)
    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    for attempt in range(retries):
        d = tempfile.mkdtemp(dir=tmp_root)
        try:
            name = rel.rsplit("/", 1)[-1]
            folder = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
            hea = Path(d) / f"{name}.hea"
            hea.write_bytes(_get(url + rel + ".hea"))
            header = wfdb.rdheader(str(hea.with_suffix("")))
            for fn in sorted(set(header.file_name or [])):
                (Path(d) / fn).write_bytes(_get(url + folder + fn))
            return wfdb.rdsamp(str(hea.with_suffix("")))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
        finally:
            shutil.rmtree(d, ignore_errors=True)
