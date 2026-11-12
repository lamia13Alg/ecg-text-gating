"""Entraînement du modèle de fusion sur MIMIC-IV-ECG.

  python scripts/train.py --config configs/default.yaml
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgfusion import metrics as M  # noqa: E402
from ecgfusion.data.dataset import SCENARIOS, ECGMultimodalDataset, FeatureNormalizer  # noqa: E402
from ecgfusion.engine import (apply_overrides, build_model, build_tokenizer,  # noqa: E402
                              make_loader, predict, seed_everything, to_device)
from ecgfusion.labels import LABELS  # noqa: E402
from ecgfusion.losses import build_loss, class_weights_from_prevalence  # noqa: E402


def val_score(model, loaders, device, label_mask):
    scores = {}
    for name, loader in loaders.items():
        r = predict(model, loader, device)
        df = M.per_label_metrics(r["labels"], M.sigmoid(r["logits"]), np.full(len(LABELS), 0.5),
                                 LABELS, label_mask, min_pos=1)
        scores[name] = float(df["auroc"].mean())
    return float(np.mean(list(scores.values()))), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="surcharges, ex. train.batch_size=64")
    ap.add_argument("--resume", default=None,
                    help="dossier d'un run précédent contenant last.pt (reprise après coupure Kaggle)")
    ap.add_argument("--max_hours", type=float, default=None,
                    help="arrêt propre avant la limite de session (ex. 11 sur Kaggle)")
    args = ap.parse_args()
    t_start = time.time()
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.set)
    seed_everything(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), allow_unicode=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, dc = cfg["train"], cfg["data"]

    # ---------------- données ----------------
    cache = Path(dc["train_cache"])
    meta = pd.read_csv(cache / "meta.csv", keep_default_na=False)
    feats = np.load(cache / "features.npy")
    tr_mask = (meta["split"] == "train").values & meta["valid"].astype(str).str.lower().eq("true").values
    feat_norm = FeatureNormalizer().fit(feats[tr_mask])
    feat_norm.save(out / "feature_norm.json")

    tok = build_tokenizer(cfg)
    signal_norm = dc.get("signal_norm", "lead")
    train_scen = SCENARIOS[tc.get("train_scenario", "full")]   # "signal_only" : référence unimodale
    train_ds = ECGMultimodalDataset(cache, "train", feat_norm, dc["text_policy"], train=True,
                                    scenario=train_scen, signal_norm=signal_norm,
                                    p_drop_text=tc["p_drop_text"], p_drop_features=tc["p_drop_features"],
                                    p_drop_leads=tc["p_drop_leads"], max_records=dc["max_train_records"])
    val_loaders = {s: make_loader(ECGMultimodalDataset(cache, "val", feat_norm, dc["text_policy"],
                                                       scenario=SCENARIOS[s], signal_norm=signal_norm),
                                  tok, cfg, shuffle=False)
                   for s in tc["val_scenarios"]}
    train_loader = make_loader(train_ds, tok, cfg, shuffle=True)
    label_mask = train_ds.label_mask
    y_train = np.load(cache / "labels.npy")[train_ds.idx]
    print(f"[data] train={len(train_ds)}  val={len(next(iter(val_loaders.values())).dataset)}")
    print("[data] prévalences train :", dict(zip(LABELS, y_train.mean(0).round(3))))

    # ---------------- modèle ----------------
    model = build_model(cfg).to(device)
    cw = class_weights_from_prevalence(y_train, tc["class_weight_power"])
    criterion = build_loss(tc["loss"], class_weights=cw, **tc.get("loss_kwargs", {})).to(device)
    aux_criterion = build_loss("bce", class_weights=cw).to(device)

    text_params = [p for n, p in model.named_parameters() if n.startswith("text_enc.bert") and p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith("text_enc.bert") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": other, "lr": tc["lr"]},
                             {"params": text_params, "lr": tc["lr_text"]}],
                            weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(train_loader)
    warm = max(1, int(tc["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params / 1e6:.1f} M paramètres entraînables sur {device}")

    # ---------------- boucle ----------------
    best, patience, history, start_epoch = -1.0, 0, [], 1
    if args.resume:
        rdir = Path(args.resume)
        ck = torch.load(rdir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        best, patience, start_epoch = ck["best"], ck["patience"], ck["epoch"] + 1
        if (rdir / "history.csv").exists():
            history = pd.read_csv(rdir / "history.csv").to_dict("records")
        if rdir.resolve() != out.resolve() and (rdir / "best.pt").exists():
            shutil.copy(rdir / "best.pt", out / "best.pt")
        print(f"[resume] reprise à l'époque {start_epoch} (meilleur val = {best:.4f})")

    stop_reason = "epochs"
    for epoch in range(start_epoch, tc["epochs"] + 1):
        if patience >= tc["patience"]:
            stop_reason = "early stopping"
            break
        model.train()
        t0, run = time.time(), []
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                o = model(batch)
                loss = criterion(o["logits"].float(), batch["labels"], batch["label_mask"])
                for k, aux in enumerate(o["aux_logits"]):
                    loss = loss + tc["aux_weight"] / 3 * aux_criterion(
                        aux.float(), batch["labels"], batch["label_mask"], sample_mask=batch["present"][:, k])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
            step = len(run)
            if step % max(1, len(train_loader) // 10) == 0:
                el = time.time() - t0
                print(f"   époque {epoch} : {100 * step / len(train_loader):3.0f} %  "
                      f"loss={np.mean(run[-50:]):.4f}  {step * tc['batch_size'] / el:.0f} ECG/s  "
                      f"(fin d'époque dans ~{el / step * (len(train_loader) - step) / 60:.0f} min)", flush=True)

        score, per = val_score(model, val_loaders, device, label_mask)
        rec = {"epoch": epoch, "train_loss": float(np.mean(run)), "val_score": score,
               **{f"val_auroc_{k}": v for k, v in per.items()}, "time_s": time.time() - t0}
        history.append(rec)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[epoch {epoch:02d}] loss={rec['train_loss']:.4f}  val={score:.4f}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in per.items())
              + f"  ({rec['time_s'] / 60:.1f} min)", flush=True)
        if score > best:
            best, patience = score, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            patience += 1
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "patience": patience}, out / "last.pt")
        if patience >= tc["patience"]:
            stop_reason = "early stopping"
            break
        if args.max_hours and epoch < tc["epochs"]:
            elapsed = (time.time() - t_start) / 3600
            if elapsed + 1.3 * rec["time_s"] / 3600 > args.max_hours:
                stop_reason = "budget de temps atteint (relancer avec --resume)"
                break
    print(f"[stop] {stop_reason}", flush=True)

    # ---------------- calibration (validation interne uniquement) ----------------
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    calib = {}
    for s, loader in val_loaders.items():
        r = predict(model, loader, device)
        T = M.fit_temperatures(r["logits"], r["labels"], label_mask)
        th = M.fit_thresholds(r["labels"], M.sigmoid(r["logits"] / T), label_mask)
        calib[s] = {"temperature": T.tolist(), "threshold": th.tolist()}
    json.dump(calib, open(out / "calibration.json", "w"), indent=2)
    finished = not stop_reason.startswith("budget")
    json.dump({"finished": finished, "stop_reason": stop_reason, "last_epoch": len(history),
               "best_val": best}, open(out / "status.json", "w"), indent=2)
    print(f"[done] meilleur score val = {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
