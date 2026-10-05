"""Tasks 4.3/4.4 — train AE-MSE and Task-driven AE codecs.

AE-MSE : L = MSE(x, x_hat)
Task-AE: L = MSE(x, x_hat) + lambda(epoch) * CE(classifier(x_hat), y)
         lambda = 0 for epochs 1-20, linear 0 -> lambda_max over epochs 21-40, lambda_max after 40.
The classifier of the same fold/seed is frozen (eval mode, requires_grad=False) but is NOT wrapped
in torch.no_grad(), so the classification gradient flows through it into decoder and encoder.

Checkpoint selection uses Val only:
  AE-MSE  -> lowest Val MSE
  Task-AE -> highest Val accuracy among epochs with lambda = lambda_max (ties: lower Val MSE)

Usage:
  python train_codec.py --methods AE-MSE Task-AE --ds 128 --folds 1 --seeds 0
  python train_codec.py --methods Task-AE --ds 128 --folds 1 2 3 4 5 6 --lams 0 0.1 0.5 2 --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from classifier import load_classifier
from codec import Codec, count_params
from dataset import load_fold, tensor_dataset
from utils import CKPT_DIR, LOG_DIR, ensure_dirs, save_json, set_seed

CFG = dict(epochs=60, batch_size=64, lr=1e-3, optimizer="Adam",
           warmup_end=20, ramp_end=40)
# (warmup_end, ramp_end, epochs). "main" is the spec schedule; "slow" is a follow-up ablation.
SCHEDULES = {"main": (20, 40, 60), "slow": (30, 60, 90)}


def lam_at(epoch: int, method: str, lam_max: float, schedule: str = "main") -> float:
    if method == "AE-MSE":
        return 0.0
    warm, ramp, _ = SCHEDULES[schedule]
    if epoch <= warm:
        return 0.0
    if epoch <= ramp:
        return lam_max * (epoch - warm) / (ramp - warm)
    return lam_max


def ckpt_name(method: str, d: int, fold: int, seed: int, lam_max: float = 1.0,
              clf_tag: str = "", schedule: str = "main") -> str:
    tag = "" if (method == "AE-MSE" or lam_max == 1.0) else f"_lam{lam_max:g}"
    tag += f"_{clf_tag}" if clf_tag else ""
    tag += "" if (method == "AE-MSE" or schedule == "main") else f"_{schedule}"
    return f"{method}_d{d}_fold{fold}_seed{seed}{tag}"


def frozen_classifier(fold: int, seed: int, clf_tag: str = "") -> nn.Module:
    sub = "classifiers" if not clf_tag else f"classifiers_{clf_tag}"
    clf = load_classifier(CKPT_DIR / sub / f"fold_{fold}_seed_{seed}.pt")
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf


def gradient_check(codec: Codec, clf: nn.Module, xb, yb) -> dict:
    """Backprop CE(classifier(x_hat), y) alone and report where gradients arrive."""
    codec.zero_grad()
    x_hat, _ = codec(xb)
    F.cross_entropy(clf(x_hat), yb).backward()
    enc_g = sum(p.grad.norm().item() ** 2 for p in codec.encoder.parameters() if p.grad is not None) ** 0.5
    dec_g = sum(p.grad.norm().item() ** 2 for p in codec.decoder.parameters() if p.grad is not None) ** 0.5
    clf_has_grad = any(p.grad is not None for p in clf.parameters())
    codec.zero_grad()
    return dict(encoder_grad_norm=enc_g, decoder_grad_norm=dec_g,
                classifier_params_received_grad=clf_has_grad,
                classifier_requires_grad=any(p.requires_grad for p in clf.parameters()),
                classifier_training_mode=clf.training)


@torch.no_grad()
def evaluate_split(codec: Codec, clf: nn.Module, split) -> tuple[float, float]:
    codec.eval()
    dev = next(codec.parameters()).device
    x = torch.from_numpy(split.x).unsqueeze(1).to(dev)
    x_hat, _ = codec(x)  # fake-quant forward == real INT8 round trip numerically
    mse = F.mse_loss(x_hat, x).item()
    acc = (clf(x_hat).argmax(1).cpu().numpy() == split.y).mean()
    return mse, float(acc)


def train_one(method: str, d: int, fold: int, seed: int, lam_max: float = 1.0,
              threads: int = 2, verbose: bool = False, clf_tag: str = "", schedule: str = "main") -> dict:
    torch.set_num_threads(threads)
    set_seed(seed)
    sp, _, _ = load_fold(fold)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = frozen_classifier(fold, seed, clf_tag).to(dev)
    n_epochs = SCHEDULES[schedule][2] if method != "AE-MSE" else CFG["epochs"]
    codec = Codec(d).to(dev)
    opt = torch.optim.Adam(codec.parameters(), lr=CFG["lr"])
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(tensor_dataset(sp["train"]), batch_size=CFG["batch_size"], shuffle=True, generator=g)
    xb0, yb0 = (t.to(dev) for t in next(iter(loader)))
    grad_info = gradient_check(codec, clf, xb0, yb0) if method != "AE-MSE" else None

    history, best = [], None
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        lam = lam_at(ep, method, lam_max, schedule)
        codec.train()
        tot = dict(rec=0.0, cls=0.0)
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            x_hat, _ = codec(xb)
            l_rec = F.mse_loss(x_hat, xb)
            loss = l_rec
            if method != "AE-MSE":
                l_cls = F.cross_entropy(clf(x_hat), yb)  # no torch.no_grad(): gradient reaches the codec
                loss = l_rec + lam * l_cls
                tot["cls"] += l_cls.item() * len(yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["rec"] += l_rec.item() * len(yb)
        n = len(sp["train"].y)
        val_mse, val_acc = evaluate_split(codec, clf, sp["val"])
        history.append(dict(epoch=ep, lam=lam, train_rec=tot["rec"] / n, train_cls=tot["cls"] / n,
                            val_mse=val_mse, val_acc=val_acc))
        if method == "AE-MSE":
            better = best is None or val_mse < best["val_mse"]
        else:
            eligible = lam == lam_max
            better = eligible and (best is None or (val_acc, -val_mse) > (best["val_acc"], -best["val_mse"]))
        if better:
            best = dict(epoch=ep, val_mse=val_mse, val_acc=val_acc,
                        state=({k: v.detach().cpu().clone() for k, v in codec.state_dict().items()}))
        if verbose:
            print(f"  ep {ep:2d} lam={lam:.2f} rec={tot['rec']/n:.4f} cls={tot['cls']/n:.4f} "
                  f"val_mse={val_mse:.4f} val_acc={val_acc:.4f}", flush=True)

    name = ckpt_name(method, d, fold, seed, lam_max, clf_tag, schedule)
    config = dict(CFG, device=dev, method=method, d=d, r=codec.r, fold=fold, seed=seed, lam_max=lam_max,
                  epochs=n_epochs, schedule=schedule, clf_tag=clf_tag,
                  classifier=f"{'classifiers_' + clf_tag if clf_tag else 'classifiers'}/fold_{fold}_seed_{seed}.pt",
                  encoder_params=count_params(codec.encoder), decoder_params=count_params(codec.decoder))
    torch.save({"state_dict": best["state"], "config": config, "best_epoch": best["epoch"],
                "val_mse": best["val_mse"], "val_acc": best["val_acc"]}, CKPT_DIR / "codecs" / f"{name}.pt")
    (LOG_DIR / "codecs").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(LOG_DIR / "codecs" / f"{name}_history.csv", index=False)
    info = dict(name=name, **{k: config[k] for k in ("method", "d", "fold", "seed", "lam_max")},
                best_epoch=best["epoch"], val_mse=best["val_mse"], val_acc=best["val_acc"],
                first_train_rec=history[0]["train_rec"], last_train_rec=history[-1]["train_rec"],
                seconds=round(time.time() - t0, 1), gradient_check=grad_info)
    save_json(info, LOG_DIR / "codecs" / f"{name}.json")
    print(f"{name}: best ep {best['epoch']} val_mse={best['val_mse']:.4f} val_acc={best['val_acc']:.4f} "
          f"({info['seconds']}s)", flush=True)
    return info


def _job(args):
    return train_one(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["AE-MSE", "Task-AE"])
    ap.add_argument("--ds", type=int, nargs="+", default=[128])
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0], help="lambda_max values (Task-AE)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose checkpoint exists")
    ap.add_argument("--clf-tag", default="", help="use checkpoints/classifiers_<tag> (follow-up classifier)")
    ap.add_argument("--schedule", default="main", choices=list(SCHEDULES))
    args = ap.parse_args()
    ensure_dirs()
    jobs = []
    for m, d, f, s in itertools.product(args.methods, args.ds, args.folds, args.seeds):
        for lam in (args.lams if m == "Task-AE" else [0.0]):
            jobs.append((m, d, f, s, lam if m == "Task-AE" else 1.0, args.threads, args.verbose,
                         args.clf_tag, args.schedule))
    if args.skip_existing:
        jobs = [j for j in jobs if not (CKPT_DIR / "codecs" / f"{ckpt_name(*j[:5], *j[7:])}.pt").exists()]
    print(f"{len(jobs)} codec trainings", flush=True)
    if args.workers == 1:
        res = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_job, jobs))
    print(pd.DataFrame(res).drop(columns=["gradient_check"]).to_string(index=False))


if __name__ == "__main__":
    main()
