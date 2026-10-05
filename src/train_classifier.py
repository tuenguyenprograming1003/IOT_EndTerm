"""Train the reference classifier for each LOSO fold.

Checkpoint selection uses Val accuracy only; Test is evaluated once with the selected checkpoint.
Defaults reproduce the locked main experiment (V1, spec architecture).

Usage:
  python train_classifier.py --folds 1 2 3 4 5 6 --seeds 0                       # main (V1)
  python train_classifier.py --tag v2search_x --arch t1 --dropout 0.3 --aug spec --no-test   # Val-only search
  python train_classifier.py --tag v2 --arch t1 --dropout 0.3 --aug spec+speed     # locked V2 follow-up
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn

from classifier import build_classifier, predict
from dataset import load_fold, load_norm
from features import normalize
from utils import CKPT_DIR, LOG_DIR, PRED_DIR, ensure_dirs, save_json, set_seed

CFG = dict(epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4, optimizer="Adam")


def clf_dir(tag: str = ""):
    d = CKPT_DIR / ("classifiers" if not tag else f"classifiers_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_one(fold: int, seed: int, device: str | None = None, verbose: bool = True, tag: str = "",
              arch: dict | None = None, aug: str = "none", cfg: dict | None = None,
              eval_test: bool = True, patience: int | None = None, threads: int | None = None) -> dict:
    cfg = dict(CFG, **(cfg or {}))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if threads:
        torch.set_num_threads(threads)
    set_seed(seed)
    sp, _, _ = load_fold(fold)
    x_tr = torch.from_numpy(sp["train"].x).unsqueeze(1)
    y_tr = torch.from_numpy(sp["train"].y)
    x_tr_aug = None
    if "speed" in aug:  # waveform-level copies of the *Train* recordings only
        from augment import load_aug_cache

        A, aidx = load_aug_cache()
        mu, sigma = load_norm(fold)
        Aa = A[[aidx[f] for f in sp["train"].meta.filename]]          # (N, K, 64, 32)
        x_tr_aug = torch.from_numpy(normalize(Aa, mu, sigma)).unsqueeze(2)  # (N, K, 1, 64, 32)
    g = torch.Generator().manual_seed(seed)
    model = build_classifier(arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    best, history, since_best = (-1.0, -1, None), [], 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = 0.0
        perm = torch.randperm(len(y_tr), generator=g)
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            xb, yb = x_tr[b], y_tr[b]
            if x_tr_aug is not None:  # each sample: original or one of K speed/noise copies, uniformly
                k = torch.randint(0, x_tr_aug.shape[1] + 1, (len(b),), generator=g)
                use = k > 0
                if use.any():
                    xb = xb.clone()
                    xb[use] = x_tr_aug[b[use], k[use] - 1]
            if "spec" in aug:
                from augment import spec_augment

                xb = spec_augment(xb, g)
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        model.eval()
        val_acc = (predict(model, sp["val"].x).numpy() == sp["val"].y).mean()
        history.append(dict(epoch=ep, train_loss=tot / len(y_tr), val_acc=float(val_acc)))
        if val_acc > best[0]:
            best = (float(val_acc), ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            since_best = 0
        else:
            since_best += 1
            if patience and since_best >= patience:
                break
    model.load_state_dict(best[2])
    model.eval()
    name = f"fold_{fold}_seed_{seed}"
    torch.save({"state_dict": best[2], "fold": fold, "seed": seed, "best_epoch": best[1],
                "val_acc": best[0], "config": dict(cfg, device=device), "arch": arch, "aug": aug, "tag": tag},
               clf_dir(tag) / f"{name}.pt")
    pd.DataFrame(history).to_csv(LOG_DIR / f"classifier{'_' + tag if tag else ''}_{name}_history.csv", index=False)
    res = dict(fold=fold, seed=seed, best_epoch=best[1], epochs_run=len(history), val_acc=best[0])
    if eval_test:
        test_pred = predict(model, sp["test"].x).numpy()
        res["test_acc"] = float((test_pred == sp["test"].y).mean())
        meta = sp["test"].meta
        res["preds"] = pd.DataFrame(dict(filename=meta.filename, speaker=meta.speaker, label=sp["test"].y,
                                         prediction=test_pred, fold=fold, seed=seed,
                                         correct=(test_pred == sp["test"].y).astype(int)))
    if verbose:
        msg = f"[{tag or 'main'}] fold {fold} seed {seed}: best epoch {best[1]}/{len(history)} val_acc={best[0]:.4f}"
        if eval_test:
            msg += f" test({sp['test'].meta.speaker.iloc[0]}) acc={res['test_acc']:.4f}"
        print(msg, flush=True)
    return res


def _job(kw):
    return train_one(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="", help="'' = locked main experiment; otherwise separate ckpt dir/logs")
    ap.add_argument("--arch", default=None, choices=[None, "v1", "t1", "t2"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--aug", default="none", choices=["none", "spec", "speed", "spec+speed"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--wd", type=float, default=CFG["weight_decay"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--no-test", action="store_true", help="Val-only (model selection): Test is not evaluated")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    ensure_dirs()
    arch = dict(name=args.arch, dropout=args.dropout) if args.arch else None
    cfg = dict(lr=args.lr, weight_decay=args.wd, epochs=args.epochs)
    jobs = [dict(fold=f, seed=s, tag=args.tag, arch=arch, aug=args.aug, cfg=cfg, eval_test=not args.no_test,
                 patience=args.patience, threads=args.threads or (1 if args.workers > 1 else 4))
            for s in args.seeds for f in args.folds]
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_job, jobs))
    else:
        res = [_job(j) for j in jobs]
    suffix = f"_{args.tag}" if args.tag else ""
    if not args.no_test:
        preds = pd.concat([r.pop("preds") for r in res], ignore_index=True)
        out = PRED_DIR / f"classifier_original_predictions{suffix}.csv"
        if out.exists():  # merge with predictions of other folds/seeds already on disk
            old = pd.read_csv(out)
            keep = ~old.set_index(["fold", "seed"]).index.isin(list({(r["fold"], r["seed"]) for r in res}))
            preds = pd.concat([old[keep], preds], ignore_index=True)
        preds.sort_values(["seed", "fold", "filename"]).to_csv(out, index=False)
    df = pd.DataFrame(res)
    save_json({"config": dict(CFG, **cfg), "arch": arch, "aug": args.aug, "patience": args.patience,
               "test_evaluated": not args.no_test, "runs": res, "seconds": round(time.time() - t0, 1)},
              LOG_DIR / f"classifier_train{suffix}_seeds_{'_'.join(map(str, args.seeds))}.json")
    print(df.to_string(index=False))
    print(f"mean val_acc = {df.val_acc.mean():.4f}" + (f"   mean test_acc = {df.test_acc.mean():.4f}" if "test_acc" in df else ""))


if __name__ == "__main__":
    main()
