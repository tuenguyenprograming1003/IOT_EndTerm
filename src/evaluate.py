"""Evaluate all methods on the Test split of every fold/seed and write the result CSVs.

Methods: Original (FP32), INT8 (direct), DCT d, Low-Mel d, AE-MSE d, Task-AE d.
The same classifier (fold, seed) is used for every method of that fold/seed.
Codec outputs go through the real path: encoder -> int8 -> packet bytes -> parse -> decoder.

Outputs:
  results/predictions/all_predictions.csv
  results/summary/results_summary.csv          (one row per method/fold/seed/d)
  results/summary/results_summary_mean.csv     (mean ± std over folds, seed 0, and over seeds at d=128)
  results/summary/lambda_ablation_val.csv      (with --ablation; Val split only)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import packet
from baselines import dct_keep, direct_int8, fit_dct_clip, lowmel_keep
from classifier import load_classifier, predict
from codec import Codec
from dataset import load_fold
from features import denormalize
from utils import (CKPT_DIR, D_BUDGETS, FP32_BYTES, INT8_BYTES, N_FOLDS, PRED_DIR, SUMMARY_DIR,
                   ensure_dirs)


def load_codec(name: str) -> Codec | None:
    path = CKPT_DIR / "codecs" / f"{name}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = Codec(ck["config"]["d"])
    c.load_state_dict(ck["state_dict"])
    return c.eval()


def codec_roundtrip(codec: Codec, x: np.ndarray, method: str) -> np.ndarray:
    """Node: encode -> INT8 -> packet bytes. Gateway: parse bytes -> dequantize -> decode."""
    q = codec.encode_int8(torch.from_numpy(x).unsqueeze(1)).numpy()
    bufs = [packet.encode(qi, method, codec.d) for qi in q]
    assert all(len(b) == codec.d + packet.HEADER.size for b in bufs)
    q_rx = np.stack([packet.decode(b).payload for b in bufs])
    return codec.decode_int8(torch.from_numpy(q_rx))[:, 0].numpy()


def per_sample_metrics(x, x_hat, mu, sigma):
    mse = ((x - x_hat) ** 2).mean(axis=(1, 2))
    S, S_hat = denormalize(x, mu, sigma), denormalize(x_hat, mu, sigma)
    prd = 100 * np.sqrt(((S - S_hat) ** 2).sum(axis=(1, 2)) / (S ** 2).sum(axis=(1, 2)))
    return mse, prd


def reconstructions(split_x, d_list, x_train, fold, seed, tag=""):
    """Yield (method, d, bytes, x_hat) for every method available for this fold/seed."""
    yield "Original", 2048, FP32_BYTES, split_x
    yield "INT8", 2048, INT8_BYTES, direct_int8(split_x)
    for d in d_list:
        yield "DCT", d, d + 4, dct_keep(split_x, d, clip=fit_dct_clip(x_train, d))
        yield "Low-Mel", d, d + 4, lowmel_keep(split_x, d)
        for method in ("AE-MSE", "Task-AE"):
            c = load_codec(f"{method}_d{d}_fold{fold}_seed{seed}" + (f"_{tag}" if tag else ""))
            if c is not None:
                yield method, d, d + 4, codec_roundtrip(c, split_x, method)


def evaluate_all(seeds, tag="", ds=D_BUDGETS):
    rows, preds = [], []
    for seed in seeds:
        for fold in range(1, N_FOLDS + 1):
            ck = CKPT_DIR / ("classifiers" if not tag else f"classifiers_{tag}") / f"fold_{fold}_seed_{seed}.pt"
            if not ck.exists():
                continue
            clf = load_classifier(ck)
            sp, mu, sigma = load_fold(fold)
            te = sp["test"]
            test_spk = te.meta.speaker.iloc[0]
            A0 = None
            for method, d, nbytes, x_hat in reconstructions(te.x, ds, sp["train"].x, fold, seed, tag):
                yhat = predict(clf, x_hat).numpy()
                mse, prd = per_sample_metrics(te.x, x_hat, mu, sigma)
                acc = float((yhat == te.y).mean())
                if method == "Original":
                    A0 = acc
                preds.append(pd.DataFrame(dict(
                    filename=te.meta.filename, speaker=te.meta.speaker, label=te.y, prediction=yhat,
                    correct=(yhat == te.y).astype(int), method=method, fold=fold, test_speaker=test_spk,
                    seed=seed, d=d, bytes=nbytes, mse=mse, prd_spec=prd)))
                rows.append(dict(method=method, fold=fold, test_speaker=test_spk, seed=seed, d=d,
                                 bytes=nbytes, CR_INT8=INT8_BYTES / nbytes, CR_FP32=FP32_BYTES / nbytes,
                                 Accuracy=acc, Macro_F1=f1_score(te.y, yhat, average="macro"),
                                 Delta_A_pp=100 * (acc - A0), R_acc=100 * acc / A0,
                                 MSE=float(mse.mean()), PRD_spec=float(prd.mean())))
            print(f"seed {seed} fold {fold} ({test_spk}) done", flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def mean_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["Accuracy", "Macro_F1", "Delta_A_pp", "R_acc", "MSE", "PRD_spec"]
    out = []
    s0 = summary[summary.seed == 0]
    g = s0.groupby(["method", "d", "bytes"])
    t = g[cols].agg(["mean", "std"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.join(g.size().rename("n_runs")).reset_index()
    t.insert(0, "scope", "seed 0, 6 folds")
    out.append(t)
    m = summary[(summary.d.isin([128, 2048]))]
    seeds_avail = m.groupby(["method", "d"]).seed.nunique()
    if (seeds_avail > 1).any():
        g = m.groupby(["method", "d", "bytes"])
        t = g[cols].agg(["mean", "std"])
        t.columns = [f"{a}_{b}" for a, b in t.columns]
        t = t.join(g.size().rename("n_runs")).reset_index()
        t.insert(0, "scope", "all seeds x 6 folds")
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["CR_INT8"] = INT8_BYTES / t.bytes
    t["CR_FP32"] = FP32_BYTES / t.bytes
    return t


def lambda_ablation(lams=(0.0, 0.1, 0.5, 1.0, 2.0), d=128, seed=0):
    """Val-split accuracy/MSE of Task-AE trained with different lambda_max (Test is not touched)."""
    rows = []
    for fold in range(1, N_FOLDS + 1):
        clf = load_classifier(CKPT_DIR / "classifiers" / f"fold_{fold}_seed_{seed}.pt")
        sp, mu, sigma = load_fold(fold)
        va = sp["val"]
        for lam in lams:
            name = f"Task-AE_d{d}_fold{fold}_seed{seed}" + ("" if lam == 1.0 else f"_lam{lam:g}")
            c = load_codec(name)
            if c is None:
                continue
            x_hat = codec_roundtrip(c, va.x, "Task-AE")
            yhat = predict(clf, x_hat).numpy()
            mse, prd = per_sample_metrics(va.x, x_hat, mu, sigma)
            rows.append(dict(lambda_max=lam, fold=fold, val_speaker=va.meta.speaker.iloc[0], seed=seed, d=d,
                             Val_Accuracy=float((yhat == va.y).mean()),
                             Val_Macro_F1=f1_score(va.y, yhat, average="macro"),
                             Val_MSE=float(mse.mean()), Val_PRD_spec=float(prd.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_DIR / "lambda_ablation_val.csv", index=False)
    print(df.groupby("lambda_max")[["Val_Accuracy", "Val_MSE"]].agg(["mean", "std"]).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--tag", default="", help="follow-up classifier tag (classifiers_<tag>, codecs *_<tag>)")
    ap.add_argument("--ds", type=int, nargs="+", default=list(D_BUDGETS))
    args = ap.parse_args()
    ensure_dirs()
    if args.ablation:
        lambda_ablation()
        return
    summary, preds = evaluate_all(args.seeds, args.tag, args.ds)
    sfx = f"_{args.tag}" if args.tag else ""
    summary["experiment"] = args.tag or "main"
    preds.to_csv(PRED_DIR / f"all_predictions{sfx}.csv", index=False)
    summary.to_csv(SUMMARY_DIR / f"results_summary{sfx}.csv", index=False)
    mt = mean_table(summary)
    mt.to_csv(SUMMARY_DIR / f"results_summary_mean{sfx}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(mt[["scope", "method", "d", "bytes", "Accuracy_mean", "Accuracy_std", "Macro_F1_mean",
                  "Delta_A_pp_mean", "R_acc_mean", "MSE_mean", "PRD_spec_mean", "n_runs"]].to_string(index=False))


if __name__ == "__main__":
    main()
