"""Follow-up extensions on top of the V2 classifier (d=128, seed 0). Reported separately from the main result.

1. Per-fold lambda selection on Validation: for each fold choose lambda_max in {0.1, 0.5, 1, 2} with the
   highest Val accuracy (ties -> 1, then the smaller lambda), lock it, then evaluate on Test once.
2. Slow lambda schedule ablation (lambda 0 for epochs 1-30, ramp 31-60, 1 after; 90 epochs) vs the
   spec schedule (20/40/60), both with lambda_max = 1.
Outputs: results/summary/followup_<tag>_lambda_val_selection.csv, followup_<tag>_schedule_ablation.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from classifier import load_classifier, predict
from dataset import load_fold
from evaluate import codec_roundtrip, load_codec
from utils import CKPT_DIR, SUMMARY_DIR

LAMS = (0.1, 0.5, 1.0, 2.0)


def acc_mse(codec, clf, split):
    xh = codec_roundtrip(codec, split.x, "Task-AE")
    return float((predict(clf, xh).numpy() == split.y).mean()), float(((xh - split.x) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    t = a.tag
    sel_rows, sch_rows = [], []
    for fold in range(1, 7):
        clf = load_classifier(CKPT_DIR / f"classifiers_{t}" / f"fold_{fold}_seed_0.pt")
        sp, _, _ = load_fold(fold)
        cand = []
        for lam in LAMS:
            name = f"Task-AE_d128_fold{fold}_seed0" + ("" if lam == 1.0 else f"_lam{lam:g}") + f"_{t}"
            c = load_codec(name)
            if c is None:
                continue
            va, vm = acc_mse(c, clf, sp["val"])
            cand.append(dict(lam=lam, val_acc=va, val_mse=vm, codec=c))
        # selection uses Validation only
        best = max(cand, key=lambda r: (r["val_acc"], r["lam"] == 1.0, -r["lam"]))
        te_sel, _ = acc_mse(best["codec"], clf, sp["test"])
        fixed = next(r for r in cand if r["lam"] == 1.0)
        te_fix, _ = acc_mse(fixed["codec"], clf, sp["test"])
        sel_rows.append(dict(fold=fold, test_speaker=sp["test"].meta.speaker.iloc[0],
                             val_acc_by_lambda={r["lam"]: round(r["val_acc"], 4) for r in cand},
                             selected_lambda=best["lam"], test_acc_selected=te_sel, test_acc_lambda1=te_fix))
        slow = load_codec(f"Task-AE_d128_fold{fold}_seed0_{t}_slow")
        if slow is not None:
            for sched, c in (("main 20/40/60", fixed["codec"]), ("slow 30/60/90", slow)):
                va, vm = acc_mse(c, clf, sp["val"])
                ta, tm = acc_mse(c, clf, sp["test"])
                sch_rows.append(dict(fold=fold, schedule=sched, val_acc=va, val_mse=vm, test_acc=ta, test_mse=tm))
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(SUMMARY_DIR / f"followup_{t}_lambda_val_selection.csv", index=False)
    print(sel.to_string(index=False))
    print(f"mean Test acc: per-fold Val-selected lambda {sel.test_acc_selected.mean():.4f} | fixed lambda=1 {sel.test_acc_lambda1.mean():.4f}")
    if sch_rows:
        sch = pd.DataFrame(sch_rows)
        sch.to_csv(SUMMARY_DIR / f"followup_{t}_schedule_ablation.csv", index=False)
        print(sch.groupby("schedule")[["val_acc", "val_mse", "test_acc", "test_mse"]].mean().round(4))


if __name__ == "__main__":
    main()
