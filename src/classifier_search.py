"""Follow-up experiment — classifier V2 search on Train/Val only (Test is never evaluated here).

Selection criterion: mean Val accuracy over the 6 LOSO folds (seed 0).
Output: results/summary/classifier_v2_search_val.csv (one row per config x fold) and a ranked summary.
Usage: python classifier_search.py --stage A --workers 7
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from train_classifier import train_one
from utils import SUMMARY_DIR, ensure_dirs

STAGES = {
    "A": [  # architecture x regularisation x augmentation
        dict(name="A1_v1_base", arch="v1", dropout=0.0, aug="none", epochs=60),
        dict(name="A2_v1_d03_spec", arch="v1", dropout=0.3, aug="spec", epochs=100),
        dict(name="A3_t1_d03_spec", arch="t1", dropout=0.3, aug="spec", epochs=100),
        dict(name="A4_t2_d03_spec", arch="t2", dropout=0.3, aug="spec", epochs=100),
        dict(name="A5_t1_d03_specspeed", arch="t1", dropout=0.3, aug="spec+speed", epochs=100),
        dict(name="A6_v1_d03_specspeed", arch="v1", dropout=0.3, aug="spec+speed", epochs=100),
        dict(name="A7_t1_d03_none", arch="t1", dropout=0.3, aug="none", epochs=60),
        dict(name="A8_t2_d03_specspeed", arch="t2", dropout=0.3, aug="spec+speed", epochs=100),
    ],
    "B": [  # longer training (best epochs of A8/A5 were 91-99/100) + lr / wd / dropout around A8
        dict(name="B1_t2_d03_lr1e-3_wd1e-4", arch="t2", dropout=0.3, aug="spec+speed", epochs=200, patience=40),
        dict(name="B2_t2_d03_lr1e-3_wd1e-3", arch="t2", dropout=0.3, aug="spec+speed", epochs=200, patience=40, wd=1e-3),
        dict(name="B3_t2_d03_lr3e-4_wd1e-4", arch="t2", dropout=0.3, aug="spec+speed", epochs=200, patience=40, lr=3e-4),
        dict(name="B4_t2_d05_lr1e-3_wd1e-4", arch="t2", dropout=0.5, aug="spec+speed", epochs=200, patience=40),
        dict(name="B5_t1_d03_lr1e-3_wd1e-4", arch="t1", dropout=0.3, aug="spec+speed", epochs=200, patience=40),
        dict(name="B6_t2_d05_lr1e-3_wd1e-3", arch="t2", dropout=0.5, aug="spec+speed", epochs=200, patience=40, wd=1e-3),
    ],
}


def run(job):
    c, fold = job
    r = train_one(fold=fold, seed=0, verbose=True, tag=f"search_{c['name']}",
                  arch=dict(name=c["arch"], dropout=c["dropout"]), aug=c["aug"],
                  cfg=dict(epochs=c["epochs"], lr=c.get("lr", 1e-3), weight_decay=c.get("wd", 1e-4)),
                  eval_test=False, patience=c.get("patience"), threads=1)
    return dict(config=c["name"], **{k: c[k] for k in ("arch", "dropout", "aug", "epochs")},
                patience=c.get("patience"), epochs_run=r["epochs_run"],
                lr=c.get("lr", 1e-3), wd=c.get("wd", 1e-4), fold=fold,
                val_acc=r["val_acc"], best_epoch=r["best_epoch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    ensure_dirs()
    jobs = [(c, f) for c in STAGES[a.stage] for f in range(1, 7)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    df = pd.DataFrame(rows)
    out = SUMMARY_DIR / "classifier_v2_search_val.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.config.isin(df.config)], df], ignore_index=True)
    df.to_csv(out, index=False)
    piv = df.pivot_table(index="config", columns="fold", values="val_acc")
    piv["mean"] = piv.mean(1)
    print(piv.sort_values("mean", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
