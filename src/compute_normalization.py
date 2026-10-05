"""Task 1.8 — per-fold, Train-only per-mel-bin normalisation statistics.

mu_f, sigma_f are computed from the Train split of each fold only (over samples and frames)
and saved to data/processed/fold_{k}_normalization.npz. Val/Test reuse these statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset import load_logmel_cache
from features import fit_normalization, normalize
from utils import LOG_DIR, N_FOLDS, PROCESSED_DIR, SPLITS_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    S_all, index = load_logmel_cache()
    lines = []
    for fold in range(1, N_FOLDS + 1):
        tr = pd.read_csv(SPLITS_DIR / f"fold_{fold}_train.csv")
        S_tr = S_all[[index[f] for f in tr.filename]]
        mu, sigma = fit_normalization(S_tr)
        assert mu.shape == (64,) and sigma.shape == (64,)
        np.savez(PROCESSED_DIR / f"fold_{fold}_normalization.npz", mu=mu, sigma=sigma,
                 train_speakers=np.array(sorted(tr.speaker.unique())), n_train=len(tr))
        x_tr = normalize(S_tr, mu, sigma)
        msg = [f"Fold {fold}: train speakers={sorted(tr.speaker.unique())} n={len(tr)} "
               f"sigma[min,max]=[{sigma.min():.4f},{sigma.max():.4f}] "
               f"train-normalised mean={x_tr.mean():+.2e} std={x_tr.std():.4f}"]
        for split in ("val", "test"):
            df = pd.read_csv(SPLITS_DIR / f"fold_{fold}_{split}.csv")
            x = normalize(S_all[[index[f] for f in df.filename]], mu, sigma)
            msg.append(f"   {split}: mean={x.mean():+.4f} std={x.std():.4f} (uses Train stats)")
        lines.extend(msg)
    text = "\n".join(lines)
    print(text)
    (LOG_DIR / "normalization_check.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
