"""Fold-level data access: cached log-mel features + Train-only normalisation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import normalize
from utils import PROCESSED_DIR, SPLITS_DIR


def load_logmel_cache() -> tuple[np.ndarray, dict[str, int]]:
    z = np.load(PROCESSED_DIR / "logmel_all.npz", allow_pickle=True)
    S = z["S"]
    index = {f: i for i, f in enumerate(z["filename"])}
    return S, index


def load_norm(fold: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(PROCESSED_DIR / f"fold_{fold}_normalization.npz")
    return z["mu"], z["sigma"]


@dataclass
class Split:
    meta: pd.DataFrame   # filename, label, speaker, ...
    S: np.ndarray        # raw log-mel (N, 64, 32)
    x: np.ndarray        # normalised log-mel (N, 64, 32)
    y: np.ndarray        # labels (N,)


def load_fold(fold: int) -> tuple[dict[str, Split], np.ndarray, np.ndarray]:
    S_all, index = load_logmel_cache()
    mu, sigma = load_norm(fold)
    out = {}
    for split in ("train", "val", "test"):
        meta = pd.read_csv(SPLITS_DIR / f"fold_{fold}_{split}.csv")
        S = S_all[[index[f] for f in meta.filename]]
        out[split] = Split(meta, S, normalize(S, mu, sigma), meta.label.to_numpy(np.int64))
    return out, mu, sigma


def tensor_dataset(split: Split):
    import torch
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.from_numpy(split.x).unsqueeze(1), torch.from_numpy(split.y))
