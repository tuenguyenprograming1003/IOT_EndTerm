"""Shared paths and helpers for the C3 log-mel compression project."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FSDD_DIR = DATA_DIR / "fsdd"
RECORDINGS_DIR = FSDD_DIR / "recordings"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"
PRED_DIR = RESULTS_DIR / "predictions"
SUMMARY_DIR = RESULTS_DIR / "summary"

N_FOLDS = 6
D_BUDGETS = (64, 128, 256)
HEADER_BYTES = 4
FP32_BYTES = 64 * 32 * 4 + HEADER_BYTES  # 8196
INT8_BYTES = 64 * 32 + HEADER_BYTES      # 2052


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, SPLITS_DIR, FIG_DIR, LOG_DIR, PRED_DIR, SUMMARY_DIR,
              CKPT_DIR / "classifiers", CKPT_DIR / "codecs"):
        d.mkdir(parents=True, exist_ok=True)
