"""Task 1.5 — log-mel extraction (64 mel x 32 frames) and Task 1.8 normalisation helpers.

Pipeline: 8000 samples -> mel power P (64x32) -> log(P + 1e-6) -> per-recording CMVN -> S.
Per fold, S is further normalised per mel bin with Train-only statistics.
"""
from __future__ import annotations

import numpy as np
import librosa

MEL_CFG = dict(
    sr=8000,
    n_fft=256,
    win_length=256,
    hop_length=256,
    window="hann",
    center=True,
    pad_mode="constant",
    n_mels=64,
    fmin=50.0,
    fmax=4000.0,
    power=2.0,
    htk=False,
    norm="slaney",
)
N_MELS, N_FRAMES = 64, 32
EPS_SIGMA = 1e-6


def mel_power(audio: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(y=audio, **MEL_CFG).astype(np.float32)


LOG_EPS = 1e-6


def logmel_raw(audio: np.ndarray) -> np.ndarray:
    """S_log = log(P + 1e-6), shape (64, 32).

    Deviation from the original spec (log1p(P)), approved by the project owner: for float audio
    in [-1, 1], P << 1 so log1p(P) ~= P (linear power, speaker-gain dependent).
    See results/logs/diagnostics/feature_variant_diagnostic.md.
    """
    S = np.log(mel_power(audio) + LOG_EPS).astype(np.float32)
    assert S.shape == (N_MELS, N_FRAMES), S.shape
    return S


def cmvn(S: np.ndarray) -> np.ndarray:
    """Per-recording CMVN over all 64x32 values of each recording (removes recording gain)."""
    m = S.mean(axis=(-2, -1), keepdims=True)
    s = S.std(axis=(-2, -1), keepdims=True)
    return ((S - m) / (s + EPS_SIGMA)).astype(np.float32)


def logmel(audio: np.ndarray) -> np.ndarray:
    """Project log-mel feature S = CMVN(log(P + 1e-6)), shape (64, 32).

    This S is the 'pre-normalisation' domain used for PRD_spec; the Train-only per-mel-bin
    normalisation (Task 1.8) is applied on top of it.
    """
    return cmvn(logmel_raw(audio))


def fit_normalization(S_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-mel-bin mean/std over Train samples and frames only. S_train: (N, 64, 32)."""
    mu = S_train.mean(axis=(0, 2)).astype(np.float32)
    sigma = S_train.std(axis=(0, 2)).astype(np.float32)
    return mu, sigma


def normalize(S: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((S - mu[:, None]) / np.maximum(sigma, EPS_SIGMA)[:, None]).astype(np.float32)


def denormalize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x * np.maximum(sigma, EPS_SIGMA)[:, None] + mu[:, None]).astype(np.float32)


if __name__ == "__main__":
    # Extract log-mel for every recording and cache to data/processed/logmel_all.npz
    import warnings

    import pandas as pd

    from audio import load_fixed
    from utils import MANIFEST_PATH, PROCESSED_DIR, ROOT, ensure_dirs, save_json

    ensure_dirs()
    df = pd.read_csv(MANIFEST_PATH)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        feats = np.stack([logmel(load_fixed(ROOT / p)) for p in df["path"]])
        warns = sorted({str(x.message) for x in w})
    assert feats.shape == (len(df), 64, 32)
    np.savez_compressed(
        PROCESSED_DIR / "logmel_all.npz",
        S=feats,
        filename=df["filename"].to_numpy(),
        label=df["label"].to_numpy(),
        speaker=df["speaker"].to_numpy(),
    )
    save_json({"mel_config": MEL_CFG, "log": "log(P + 1e-6)", "per_recording": "CMVN over 64x32",
               "spec_deviation": "spec used log1p(P); changed to log(P+1e-6)+CMVN (owner decision 2026-10-05)",
               "shape": [64, 32],
               "n_samples": int(len(df)), "librosa_version": librosa.__version__,
               "warnings": warns}, PROCESSED_DIR / "feature_config.json")
    print("features:", feats.shape, feats.dtype, "min", feats.min(), "max", feats.max())
    print("librosa warnings:", warns or "none")
