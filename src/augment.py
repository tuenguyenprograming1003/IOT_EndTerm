"""Follow-up experiment — Train-only data augmentation for the classifier V2 search.

Waveform-level copies (precomputed once, cached): speed perturbation (resample, factors 0.9/0.95/1.05/1.1,
changes tempo + pitch ~ vocal-tract/speaker variation) + white noise at random SNR 20–40 dB, then the
unchanged feature pipeline (fix to 8000 samples -> log-mel 64x32 -> CMVN).
Copies of a recording are only ever used when that recording is in the Train split of a fold;
normalisation statistics are still the fold's original Train mu/sigma.
Random gain is not used: per-recording CMVN removes it.

Feature-level (on the fly, Train batches only): random time shift (roll ±3 frames), SpecAugment
(1 frequency mask <= 8 bins, 1 time mask <= 4 frames, filled with 0 = Train mean), Gaussian noise sigma 0.1.
"""
from __future__ import annotations

import librosa
import numpy as np
import pandas as pd
import torch

from audio import fix_length, load_wav
from features import logmel
from utils import MANIFEST_PATH, PROCESSED_DIR, ROOT

SPEEDS = (0.9, 0.95, 1.05, 1.1)
CACHE = PROCESSED_DIR / "logmel_aug_speed_noise.npz"


def make_copy(y: np.ndarray, speed: float, rng: np.random.Generator) -> np.ndarray:
    ys = librosa.resample(y, orig_sr=8000, target_sr=int(round(8000 / speed)))
    snr_db = rng.uniform(20, 40)
    p = np.mean(ys ** 2) + 1e-12
    ys = ys + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), size=ys.shape)
    return fix_length(ys.astype(np.float32))


def build_cache() -> None:
    man = pd.read_csv(MANIFEST_PATH)
    rng = np.random.default_rng(1234)
    feats = np.zeros((len(man), len(SPEEDS), 64, 32), np.float32)
    for i, p in enumerate(man.path):
        y, _ = load_wav(ROOT / p)
        for k, s in enumerate(SPEEDS):
            feats[i, k] = logmel(make_copy(y, s, rng))
    np.savez_compressed(CACHE, S=feats, filename=man.filename.to_numpy(), speeds=np.array(SPEEDS))
    print("augmented cache:", feats.shape)


def load_aug_cache() -> tuple[np.ndarray, dict[str, int]]:
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE, allow_pickle=True)
    return z["S"], {f: i for i, f in enumerate(z["filename"])}


def spec_augment(xb: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """xb: (B,1,64,32) normalised log-mel. Returns augmented copy."""
    B = xb.shape[0]
    out = xb.clone()
    shifts = torch.randint(-3, 4, (B,), generator=g)
    for i in range(B):
        if shifts[i]:
            out[i] = torch.roll(out[i], int(shifts[i]), dims=-1)
        f = int(torch.randint(0, 9, (1,), generator=g))
        f0 = int(torch.randint(0, 64 - f + 1, (1,), generator=g))
        out[i, :, f0:f0 + f, :] = 0
        t = int(torch.randint(0, 5, (1,), generator=g))
        t0 = int(torch.randint(0, 32 - t + 1, (1,), generator=g))
        out[i, :, :, t0:t0 + t] = 0
    return out + 0.1 * torch.randn(out.shape, generator=g)


if __name__ == "__main__":
    build_cache()
