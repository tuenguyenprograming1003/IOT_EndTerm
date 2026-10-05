"""Task 1.4 — load FSDD audio and fix it to exactly 1 s (8000 samples @ 8 kHz).

Rules:
  N <  8000 -> zero-pad at the end   (PAD)
  N == 8000 -> keep                  (KEEP)
  N >  8000 -> centre crop, start = (N - 8000) // 2   (CROP)
No resampling is performed: files whose sample rate is not 8000 Hz raise an error.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

SAMPLE_RATE = 8000
TARGET_LEN = 8000


def load_wav(path) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 mono. Multi-channel input is averaged to mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    return audio.astype(np.float32), sr


def length_action(n: int, target: int = TARGET_LEN) -> str:
    if n < target:
        return "PAD"
    if n > target:
        return "CROP"
    return "KEEP"


def fix_length(audio: np.ndarray, target: int = TARGET_LEN) -> np.ndarray:
    n = len(audio)
    if n < target:
        out = np.zeros(target, dtype=np.float32)
        out[:n] = audio
    elif n > target:
        start = (n - target) // 2
        out = audio[start:start + target]
    else:
        out = audio
    return np.ascontiguousarray(out, dtype=np.float32)


def load_fixed(path) -> np.ndarray:
    audio, sr = load_wav(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE}; resampling is not allowed")
    out = fix_length(audio)
    assert len(out) == TARGET_LEN and out.dtype == np.float32
    return out


if __name__ == "__main__":
    # Test over every file in the manifest and save pad/crop statistics.
    import pandas as pd

    from utils import LOG_DIR, MANIFEST_PATH, ROOT, ensure_dirs

    ensure_dirs()
    df = pd.read_csv(MANIFEST_PATH)
    for p in df["path"]:
        a = load_fixed(ROOT / p)
        assert len(a) == 8000
        assert a.dtype == np.float32
    df["action"] = df["num_samples"].map(length_action)
    overall = df["action"].value_counts().reindex(["PAD", "KEEP", "CROP"], fill_value=0)
    by_spk = pd.crosstab(df["speaker"], df["action"]).reindex(columns=["PAD", "KEEP", "CROP"], fill_value=0)
    by_dig = pd.crosstab(df["label"], df["action"]).reindex(columns=["PAD", "KEEP", "CROP"], fill_value=0)
    overall.to_csv(LOG_DIR / "audio_length_actions_overall.csv", header=["count"])
    by_spk.to_csv(LOG_DIR / "audio_length_actions_by_speaker.csv")
    by_dig.to_csv(LOG_DIR / "audio_length_actions_by_digit.csv")
    print(f"All {len(df)} files -> 8000 float32 samples: PASS")
    print(overall.to_string())
    print(by_spk.to_string())
    print(by_dig.to_string())
