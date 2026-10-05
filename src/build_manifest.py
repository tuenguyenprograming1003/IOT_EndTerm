"""Task 1.2 — scan FSDD recordings and build data/manifest.csv."""
from __future__ import annotations

import re

import pandas as pd
import soundfile as sf

from utils import MANIFEST_PATH, RECORDINGS_DIR, ROOT, sha256_file

NAME_RE = re.compile(r"^(\d+)_([A-Za-z]+)_(\d+)\.wav$")


def build() -> pd.DataFrame:
    rows = []
    for p in sorted(RECORDINGS_DIR.glob("*.wav")):
        m = NAME_RE.match(p.name)
        row = dict(filename=p.name, path=str(p.relative_to(ROOT)),
                   label=None, speaker=None, recording_index=None,
                   sample_rate=None, num_samples=None, duration_sec=None,
                   channels=None, subtype=None, sha256=sha256_file(p),
                   name_ok=bool(m), readable=True, error="")
        if m:
            row["label"], row["speaker"], row["recording_index"] = int(m[1]), m[2], int(m[3])
        try:
            info = sf.info(str(p))
            row.update(sample_rate=info.samplerate, num_samples=info.frames,
                       duration_sec=info.frames / info.samplerate,
                       channels=info.channels, subtype=info.subtype)
            if info.frames == 0:
                raise ValueError("empty file")
            sf.read(str(p), dtype="float32")  # make sure the payload decodes
        except Exception as e:  # corrupted file
            row["readable"], row["error"] = False, repr(e)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {MANIFEST_PATH} ({len(df)} rows)")
