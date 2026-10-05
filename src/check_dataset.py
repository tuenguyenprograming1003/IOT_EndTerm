"""Task 1.2 checks — validate data/manifest.csv and write a log to results/logs/."""
from __future__ import annotations

import sys

import pandas as pd

from utils import LOG_DIR, MANIFEST_PATH, ensure_dirs


def main() -> int:
    ensure_dirs()
    df = pd.read_csv(MANIFEST_PATH)
    lines, failures = [], []

    def out(s=""):
        lines.append(str(s))

    def check(name, ok):
        out(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    out("=== FSDD dataset check ===")
    out(f"Total files: {len(df)}")
    out(f"Speakers ({df.speaker.nunique()}): {sorted(df.speaker.dropna().unique())}")
    out(f"Labels ({df.label.nunique()}): {sorted(int(x) for x in df.label.dropna().unique())}")
    out("\nFiles per speaker:\n" + df.speaker.value_counts().sort_index().to_string())
    out("\nFiles per digit:\n" + df.label.value_counts().sort_index().to_string())
    ct = pd.crosstab(df.speaker, df.label)
    out("\nFiles per speaker x digit:\n" + ct.to_string())
    out("\nSample rates: " + str(df.sample_rate.value_counts().to_dict()))
    out("Channels: " + str(df.channels.value_counts().to_dict()))
    out("Subtypes: " + str(df.subtype.value_counts().to_dict()))
    d = df.duration_sec
    out(f"Duration (s): min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f} std={d.std():.4f} median={d.median():.4f}")
    n = df.num_samples
    out(f"num_samples: min={n.min()} max={n.max()}  PAD(<8000)={(n < 8000).sum()} KEEP(=8000)={(n == 8000).sum()} CROP(>8000)={(n > 8000).sum()}")

    dup_names = df[df.filename.duplicated(keep=False)]
    dup_hash = df[df.sha256.duplicated(keep=False)].sort_values("sha256")
    out("")
    check("all filenames match {digit}_{speaker}_{index}.wav", bool(df.name_ok.all()))
    check("all files readable (no corrupt files)", bool(df.readable.all()))
    check("total files == 3000", len(df) == 3000)
    check("6 speakers", df.speaker.nunique() == 6)
    check("10 classes 0..9", set(df.label.astype(int)) == set(range(10)))
    check("no label outside 0..9", bool(df.label.between(0, 9).all()))
    check("all sample rates == 8000", bool((df.sample_rate == 8000).all()))
    check("all mono", bool((df.channels == 1).all()))
    check("no duplicate filenames", dup_names.empty)
    check("500 files per speaker", bool((df.speaker.value_counts() == 500).all()))
    check("300 files per digit", bool((df.label.value_counts() == 300).all()))
    check("50 files per speaker x digit", bool((ct.values == 50).all()))
    check("recording_index 0..49 unique per speaker x digit",
          bool(df.groupby(["speaker", "label"]).recording_index.apply(lambda s: sorted(s) == list(range(50))).all()))
    # Identical audio content is reported but not counted as a hard failure of the dataset.
    out(f"[{'PASS' if dup_hash.empty else 'WARN'}] no duplicate content (sha256): {len(dup_hash)} files share a hash")
    if not dup_hash.empty:
        out(dup_hash[["filename", "sha256"]].to_string(index=False))
    if not df.readable.all():
        out(df.loc[~df.readable, ["filename", "error"]].to_string(index=False))

    out(f"\nRESULT: {'ALL CHECKS PASS' if not failures else 'FAILED: ' + '; '.join(failures)}")
    text = "\n".join(lines)
    print(text)
    (LOG_DIR / "dataset_check.log").write_text(text + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
