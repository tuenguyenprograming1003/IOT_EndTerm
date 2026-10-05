"""Task 1.7 — 6 Leave-One-Speaker-Out folds (speakers sorted alphabetically).

Fold k (1-based): Test = s_k, Val = s_{k+1 mod 6}, Train = remaining 4 speakers.
"""
from __future__ import annotations

import pandas as pd

from utils import LOG_DIR, MANIFEST_PATH, N_FOLDS, SPLITS_DIR, ensure_dirs, save_json

COLS = ["filename", "path", "label", "speaker", "recording_index", "num_samples"]


def fold_speakers(speakers: list[str], fold: int) -> dict[str, list[str]]:
    i = fold - 1
    test, val = speakers[i], speakers[(i + 1) % len(speakers)]
    train = [s for s in speakers if s not in (test, val)]
    return {"train": train, "val": [val], "test": [test]}


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(MANIFEST_PATH)
    speakers = sorted(df.speaker.unique())
    assert len(speakers) == N_FOLDS
    summary, lines = {}, []
    for fold in range(1, N_FOLDS + 1):
        spk = fold_speakers(speakers, fold)
        parts = {}
        for name, s in spk.items():
            part = df[df.speaker.isin(s)][COLS].sort_values("filename").reset_index(drop=True)
            part.to_csv(SPLITS_DIR / f"fold_{fold}_{name}.csv", index=False)
            parts[name] = part
        tr, va, te = (set(parts[k].speaker) for k in ("train", "val", "test"))
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        files = [set(parts[k].filename) for k in ("train", "val", "test")]
        assert files[0].isdisjoint(files[1]) and files[0].isdisjoint(files[2]) and files[1].isdisjoint(files[2])
        assert sum(len(p) for p in parts.values()) == len(df)
        summary[f"fold_{fold}"] = {k: {"speakers": spk[k], "n": len(parts[k]),
                                       "per_digit": parts[k].label.value_counts().sort_index().tolist()}
                                   for k in spk}
        lines.append(f"Fold {fold}: Test={spk['test'][0]:<9} Val={spk['val'][0]:<9} "
                     f"Train={','.join(spk['train'])}  sizes={len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}  "
                     f"speaker-disjoint=PASS")
    save_json({"speakers_sorted": speakers, "folds": summary}, SPLITS_DIR / "splits_summary.json")
    text = "\n".join(lines)
    print(text)
    (LOG_DIR / "splits_check.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
