"""Task 6.1 — real byte-stream round trip in two separate processes.

  python packet_roundtrip.py node    --method Task-AE --d 128 --fold 1 --seed 0
      encodes every Test recording of the fold, writes one binary packet per recording to
      results/packets/<run>/<filename>.bin plus labels.csv (filename, label only — no features).
  python packet_roundtrip.py gateway --method Task-AE --d 128 --fold 1 --seed 0
      runs in a fresh process: reads only the .bin files, parses header, dequantizes, decodes,
      classifies, and compares with the in-memory reference pipeline.
  python packet_roundtrip.py both ...   runs node then gateway as two subprocesses.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from utils import CKPT_DIR, LOG_DIR, RESULTS_DIR, ensure_dirs, save_json


def run_dir(a) -> "Path":
    return RESULTS_DIR / "packets" / f"{a.method}_d{a.d}_fold{a.fold}_seed{a.seed}"


def node(a):
    import packet
    from dataset import load_fold
    from evaluate import load_codec

    codec = load_codec(f"{a.method}_d{a.d}_fold{a.fold}_seed{a.seed}")
    sp, _, _ = load_fold(a.fold)
    te = sp["test"]
    out = run_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    q = codec.encode_int8(torch.from_numpy(te.x).unsqueeze(1)).numpy()
    for fn, qi in zip(te.meta.filename, q):
        (out / f"{fn}.bin").write_bytes(packet.encode(qi, a.method, a.d))
    te.meta[["filename", "label"]].to_csv(out / "labels.csv", index=False)
    del te, q, sp  # the original features are not passed to the gateway
    print(f"[node] wrote {len(list(out.glob('*.bin')))} packets to {out}")


def gateway(a):
    import packet
    from classifier import load_classifier, predict
    from dataset import load_fold
    from evaluate import codec_roundtrip, load_codec

    out = run_dir(a)
    labels = pd.read_csv(out / "labels.csv")
    codec = load_codec(f"{a.method}_d{a.d}_fold{a.fold}_seed{a.seed}")
    clf = load_classifier(CKPT_DIR / "classifiers" / f"fold_{a.fold}_seed_{a.seed}.pt")
    sizes, payloads = [], []
    for fn in labels.filename:
        buf = (out / f"{fn}.bin").read_bytes()
        sizes.append(len(buf))
        p = packet.decode(buf)
        assert p.config_id == a.d and p.method_id == packet.METHOD_IDS[a.method]
        payloads.append(p.payload)
    q = np.stack(payloads)
    x_hat = codec.decode_int8(torch.from_numpy(q))[:, 0].numpy()
    pred_rx = predict(clf, x_hat).numpy()
    # Reference: in-memory pipeline (loads features independently, only for comparison)
    sp, _, _ = load_fold(a.fold)
    pred_ref = predict(clf, codec_roundtrip(codec, sp["test"].x, a.method)).numpy()
    assert list(sp["test"].meta.filename) == list(labels.filename)
    res = dict(method=a.method, d=a.d, fold=a.fold, seed=a.seed, n_packets=len(sizes),
               packet_bytes_min=min(sizes), packet_bytes_max=max(sizes), expected_bytes=a.d + 4,
               bytes_ok=set(sizes) == {a.d + 4},
               accuracy_from_packets=float((pred_rx == labels.label.to_numpy()).mean()),
               prediction_agreement_with_reference=float((pred_rx == pred_ref).mean()))
    save_json(res, LOG_DIR / f"packet_roundtrip_{out.name}.json")
    print("[gateway]", res)
    assert res["bytes_ok"] and res["prediction_agreement_with_reference"] == 1.0
    print("[gateway] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["node", "gateway", "both"])
    ap.add_argument("--method", default="Task-AE")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ensure_dirs()
    if a.role == "node":
        node(a)
    elif a.role == "gateway":
        gateway(a)
    else:
        common = ["--method", a.method, "--d", str(a.d), "--fold", str(a.fold), "--seed", str(a.seed)]
        for role in ("node", "gateway"):
            subprocess.run([sys.executable, __file__, role, *common], check=True)


if __name__ == "__main__":
    main()
