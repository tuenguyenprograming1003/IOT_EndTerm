"""Section 38 — pre-flash verification: compile the firmware C kernels for the host and compare
against the PyTorch reference on every Test recording of the export fold.

Checks (per d):
  Level 1: Python normalised log-mel -> C encoder -> int8   vs  PyTorch encoder -> int8
  Level 2: audio -> C preprocessing -> C encoder -> int8     vs  Python features + PyTorch encoder
  Gateway: C int8 latents -> packet -> PyTorch decoder -> classifier; prediction agreement
Output: results/hardware/host_c_consistency.csv
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import pandas as pd
import torch

import packet
from audio import load_fixed
from classifier import load_classifier, predict
from dataset import load_fold
from evaluate import load_codec
from quantizer import quantize
from utils import CKPT_DIR, RESULTS_DIR, ROOT

ESP = ROOT / "esp32"


def build(d: int, fold: int) -> str:
    exe = ESP / "host_test" / f"host_test_d{d}"
    src = [ESP / "host_test" / "host_test.c", ESP / "firmware" / "main" / "encoder.c",
           ESP / "firmware" / "main" / "preproc.c", ESP / "model" / f"model_d{d}.c",
           ESP / "model" / f"preproc_fold{fold}.c", ESP / "model" / "test_vectors.c"]
    subprocess.run(["cc", "-O2", "-std=c11", "-D_DEFAULT_SOURCE", "-I", str(ESP / "firmware" / "main"),
                    *map(str, src), "-lm", "-o", str(exe)], check=True)
    return str(exe)


def run(exe: str, mode: str, payload: np.ndarray) -> bytes:
    return subprocess.run([exe, mode], input=payload.astype(np.float32).tobytes(),
                          capture_output=True, check=True).stdout


def verify(d: int, fold: int, seed: int) -> dict:
    exe = build(d, fold)
    sp, _, _ = load_fold(fold)
    te = sp["test"]
    codec = load_codec(f"Task-AE_d{d}_fold{fold}_seed{seed}")
    clf = load_classifier(CKPT_DIR / "classifiers" / f"fold_{fold}_seed_{seed}.pt")
    with torch.no_grad():
        q_ref = quantize(codec.encoder(torch.from_numpy(te.x).unsqueeze(1))).numpy().reshape(len(te.x), -1)
    n = len(te.x)
    q_l1 = np.frombuffer(run(exe, "L", te.x), np.int8).reshape(n, d)
    audio = np.stack([load_fixed(ROOT / p) for p in te.meta.path])
    out = np.frombuffer(run(exe, "A", audio), np.uint8).reshape(n, 2048 * 4 + d)
    x_l2 = out[:, :8192].copy().view(np.float32).reshape(n, 64, 32)
    q_l2 = out[:, 8192:].copy().view(np.int8)

    def gateway(q):
        bufs = [packet.encode(qi, "Task-AE", d) for qi in q]
        rx = np.stack([packet.decode(b).payload for b in bufs])
        return predict(clf, codec.decode_int8(torch.from_numpy(rx))[:, 0]).numpy()

    p_ref, p_l1, p_l2 = gateway(q_ref), gateway(q_l1), gateway(q_l2)
    diff1 = np.abs(q_l1.astype(int) - q_ref)
    diff2 = np.abs(q_l2.astype(int) - q_ref)
    res = dict(d=d, fold=fold, seed=seed, n=n,
               L1_exact_value_ratio=float((diff1 == 0).mean()), L1_max_abs_q_diff=int(diff1.max()),
               L1_mae_q=float(diff1.mean()), L1_exact_vector_ratio=float((diff1.max(1) == 0).mean()),
               L1_prediction_agreement=float((p_l1 == p_ref).mean()),
               L2_logmel_max_abs_err=float(np.abs(x_l2 - te.x).max()),
               L2_logmel_mean_abs_err=float(np.abs(x_l2 - te.x).mean()),
               L2_exact_value_ratio=float((diff2 == 0).mean()), L2_max_abs_q_diff=int(diff2.max()),
               L2_prediction_agreement=float((p_l2 == p_ref).mean()),
               acc_reference=float((p_ref == te.y).mean()), acc_L1_c=float((p_l1 == te.y).mean()),
               acc_L2_c=float((p_l2 == te.y).mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (RESULTS_DIR / "hardware").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([verify(d, a.fold, a.seed) for d in a.ds])
    df.to_csv(RESULTS_DIR / "hardware" / "host_c_consistency.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
