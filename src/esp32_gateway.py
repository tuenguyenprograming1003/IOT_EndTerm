"""Sections 40–45 — PC gateway for the ESP32-S3 node.

  python esp32_gateway.py --port /dev/cu.usbserial-XXXX --d 128 [--mode L|A|both] [--n 500] [--start reset]

1. (optionally) resets the board, captures the boot log until [READY]
   -> results/hardware/serial_logs/<run>.log, parses [MODEL]/[BOARD]/[MEM]/[CHECK]/[RUN]
   -> appends per-run rows to results/hardware/hardware_benchmark.csv
2. streams the Test recordings of the export fold to the board:
     mode L: normalised log-mel (Python features) -> board encoder      (Level 1)
     mode A: raw 1-s int16 audio -> board preprocessing + encoder        (Level 2)
   receives the real packet bytes, parses them on the PC, dequantizes, decodes, classifies
   -> results/hardware/hardware_predictions.csv
The port can also be "socket://localhost:5555" for the ESP-IDF QEMU target (protocol test only;
QEMU latencies are not hardware measurements).
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import packet
from audio import fix_length
from classifier import load_classifier, predict
from dataset import load_fold
from evaluate import load_codec
from quantizer import quantize
from utils import CKPT_DIR, RESULTS_DIR, ROOT

HW = RESULTS_DIR / "hardware"


def open_port(port: str, baud: int):
    import serial

    ser = serial.serial_for_url(port, baudrate=baud, timeout=1, do_not_open=True)
    ser.dtr = False  # do not toggle EN/IO0 when opening (USB-Serial-JTAG maps DTR/RTS to reset/boot)
    ser.rts = False
    ser.open()
    return ser


def reset_board(ser) -> None:
    """Classic esptool auto-reset: EN low via RTS, IO0 high via DTR."""
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    time.sleep(0.05)


def read_until_ready(ser, log_f, timeout_s: float = 600, poke_after: float | None = None) -> list[str]:
    lines, t0, last = [], time.time(), time.time()
    while time.time() - t0 < timeout_s:
        raw = ser.readline()
        if not raw:
            if poke_after and time.time() - last > poke_after:
                ser.write(b"I")  # board is idle at [READY] (boot output already consumed): ask for info
                last = time.time()
            continue
        last = time.time()
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        log_f.write(line + "\n")
        lines.append(line)
        if line.startswith("[READY]"):
            return lines
    raise TimeoutError("board did not print [READY]")


def parse_boot(lines: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[dict]]:
    kv, mem, checks, runs = {}, [], [], []
    for ln in lines:
        m = re.match(r"^([a-z_]+)=(.*)$", ln)
        if m:
            kv[m[1]] = m[2]
        if ln.startswith("[MEM] stage="):
            mem.append(dict(re.findall(r"(\w+)=(\S+)", ln[6:])))
        elif ln.startswith("[MEM] activation_buffers_bytes="):
            kv["activation_buffers_bytes"] = ln.split("=")[1]
        elif ln.startswith("[CHECK] sample="):
            checks.append(dict(re.findall(r"(\w+)=(\S+)", ln[8:])))
        elif ln.startswith("[RUN] "):
            i, pre, enc, q, pk = map(int, ln.split()[1:6])
            runs.append(dict(run=i, preprocess_ms=pre / 1e3, encoder_ms=enc / 1e3,
                             quantization_ms=q / 1e3, packetization_ms=pk / 1e3))
    return kv, pd.DataFrame(runs), pd.DataFrame(checks), mem


def stream(ser, d: int, fold: int, seed: int, mode: str, n: int, log_f) -> pd.DataFrame:
    sp, _, _ = load_fold(fold)
    te = sp["test"]
    idx = np.arange(len(te.y))[:n]
    codec = load_codec(f"Task-AE_d{d}_fold{fold}_seed{seed}")
    clf = load_classifier(CKPT_DIR / "classifiers" / f"fold_{fold}_seed_{seed}.pt")
    with torch.no_grad():
        q_py = quantize(codec.encoder(torch.from_numpy(te.x[idx]).unsqueeze(1))).numpy().reshape(len(idx), -1)
    pred_py = predict(clf, codec.decode_int8(torch.from_numpy(q_py))[:, 0]).numpy()
    rows = []
    for k, i in enumerate(idx):
        if mode == "L":
            ser.write(b"L" + te.x[i].astype("<f4").tobytes())
        else:
            pcm, sr = sf.read(str(ROOT / te.meta.path.iloc[i]), dtype="int16")
            assert sr == 8000
            ser.write(b"A" + fix_length(pcm.astype(np.float32)).astype("<i2").tobytes())
        while True:
            line = ser.readline().decode("utf-8", "replace").strip()
            if line:
                log_f.write(line + "\n")
            if line.startswith("[PKT]"):
                break
            if not line and not ser.in_waiting:
                raise TimeoutError(f"no [PKT] for sample {k}")
        f = dict(re.findall(r"(\w+)=(\S+)", line[6:]))
        buf = bytes.fromhex(f["pkt"])
        p = packet.decode(buf)
        assert p.config_id == d
        x_hat = codec.decode_int8(torch.from_numpy(p.payload[None]))[:, 0]
        pred_hw = int(predict(clf, x_hat)[0])
        rows.append(dict(filename=te.meta.filename.iloc[i], label=int(te.y[i]), prediction_python=int(pred_py[k]),
                         prediction_esp32_packet=pred_hw, match=int(pred_hw == pred_py[k]), d=d,
                         payload_bytes=len(p.payload), packet_bytes=len(buf),
                         encoder_latency_ms=int(f["enc_us"]) / 1e3, preprocess_latency_ms=int(f["pre_us"]) / 1e3,
                         latent_exact_ratio=float((p.payload == q_py[k]).mean()),
                         latent_max_abs_diff=int(np.abs(p.payload.astype(int) - q_py[k]).max()),
                         mode=mode, fold=fold, seed=seed))
        if k % 50 == 0:
            print(f"  {k}/{len(idx)} match so far {np.mean([r['match'] for r in rows]):.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["L", "A", "both"], default="both")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--start", choices=["auto", "reset", "fresh", "running"], default="auto",
                    help="auto: read boot output (or poke an idle board), then request self-check/benchmark "
                         "if they were missed; reset: toggle RTS/DTR to reboot (UART bridge boards); "
                         "fresh: board is booting now (e.g. QEMU); running: board at [READY], rerun benchmark")
    ap.add_argument("--build-dir", default=None, help="firmware build dir (default esp32/firmware/build_d<d>)")
    ap.add_argument("--board", default="ESP32-S3")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. qemu)")
    a = ap.parse_args()
    (HW / "serial_logs").mkdir(parents=True, exist_ok=True)
    run_id = f"d{a.d}_{datetime.now():%Y%m%d_%H%M%S}{('_' + a.tag) if a.tag else ''}"
    ser = open_port(a.port, a.baud)
    with open(HW / "serial_logs" / f"{run_id}.log", "w") as log_f:
        if a.start == "auto":
            lines = read_until_ready(ser, log_f, 900, poke_after=5)
            if not any(l.startswith("[CHECK] sample=") for l in lines):
                ser.write(b"C")
                lines += read_until_ready(ser, log_f, 300)
            if not any(l.startswith("[RUN] ") for l in lines):
                ser.write(b"B")
                lines += read_until_ready(ser, log_f, 900)
        elif a.start == "reset":
            reset_board(ser)
        elif a.start == "running":
            ser.write(b"I")  # board already running: ask for info + rerun benchmark
            read_until_ready(ser, log_f, 30)
            ser.write(b"B")
        if a.start != "auto":
            lines = read_until_ready(ser, log_f)
        kv, runs, checks, mem = parse_boot(lines)
        fw = (ROOT / a.build_dir if a.build_dir else ROOT / "esp32" / "firmware" / f"build_d{a.d}") / "c3_node.bin"
        if not runs.empty:
            last_mem = mem[-1] if mem else {}
            runs = runs.assign(board=a.board, cpu_mhz=int(kv.get("cpu_mhz", 0)), method="Task-AE", d=a.d)
            runs["total_node_ms"] = runs.encoder_ms + runs.quantization_ms + runs.packetization_ms
            runs["end_to_end_node_ms"] = runs.total_node_ms + runs.preprocess_ms
            runs["free_heap_bytes"] = int(last_mem.get("free_heap", 0))
            runs["min_free_heap_bytes"] = int(last_mem.get("min_free_heap", 0))
            runs["psram_free_bytes"] = int(last_mem.get("psram_free", 0))
            runs["model_size_bytes"] = int(kv.get("model_size_bytes", 0))
            runs["firmware_size_bytes"] = fw.stat().st_size if fw.exists() else None
            runs["payload_bytes"], runs["packet_bytes"] = a.d, a.d + 4
            runs["activation_buffers_bytes"] = int(kv.get("activation_buffers_bytes", 0))
            runs["run_id"] = run_id
            cols = ["board", "cpu_mhz", "method", "d", "run", "preprocess_ms", "encoder_ms", "quantization_ms",
                    "packetization_ms", "total_node_ms", "end_to_end_node_ms", "free_heap_bytes",
                    "min_free_heap_bytes", "psram_free_bytes", "model_size_bytes", "firmware_size_bytes",
                    "payload_bytes", "packet_bytes", "activation_buffers_bytes", "run_id"]
            out = HW / f"hardware_benchmark{('_' + a.tag) if a.tag else ''}.csv"
            runs[cols].to_csv(out, mode="a", header=not out.exists(), index=False)
            s = runs.encoder_ms
            print(f"[bench] encoder_ms median={s.median():.3f} p95={s.quantile(.95):.3f} "
                  f"end_to_end median={runs.end_to_end_node_ms.median():.3f}")
        pd.DataFrame(mem).assign(run_id=run_id, d=a.d).to_csv(HW / "serial_logs" / f"{run_id}_mem.csv", index=False)
        checks.assign(run_id=run_id, d=a.d).to_csv(HW / "serial_logs" / f"{run_id}_check.csv", index=False)
        print(f"[check] {len(checks)} samples, PASS={sum(c.get('status') == 'PASS' for c in checks.to_dict('records'))}")
        modes = ["L", "A"] if a.mode == "both" else [a.mode]
        preds = pd.concat([stream(ser, a.d, a.fold, a.seed, m, a.n, log_f) for m in modes], ignore_index=True)
    preds["run_id"] = run_id
    out = HW / f"hardware_predictions{('_' + a.tag) if a.tag else ''}.csv"
    preds.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(preds.groupby("mode")[["match", "latent_exact_ratio", "encoder_latency_ms"]].mean())
    print("accuracy (esp32 packets):", preds.groupby("mode").apply(
        lambda g: (g.label == g.prediction_esp32_packet).mean(), include_groups=False).to_dict())


if __name__ == "__main__":
    main()
