# ESP32-S3 node — encoder + INT8 latent + packet

Training platform: CPU (PC, PyTorch). Deployment platform: ESP32-S3 (inference only).
The board runs **[preprocessing →] encoder → INT8 quantisation → 4-byte header + d-byte packet**;
the PC gateway parses the packet, dequantises, runs the decoder and the classifier.

```
esp32/
├── firmware/          ESP-IDF project (main/main.c, encoder.c, preproc.c, model.h)
├── model/             generated C sources: model_d{64,128,256}.c, preproc_fold1.c, test_vectors.c
├── host_test/         host build of the same C kernels (pre-flash verification)
└── README.md
hardware_test_vectors/ 20 vectors (2 per digit, all 6 speakers): input_audio.npy, input_logmel.npy,
                       expected_latent_float.npy, expected_latent_int8.bin (d=128), metadata.json
exported_models/       encoder_d{d}.onnx / .pt (archival), export_info.json
```

## Runtime / backend

Plain C float32 kernels (`encoder.c`), weights compiled into Flash as `const float` arrays —
no TFLite Micro / ESP-DL. Reason: the encoder is 4 conv layers (~93 k params, ~4.9 M MAC);
a direct C implementation is bit-exact against PyTorch (verified below) and avoids a
PyTorch→ONNX→TF→TFLite conversion chain. Weights and activations are **float32**; only the
**latent** is INT8. Do not describe the model as an INT8 model.

| item | value |
|---|---|
| framework | ESP-IDF v5.5.5 |
| compiler | xtensa-esp-elf GCC 14.2.0 (esp-14.2.0_20260121) |
| optimisation | `-O2` (`CONFIG_COMPILER_OPTIMIZATION_PERF`) |
| CPU clock | 240 MHz |
| console | `build_d*`: UART0 (GPIO43/44), 921600 baud; `build_usb_d*` (measured): USB-Serial-JTAG |
| PSRAM | enabled, octal mode, `SPIRAM_IGNORE_NOTFOUND` (boards without PSRAM still boot) |
| flash setting | 4 MB, single-app-large partition (1.5 MB app) — works on N16R8 (16 MB) |
| board (measured) | ESP32-S3 N16R8: chip ESP32-S3 (QFN56) rev v0.2, 16 MB flash, 8 MB embedded octal PSRAM |
| port used | native USB (USB-Serial-JTAG), `/dev/cu.usbmodem1101` |

## Pre-flash verification (done, no board needed)

```bash
.venv/bin/python src/export_encoder.py            # regenerate esp32/model/*.c and test vectors
.venv/bin/python src/verify_c_encoder.py          # host C vs PyTorch on all 500 Test recordings of fold 1
```
Result → `results/hardware/host_c_consistency.csv`.

The firmware was also booted in the ESP-IDF QEMU ESP32-S3 emulator and the full serial protocol
was exercised (`results/hardware/*_qemu.*`). **QEMU timings are not hardware measurements.**

## Build

Native-USB boards (USB-Serial-JTAG console; this is what was measured):
```bash
idf.py -B build_usb_d128 -DENC_D=128 -DSDKCONFIG=build_usb_d128/sdkconfig \
  "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.defaults.usbjtag" -p /dev/cu.usbmodemXXXX flash
cd ../../src && ../.venv/bin/python esp32_gateway.py --port /dev/cu.usbmodemXXXX --baud 115200 --d 128 \
  --mode both --n 500 --start auto --build-dir esp32/firmware/build_usb_d128 --board "ESP32-S3 N16R8"
```

UART-bridge boards:

```bash
source ~/.espressif/tools/activate_idf_v5.5.5.sh
cd esp32/firmware
idf.py -B build_d128 -DENC_D=128 build      # also: -B build_d64 -DENC_D=64, -B build_d256 -DENC_D=256
```

## Flash + measure (needs the board)

1. Connect the DevKit's **UART** USB port (CP210x/CH34x bridge), find the port: `ls /dev/cu.usb*`
2. Flash: `idf.py -B build_d128 -p /dev/cu.usbserial-XXXX flash`
3. Run the gateway (resets the board, captures boot log + 1000-run benchmark, streams the 500
   Test recordings of fold 1 in both modes, decodes packets on the PC):
   ```bash
   cd src
   ../.venv/bin/python esp32_gateway.py --port /dev/cu.usbserial-XXXX --d 128 --mode both --n 500 \
       --board "ESP32-S3-DevKitC-1 N16R8"
   ```
4. Repeat steps 2–3 for d=64 and d=256 (`build_d64`, `build_d256`).
5. Run `notebooks/08_esp32_hardware_analysis.ipynb` (TAG = '').

Outputs: `results/hardware/hardware_benchmark.csv`, `hardware_predictions.csv`,
`serial_logs/<run>.log` (+ `_check.csv`, `_mem.csv`).

## Serial protocol (after `[READY]`)

| host → node | node → host |
|---|---|
| `'L'` + 2048 float32 LE (normalised log-mel, fold-1 Train stats) | `[PKT] seq=.. mode=L bytes=d+4 pkt=<hex> pre_us=0 enc_us=.. q_us=.. pkt_us=..` |
| `'A'` + 8000 int16 LE (1-s PCM, already padded/cropped) | same, with on-node preprocessing time `pre_us` |
| `'B'` | rerun benchmark (`[RUN] i pre_us enc_us q_us pkt_us` × 1000) then `[READY]` |
| `'I'` | `[MODEL]`, `[BOARD]`, `[MEM]` then `[READY]` |

Packet = `version u8 (1) | method_id u8 (2 = Task-AE) | config_id u16 LE (= d) | d × int8`.
