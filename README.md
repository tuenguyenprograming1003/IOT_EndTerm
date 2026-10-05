# C3 — Task-driven Log-Mel Feature Compression on FSDD

Nén đặc trưng log-mel định hướng nhiệm vụ cho phân loại chữ số nói (FSDD), đánh giá theo người nói
chưa xuất hiện trong Train (6 lượt Leave-One-Speaker-Out), so sánh ở cùng ngân sách byte, và triển khai
encoder + INT8 latent + packet trên ESP32-S3.

## 1. Project overview

```
audio (8 kHz) → 1 s (pad/crop) → log-mel 64×32 → Train-only normalisation → x
x → [codec encoder] → z ∈ [-1,1]^d → INT8 q → packet (4 B header + d B) → [decoder] → x̂ → classifier → digit
```

Methods compared at d ∈ {64, 128, 256} (packet = d + 4 = 68 / 132 / 260 bytes):

| method | description |
|---|---|
| Original (FP32) | x sent as float32, 8196 B — gives A0 |
| INT8 | x quantised to INT8 (fixed range ±8), 2052 B |
| DCT | 2-D DCT-II (ortho), first d zig-zag coefficients, per-coefficient INT8 ranges fitted on Train |
| Low-Mel | first m = d/32 mel rows (all 32 frames) as INT8, other rows = 0 |
| AE-MSE | conv autoencoder, L = MSE |
| Task-AE | same autoencoder, L = MSE + λ·CE(frozen classifier(x̂), y), λ: 0 (ep 1–20) → ramp (21–40) → 1 |

All methods of one fold/seed use the same split and the same frozen classifier.

### Deviation from the original specification (owner-approved, 2026-10-05)

The spec defined `S = log(1 + P)`. With float audio in [-1, 1], P ≪ 1 so `log1p(P) ≈ P`
(95.3 % of values < 0.01): the feature is linear power and depends on each speaker's recording gain.
The reference classifier reached only 21.8 % mean LOSO accuracy. The project uses instead
**S = CMVN(log(P + 1e-6))** (per-recording mean/variance normalisation over the 64×32 values),
followed by the Train-only per-mel-bin normalisation of Task 1.8. Evidence:
`results/logs/diagnostics/feature_variant_diagnostic.md`. PRD_spec is computed in this S domain.

## 2. Dataset version

FSDD, https://github.com/Jakobovski/free-spoken-digit-dataset, commit
`26eb9aaf76e81b692f806f9140c2d2777410d7a1` (`v1.0.10-27-g26eb9aa`), 3000 recordings, 6 speakers ×
10 digits × 50. Details: `data/DATASET_VERSION.txt`, checks: `results/logs/dataset_check.log`.

## 3. Environment setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # Python 3.13, CPU is sufficient
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git data/fsdd
git -C data/fsdd checkout 26eb9aaf76e81b692f806f9140c2d2777410d7a1
```
ESP32 part: ESP-IDF v5.5.5 (see `esp32/README.md`).

## 4. Folder structure

```
data/            fsdd/ (git clone), manifest.csv, DATASET_VERSION.txt, splits/, processed/
src/             all pipeline code (see below)
notebooks/       01–08 (EDA, features, classifier, baselines, AE-MSE, Task-AE, analysis, ESP32)
checkpoints/     classifiers/fold_{k}_seed_{s}.pt, codecs/{method}_d{d}_fold{k}_seed{s}[_lam{λ}].pt
results/         predictions/, summary/, figures/, logs/, packets/, hardware/
papers/          14 PDFs + tech_docs/, README.md, paper_tracker.csv
esp32/           firmware/, model/, host_test/, README.md
exported_models/ encoder_d{d}.onnx/.pt
hardware_test_vectors/
run_training_matrix.sh
```

| src file | role |
|---|---|
| build_manifest.py / check_dataset.py | Task 1.2 manifest + checks |
| audio.py | Task 1.4 load + fix to 8000 samples (`python audio.py` = test + pad/crop stats) |
| features.py | Task 1.5 log-mel (+ cache `data/processed/logmel_all.npz`) |
| create_splits.py | Task 1.7 six LOSO folds |
| compute_normalization.py | Task 1.8 Train-only μ/σ per fold |
| dataset.py | fold loading |
| classifier.py / train_classifier.py | reference CNN |
| baselines.py | INT8, DCT, Low-Mel |
| quantizer.py / codec.py / train_codec.py | INT8 latent (fake-quant + STE), codec, AE-MSE / Task-AE training |
| packet.py / packet_roundtrip.py | 4-byte header packet, two-process binary round trip |
| evaluate.py | all metrics + CSVs (+ `--ablation`) |
| export_encoder.py / verify_c_encoder.py / esp32_gateway.py | ESP32-S3 export, host C check, serial gateway |
| plotstyle.py / utils.py | shared figure style, paths |

## 5. Data preprocessing

```bash
cd src
../.venv/bin/python build_manifest.py && ../.venv/bin/python check_dataset.py
../.venv/bin/python audio.py          # asserts len == 8000 & float32 for all files; pad/crop stats
../.venv/bin/python features.py       # log-mel cache, asserts shape (64, 32)
```

## 6. Recreate splits and normalisation

```bash
../.venv/bin/python create_splits.py          # data/splits/fold_{1..6}_{train,val,test}.csv
../.venv/bin/python compute_normalization.py  # data/processed/fold_{1..6}_normalization.npz (mu, sigma: (64,))
```
Speakers sorted alphabetically: george, jackson, lucas, nicolas, theo, yweweler. Fold k: Test = s_k,
Val = s_{k+1}, Train = other 4 (2000 / 500 / 500).

## 7. Train classifier

```bash
../.venv/bin/python train_classifier.py --folds 1 2 3 4 5 6 --seeds 0 1 2
```
Adam 1e-3, wd 1e-4, batch 64, 60 epochs, checkpoint = best Val accuracy.

## 8. Train codecs

```bash
# pilot (fold 1, d=128)
../.venv/bin/python train_codec.py --methods AE-MSE Task-AE --ds 128 --folds 1 --seeds 0 --verbose
# everything (classifiers, 36 main runs, seeds 1-2 at d=128, λ ablation, evaluation)
bash run_training_matrix.sh        # from the project root; WORKERS=6 for 6 parallel CPU processes
```

## 9. Baselines

Computed inside `evaluate.py` (`baselines.py`); illustrated in `notebooks/04_baselines.ipynb`.

## 10. Evaluate

```bash
../.venv/bin/python evaluate.py --seeds 0 1 2     # all_predictions.csv, results_summary.csv, results_summary_mean.csv
../.venv/bin/python evaluate.py --ablation        # lambda_ablation_val.csv (Validation only)
../.venv/bin/python packet_roundtrip.py both --method Task-AE --d 128 --fold 1 --seed 0
```
Metrics: Accuracy, Macro-F1, ΔA_pp = Acomp − A0 (percentage points), R_acc = 100·Acomp/A0,
MSE (normalised domain), PRD_spec = 100·‖S − Ŝ‖/‖S‖ (log-mel representation error only — not audio quality),
CR_INT8 = 2052/(d+4), CR_FP32 = 8196/(d+4).

## 11. Regenerate figures

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

## 12. Expected output files

- `data/manifest.csv`, `data/DATASET_VERSION.txt`, `data/splits/*.csv`, `data/processed/*.npz`
- `checkpoints/classifiers/*.pt` (18), `checkpoints/codecs/*.pt` (36 + 24 + 24)
- `results/predictions/classifier_original_predictions.csv`, `results/predictions/all_predictions.csv`
- `results/summary/results_summary.csv`, `results_summary_mean.csv`, `report_table_seed0.csv`, `lambda_ablation_val.csv`
- `results/figures/*.png` (dataset, features, accuracy_vs_bytes, task_gain_by_speaker, mse_vs_accuracy,
  compression_vs_retention, accuracy_by_speaker, confusion_matrices, lambda_ablation, hw_*)
- `results/logs/*` (checks, histories, gradient checks, packet round trip)
- `results/hardware/*` (ESP32-S3; see `esp32/README.md`)

## Results and report

* `RESULTS.md` — result summary (main experiment, follow-up V2, ESP32-S3 hardware).
* `report/BAO_CAO_C3.md` — report draft (Vietnamese) with all figures in `report/figures/`;
  `report/PHU_LUC_SO_LIEU.md` — detailed numeric appendix generated by `src/make_report_appendix.py`.
* `bash make_report_package.sh` rebuilds `report_C3.zip` (markdown + figures).

## Not included in this repository

* `data/fsdd/` — clone FSDD at the commit in `data/DATASET_VERSION.txt` (see section 3).
* `papers/*.pdf`, `papers/tech_docs/*` — third-party papers/docs are not redistributed; download them
  from the links in `papers/README.md` / `papers/paper_tracker.csv` using the listed file names.
* `.venv/`, ESP-IDF build folders (`esp32/firmware/build*`) — recreate with the commands above and in
  `esp32/README.md`.
