#!/usr/bin/env bash
# Full training matrix (after Phase-1 data + pilot). Run from project root: bash run_training_matrix.sh
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
W=${WORKERS:-5}
# 1) classifiers (6 folds x seeds 0,1,2)
[ -f ../checkpoints/classifiers/fold_6_seed_2.pt ] || $PY -W ignore train_classifier.py --folds 1 2 3 4 5 6 --seeds 0 1 2
# 2) main matrix: 6 folds x d={64,128,256} x {AE-MSE, Task-AE}, seed 0  (36 runs)
$PY -W ignore train_codec.py --methods AE-MSE Task-AE --ds 64 128 256 --folds 1 2 3 4 5 6 --seeds 0 --workers $W --threads 1 --skip-existing
# 3) multi-seed at d=128: seeds 1,2  (24 runs)
$PY -W ignore train_codec.py --methods AE-MSE Task-AE --ds 128 --folds 1 2 3 4 5 6 --seeds 1 2 --workers $W --threads 1 --skip-existing
# 4) lambda ablation at d=128, seed 0 (lambda_max = 0, 0.1, 0.5, 2; lambda=1 is the main run)
$PY -W ignore train_codec.py --methods Task-AE --ds 128 --folds 1 2 3 4 5 6 --seeds 0 --lams 0 0.1 0.5 2 --workers $W --threads 1 --skip-existing
# 5) evaluation
$PY -W ignore evaluate.py --seeds 0 1 2
$PY -W ignore evaluate.py --ablation
echo MATRIX_DONE
