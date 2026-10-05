#!/usr/bin/env bash
# Follow-up V2 extensions (blocks 4-6), run after run_followup_v2.sh. From the project root.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
W=${WORKERS:-6}
TAG=v2
codec() { $PY -W ignore train_codec.py --clf-tag $TAG --folds 1 2 3 4 5 6 --workers $W --threads 1 --skip-existing "$@"; }
echo "== BLOCK 4: d=512 seed 0 =="
codec --methods AE-MSE Task-AE --ds 512 --seeds 0
$PY -W ignore evaluate.py --tag $TAG --seeds 0 1 2 --ds 64 128 256 512
echo "== EVAL DONE: d=64 128 256 512 =="
echo "== BLOCK 5: lambda grid (per-fold Val selection) =="
codec --methods Task-AE --ds 128 --seeds 0 --lams 0.1 0.5 2
echo "== BLOCK 6: slow lambda schedule =="
codec --methods Task-AE --ds 128 --seeds 0 --schedule slow
$PY -W ignore followup_extensions.py --tag $TAG
echo EXT_DONE
