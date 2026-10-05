#!/usr/bin/env bash
# FOLLOW-UP EXPERIMENT (post-hoc, after the main locked experiment was evaluated on Test).
# Classifier V2 config is chosen on Train/Val only (src/classifier_search.py -> src/lock_v2_config.py).
# Ordered core-first; results are re-evaluated after each block so the run can be stopped at any block.
# Run from the project root:  CLF_ARGS="..." bash run_followup_v2.sh     (or leave CLF_ARGS unset to lock from search)
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
W=${WORKERS:-6}
TAG=v2
if [ -z "${CLF_ARGS:-}" ]; then
  CLF_ARGS=$($PY lock_v2_config.py | sed -n 's/^CLF_ARGS=//p')
fi
echo "V2 classifier config: $CLF_ARGS" | tee ../results/logs/followup_v2_config.txt
clf() { for s in "$@"; do
  [ -f ../checkpoints/classifiers_${TAG}/fold_6_seed_${s}.pt ] || \
    $PY -W ignore train_classifier.py --tag $TAG $CLF_ARGS --folds 1 2 3 4 5 6 --seeds $s --workers $W
done; }
codec() { $PY -W ignore train_codec.py --clf-tag $TAG --folds 1 2 3 4 5 6 --workers $W --threads 1 --skip-existing "$@"; }
evaluate() { $PY -W ignore evaluate.py --tag $TAG --seeds 0 1 2 --ds "$@"; echo "== EVAL DONE: d=$* =="; }

echo "== BLOCK 1 (core): V2 classifier seed 0 + codecs d=128 seed 0 =="
clf 0
codec --methods AE-MSE Task-AE --ds 128 --seeds 0
evaluate 128
echo "== BLOCK 2: codecs d=64, 256 seed 0 =="
codec --methods AE-MSE Task-AE --ds 64 256 --seeds 0
evaluate 64 128 256
echo "== BLOCK 3: multi-seed d=128 (classifier + codecs seeds 1,2) =="
clf 1 2
codec --methods AE-MSE Task-AE --ds 128 --seeds 1 2
evaluate 64 128 256
# Blocks 4-6 were first dropped, then run separately on request (2026-10-05): see run_followup_v2_ext.sh
#   codec --methods AE-MSE Task-AE --ds 512 --seeds 0 && evaluate 64 128 256 512      # block 4: d=512
#   codec --methods Task-AE --ds 128 --seeds 0 --lams 0.1 0.5 2                        # block 5: lambda grid
#   codec --methods Task-AE --ds 128 --seeds 0 --schedule slow                         # block 6: slow schedule
#   $PY -W ignore followup_extensions.py --tag $TAG
echo FOLLOWUP_DONE
