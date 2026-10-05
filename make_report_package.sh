#!/usr/bin/env bash
# Copy report figures next to report/BAO_CAO_C3.md and build report_C3.zip (markdown + figures).
# QEMU-emulator figures (*_qemu.png) are excluded: they are not hardware measurements.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf report/figures && mkdir -p report/figures
find results/figures -name '*.png' ! -name '*_qemu.png' -exec cp {} report/figures/ \;
rm -f report_C3.zip report_C3_figures.zip
(cd report && zip -q -r ../report_C3.zip BAO_CAO_C3.md PHU_LUC_SO_LIEU.md figures)
(cd report && zip -q -r ../report_C3_figures.zip figures)
echo "figures: $(ls report/figures | wc -l | tr -d ' ')"
ls -lh report_C3.zip report_C3_figures.zip
