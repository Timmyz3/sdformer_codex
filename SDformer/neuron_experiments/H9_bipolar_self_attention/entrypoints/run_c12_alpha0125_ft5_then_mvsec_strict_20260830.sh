#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
STATUS="$REPO/neuron_experiments/H9_bipolar_self_attention/results/c12_alpha0125_ft5_then_mvsec_strict_20260830.log"

cd "$REPO"
mkdir -p "$(dirname "$STATUS")"
exec >>"$STATUS" 2>&1

echo "[$(date -u +%FT%TZ)] START C12 alpha=1/8 true resume5"
"$PY" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_dsec_c12_alpha0125_resume5_20260830.py
echo "[$(date -u +%FT%TZ)] END C12 alpha=1/8 true resume5"

echo "[$(date -u +%FT%TZ)] START MVSEC strict same-parent C00"
"$PY" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_mvsec_strict_c00_continuation_20260830.py
echo "[$(date -u +%FT%TZ)] ALL COMPLETE"
