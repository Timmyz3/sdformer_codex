#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

echo "[1/4] MVSEC indoor_flying download (train/eval sequences)"
python3 scripts/download_mvsec_indoor_flying.py \
  --sequence indoor_flying1 \
  --sequence indoor_flying2 \
  --sequence indoor_flying3

echo "[2/4] MVSEC preprocess -> MVSEC_test/dt1"
python3 scripts/prepare_mvsec_dt1.py \
  --sequence indoor_flying1 \
  --sequence indoor_flying2 \
  --sequence indoor_flying3 \
  --encode-only

echo "[3/4] MDR training-set check"
if ! python3 scripts/download_mdr.py --check-only; then
  echo "MDR not ready. Manual step required:"
  python3 scripts/download_mdr.py
  exit 2
fi

echo "[4/4] Launch MDR baseline training (validates on MVSEC indoor_flying3)"
bash neuron_experiments/H9_bipolar_self_attention/entrypoints/launch_mdr_baseline_train.sh