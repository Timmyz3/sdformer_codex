#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

echo "[watch] waiting for MDR training npz files..."
while true; do
  if python3 scripts/download_mdr.py --check-only >/dev/null 2>&1; then
    echo "[watch] MDR detected, organizing batches..."
    cd third_party/SDformerFlow
    python MDR_dataloader/MDR_menage.py -dt 1
    cd "$REPO"
    echo "[watch] launching MDR baseline training (MVSEC val on indoor_flying3)"
    bash neuron_experiments/H9_bipolar_self_attention/entrypoints/launch_mdr_baseline_train.sh
    exit 0
  fi
  sleep 300
done