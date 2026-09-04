#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

export SDFORMER_SNN_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE

python3 scripts/download_mdr.py --check-only
python3 - <<'PY'
from pathlib import Path
mdr = Path("third_party/SDformerFlow/data/Datasets/MDR/dt1/train/events1")
if not any(mdr.rglob("*.npz")):
    raise SystemExit("MDR training set missing. Download from Baidu Pan first (see scripts/download_mdr.py).")
PY

cd third_party/SDformerFlow
python train_mdr_supervised_SNN.py \
  --config ../../../configs/generated/train_mdr_baseline_mvsec_route.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow