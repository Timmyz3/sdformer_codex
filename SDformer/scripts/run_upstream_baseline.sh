#!/usr/bin/env bash
# Launch paper-consistent baseline training using the upstream training script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${ROOT_DIR}/experiments/baseline_stride_upstream"
mkdir -p "${OUTPUT_DIR}"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=torch
export PYTHONPATH="${ROOT_DIR}/third_party/SDformerFlow:${PYTHONPATH:-}"

exec python -u -m train_flow_parallel_supervised_SNN \
    --config "${ROOT_DIR}/configs/generated/upstream_baseline_stride.yml" \
    --save_path "${OUTPUT_DIR}/checkpoint_epoch{}.pth" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
