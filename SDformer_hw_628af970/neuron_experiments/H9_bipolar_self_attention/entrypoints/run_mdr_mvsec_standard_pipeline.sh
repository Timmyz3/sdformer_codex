#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

export SDFORMER_SNN_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export SDFORMER_MDR_DETECT_ANOMALY="${SDFORMER_MDR_DETECT_ANOMALY:-0}"

ARCHIVE_DIR="${ARCHIVE_DIR:-/root/private_data/mdr/train}"
CONFIG="${CONFIG:-configs/generated/train_mdr_baseline_mvsec_route_fast.yml}"
MLFLOW_URI="${MLFLOW_URI:-file:///root/private_data/sdformer_mlflow}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/sdformerflow/bin/python}"

echo "=== MDR archive preparation $(date -Iseconds) ==="
"$PYTHON_BIN" scripts/prepare_mdr_from_archives.py --archive-dir "$ARCHIVE_DIR" --dt 1

echo "=== MDR dataset smoke test $(date -Iseconds) ==="
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

repo = Path("/root/private_data/work/sdformer_codex/SDformer")
train = repo / "third_party/SDformerFlow/data/Datasets/MDR/dt1/train"
checks = {
    "events1": "*.npz",
    "events2": "*.npz",
    "best_density_events1": "*.npz",
    "best_density_events2": "*.npz",
    "flow": "*.flo",
}
for name, pattern in checks.items():
    root = train / name
    sample = next(root.rglob(pattern), None)
    print(f"[mdr-smoke] {name}: {sample}")
    if sample is None:
        raise SystemExit(f"MDR {name} is empty after archive preparation")
PY

echo "=== MDR baseline training -> MVSEC validation $(date -Iseconds) ==="
cd third_party/SDformerFlow
"$PYTHON_BIN" train_mdr_supervised_SNN.py \
  --config "../../${CONFIG}" \
  --path_mlflow "$MLFLOW_URI"
