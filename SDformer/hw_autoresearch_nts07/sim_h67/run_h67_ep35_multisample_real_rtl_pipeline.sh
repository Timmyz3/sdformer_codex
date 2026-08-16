#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PY="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
SAMPLES="${SAMPLES:-10}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-8192}"

CONFIG="${CONFIG:-$REPO/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml}"
CHECKPOINT="${CHECKPOINT:-$REPO/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth}"
TAG="${TAG:-h67_ep35_multisample${SAMPLES}_t450_real_rtl}"
PROFILE_DIR="${PROFILE_DIR:-$ROOT/results/${TAG}_profile}"
TRACE_DIR="${TRACE_DIR:-$ROOT/results/${TAG}_bit_trace}"
AUDIT_DIR="${AUDIT_DIR:-$ROOT/results/${TAG}_bit_trace_audit}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_h67/vectors/${TAG}}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/${TAG}}"

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]] || (( SAMPLES < 2 || SAMPLES > 16 )); then
  echo "FAIL: SAMPLES must be an integer in [2,16], got '$SAMPLES'" >&2
  exit 2
fi
for path in "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "FAIL: missing frozen input: $path" >&2
    exit 2
  fi
done
for path in "$PROFILE_DIR" "$TRACE_DIR" "$AUDIT_DIR" "$VECTOR_DIR" "$RESULT_DIR"; do
  if [[ -e "$path" ]]; then
    echo "FAIL: refusing to overwrite existing artifact path: $path" >&2
    exit 2
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "BLOCKED: nvidia-smi is unavailable; real Q/K trace capture requires GPU inference" >&2
  exit 75
fi

gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if ! [[ "$gpu_used_mib" =~ ^[0-9]+$ ]]; then
  echo "BLOCKED: could not parse GPU memory usage: '$gpu_used_mib'" >&2
  exit 75
fi
if (( gpu_used_mib > MAX_GPU_USED_MIB )); then
  echo "BLOCKED: GPU uses ${gpu_used_mib} MiB, limit is ${MAX_GPU_USED_MIB} MiB; no trace capture started" >&2
  exit 75
fi

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
"$PY" neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$PROFILE_DIR" \
  --samples "$SAMPLES" \
  --num-workers 0 \
  --ordered-trace \
  --bit-trace-dir "$TRACE_DIR" \
  --bit-trace-samples "$SAMPLES" \
  --bit-trace-windows 1 \
  --bit-trace-all-blocks

cd "$ROOT"
"$PY" scripts/audit_h67_bit_trace.py \
  --manifest "$TRACE_DIR/manifest.json" \
  --output-dir "$AUDIT_DIR" \
  --require-four-stages \
  --require-records "$((SAMPLES * 12))"

"$PY" scripts/generate_h67_multisample_checkpoint_row_vectors.py \
  --manifest "$TRACE_DIR/manifest.json" \
  --output-dir "$VECTOR_DIR" \
  --expected-tokens 450

VECTOR_DIR="$VECTOR_DIR" \
RESULT_DIR="$RESULT_DIR" \
MAX_SAMPLES="$SAMPLES" \
  bash sim_h67/run_h67_rqtb_multisample_real_rtl.sh

echo "PASS H67 ep35 multisample real RTL pipeline samples=$SAMPLES"
