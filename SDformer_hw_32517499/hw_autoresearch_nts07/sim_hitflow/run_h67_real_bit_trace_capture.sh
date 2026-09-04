#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PY="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-8192}"
PROFILE_DIR="$ROOT/results/h67_real_bit_trace_profile_20260717"
TRACE_DIR="$ROOT/results/h67_real_bit_trace_20260717"
AUDIT_DIR="$ROOT/results/h67_real_bit_trace_audit_20260717"
CONFIG="$REPO/neuron_experiments/H9_bipolar_self_attention/configs/generated/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_dyadic_int8_deploy_rtl_exact.yml"
CHECKPOINT="$REPO/neuron_experiments/H9_bipolar_self_attention/results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: 未找到nvidia-smi，不能启动真实网络trace采集" >&2
  exit 1
fi
gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( gpu_used_mib > MAX_GPU_USED_MIB )); then
  echo "BLOCKED: GPU已使用${gpu_used_mib}MiB，阈值${MAX_GPU_USED_MIB}MiB；不与训练抢卡" >&2
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
  --samples 1 \
  --num-workers 0 \
  --ordered-trace \
  --bit-trace-dir "$TRACE_DIR" \
  --bit-trace-samples 1 \
  --bit-trace-windows 1

cd "$ROOT"
"$PY" scripts/audit_h67_bit_trace.py \
  --manifest "$TRACE_DIR/manifest.json" \
  --output-dir "$AUDIT_DIR" \
  --require-four-stages

echo "PASS: H67真实四stage位级trace采集与数据质量审计完成"
