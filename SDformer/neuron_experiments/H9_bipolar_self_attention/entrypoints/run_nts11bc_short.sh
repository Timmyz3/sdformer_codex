#!/usr/bin/env bash
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/envs/sdformerflow/bin/python
RESUME="${REPO}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${EXP}/results/nts11bc_short_${STAMP}.log"

exec >>"${LOG}" 2>&1
echo "=== 11bc short test start $(date -Is) ==="

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
"${PY}" "${EXP}/entrypoints/make_nts11bc_unified_attn_config.py"
CONFIG="${EXP}/configs/generated/nts11bc_hw_h60_all12_ds_w720_fastlr_s1224.yml"
echo "config=${CONFIG}"
echo "resume=${RESUME}"

wait_for_gpu() {
  while true; do
    if pgrep -f "eval_DSEC_flow_SNN.py" >/dev/null 2>&1; then
      echo "[wait] eval in progress $(date -Is)"
      sleep 60
      continue
    fi
    if pgrep -f "entrypoints/train.py" >/dev/null 2>&1; then
      echo "[wait] train in progress $(date -Is)"
      sleep 60
      continue
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used}" && "${used}" -gt 8000 ]]; then
      echo "[wait] GPU mem ${used} MiB $(date -Is)"
      sleep 30
      continue
    fi
    break
  done
}

wait_for_gpu
"${PY}" -u "${EXP}/entrypoints/verify_nts11_chain.py" "${CONFIG}"

wait_for_gpu
"${PY}" -u "${EXP}/entrypoints/rapid_screen.py" \
  --config "${CONFIG}" \
  --steps 1224 \
  --prev-runid "${RESUME}" \
  --batch-size 8 \
  --workers 8 \
  --prefetch-factor 4 \
  --valid-samples 10 \
  --confirm-steps 1224 \
  --no-promote-valid40 \
  --tag nts11bc_short

echo "=== 11bc short test done $(date -Is) ==="
ls -lt "${EXP}/results"/nts11bc_short_* 2>/dev/null | head -5 || true