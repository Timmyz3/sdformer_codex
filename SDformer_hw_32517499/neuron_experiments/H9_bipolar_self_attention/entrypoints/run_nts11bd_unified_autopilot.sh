#!/usr/bin/env bash
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/envs/sdformerflow/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${EXP}/results/nts11bd_unified_autopilot_${STAMP}.log"

exec >>"${LOG}" 2>&1
echo "=== 11bd unified-attn autopilot start $(date -Is) ==="
echo "python=${PY}"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNIFIED_ATTN_PYTHON="${PY}"

wait_for_gpu() {
  while true; do
    if pgrep -f "eval_DSEC_flow_SNN.py" >/dev/null 2>&1; then
      echo "[wait] eval in progress $(date -Is)"
      sleep 60
      continue
    fi
    other=$(pgrep -f "entrypoints/train.py" | grep -v "$$" || true)
    if [[ -n "${other}" ]]; then
      echo "[wait] train pid(s)=${other} $(date -Is)"
      sleep 60
      continue
    fi
    other=$(pgrep -f "rapid_screen.py" | grep -v "$$" || true)
    if [[ -n "${other}" ]]; then
      echo "[wait] rapid_screen pid(s)=${other} $(date -Is)"
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
cd "${REPO}"
"${PY}" -u "${EXP}/entrypoints/run_nts11_unified_attn_autopilot.py"
echo "=== 11bd unified-attn autopilot done $(date -Is) ==="