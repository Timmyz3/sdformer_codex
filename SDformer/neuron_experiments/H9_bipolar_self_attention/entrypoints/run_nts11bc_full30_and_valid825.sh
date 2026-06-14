#!/usr/bin/env bash
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/envs/sdformerflow/bin/python
RESUME="${REPO}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${EXP}/results/nts11bc_hw_h60_all12_ds_w720_fastlr_full30_bs8_${STAMP}_setsid"
LOG="${RUN_DIR}/pipeline.log"

mkdir -p "${RUN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== 11bc unified-attn full30+valid825 start $(date -Is) ==="

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
"${PY}" "${EXP}/entrypoints/make_nts11bc_unified_attn_config.py"
CONFIG="${EXP}/configs/generated/nts11bc_hw_h60_all12_ds_w720_fastlr_full30.yml"
echo "config=${CONFIG}"
echo "resume=${RESUME}"
echo "run_dir=${RUN_DIR}"

wait_for_gpu() {
  while true; do
    if pgrep -f "eval_DSEC_flow_SNN.py" >/dev/null 2>&1; then
      echo "[wait] valid825 eval in progress $(date -Is)"
      sleep 90
      continue
    fi
    if pgrep -f "entrypoints/train.py" >/dev/null 2>&1; then
      echo "[wait] another train.py running $(date -Is)"
      sleep 90
      continue
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used}" && "${used}" -gt 8000 ]]; then
      echo "[wait] GPU mem ${used} MiB $(date -Is)"
      sleep 60
      continue
    fi
    break
  done
}

echo "=== verify chain ==="
"${PY}" -u "${EXP}/entrypoints/verify_nts11_chain.py" "${CONFIG}" || {
  echo "verify failed; abort"
  exit 1
}

wait_for_gpu
echo "=== train full30 ==="
"${PY}" -u "${EXP}/entrypoints/train.py" \
  --config "${CONFIG}" \
  --prev_runid "${RESUME}" \
  --save_path "${RUN_DIR}/checkpoint_epoch{}.pth"

echo "=== train done $(date -Is) ==="
ls -la "${RUN_DIR}"/checkpoint_epoch*.pth || true

wait_for_gpu
echo "=== standard valid825 ==="
"${PY}" -u "${EXP}/entrypoints/run_h9_standard_valid825_eval.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --epoch 9 --epoch 14 --epoch 19 --epoch 24 --epoch 28 --epoch 29

echo "=== 11bc pipeline complete $(date -Is) ==="