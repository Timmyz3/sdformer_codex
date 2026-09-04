#!/usr/bin/env bash
# NTS-11-lite qkonly ablation: full30 + standard valid825 from NB0
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/bin/python3
CONFIG="${EXP}/configs/generated/nts11lite_u12_qkonly_w720_fastlr_full30.yml"
RESUME="${REPO}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${EXP}/results/nts11lite_u12_qkonly_w720_fastlr_full30_bs8_${STAMP}_setsid"
LOG="${RUN_DIR}/pipeline.log"

mkdir -p "${RUN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== 11lite qkonly full30+valid825 start $(date -Is) ==="
echo "config=${CONFIG}"
echo "resume=${RESUME}"
echo "run_dir=${RUN_DIR}"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

wait_for_gpu() {
  while true; do
    if pgrep -f "eval_DSEC_flow_SNN.py" >/dev/null 2>&1; then
      echo "[wait] valid825 eval in progress $(date -Is)"
      sleep 90
      continue
    fi
    other=$(pgrep -af "entrypoints/train.py" 2>/dev/null | grep -v "${RUN_DIR}" || true)
    if [[ -n "${other}" ]]; then
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
"${PY}" -u "${EXP}/entrypoints/verify_nts11_chain.py" "${CONFIG}"

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

echo "=== valid825 done $(date -Is) ==="
echo "=== 11lite qkonly pipeline complete ==="