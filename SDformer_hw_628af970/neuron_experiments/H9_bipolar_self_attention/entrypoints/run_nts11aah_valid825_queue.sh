#!/usr/bin/env bash
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
CONFIG="${EXP}/configs/generated/nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15.yml"
RUN_DIR="${EXP}/results/nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15_bs8_20260612_194020_setsid"
PY=/opt/conda/envs/sdformerflow/bin/python
LOG="${RUN_DIR}/valid825_queue.log"

exec >>"${LOG}" 2>&1
echo "=== 11aah valid825 queue start $(date -Is) ==="

wait_for_gpu() {
  while true; do
    if pgrep -f "eval_DSEC_flow_SNN.py" >/dev/null 2>&1; then
      echo "[wait] eval in progress $(date -Is)"
      sleep 120
      continue
    fi
    if pgrep -f "nts11u.*train.py" >/dev/null 2>&1; then
      echo "[wait] 11u train in progress $(date -Is)"
      sleep 120
      continue
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used}" && "${used}" -gt 45000 ]]; then
      echo "[wait] GPU mem ${used} MiB $(date -Is)"
      sleep 60
      continue
    fi
    break
  done
}

wait_for_gpu
cd "${REPO}"
"${PY}" -u "${EXP}/entrypoints/run_h9_standard_valid825_eval.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --epoch 0 --epoch 4 --epoch 9 --epoch 14
echo "=== 11aah valid825 queue done $(date -Is) ==="