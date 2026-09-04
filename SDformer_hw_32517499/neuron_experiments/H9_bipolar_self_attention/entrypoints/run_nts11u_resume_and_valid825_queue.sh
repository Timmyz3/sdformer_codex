#!/usr/bin/env bash
# Wait for GPU-heavy jobs (11aah finetune), then:
#   1) standard valid825 on 11u ep16/17/18
#   2) resume 11u full30 from ep18 -> ep29
#   3) standard valid825 on ep19/24/29
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
CONFIG="${EXP}/configs/nts11u_hw_h60_s23_scope_ffn_all_ternary_scope_full30_20260612_130819.yml"
RUN_DIR="${EXP}/results/nts11u_hw_h60_s23_scope_ffn_all_ternary_scope_full30_bs8_20260612_130819_setsid"
PY=/opt/conda/envs/sdformerflow/bin/python
LOG="${RUN_DIR}/resume_valid825_queue.log"

exec >>"${LOG}" 2>&1
echo "=== nts11u resume+valid825 queue start $(date -Is) ==="

wait_for_gpu() {
  while true; do
    if pgrep -f "nts11aah.*train.py" >/dev/null 2>&1; then
      echo "[wait] 11aah still training $(date -Is)"
      sleep 300
      continue
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used}" && "${used}" -gt 45000 ]]; then
      echo "[wait] GPU mem ${used} MiB > 45G $(date -Is)"
      sleep 120
      continue
    fi
    break
  done
}

run_valid825() {
  local epochs=("$@")
  local args=()
  for ep in "${epochs[@]}"; do
    args+=(--epoch "${ep}")
  done
  wait_for_gpu
  echo "[eval] valid825 epochs: ${epochs[*]} $(date -Is)"
  cd "${REPO}"
  "${PY}" -u "${EXP}/entrypoints/run_h9_standard_valid825_eval.py" \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    "${args[@]}"
}

resume_train() {
  wait_for_gpu
  echo "[train] resume 11u from checkpoint_epoch18 $(date -Is)"
  cd "${REPO}"
  "${PY}" -u "${EXP}/entrypoints/train.py" \
    --config "${CONFIG}" \
    --resume \
    --prev_runid "${RUN_DIR}/checkpoint_epoch18.pth" \
    --save_path "${RUN_DIR}/checkpoint_epoch{}.pth"
}

run_valid825 16 17 18
resume_train
run_valid825 19 24 29
echo "=== nts11u resume+valid825 queue done $(date -Is) ==="