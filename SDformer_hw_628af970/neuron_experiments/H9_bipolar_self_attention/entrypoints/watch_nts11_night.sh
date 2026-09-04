#!/usr/bin/env bash
# Night watchdog: heartbeat + autopilot liveness. Safe to run in background.
set -uo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
EXP="${ROOT}/neuron_experiments/H9_bipolar_self_attention"
DRIVER="${EXP}/results/nts11_scope_autopilot_20260612_020946"
STATUS="${DRIVER}/status.log"
SHORT_GLOB="${EXP}/results/nts11_scope_short_*"
INTERVAL=600

log() {
  echo "[$(date -Iseconds)] [watch] $*" | tee -a "${STATUS}"
}

count_short_done() {
  local dir
  dir=$(ls -td ${SHORT_GLOB} 2>/dev/null | head -1 || true)
  [[ -n "${dir}" && -d "${dir}/runs" ]] || { echo 0; return; }
  find "${dir}/runs" -name 'checkpoint_epoch0.pth' 2>/dev/null | wc -l
}

while true; do
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
  done=$(count_short_done)
  if pgrep -f 'run_nts11_scope_autopilot.py' >/dev/null; then
    ap=alive
  else
    ap=dead
  fi
  if pgrep -f 'rapid_screen.py.*nts11_scope_short' >/dev/null; then
    rs=alive
  elif pgrep -f 'entrypoints/train.py.*nts11_scope_short' >/dev/null; then
    rs=train
  elif pgrep -f 'entrypoints/train.py.*scope_full30' >/dev/null; then
    rs=full_train
  elif pgrep -f 'eval_DSEC_flow_SNN.py.*nts11' >/dev/null; then
    rs=valid825
  else
    rs=idle
  fi
  latest_train=""
  dir=$(ls -td ${SHORT_GLOB} 2>/dev/null | head -1 || true)
  if [[ -n "${dir}" ]]; then
    latest_train=$(grep -h 'it/s' "${dir}"/runs/*/train.log 2>/dev/null | tail -1 | tr -d '\r' | cut -c1-120)
  fi
  log "gpu=${gpu} short_done=${done}/12 autopilot=${ap} stage=${rs} train_tail=${latest_train}"

  if [[ "${ap}" == "dead" && "${rs}" == "idle" && "${done}" -lt 12 ]]; then
    log "WARN autopilot died before short sweep complete (done=${done})"
  fi
  if [[ "${ap}" == "dead" && "${rs}" == "idle" && -f "${DRIVER}/rapid_screen.log" ]]; then
    if grep -q 'NTS-11 scope autopilot complete' "${STATUS}" 2>/dev/null; then
      log "autopilot finished successfully"
      break
    fi
  fi
  sleep "${INTERVAL}"
done