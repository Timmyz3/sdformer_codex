#!/usr/bin/env bash
# Delete checkpoint .pth only from pre-NTS/NTX experiments.
# NEVER touches nts*/ntx* dirs, configs, entrypoints, or code.
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
RES="${ROOT}/neuron_experiments/H9_bipolar_self_attention/results"
NEURON_RES="${ROOT}/neuron_experiments"
LOG="${RES}/checkpoint_cleanup_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "[ckpt-only] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[ckpt-only] before: $(df -h /root/private_data | tail -1)"

is_protected() {
  local base
  base=$(basename "$1")
  [[ "${base}" == nts* || "${base}" == ntx* ]]
}

purge_ckpts_in_dir() {
  local d="$1"
  [[ -d "${d}" ]] || return 0
  if is_protected "${d}"; then
    echo "[ckpt-only] skip protected ${d}"
    return 0
  fi
  local n=0
  local f
  for f in "${d}"/checkpoint_epoch*.pth; do
    [[ -e "${f}" ]] || continue
    rm -f "${f}"
    n=$((n + 1))
  done
  if (( n > 0 )); then
    echo "[ckpt-only] removed ${n} ckpt(s) in ${d}"
  fi
}

# H9 results: early h*/nsc*/st_*/j*/faps*/profile_* etc.
for d in "${RES}"/*; do
  [[ -d "${d}" ]] || continue
  purge_ckpts_in_dir "${d}"
done

# E/F neuron_experiment legacy full runs (configs/code untouched)
for d in "${NEURON_RES}"/E*/results/* "${NEURON_RES}"/F*/results/*; do
  [[ -d "${d}" ]] || continue
  purge_ckpts_in_dir "${d}"
done

# baseline_stride_upstream: keep epoch59 only
BASE="${ROOT}/experiments/baseline_stride_upstream"
if [[ -d "${BASE}" ]]; then
  echo "[ckpt-only] baseline keep checkpoint_epoch59.pth only"
  find "${BASE}" -maxdepth 1 -name 'checkpoint_epoch*_state_dict.pth' -delete
  for f in "${BASE}"/checkpoint_epoch*.pth; do
    [[ -e "${f}" ]] || continue
    [[ "$(basename "${f}")" == "checkpoint_epoch59.pth" ]] && continue
    rm -f "${f}"
  done
fi

echo "[ckpt-only] after: $(df -h /root/private_data | tail -1)"
echo "[ckpt-only] log=${LOG}"
echo "[ckpt-only] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"