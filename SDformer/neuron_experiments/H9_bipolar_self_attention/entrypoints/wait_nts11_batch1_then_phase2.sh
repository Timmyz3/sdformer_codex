#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
LOG="${ROOT}/neuron_experiments/H9_bipolar_self_attention/results/nts11_phase2_waiter.log"
PHASE2="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nts11_phase2_short.sh"

{
  echo "[$(date -Is)] waiting for nts11_two_neuron rapid_screen to finish"
  while pgrep -f "rapid_screen.py.*tag nts11_two_neuron" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date -Is)] batch1 done; launching phase2"
  exec "${PHASE2}"
} >> "${LOG}" 2>&1