#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_bl_veto_prior.vvp"
LOG="out/sim/c1s_bl_veto_prior_sim.log"

echo "[sim_bl_veto_prior] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_bl_veto_prior.sv \
  rtl_c1star/c1s_wake_merge.sv \
  tb_c1star/tb_c1s_bl_veto_prior.sv

echo "[sim_bl_veto_prior] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all BL-VetoPrior cases" "$LOG"; then
  echo "[sim_bl_veto_prior] PASS"
  exit 0
else
  echo "[sim_bl_veto_prior] FAIL — see $LOG"
  exit 1
fi
