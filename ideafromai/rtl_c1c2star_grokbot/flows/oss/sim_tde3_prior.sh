#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate C1* TDE3-Prior + wake_merge
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_tde3_prior.vvp"
LOG="out/sim/c1s_tde3_prior_sim.log"

echo "[sim_tde3_prior] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_tde3_prior.sv \
  rtl_c1star/c1s_wake_merge.sv \
  tb_c1star/tb_c1s_tde3_prior.sv

echo "[sim_tde3_prior] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all TDE3-Prior / wake_merge cases" "$LOG"; then
  echo "[sim_tde3_prior] PASS"
  exit 0
else
  echo "[sim_tde3_prior] FAIL — see $LOG"
  exit 1
fi
