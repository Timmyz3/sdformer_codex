#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate C2* BiSAT-Agg (Card H1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c2s_bisat_agg.vvp"
LOG="out/sim/c2s_bisat_agg_sim.log"

echo "[sim_bisat_agg] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c2star/c2s_bisat_agg.sv \
  tb_c2star/tb_c2s_bisat_agg.sv

echo "[sim_bisat_agg] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all BiSAT-Agg cases" "$LOG"; then
  echo "[sim_bisat_agg] PASS"
  exit 0
else
  echo "[sim_bisat_agg] FAIL — see $LOG"
  exit 1
fi
