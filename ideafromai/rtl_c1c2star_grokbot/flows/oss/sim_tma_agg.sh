#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate C2* TMA-Agg
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c2s_tma_agg.vvp"
LOG="out/sim/c2s_tma_agg_sim.log"

echo "[sim_tma_agg] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c2star/c2s_tma_agg.sv \
  tb_c2star/tb_c2s_tma_agg.sv

echo "[sim_tma_agg] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all TMA-Agg cases" "$LOG"; then
  echo "[sim_tma_agg] PASS"
  exit 0
else
  echo "[sim_tma_agg] FAIL — see $LOG"
  exit 1
fi
