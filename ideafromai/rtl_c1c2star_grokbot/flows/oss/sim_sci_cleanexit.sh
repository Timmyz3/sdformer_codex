#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate C1* SCI-CleanExit (Card H3)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_sci_cleanexit.vvp"
LOG="out/sim/c1s_sci_cleanexit_sim.log"

echo "[sim_sci_cleanexit] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_sci_cleanexit.sv \
  tb_c1star/tb_c1s_sci_cleanexit.sv

echo "[sim_sci_cleanexit] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all SCI-CleanExit cases" "$LOG"; then
  echo "[sim_sci_cleanexit] PASS"
  exit 0
else
  echo "[sim_sci_cleanexit] FAIL — see $LOG"
  exit 1
fi
