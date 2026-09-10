#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate Card A OP-STW with iverilog+vvp; dump VCD + log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_op_stw.vvp"
LOG="out/sim/c1s_op_stw_sim.log"
VCD="out/waves/c1s_op_stw.vcd"

echo "[sim_op_stw] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_op_stw_predictor.sv \
  tb_c1star/tb_c1s_op_stw_predictor.sv

echo "[sim_op_stw] run → $VCD"
# TB writes $dumpfile to out/waves/c1s_op_stw.vcd when DUMP_VCD is defined
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all 3 OP-STW cases" "$LOG"; then
  echo "[sim_op_stw] PASS"
  exit 0
else
  echo "[sim_op_stw] FAIL — see $LOG"
  exit 1
fi
