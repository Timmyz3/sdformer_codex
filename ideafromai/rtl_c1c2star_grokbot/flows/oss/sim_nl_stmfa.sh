#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_nl_stmfa.vvp"
LOG="out/sim/c1s_nl_stmfa_sim.log"

echo "[sim_nl_stmfa] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_nl_stmfa.sv \
  rtl_c1star/c1s_wake_merge.sv \
  tb_c1star/tb_c1s_nl_stmfa.sv

echo "[sim_nl_stmfa] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all NL-STMFA cases" "$LOG"; then
  echo "[sim_nl_stmfa] PASS"
  exit 0
else
  echo "[sim_nl_stmfa] FAIL — see $LOG"
  exit 1
fi
