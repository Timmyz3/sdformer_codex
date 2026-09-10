#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_tid_deblur_loop.vvp"
LOG="out/sim/c1s_tid_deblur_loop_sim.log"
TAG="sim_tid_deblur_loop"

echo "[$TAG] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_tid_deblur_loop.sv \
  tb_c1star/tb_c1s_tid_deblur_loop.sv

echo "[$TAG] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all TID-DeblurLoop cases" "$LOG"; then
  echo "[$TAG] PASS"
  exit 0
else
  echo "[$TAG] FAIL — see $LOG"
  exit 1
fi
