#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c1s_ogec_sim.log
iverilog -g2012 -o out/sim/c1s_ogec.vvp \
  rtl_c1star/c1s_ogec_gate.sv \
  tb_c1star/tb_c1s_ogec_gate.sv
vvp out/sim/c1s_ogec.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all OGEC cases" "$LOG"
echo "[sim_ogec] PASS"
