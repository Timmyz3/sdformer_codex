#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c1s_prrc_sim.log
iverilog -g2012 -o out/sim/c1s_prrc.vvp \
  rtl_c1star/c1s_prrc_ledger.sv \
  tb_c1star/tb_c1s_prrc_ledger.sv
vvp out/sim/c1s_prrc.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all PRRC cases" "$LOG"
echo "[sim_prrc] PASS"
