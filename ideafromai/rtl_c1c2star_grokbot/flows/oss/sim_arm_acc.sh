#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c2s_arm_acc_sim.log
iverilog -g2012 -o out/sim/c2s_arm_acc.vvp \
  rtl_c2star/c2s_arm_acc.sv \
  tb_c2star/tb_c2s_arm_acc.sv
vvp out/sim/c2s_arm_acc.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all ARM-Acc cases" "$LOG"
echo "[sim_arm_acc] PASS"
