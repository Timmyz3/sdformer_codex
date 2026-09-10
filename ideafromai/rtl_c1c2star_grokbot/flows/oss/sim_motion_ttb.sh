#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate Motion-TTB packer with iverilog+vvp
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c2s_motion_ttb_sim.log
iverilog -g2012 -o out/sim/c2s_motion_ttb.vvp \
  rtl_c2star/c2s_motion_ttb_packer.sv \
  tb_c2star/tb_c2s_motion_ttb_packer.sv
vvp out/sim/c2s_motion_ttb.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all Motion-TTB packer cases" "$LOG"
echo "[sim_motion_ttb] PASS"
