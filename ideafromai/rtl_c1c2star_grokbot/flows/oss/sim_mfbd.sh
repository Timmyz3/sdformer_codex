#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c2s_mfbd_sim.log
iverilog -g2012 -o out/sim/c2s_mfbd.vvp \
  rtl_c2star/c2s_mfbd.sv \
  tb_c2star/tb_c2s_mfbd.sv
vvp out/sim/c2s_mfbd.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all MFBD cases" "$LOG"
echo "[sim_mfbd] PASS"
