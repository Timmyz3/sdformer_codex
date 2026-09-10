#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c2s_back_pipe_sim.log
iverilog -g2012 -o out/sim/c2s_back_pipe.vvp \
  rtl_c2star/c2s_hbg_rp_packetizer.sv \
  rtl_c2star/c2s_smam_rp.sv \
  rtl_c2star/c2s_sth_gate.sv \
  rtl_c2star/c2s_back_pipe.sv \
  tb_c2star/tb_c2s_back_pipe.sv
vvp out/sim/c2s_back_pipe.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all back_pipe cases" "$LOG"
echo "[sim_back_pipe] PASS"
