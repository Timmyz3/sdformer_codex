#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves
LOG=out/sim/c1s_front_pipe_sim.log
iverilog -g2012 -o out/sim/c1s_front_pipe.vvp \
  rtl_c1star/c1s_op_stw_predictor.sv \
  rtl_c1star/c1s_ecp_qkv_predictor.sv \
  rtl_c1star/c1s_mw_delta_buf.sv \
  rtl_c1star/c1s_front_pipe.sv \
  tb_c1star/tb_c1s_front_pipe.sv
vvp out/sim/c1s_front_pipe.vvp +DUMP_VCD 2>&1 | tee "$LOG"
grep -q "PASS: all front_pipe cases" "$LOG"
echo "[sim_front_pipe] PASS"
