#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim
LOG=out/sim/c1s_ecp_qkv_sim.log
iverilog -g2012 -o out/sim/c1s_ecp_qkv.vvp \
  rtl_c1star/c1s_ecp_qkv_predictor.sv \
  tb_c1star/tb_c1s_ecp_qkv_predictor.sv
vvp out/sim/c1s_ecp_qkv.vvp 2>&1 | tee "$LOG"
grep -q "PASS: all ECP-QKV cases" "$LOG"
echo "[sim_ecp_qkv] PASS"
