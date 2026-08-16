#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_ttx/build"
mkdir -p "$BUILD"

cd "$ROOT"
iverilog -g2012 -s tb_ttx_row_engine \
  -o "$BUILD/tb_ttx_row_engine.vvp" \
  -f rtl_ttx/filelist.f tb_ttx/tb_ttx_row_engine.sv
vvp "$BUILD/tb_ttx_row_engine.vvp"

iverilog -g2012 -s tb_ttx_scheduler \
  -o "$BUILD/tb_ttx_scheduler.vvp" \
  rtl_ttx/ttx_descriptor_scheduler.sv tb_ttx/tb_ttx_scheduler.sv
vvp "$BUILD/tb_ttx_scheduler.vvp"

python3 scripts/gate_quant_reference.py \
  --vectors "$BUILD/gate_quant_vectors.txt" \
  --output results/gate_quant_reference.json
iverilog -g2012 -s tb_ttx_gate_quant_q17 \
  -o "$BUILD/tb_ttx_gate_quant_q17.vvp" \
  rtl_ttx/ttx_ceil_log2_u32.sv rtl_ttx/ttx_gate_quant_q17.sv \
  tb_ttx/tb_ttx_gate_quant_q17.sv
vvp "$BUILD/tb_ttx_gate_quant_q17.vvp" +VECTORS="$BUILD/gate_quant_vectors.txt"
