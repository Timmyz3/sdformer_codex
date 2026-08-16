#!/usr/bin/env bash
# Medium-scale G1 projection TB only (new file; does not edit GPT run_projection_g1_checks.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/projection_g1_medium"
mkdir -p "$BUILD"
cd "$ROOT"

echo "[medium] iverilog tb_hitflow_g1_projection_top_medium"
iverilog -g2012 -Wall -s tb_hitflow_g1_projection_top_medium \
  -o "$BUILD/tb_g1_top_medium.vvp" \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  rtl_hitflow/hitflow_g1_projection_top.sv \
  tb_hitflow/tb_hitflow_g1_projection_top_medium.sv
vvp "$BUILD/tb_g1_top_medium.vvp"

echo "[medium] PASS"
