#!/usr/bin/env bash
# POW2-safe G1 projection scale checks (new file; GPT RTL/TB untouched).
# Uses rtl_hitflow_patch/*_pow2safe.sv + tb_hitflow/*_pow2_{32,64}.sv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/projection_g1_pow2safe"
mkdir -p "$BUILD"
cd "$ROOT"

RTL_COMMON=(
  rtl_hitflow/hitflow_nmf_g1_builder.sv
  rtl_hitflow/hitflow_gate_product_engine.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv
  rtl_hitflow_patch/hitflow_g1_projection_top_pow2safe.sv
)

run_one() {
  local top="$1"
  local tb="$2"
  echo "[pow2safe] iverilog $top"
  iverilog -g2012 -Wall -s "$top" \
    -o "$BUILD/${top}.vvp" \
    "${RTL_COMMON[@]}" \
    "$tb"
  vvp "$BUILD/${top}.vvp"
}

run_one tb_hitflow_g1_projection_top_pow2_32 \
  tb_hitflow/tb_hitflow_g1_projection_top_pow2_32.sv
run_one tb_hitflow_g1_projection_top_pow2_64 \
  tb_hitflow/tb_hitflow_g1_projection_top_pow2_64.sv

echo "[pow2safe] ALL PASS (T=32 and T=64)"
