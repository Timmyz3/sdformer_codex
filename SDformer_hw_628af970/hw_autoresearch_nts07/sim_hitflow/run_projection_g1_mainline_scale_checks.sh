#!/usr/bin/env bash
# 主线 G1 尺度回归：验证 power-of-two 修复及 H67 的 162x32 接口规模。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/projection_g1_mainline_scale"
mkdir -p "$BUILD"
cd "$ROOT"

echo "[mainline] accumulator TOKENS=32"
iverilog -g2012 -Wall -DACC_DUT=hitflow_banked_accumulator \
  -s tb_hitflow_banked_accumulator_pow2safe \
  -o "$BUILD/tb_acc_main_t32.vvp" \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  tb_hitflow/tb_hitflow_banked_accumulator_pow2safe.sv
vvp "$BUILD/tb_acc_main_t32.vvp"

run_top() {
  local tokens="$1"
  local lanes="$2"
  local output="$BUILD/tb_g1_main_t${tokens}_l${lanes}.vvp"
  echo "[mainline] G1 TOKENS=$tokens LANES=$lanes"
  iverilog -g2012 -Wall \
    -DTB_TOKENS="$tokens" -DTB_LANES="$lanes" \
    -DG1_DUT=hitflow_g1_projection_top \
    -s tb_hitflow_g1_projection_top_pow2_64 \
    -o "$output" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow/hitflow_banked_accumulator.sv \
    rtl_hitflow/hitflow_g1_projection_top.sv \
    tb_hitflow/tb_hitflow_g1_projection_top_pow2_64.sv
  vvp "$output"
}

run_top 32 8
run_top 64 8
run_top 162 32

echo "[mainline] ALL PASS (T32/L8, T64/L8, T162/L32)"
