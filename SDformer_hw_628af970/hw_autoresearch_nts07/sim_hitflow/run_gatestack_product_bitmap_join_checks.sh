#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_product_bitmap_join"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_product_bitmap_join \
  -o "$BUILD/tb_join.vvp" \
  rtl_hitflow/gatestack_product_bitmap_join.sv \
  tb_hitflow/tb_gatestack_product_bitmap_join.sv
vvp "$BUILD/tb_join.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_product_bitmap_join \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_product_bitmap_join.sv \
  verif_hitflow/gatestack_product_bitmap_join_assertions.sv \
  verif_hitflow/bind_gatestack_product_bitmap_join_assertions.sv \
  tb_hitflow/tb_gatestack_product_bitmap_join.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_product_bitmap_join" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_product_bitmap_join.sv; hierarchy -check -top gatestack_product_bitmap_join; proc; opt; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack product-bitmap join；Verilator 0 warning/error"
