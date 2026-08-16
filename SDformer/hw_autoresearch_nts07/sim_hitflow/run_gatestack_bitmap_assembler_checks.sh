#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_bitmap_assembler"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_destination_bitmap_assembler \
  -o "$BUILD/tb_bitmap.vvp" \
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv \
  tb_hitflow/tb_gatestack_destination_bitmap_assembler.sv
vvp "$BUILD/tb_bitmap.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_destination_bitmap_assembler \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv \
  verif_hitflow/gatestack_destination_bitmap_assembler_assertions.sv \
  verif_hitflow/bind_gatestack_destination_bitmap_assembler_assertions.sv \
  tb_hitflow/tb_gatestack_destination_bitmap_assembler.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_destination_bitmap_assembler" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_destination_bitmap_assembler.sv; hierarchy -check -top gatestack_destination_bitmap_assembler; proc; opt; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack destination bitmap assembler；Verilator 0 warning/error"
