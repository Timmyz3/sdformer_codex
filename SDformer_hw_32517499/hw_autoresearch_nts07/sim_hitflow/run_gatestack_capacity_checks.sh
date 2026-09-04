#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_capacity"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_capacity_mode_selector \
  -o "$BUILD/tb_capacity.vvp" \
  rtl_hitflow/gatestack_capacity_mode_selector.sv \
  tb_hitflow/tb_gatestack_capacity_mode_selector.sv
vvp "$BUILD/tb_capacity.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_capacity_mode_selector \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_capacity_mode_selector.sv \
  verif_hitflow/gatestack_capacity_mode_selector_assertions.sv \
  verif_hitflow/bind_gatestack_capacity_mode_selector_assertions.sv \
  tb_hitflow/tb_gatestack_capacity_mode_selector.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_capacity_mode_selector" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_capacity_mode_selector.sv; hierarchy -check -top gatestack_capacity_mode_selector; proc; opt; memory -nomap; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack capacity selector；Verilator 0 warning/error"
