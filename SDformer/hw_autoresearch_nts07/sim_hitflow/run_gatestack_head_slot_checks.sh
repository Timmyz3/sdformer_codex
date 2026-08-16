#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_head_slot"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_head_slot_sram_adapter \
  -o "$BUILD/tb_head_slot.vvp" \
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv \
  tb_hitflow/tb_gatestack_head_slot_sram_adapter.sv
vvp "$BUILD/tb_head_slot.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_head_slot_sram_adapter \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv \
  verif_hitflow/gatestack_head_slot_sram_adapter_assertions.sv \
  verif_hitflow/bind_gatestack_head_slot_sram_adapter_assertions.sv \
  tb_hitflow/tb_gatestack_head_slot_sram_adapter.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_head_slot_sram_adapter" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_head_slot_sram_adapter.sv; hierarchy -check -top gatestack_head_slot_sram_adapter; proc; opt; memory -nomap; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack head-slot adapter；Verilator 0 warning/error"
