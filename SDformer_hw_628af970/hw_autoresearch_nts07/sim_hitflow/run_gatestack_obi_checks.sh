#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_obi"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall \
  -s tb_gatestack_obi_iterator \
  -o "$BUILD/tb_gatestack_obi_iterator.vvp" \
  rtl_hitflow/gatestack_obi_iterator.sv \
  tb_hitflow/tb_gatestack_obi_iterator.sv
vvp "$BUILD/tb_gatestack_obi_iterator.vvp" | tee "$BUILD/iverilog.log"

verilator --lint-only --timing -Wall \
  --top-module gatestack_obi_iterator \
  rtl_hitflow/gatestack_obi_iterator.sv \
  >"$BUILD/verilator_lint.log" 2>&1

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_obi_iterator \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_obi_iterator.sv \
  verif_hitflow/gatestack_obi_iterator_assertions.sv \
  verif_hitflow/bind_gatestack_obi_iterator_assertions.sv \
  tb_hitflow/tb_gatestack_obi_iterator.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_obi_iterator" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_obi_iterator.sv; hierarchy -check -top gatestack_obi_iterator; proc; opt; memory -nomap; check; stat"

if grep -Eq '%Warning|%Error' \
  "$BUILD/verilator_lint.log" "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack OBI iverilog + Verilator lint/assert + Yosys；Verilator 0 warning/error"
