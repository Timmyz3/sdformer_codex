#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_term_fork"
mkdir -p "$BUILD"; cd "$ROOT"
iverilog -g2012 -Wall -s tb_gatestack_term_fork -o "$BUILD/tb.vvp" \
  rtl_hitflow/gatestack_term_fork.sv tb_hitflow/tb_gatestack_term_fork.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
verilator --binary --timing --assert -Wall --top-module tb_gatestack_term_fork \
  -Mdir "$BUILD/verilator_obj" rtl_hitflow/gatestack_term_fork.sv \
  verif_hitflow/gatestack_term_fork_assertions.sv \
  verif_hitflow/bind_gatestack_term_fork_assertions.sv \
  tb_hitflow/tb_gatestack_term_fork.sv >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_term_fork" | tee "$BUILD/verilator.log"
yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_term_fork.sv; hierarchy -check -top gatestack_term_fork; proc; opt; check; stat"
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2; exit 1
fi
echo "PASS: GateStack term fork；Verilator 0 warning/error"
