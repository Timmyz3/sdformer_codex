#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_raw_issue_adapter"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_raw_issue_adapter \
  -o "$BUILD/tb_adapter.vvp" \
  rtl_hitflow/gatestack_raw_issue_adapter.sv \
  tb_hitflow/tb_gatestack_raw_issue_adapter.sv
vvp "$BUILD/tb_adapter.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_raw_issue_adapter \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_raw_issue_adapter.sv \
  verif_hitflow/gatestack_raw_issue_adapter_assertions.sv \
  verif_hitflow/bind_gatestack_raw_issue_adapter_assertions.sv \
  tb_hitflow/tb_gatestack_raw_issue_adapter.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_raw_issue_adapter" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_raw_issue_adapter.sv; hierarchy -check -top gatestack_raw_issue_adapter; proc; opt; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack RAW issue adapter；Verilator 0 warning/error"
