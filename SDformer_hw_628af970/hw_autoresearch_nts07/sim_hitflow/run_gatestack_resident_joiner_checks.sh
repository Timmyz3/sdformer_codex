#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_resident_joiner"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_resident_replay_joiner \
  -o "$BUILD/tb_joiner.vvp" \
  rtl_hitflow/gatestack_resident_replay_joiner.sv \
  tb_hitflow/tb_gatestack_resident_replay_joiner.sv
vvp "$BUILD/tb_joiner.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_resident_replay_joiner \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_resident_replay_joiner.sv \
  verif_hitflow/gatestack_resident_replay_joiner_assertions.sv \
  verif_hitflow/bind_gatestack_resident_replay_joiner_assertions.sv \
  tb_hitflow/tb_gatestack_resident_replay_joiner.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_resident_replay_joiner" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_resident_replay_joiner.sv; hierarchy -check -top gatestack_resident_replay_joiner; proc; opt; memory -nomap; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack resident replay joiner；Verilator 0 warning/error"
