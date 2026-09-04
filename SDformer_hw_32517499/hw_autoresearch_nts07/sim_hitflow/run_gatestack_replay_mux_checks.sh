#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_replay_mux"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_replay_mux \
  -o "$BUILD/tb_mux.vvp" \
  rtl_hitflow/gatestack_replay_mux.sv \
  tb_hitflow/tb_gatestack_replay_mux.sv
vvp "$BUILD/tb_mux.vvp" | tee "$BUILD/iverilog.log"

iverilog -g2012 -Wall -s tb_gatestack_replay_mux_sources2 \
  -o "$BUILD/tb_mux_sources2.vvp" \
  rtl_hitflow/gatestack_replay_mux.sv \
  tb_hitflow/tb_gatestack_replay_mux_sources2.sv
vvp "$BUILD/tb_mux_sources2.vvp" | tee "$BUILD/iverilog_sources2.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_replay_mux \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_replay_mux.sv \
  verif_hitflow/gatestack_replay_mux_assertions.sv \
  verif_hitflow/bind_gatestack_replay_mux_assertions.sv \
  tb_hitflow/tb_gatestack_replay_mux.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_replay_mux" \
  | tee "$BUILD/verilator_assert.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_replay_mux_sources2 \
  -Mdir "$BUILD/verilator_sources2_obj" \
  rtl_hitflow/gatestack_replay_mux.sv \
  verif_hitflow/gatestack_replay_mux_assertions.sv \
  verif_hitflow/bind_gatestack_replay_mux_assertions.sv \
  tb_hitflow/tb_gatestack_replay_mux_sources2.sv \
  >"$BUILD/verilator_sources2_build.log" 2>&1
"$BUILD/verilator_sources2_obj/Vtb_gatestack_replay_mux_sources2" \
  | tee "$BUILD/verilator_sources2_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_replay_mux.sv; hierarchy -check -top gatestack_replay_mux; proc; opt; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi
if grep -Eq '%Warning|%Error' "$BUILD/verilator_sources2_build.log"; then
  echo "FAIL: SOURCES=2 Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack replay mux；Verilator 0 warning/error"
