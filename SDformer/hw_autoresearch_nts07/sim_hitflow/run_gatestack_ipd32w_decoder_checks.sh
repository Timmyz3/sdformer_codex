#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ipd32w_decoder"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_ipd32w_replay_decoder \
  -o "$BUILD/tb_decoder.vvp" \
  rtl_hitflow/gatestack_ipd32w_replay_decoder.sv \
  tb_hitflow/tb_gatestack_ipd32w_replay_decoder.sv
vvp "$BUILD/tb_decoder.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ipd32w_replay_decoder \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_ipd32w_replay_decoder.sv \
  verif_hitflow/gatestack_ipd32w_replay_decoder_assertions.sv \
  verif_hitflow/bind_gatestack_ipd32w_replay_decoder_assertions.sv \
  tb_hitflow/tb_gatestack_ipd32w_replay_decoder.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_ipd32w_replay_decoder" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_ipd32w_replay_decoder.sv; hierarchy -check -top gatestack_ipd32w_replay_decoder; proc; opt; memory -nomap; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack IPD32W decoder；Verilator 0 warning/error"
