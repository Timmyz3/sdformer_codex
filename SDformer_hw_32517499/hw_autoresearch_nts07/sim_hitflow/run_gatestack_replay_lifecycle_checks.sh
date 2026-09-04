#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
BUILD="$ROOT/build_hitflow/gatestack_replay_lifecycle";mkdir -p "$BUILD";cd "$ROOT"
iverilog -g2012 -Wall -s tb_gatestack_replay_lifecycle_manager -o "$BUILD/tb.vvp" \
 rtl_hitflow/gatestack_replay_lifecycle_manager.sv \
 tb_hitflow/tb_gatestack_replay_lifecycle_manager.sv
vvp "$BUILD/tb.vvp"|tee "$BUILD/iverilog.log"
verilator --binary --timing --assert -Wall \
 --top-module tb_gatestack_replay_lifecycle_manager -Mdir "$BUILD/verilator_obj" \
 rtl_hitflow/gatestack_replay_lifecycle_manager.sv \
 verif_hitflow/gatestack_replay_lifecycle_manager_assertions.sv \
 verif_hitflow/bind_gatestack_replay_lifecycle_manager_assertions.sv \
 tb_hitflow/tb_gatestack_replay_lifecycle_manager.sv \
 >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_replay_lifecycle_manager"|tee "$BUILD/verilator.log"
yosys -q -l "$BUILD/yosys.log" -p \
 "read_verilog -sv rtl_hitflow/gatestack_replay_lifecycle_manager.sv; hierarchy -check -top gatestack_replay_lifecycle_manager; proc; opt; check; stat"
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log";then echo FAIL >&2;exit 1;fi
echo "PASS: GateStack replay lifecycle；Verilator 0 warning/error"
