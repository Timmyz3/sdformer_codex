#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_output_tile_scheduler"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_output_tile_scheduler \
  -o "$BUILD/tb.vvp" \
  rtl_hitflow/gatestack_output_tile_scheduler.sv \
  tb_hitflow/tb_gatestack_output_tile_scheduler.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

iverilog -g2012 -Wall -s tb_gatestack_output_tile_scheduler_stage_sweep \
  -o "$BUILD/tb_stage_sweep.vvp" \
  rtl_hitflow/gatestack_output_tile_scheduler.sv \
  tb_hitflow/tb_gatestack_output_tile_scheduler_stage_sweep.sv
vvp "$BUILD/tb_stage_sweep.vvp" | tee "$BUILD/iverilog_stage_sweep.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_output_tile_scheduler \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_output_tile_scheduler.sv \
  verif_hitflow/gatestack_output_tile_scheduler_assertions.sv \
  verif_hitflow/bind_gatestack_output_tile_scheduler_assertions.sv \
  tb_hitflow/tb_gatestack_output_tile_scheduler.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_output_tile_scheduler" | \
  tee "$BUILD/verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_output_tile_scheduler_stage_sweep \
  -Mdir "$BUILD/verilator_stage_sweep_obj" \
  rtl_hitflow/gatestack_output_tile_scheduler.sv \
  verif_hitflow/gatestack_output_tile_scheduler_assertions.sv \
  verif_hitflow/bind_gatestack_output_tile_scheduler_assertions.sv \
  tb_hitflow/tb_gatestack_output_tile_scheduler_stage_sweep.sv \
  >"$BUILD/verilator_stage_sweep_build.log" 2>&1
"$BUILD/verilator_stage_sweep_obj/Vtb_gatestack_output_tile_scheduler_stage_sweep" | \
  tee "$BUILD/verilator_stage_sweep.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_output_tile_scheduler.sv; hierarchy -check -top gatestack_output_tile_scheduler; proc; opt; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
if grep -Eq '%Warning|%Error' "$BUILD/verilator_stage_sweep_build.log"; then
  echo "FAIL: stage-sweep Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack output-tile scheduler；Verilator 0 warning/error"
