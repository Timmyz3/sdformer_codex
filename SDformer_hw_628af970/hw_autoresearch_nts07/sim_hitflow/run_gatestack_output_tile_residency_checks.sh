#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_output_tile_residency"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_output_tile_scheduler.sv
  rtl_hitflow/gatestack_descriptor_residency_cache.sv
  rtl_hitflow/gatestack_dualtag_replay_lifecycle_manager.sv
)

iverilog -g2012 -Wall \
  -s tb_gatestack_output_tile_residency_integration \
  -o "$BUILD/tb.vvp" "${RTL[@]}" \
  tb_hitflow/tb_gatestack_output_tile_residency_integration.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_output_tile_residency_integration \
  -Mdir "$BUILD/verilator_obj" "${RTL[@]}" \
  verif_hitflow/gatestack_output_tile_scheduler_assertions.sv \
  verif_hitflow/bind_gatestack_output_tile_scheduler_assertions.sv \
  verif_hitflow/gatestack_descriptor_residency_cache_assertions.sv \
  verif_hitflow/bind_gatestack_descriptor_residency_cache_assertions.sv \
  verif_hitflow/gatestack_dualtag_replay_lifecycle_assertions.sv \
  verif_hitflow/bind_gatestack_dualtag_replay_lifecycle_assertions.sv \
  tb_hitflow/tb_gatestack_output_tile_residency_integration.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_output_tile_residency_integration" | \
  tee "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_output_tile_scheduler; proc; opt; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack output-tile descriptor residency；Verilator 0 warning/error"
