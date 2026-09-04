#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_descriptor_cache_stress"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_descriptor_residency_cache_stress \
  -o "$BUILD/tb.vvp" \
  rtl_hitflow/gatestack_descriptor_residency_cache.sv \
  tb_hitflow/tb_gatestack_descriptor_residency_cache_stress.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_descriptor_residency_cache_stress \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_descriptor_residency_cache.sv \
  verif_hitflow/gatestack_descriptor_residency_cache_assertions.sv \
  verif_hitflow/bind_gatestack_descriptor_residency_cache_assertions.sv \
  tb_hitflow/tb_gatestack_descriptor_residency_cache_stress.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_descriptor_residency_cache_stress" | \
  tee "$BUILD/verilator.log"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack descriptor cache长序列release/refill压力验证"
