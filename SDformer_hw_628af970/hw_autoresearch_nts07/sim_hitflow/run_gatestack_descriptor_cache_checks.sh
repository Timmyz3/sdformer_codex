#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_descriptor_cache"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_descriptor_residency_cache \
  -o "$BUILD/tb_cache.vvp" \
  rtl_hitflow/gatestack_descriptor_residency_cache.sv \
  tb_hitflow/tb_gatestack_descriptor_residency_cache.sv
vvp "$BUILD/tb_cache.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_descriptor_residency_cache \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_descriptor_residency_cache.sv \
  verif_hitflow/gatestack_descriptor_residency_cache_assertions.sv \
  verif_hitflow/bind_gatestack_descriptor_residency_cache_assertions.sv \
  tb_hitflow/tb_gatestack_descriptor_residency_cache.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_descriptor_residency_cache" \
  | tee "$BUILD/verilator_assert.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_descriptor_residency_cache.sv; hierarchy -check -top gatestack_descriptor_residency_cache; proc; opt; memory -nomap; check; stat"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack descriptor residency cache；Verilator 0 warning/error"
