#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/banked_accumulator_overflow_atomic"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=rtl_hitflow/hitflow_banked_accumulator.sv
TB=tb_hitflow/tb_hitflow_banked_accumulator_overflow_atomic.sv

iverilog -g2012 -Wall -s tb_hitflow_banked_accumulator_overflow_atomic \
  -o "$BUILD/tb.vvp" "$RTL" "$TB" >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing -Wall \
  --top-module tb_hitflow_banked_accumulator_overflow_atomic \
  -Mdir "$BUILD/verilator_obj" "$RTL" "$TB" \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_hitflow_banked_accumulator_overflow_atomic" \
  | tee "$BUILD/verilator.log"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
grep -q 'quarantined=1' "$BUILD/iverilog.log"
grep -q 'quarantined=1' "$BUILD/verilator.log"
echo "PASS: accumulator overflow在final边界被原子隔离"
