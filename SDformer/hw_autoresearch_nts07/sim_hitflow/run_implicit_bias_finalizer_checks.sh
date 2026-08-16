#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/implicit_bias_finalizer"
RTL="$ROOT/rtl_hitflow/hitflow_implicit_bias_finalizer_accumulator.sv"
TB="$ROOT/tb_hitflow/tb_hitflow_implicit_bias_finalizer_accumulator.sv"
SVA="$ROOT/verif_hitflow/hitflow_implicit_bias_finalizer_assertions.sv"
BIND="$ROOT/verif_hitflow/bind_hitflow_implicit_bias_finalizer_assertions.sv"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_hitflow_implicit_bias_finalizer_accumulator \
  -o "$BUILD/tb.vvp" "$RTL" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
grep -q 'PASS: implicit-bias finalizer exact' "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_hitflow_implicit_bias_finalizer_accumulator \
  -Mdir "$BUILD/verilator_obj" "$RTL" "$SVA" "$BIND" "$TB" \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_hitflow_implicit_bias_finalizer_accumulator" \
  | tee "$BUILD/verilator.log"
grep -q 'PASS: implicit-bias finalizer exact' "$BUILD/verilator.log"

verilator --lint-only --timing -Wall --top-module \
  hitflow_implicit_bias_finalizer_accumulator "$RTL" \
  >"$BUILD/verilator_lint.log" 2>&1

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top hitflow_implicit_bias_finalizer_accumulator; proc; opt; memory -nomap; opt; check; stat"

echo 'RESULT suite=implicit_bias_finalizer status=PASS iverilog=PASS verilator_sva=PASS yosys=PASS'
