#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_typed_payload_serializer"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

for mode in 0 1; do
  iverilog -g2012 -Wall -s tb_gatestack_typed_payload_serializer_real \
    -Ptb_gatestack_typed_payload_serializer_real.BITMAP_BYPASS_ENABLE="$mode" \
    -o "$BUILD/tb_mode${mode}.vvp" \
    rtl_hitflow/gatestack_typed_payload_serializer.sv \
    tb_hitflow/tb_gatestack_typed_payload_serializer_real.sv \
    >"$BUILD/iverilog_mode${mode}_build.log" 2>&1
  vvp "$BUILD/tb_mode${mode}.vvp" | tee "$BUILD/iverilog_mode${mode}.log"

  verilator --binary --timing --assert -Wall \
    -GBITMAP_BYPASS_ENABLE="$mode" \
    --top-module tb_gatestack_typed_payload_serializer_real \
    -Mdir "$BUILD/verilator_mode${mode}_obj" \
    rtl_hitflow/gatestack_typed_payload_serializer.sv \
    verif_hitflow/gatestack_typed_payload_serializer_assertions.sv \
    verif_hitflow/bind_gatestack_typed_payload_serializer_assertions.sv \
    tb_hitflow/tb_gatestack_typed_payload_serializer_real.sv \
    >"$BUILD/verilator_mode${mode}_build.log" 2>&1
  "$BUILD/verilator_mode${mode}_obj/Vtb_gatestack_typed_payload_serializer_real" | \
    tee "$BUILD/verilator_mode${mode}.log"

  yosys -q -l "$BUILD/yosys_mode${mode}.log" -p \
    "read_verilog -sv rtl_hitflow/gatestack_typed_payload_serializer.sv; chparam -set BITMAP_BYPASS_ENABLE $mode gatestack_typed_payload_serializer; hierarchy -check -top gatestack_typed_payload_serializer; proc; opt; memory_collect; check; stat"
done

python3 "$LINTER" rtl_hitflow/gatestack_typed_payload_serializer.sv \
  >"$BUILD/erie_lint.log" 2>&1

if grep -Eq '%Warning|%Error' "$BUILD"/verilator_mode*_build.log; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack typed payload serializer；legacy/bitmap-bypass双模式逐word金参考及全RTL门通过"
