#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_hatf96_weight_coalescer"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"
RTL=rtl_hitflow/gatestack_hatf96_weight_coalescer.sv
TB=tb_hitflow/tb_gatestack_hatf96_weight_coalescer.sv

iverilog -g2012 -Wall -s tb_gatestack_hatf96_weight_coalescer \
  -o "$BUILD/tb.vvp" "$RTL" "$TB" >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing -Wall \
  --assert \
  --top-module tb_gatestack_hatf96_weight_coalescer \
  -Mdir "$BUILD/verilator_obj" "$RTL" "$TB" \
  verif_hitflow/gatestack_hatf96_weight_coalescer_assertions.sv \
  verif_hitflow/bind_gatestack_hatf96_weight_coalescer_assertions.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_hatf96_weight_coalescer" \
  | tee "$BUILD/verilator.log"
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top gatestack_hatf96_weight_coalescer; proc; opt; check; stat"
if grep -q '^Warning:' "$BUILD/yosys.log"; then
  grep '^Warning:' "$BUILD/yosys.log" >&2
  exit 1
fi
python "$LINTER" "$RTL" >"$BUILD/erie_lint.log" 2>&1 || true
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_lint.log"; then
  cat "$BUILD/erie_lint.log" >&2
  exit 1
fi
echo "PASS: HATF96三bank权重请求/响应聚合双模拟器、SVA、Yosys与Erie通过"
