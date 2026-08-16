#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_raw_tail_retimer"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_raw_tail_retimer \
  -o "$BUILD/tb.vvp" rtl_hitflow/gatestack_raw_tail_retimer.sv \
  tb_hitflow/tb_gatestack_raw_tail_retimer.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_raw_tail_retimer \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_raw_tail_retimer.sv \
  verif_hitflow/gatestack_raw_tail_retimer_assertions.sv \
  verif_hitflow/bind_gatestack_raw_tail_retimer_assertions.sv \
  tb_hitflow/tb_gatestack_raw_tail_retimer.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_raw_tail_retimer" \
  | tee "$BUILD/verilator_assert.log"
yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_raw_tail_retimer.sv; hierarchy -check -top gatestack_raw_tail_retimer; proc; opt; check; stat"
python "$LINTER" rtl_hitflow/gatestack_raw_tail_retimer.sv \
  >"$BUILD/erie_lint.log" 2>&1 || true
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_lint.log"; then
  echo "FAIL: Erie lint存在MUST error" >&2
  cat "$BUILD/erie_lint.log" >&2
  exit 1
fi
echo "PASS: GateStack RAW tail retimer严格回归通过"
