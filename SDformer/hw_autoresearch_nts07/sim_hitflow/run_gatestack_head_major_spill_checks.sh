#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_head_major_spill"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=rtl_hitflow/gatestack_head_major_spill_scheduler.sv
TB=tb_hitflow/tb_gatestack_head_major_spill_scheduler.sv
iverilog -g2012 -Wall -s tb_gatestack_head_major_spill_scheduler \
  -o "$BUILD/tb.vvp" "$RTL" "$TB" >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_head_major_spill_scheduler \
  -Mdir "$BUILD/verilator_obj" "$RTL" "$TB" \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_head_major_spill_scheduler" \
  | tee "$BUILD/verilator.log"
yosys -ql "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top gatestack_head_major_spill_scheduler; proc; flatten; opt; stat"
python3 "$LINTER" "$RTL" >"$BUILD/erie_lint.log" 2>&1 || true
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: head-major Verilator warning/error" >&2
  exit 1
fi
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_lint.log"; then
  echo "FAIL: head-major Erie lint error" >&2
  exit 1
fi
python3 scripts/model_gatestack_head_major_spill_real_trace.py \
  --baseline results/gatestack_real_trace_ablation_20260717/report.json \
  --yosys-log "$BUILD/yosys.log" \
  --output-dir results/gatestack_head_major_spill_20260718
echo "PASS: head-major spill scheduler双工具、lint与综合检查完成"
