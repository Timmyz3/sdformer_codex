#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ipd_cache_fill_adapter"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_ipd_cache_fill_adapter \
  -o "$BUILD/tb.vvp" rtl_hitflow/gatestack_ipd_cache_fill_adapter.sv \
  tb_hitflow/tb_gatestack_ipd_cache_fill_adapter.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ipd_cache_fill_adapter \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_ipd_cache_fill_adapter.sv \
  verif_hitflow/gatestack_ipd_cache_fill_adapter_assertions.sv \
  verif_hitflow/bind_gatestack_ipd_cache_fill_adapter_assertions.sv \
  tb_hitflow/tb_gatestack_ipd_cache_fill_adapter.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_ipd_cache_fill_adapter" | \
  tee "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_ipd_cache_fill_adapter.sv; hierarchy -check -top gatestack_ipd_cache_fill_adapter; proc; opt; check; stat"
python3 "$LINTER" rtl_hitflow/gatestack_ipd_cache_fill_adapter.sv \
  >"$BUILD/erie_lint.log" 2>&1
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack IPD cache fill adapter；Verilator/Erie 0 warning/error"
