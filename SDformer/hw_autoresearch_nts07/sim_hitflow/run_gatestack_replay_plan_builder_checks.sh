#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_replay_plan_builder"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_replay_plan_builder \
  -o "$BUILD/tb.vvp" \
  rtl_hitflow/gatestack_replay_plan_builder.sv \
  tb_hitflow/tb_gatestack_replay_plan_builder.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

iverilog -g2012 -Wall -s tb_gatestack_replay_plan_builder \
  -Ptb_gatestack_replay_plan_builder.DUT_CSR_FORMAT=0 \
  -o "$BUILD/tb_static_ipd.vvp" \
  rtl_hitflow/gatestack_replay_plan_builder.sv \
  tb_hitflow/tb_gatestack_replay_plan_builder.sv
vvp "$BUILD/tb_static_ipd.vvp" | tee "$BUILD/iverilog_static_ipd.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_replay_plan_builder \
  -Mdir "$BUILD/verilator_obj" \
  rtl_hitflow/gatestack_replay_plan_builder.sv \
  verif_hitflow/gatestack_replay_plan_builder_assertions.sv \
  verif_hitflow/bind_gatestack_replay_plan_builder_assertions.sv \
  tb_hitflow/tb_gatestack_replay_plan_builder.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_replay_plan_builder" | \
  tee "$BUILD/verilator.log"

verilator --binary --timing --assert -Wall \
  -GDUT_CSR_FORMAT=0 \
  --top-module tb_gatestack_replay_plan_builder \
  -Mdir "$BUILD/verilator_static_ipd_obj" \
  rtl_hitflow/gatestack_replay_plan_builder.sv \
  verif_hitflow/gatestack_replay_plan_builder_assertions.sv \
  verif_hitflow/bind_gatestack_replay_plan_builder_assertions.sv \
  tb_hitflow/tb_gatestack_replay_plan_builder.sv \
  >"$BUILD/verilator_static_ipd_build.log" 2>&1
"$BUILD/verilator_static_ipd_obj/Vtb_gatestack_replay_plan_builder" | \
  tee "$BUILD/verilator_static_ipd.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_replay_plan_builder.sv; hierarchy -check -top gatestack_replay_plan_builder; proc; opt; check; stat"
python3 "$LINTER" rtl_hitflow/gatestack_replay_plan_builder.sv \
  >"$BUILD/erie_lint.log" 2>&1

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
if grep -Eq '%Warning|%Error' "$BUILD/verilator_static_ipd_build.log"; then
  echo "FAIL: static-IPD Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack replay PLAN builder；Verilator/Erie 0 warning/error"
