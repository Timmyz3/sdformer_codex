#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_typed_builder_frontend"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_format_metadata_accumulator.sv
  rtl_hitflow/gatestack_typed_format_policy.sv
  rtl_hitflow/gatestack_typed_builder_frontend.sv
)

iverilog -g2012 -Wall -s tb_gatestack_typed_builder_frontend \
  -o "$BUILD/tb.vvp" "${RTL[@]}" \
  tb_hitflow/tb_gatestack_typed_builder_frontend.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_typed_builder_frontend \
  -Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" \
  verif_hitflow/gatestack_typed_builder_frontend_assertions.sv \
  verif_hitflow/bind_gatestack_typed_builder_frontend_assertions.sv \
  tb_hitflow/tb_gatestack_typed_builder_frontend.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_typed_builder_frontend" | \
  tee "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_typed_builder_frontend; proc; opt; check; stat"

for file in "${RTL[@]}"; do
  python3 "$LINTER" "$file" >>"$BUILD/erie_lint.log" 2>&1
done

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack typed builder frontend；Icarus/Verilator-SVA/Yosys/Erie通过"
