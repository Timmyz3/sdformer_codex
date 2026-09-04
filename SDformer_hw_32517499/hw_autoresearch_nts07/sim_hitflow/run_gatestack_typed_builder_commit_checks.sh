#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_typed_builder_commit"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_typed_payload_serializer.sv
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv
  rtl_hitflow/gatestack_typed_builder_commit_top.sv
)
TB=tb_hitflow/tb_gatestack_typed_builder_commit_top_real.sv

iverilog -g2012 -Wall -s tb_gatestack_typed_builder_commit_top_real \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_typed_builder_commit_top_real \
  -Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" \
  verif_hitflow/gatestack_typed_payload_serializer_assertions.sv \
  verif_hitflow/bind_gatestack_typed_payload_serializer_assertions.sv \
  verif_hitflow/gatestack_head_slot_sram_adapter_assertions.sv \
  verif_hitflow/bind_gatestack_head_slot_sram_adapter_assertions.sv \
  "$TB" >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_typed_builder_commit_top_real" | \
  tee "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_typed_builder_commit_top; proc; opt; memory_collect; check; stat"

for source in "${RTL[@]}"; do
  name="$(basename "$source" .sv)"
  python3 "$LINTER" "$source" >"$BUILD/erie_${name}.log" 2>&1
done

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi

echo "PASS: GateStack三格式Serializer到typed slot的atomic commit；逐word金参考/Icarus/Verilator-SVA/Yosys/Erie通过"
