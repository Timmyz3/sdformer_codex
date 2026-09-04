#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_onchip_typed_builder_c0"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_transposed_bitmap_bank.sv
  rtl_hitflow/gatestack_canonical_head_workspace_c0.sv
  rtl_hitflow/gatestack_typed_format_policy.sv
  rtl_hitflow/gatestack_typed_payload_serializer.sv
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv
  rtl_hitflow/gatestack_typed_builder_commit_top.sv
  rtl_hitflow/gatestack_onchip_typed_builder_c0_top.sv
)
TB=tb_hitflow/tb_gatestack_onchip_typed_builder_c0_top.sv

iverilog -g2012 -Wall -s tb_gatestack_onchip_typed_builder_c0_top \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_onchip_typed_builder_c0_top \
  -Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" \
  verif_hitflow/gatestack_canonical_head_workspace_c0_assertions.sv \
  verif_hitflow/bind_gatestack_canonical_head_workspace_c0_assertions.sv \
  verif_hitflow/gatestack_typed_payload_serializer_assertions.sv \
  verif_hitflow/bind_gatestack_typed_payload_serializer_assertions.sv \
  verif_hitflow/gatestack_head_slot_sram_adapter_assertions.sv \
  verif_hitflow/bind_gatestack_head_slot_sram_adapter_assertions.sv \
  verif_hitflow/gatestack_onchip_typed_builder_c0_top_assertions.sv \
  verif_hitflow/bind_gatestack_onchip_typed_builder_c0_top_assertions.sv \
  "$TB" >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_onchip_typed_builder_c0_top" | \
  tee "$BUILD/verilator.log"

timeout 300 yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_onchip_typed_builder_c0_top; proc; opt; memory_collect; check; stat"

for source in "${RTL[@]}"; do
  name="$(basename "$source" .sv)"
  python3 "$LINTER" "$source" >"$BUILD/erie_${name}.log" 2>&1
done

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi

echo "PASS: GateStack完整C0片上Builder；final-gate/K到typed slot三格式自动决策逐word金参考及全RTL门通过"
