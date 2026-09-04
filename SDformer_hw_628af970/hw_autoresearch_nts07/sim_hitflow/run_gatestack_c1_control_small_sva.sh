#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_c1_control_small"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_transposed_bitmap_bank.sv
  rtl_hitflow/gatestack_canonical_head_workspace_c0.sv
  rtl_hitflow/gatestack_typed_format_policy.sv
  rtl_hitflow/gatestack_typed_payload_serializer.sv
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv
  rtl_hitflow/gatestack_typed_builder_commit_top.sv
  rtl_hitflow/gatestack_onchip_typed_builder_c1_top.sv
)
SVA=(
  verif_hitflow/gatestack_canonical_head_workspace_c0_assertions.sv
  verif_hitflow/bind_gatestack_canonical_head_workspace_c0_assertions.sv
  verif_hitflow/gatestack_typed_payload_serializer_assertions.sv
  verif_hitflow/bind_gatestack_typed_payload_serializer_assertions.sv
  verif_hitflow/gatestack_head_slot_sram_adapter_assertions.sv
  verif_hitflow/bind_gatestack_head_slot_sram_adapter_assertions.sv
  verif_hitflow/gatestack_onchip_typed_builder_c1_top_assertions.sv
  verif_hitflow/bind_gatestack_onchip_typed_builder_c1_top_assertions.sv
)
TB=tb_hitflow/tb_gatestack_c1_control_small.sv

iverilog -g2012 -Wall -s tb_gatestack_c1_control_small \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --lint-only --timing --assert -Wall \
  --top-module tb_gatestack_c1_control_small \
  "${RTL[@]}" "${SVA[@]}" "$TB" >"$BUILD/verilator_build.log" 2>&1

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi

echo "PASS: GateStack C1缩参Icarus动态回归与Verilator-SVA elaboration通过"
