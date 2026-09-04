#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_c1_all45_stagebounded"
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
TB=tb_hitflow/tb_gatestack_onchip_builder_c1_all45_stagebounded.sv

iverilog -g2012 -Wall -s tb_gatestack_onchip_builder_c1_all45_stagebounded \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

echo "PASS: GateStack C1四stage全部45-head有序重叠与逐word回放Icarus通过"
