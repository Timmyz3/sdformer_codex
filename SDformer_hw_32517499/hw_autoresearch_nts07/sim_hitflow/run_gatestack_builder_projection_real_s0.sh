#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_builder_projection_real_allstages"
mkdir -p "$BUILD"
cd "$ROOT"

BUILDER_RTL=(
  rtl_hitflow/gatestack_transposed_bitmap_bank.sv
  rtl_hitflow/gatestack_canonical_head_workspace_c0.sv
  rtl_hitflow/gatestack_typed_format_policy.sv
  rtl_hitflow/gatestack_typed_payload_serializer.sv
  rtl_hitflow/gatestack_typed_builder_commit_top.sv
  rtl_hitflow/gatestack_onchip_typed_builder_c0_top.sv
  rtl_hitflow/gatestack_onchip_typed_builder_c1_top.sv
)
mapfile -t EXECUTION_RTL < rtl_hitflow/filelist_single_context_execution.f
RTL=(
  "${BUILDER_RTL[@]}"
  "${EXECUTION_RTL[@]}"
  rtl_hitflow/gatestack_builder_projection_single_context_top.sv
)
TB=tb_hitflow/tb_gatestack_builder_projection_real_s0.sv

for stage in 0 1 2 3; do
for mode in 0 1; do
  mode_build="$BUILD/s${stage}_c${mode}"
  mkdir -p "$mode_build"
  iverilog -g2012 -Wall \
    -Ptb_gatestack_builder_projection_real_s0.BUILDER_C1_ENABLE="$mode" \
    -Ptb_gatestack_builder_projection_real_s0.STAGE_ID="$stage" \
    -s tb_gatestack_builder_projection_real_s0 \
    -o "$mode_build/tb.vvp" "${RTL[@]}" "$TB" \
    >"$mode_build/iverilog_build.log" 2>&1
  timeout 1200 vvp "$mode_build/tb.vvp" \
    | tee "$mode_build/iverilog.log"

  verilator --lint-only --timing -Wall \
    -GBUILDER_C1_ENABLE="$mode" \
    -GSTAGE_ID="$stage" \
    --top-module tb_gatestack_builder_projection_real_s0 \
    "${RTL[@]}" "$TB" >"$mode_build/verilator_lint.log" 2>&1
  if grep -Eq '%Warning|%Error' "$mode_build/verilator_lint.log"; then
    echo "FAIL: C${mode} Verilator elaboration存在warning/error" >&2
    cat "$mode_build/verilator_lint.log" >&2
    exit 1
  fi
done
done

for stage in 0 1 2 3; do
for mode in 0 1; do
  result_line="$(grep '^RESULT ' "$BUILD/s${stage}_c${mode}/iverilog.log")"
  expected_compared=$((162 * (3 << stage) * 32))
  if [[ "$result_line" != *"status=PASS"* ||
        "$result_line" != *"compared=$expected_compared"* ||
        "$result_line" != *"mismatches=0"* ||
        "$result_line" != *"payload_copy=0"* ||
        "$result_line" != *"errors=0"* ]]; then
    echo "FAIL: C${mode} RESULT字段不完整: $result_line" >&2
    exit 1
  fi
done
done

for stage in 0 1 2 3; do
  checksum_c0="$(sed -n 's/.* checksum=\([^ ]*\).*/\1/p' \
    "$BUILD/s${stage}_c0/iverilog.log")"
  checksum_c1="$(sed -n 's/.* checksum=\([^ ]*\).*/\1/p' \
    "$BUILD/s${stage}_c1/iverilog.log")"
  if [[ -z "$checksum_c0" || "$checksum_c0" != "$checksum_c1" ]]; then
    echo "FAIL: S${stage} C0/C1 checksum不一致 C0=$checksum_c0 C1=$checksum_c1" >&2
    exit 1
  fi
done

echo "RESULT suite=builder_projection_real_allstages status=PASS stages=4 modes=2 mismatches=0 iverilog=PASS verilator_lint=PASS"
echo "PASS: 真实S0-S3共45个head共享typed-slot旁路，C0/C1逐元素零差异，Icarus与Verilator双模式通过"
