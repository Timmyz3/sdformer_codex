#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_projection_supertile_sweep"
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

for width in 32 64 96 128; do
  for stage in 0 1 2 3; do
    run_dir="$BUILD/w${width}_s${stage}"
    mkdir -p "$run_dir"
    iverilog -g2012 -Wall \
      -Ptb_gatestack_builder_projection_real_s0.BUILDER_C1_ENABLE=0 \
      -Ptb_gatestack_builder_projection_real_s0.STAGE_ID="$stage" \
      -Ptb_gatestack_builder_projection_real_s0.OUT_TILE="$width" \
      -s tb_gatestack_builder_projection_real_s0 \
      -o "$run_dir/tb.vvp" "${RTL[@]}" "$TB" \
      >"$run_dir/iverilog_build.log" 2>&1
    timeout 1200 vvp "$run_dir/tb.vvp" | tee "$run_dir/iverilog.log"
    result_line="$(grep '^RESULT ' "$run_dir/iverilog.log")"
    expected_compared=$((162 * (3 << stage) * 32))
    if [[ "$result_line" != *"out_tile=$width"* ||
          "$result_line" != *"status=PASS"* ||
          "$result_line" != *"compared=$expected_compared"* ||
          "$result_line" != *"mismatches=0"* ||
          "$result_line" != *"payload_copy=0"* ]]; then
      echo "FAIL: W${width} S${stage} RESULT异常: $result_line" >&2
      exit 1
    fi
  done

  # 每个宽度在最大 stage 做一次全层次参数化 elaboration/lint。
  verilator --lint-only --timing -Wall \
    -GBUILDER_C1_ENABLE=0 -GSTAGE_ID=3 -GOUT_TILE="$width" \
    --top-module tb_gatestack_builder_projection_real_s0 \
    "${RTL[@]}" "$TB" >"$BUILD/w${width}_verilator_lint.log" 2>&1
  if grep -Eq '%Warning|%Error' "$BUILD/w${width}_verilator_lint.log"; then
    echo "FAIL: W${width} Verilator存在warning/error" >&2
    cat "$BUILD/w${width}_verilator_lint.log" >&2
    exit 1
  fi
done

echo "RESULT suite=projection_supertile_sweep status=PASS widths=32,64,96,128 stages=4 mismatches=0 iverilog=PASS verilator_lint=PASS"
echo "PASS: C0真实S0-S3跨输出supertile宽度逐元素零差异"
