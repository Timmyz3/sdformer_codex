#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_builder_projection_single_context_small"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
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
TB=tb_hitflow/tb_gatestack_builder_projection_single_context_small.sv

for mode in 0 1; do
  mode_build="$BUILD/c${mode}"
  mkdir -p "$mode_build"
  iverilog -g2012 -Wall \
    -Ptb_gatestack_builder_projection_single_context_small.BUILDER_C1_ENABLE="$mode" \
    -s tb_gatestack_builder_projection_single_context_small \
    -o "$mode_build/tb.vvp" "${RTL[@]}" "$TB" \
    >"$mode_build/iverilog_build.log" 2>&1
  vvp "$mode_build/tb.vvp" | tee "$mode_build/iverilog.log"

  verilator --lint-only --timing -Wall \
    -GBUILDER_C1_ENABLE="$mode" \
    --top-module tb_gatestack_builder_projection_single_context_small \
    "${RTL[@]}" "$TB" >"$mode_build/verilator_lint.log" 2>&1
  if grep -Eq '%Warning|%Error' "$mode_build/verilator_lint.log"; then
    echo "FAIL: C${mode} Verilator存在warning/error" >&2
    cat "$mode_build/verilator_lint.log" >&2
    exit 1
  fi
done

sig_c0="$(sed -n 's/.*RESULT mode=C0 signature=\([^ ]*\).*/\1/p' \
  "$BUILD/c0/iverilog.log")"
sig_c1="$(sed -n 's/.*RESULT mode=C1 signature=\([^ ]*\).*/\1/p' \
  "$BUILD/c1/iverilog.log")"
if [[ -z "$sig_c0" || "$sig_c0" != "$sig_c1" ]]; then
  echo "FAIL: C0/C1 final输出签名不一致 C0=$sig_c0 C1=$sig_c1" >&2
  exit 1
fi

python3 "$LINTER" \
  rtl_hitflow/gatestack_builder_projection_single_context_top.sv \
  >"$BUILD/erie_top.log" 2>&1
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_top.log"; then
  echo "FAIL: Erie新顶层存在MUST错误" >&2
  cat "$BUILD/erie_top.log" >&2
  exit 1
fi

echo "PASS: C0/C1共享typed-slot无payload复制，Icarus同值输出、replay/release、projection与group闭环通过；Verilator双模式和Erie新顶层通过"
