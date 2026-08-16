#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_canonical_head_workspace_c0"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_transposed_bitmap_bank.sv
  rtl_hitflow/gatestack_canonical_head_workspace_c0.sv
)
TB=tb_hitflow/tb_gatestack_canonical_head_workspace_c0.sv

for storage in 0 1; do
for mode in 0 1; do
  iverilog -g2012 -Wall \
    -Ptb_gatestack_canonical_head_workspace_c0.DESTINATION_SCAN_MODE="$mode" \
    -Ptb_gatestack_canonical_head_workspace_c0.EXPLICIT_BITMAP_BANK_ENABLE="$storage" \
    -s tb_gatestack_canonical_head_workspace_c0 \
    -o "$BUILD/tb_storage${storage}_mode${mode}.vvp" "${RTL[@]}" "$TB" \
    >"$BUILD/iverilog_storage${storage}_mode${mode}_build.log" 2>&1
  vvp "$BUILD/tb_storage${storage}_mode${mode}.vvp" | \
    tee "$BUILD/iverilog_storage${storage}_mode${mode}.log"

  verilator --binary --timing --assert -Wall \
    -GDESTINATION_SCAN_MODE="$mode" \
    -GEXPLICIT_BITMAP_BANK_ENABLE="$storage" \
    --top-module tb_gatestack_canonical_head_workspace_c0 \
    -Mdir "$BUILD/verilator_storage${storage}_mode${mode}_obj" \
    "${RTL[@]}" \
    verif_hitflow/gatestack_canonical_head_workspace_c0_assertions.sv \
    verif_hitflow/bind_gatestack_canonical_head_workspace_c0_assertions.sv \
    "$TB" >"$BUILD/verilator_storage${storage}_mode${mode}_build.log" 2>&1
  "$BUILD/verilator_storage${storage}_mode${mode}_obj/Vtb_gatestack_canonical_head_workspace_c0" | \
    tee "$BUILD/verilator_storage${storage}_mode${mode}.log"

  if [[ "$storage" == 0 ]]; then
    timeout 300 yosys -q -l "$BUILD/yosys_mode${mode}.log" -p \
      "read_verilog -sv ${RTL[*]}; chparam -set DESTINATION_SCAN_MODE $mode gatestack_canonical_head_workspace_c0; hierarchy -check -top gatestack_canonical_head_workspace_c0; proc; opt; memory_collect; check; stat"
  fi
done
done

for source in "${RTL[@]}"; do
  name="$(basename "$source" .sv)"
  python3 "$LINTER" "$source" >"$BUILD/erie_${name}.log" 2>&1
done

if grep -Eq '%Warning|%Error' "$BUILD"/verilator_storage*_build.log; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi

echo "PASS: GateStack C0 canonical workspace；隐式/显式bitmap存储×线性/16-bit分段扫描真实金参考、Icarus、Verilator-SVA通过；默认隐式存储Yosys、全文件Erie通过"
