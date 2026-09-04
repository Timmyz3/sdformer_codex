#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_single_head_projection"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_single_head_projection_top.sv
)

iverilog -g2012 -Wall -s tb_gatestack_single_head_projection_top \
  -o "$BUILD/tb_top.vvp" "${RTL[@]}" \
  tb_hitflow/tb_gatestack_single_head_projection_top.sv
vvp "$BUILD/tb_top.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_single_head_projection_top \
  -Mdir "$BUILD/verilator_obj" "${RTL[@]}" \
  verif_hitflow/gatestack_single_head_projection_assertions.sv \
  verif_hitflow/bind_gatestack_single_head_projection_assertions.sv \
  tb_hitflow/tb_gatestack_single_head_projection_top.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_single_head_projection_top" \
  | tee "$BUILD/verilator_assert.log"

for log in "$BUILD/iverilog.log" "$BUILD/verilator_assert.log"; do
  if ! grep -Eq 'PASS: single-head req/rsp projection .*stalls=[1-9][0-9]*' \
      "$log"; then
    echo "FAIL: bias req/rsp定向覆盖（至少一拍、身份拒绝、response反压、ACC_W bias）缺失" >&2
    exit 1
  fi
done

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_single_head_projection_top; proc; opt; check; stat"

python "$LINTER" rtl_hitflow/gatestack_single_head_projection_top.sv \
  >"$BUILD/erie_lint.log" 2>&1 || true

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_lint.log"; then
  echo "FAIL: Erie lint存在MUST error" >&2
  cat "$BUILD/erie_lint.log" >&2
  exit 1
fi
echo "PASS: GateStack single-head bias req/rsp；延迟/反压/身份/ACC_W定向与Icarus/Verilator/SVA/Yosys/Erie通过"
