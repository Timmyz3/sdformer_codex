#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
BUILD="$ROOT/build_hitflow/nangate45_mapping"
mkdir -p "$BUILD"
cd "$ROOT"

if [[ ! -s "$LIB" ]]; then
  echo "缺少Nangate45 Liberty: $LIB" >&2
  exit 1
fi

COMMON=(
  rtl_hitflow/gatestack_canonical_head_workspace_c0.sv
  rtl_hitflow/gatestack_typed_format_policy.sv
  rtl_hitflow/gatestack_typed_payload_serializer.sv
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv
  rtl_hitflow/gatestack_typed_builder_commit_top.sv
)

run_one() {
  local name="$1"
  local top="$2"
  local top_rtl="$3"
  timeout 900 yosys -q -l "$BUILD/${name}.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${COMMON[*]} $top_rtl; hierarchy -check -top $top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check; stat -liberty $LIB; write_verilog -noattr $BUILD/${name}_mapped.v"
}

run_one c0 gatestack_onchip_typed_builder_c0_top \
  rtl_hitflow/gatestack_onchip_typed_builder_c0_top.sv
run_one c1 gatestack_onchip_typed_builder_c1_top \
  rtl_hitflow/gatestack_onchip_typed_builder_c1_top.sv

python3 scripts/summarize_gatestack_nangate45_mapping.py \
  --c0-log "$BUILD/c0.log" \
  --c1-log "$BUILD/c1.log" \
  --lib "$LIB" \
  --output-dir results/gatestack_nangate45_mapping_20260720

echo "PASS: GateStack C0/C1 Nangate45开放目标库逻辑映射完成；memory面积、STA和功耗不在本口径内"
