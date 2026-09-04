#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
OUT="$ROOT/results/gatestack_dctf96_2c_mapping_20260722"
MAPPING="$OUT/mapping"
TOP="gatestack_dctf96_banklocal_projection_top"
mkdir -p "$MAPPING"
cd "$ROOT"

if [[ ! -s "$LIB" ]]; then
  echo "缺少Nangate45 Liberty: $LIB" >&2
  exit 1
fi

RTL=(
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)

yosys -V >"$OUT/yosys_version.txt"
sha256sum "$LIB" "${RTL[@]}" >"$OUT/input_sha256.txt"
timeout 1800 yosys -q -l "$MAPPING/dctf96_2c.log" -p \
  "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; chparam -set Q 2 $TOP; chparam -set TOKENS 162 $TOP; chparam -set OUT_TILE 32 $TOP; chparam -set ADAPTER_CONTEXTS 2 $TOP; hierarchy -check -top $TOP; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert; stat -liberty $LIB; write_verilog -noattr $MAPPING/dctf96_2c_mapped.v"

python3 scripts/summarize_gatestack_dctf96_2c_mapping.py \
  --output-dir "$OUT" \
  --baseline results/gatestack_equal96_term_boundary_mapping_20260722/report.json
echo "PASS: DCTF96-2C开放Nangate45无约束逻辑映射完成"
