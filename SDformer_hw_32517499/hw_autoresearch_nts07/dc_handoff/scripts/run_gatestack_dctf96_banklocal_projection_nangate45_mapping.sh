#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
OUT="$ROOT/results/gatestack_dctf96_banklocal_projection_mapping_20260720"
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
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)

yosys -V >"$OUT/yosys_version.txt"
sha256sum "$LIB" "${RTL[@]}" >"$OUT/input_sha256.txt"

timeout 1200 yosys -q -l "$MAPPING/top_q2_tokens162_out32.log" -p \
  "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; chparam -set Q 2 $TOP; chparam -set TOKENS 162 $TOP; chparam -set OUT_TILE 32 $TOP; hierarchy -check -top $TOP; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert; stat -liberty $LIB; write_verilog -noattr $MAPPING/top_q2_tokens162_out32_mapped.v"

if [[ ! -s "$MAPPING/top_q2_tokens162_out32_mapped.v" ]]; then
  echo "映射网表缺失或为空" >&2
  exit 1
fi

python3 scripts/summarize_gatestack_dctf96_banklocal_projection.py \
  --output-dir "$OUT"

echo "PASS: DCTF96 bank-local projection开放Nangate45无约束逻辑映射代理完成"
