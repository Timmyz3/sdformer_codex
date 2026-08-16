#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
OUT="$ROOT/results/gatestack_equal96_term_boundary_mapping_20260722"
MAPPING="$OUT/mapping"
mkdir -p "$MAPPING"
cd "$ROOT"

if [[ ! -s "$LIB" ]]; then
  echo "缺少Nangate45 Liberty: $LIB" >&2
  exit 1
fi

# 三个候选使用同一RTL并集。hierarchy会移除未被相应顶层引用的模块，
# 从而保证前端边界一致，同时避免不同filelist造成的映射口径漂移。
RTL=(
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_transposed_bitmap_bank.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_three_independent32_term_projection_top.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)

run_mapping() {
  local name="$1"
  local top="$2"
  local chparam="$3"
  timeout 1800 yosys -q -l "$MAPPING/${name}.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; $chparam hierarchy -check -top $top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert; stat -liberty $LIB; write_verilog -noattr $MAPPING/${name}_mapped.v"
}

yosys -V >"$OUT/yosys_version.txt"
sha256sum "$LIB" "${RTL[@]}" >"$OUT/input_sha256.txt"

run_mapping central96_term gatestack_multihead_tile_projection_top \
  "chparam -set TOKENS 162 gatestack_multihead_tile_projection_top; chparam -set OUT_TILE 96 gatestack_multihead_tile_projection_top;"
run_mapping independent32x3_term gatestack_three_independent32_term_projection_top \
  "chparam -set TOKENS 162 gatestack_three_independent32_term_projection_top;"
run_mapping dctf96_term gatestack_dctf96_banklocal_projection_top \
  "chparam -set Q 2 gatestack_dctf96_banklocal_projection_top; chparam -set TOKENS 162 gatestack_dctf96_banklocal_projection_top; chparam -set OUT_TILE 32 gatestack_dctf96_banklocal_projection_top;"

python3 scripts/summarize_gatestack_equal96_term_boundary.py \
  --output-dir "$OUT"

echo "PASS: 三种96-lane结构在同term/event输入边界完成Nangate45开放无约束逻辑映射"
