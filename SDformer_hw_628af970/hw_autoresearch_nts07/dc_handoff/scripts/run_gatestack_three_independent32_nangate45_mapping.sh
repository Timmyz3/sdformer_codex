#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
BUILD="$ROOT/build_hitflow/nangate45_three_independent32_mapping"
mkdir -p "$BUILD"
cd "$ROOT"

if [[ ! -s "$LIB" ]]; then
  echo "缺少Nangate45 Liberty: $LIB" >&2
  exit 1
fi

RTL=(
  rtl_hitflow/gatestack_resident_replay_joiner.sv
  rtl_hitflow/gatestack_ipd32w_replay_decoder.sv
  rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv
  rtl_hitflow/gatestack_adaptive_csr_replay_decoder.sv
  rtl_hitflow/gatestack_raw41_replay_decoder.sv
  rtl_hitflow/gatestack_raw_tail_retimer.sv
  rtl_hitflow/gatestack_raw_issue_adapter.sv
  rtl_hitflow/gatestack_replay_mux.sv
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_routed_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_multihead_decoder_projection_top.sv
  rtl_hitflow/gatestack_three_independent32_projection_top.sv
)

run_mapping() {
  local name="$1"
  local top="$2"
  local chparam="$3"
  timeout 1800 yosys -q -l "$BUILD/${name}.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; $chparam hierarchy -check -top $top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check; stat -liberty $LIB; write_verilog -noattr $BUILD/${name}_mapped.v"
}

report_mapping() {
  local name="$1"
  local top="$2"
  local log="$BUILD/${name}.log"
  local cells area mem_v2
  cells="$(awk -v marker="=== ${top} ===" '
    $0 == marker { active=1; next }
    active && /Number of cells:/ { print $4; exit }
  ' "$log")"
  area="$(awk -v module="\\${top}" '
    $0 ~ "Chip area for module" && $0 ~ module { value=$NF }
    END { print value }
  ' "$log")"
  mem_v2="$(awk '
    $1 == "$mem_v2" { value=$2 }
    END { if (value == "") value=0; print value }
  ' "$log")"
  if [[ -z "$cells" || -z "$area" ]]; then
    echo "FAIL: 无法从 $log 提取最终映射统计" >&2
    exit 1
  fi
  echo "RESULT design=$name top=$top area=$area cells=$cells mem_v2=$mem_v2 library=$(basename "$LIB") memory_area=included_no"
}

run_mapping central96 gatestack_multihead_decoder_projection_top \
  "chparam -set OUT_TILE 96 gatestack_multihead_decoder_projection_top;"
run_mapping three_independent32 gatestack_three_independent32_projection_top ""

report_mapping central96 gatestack_multihead_decoder_projection_top
report_mapping three_independent32 gatestack_three_independent32_projection_top
python3 scripts/summarize_gatestack_equal96_mapping.py \
  --mapping-dir "$BUILD" \
  --output-dir results/gatestack_equal96_mapping_20260720
echo "PASS: Central96与3xIndependent32同RTL集合、同Nangate45 Liberty开放逻辑映射完成；\$mem_v2面积未计入area"
