#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
BUILD="$ROOT/build_hitflow/nangate45_supertile_mapping"
mkdir -p "$BUILD"
cd "$ROOT"

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
)

for width in 32 64 96 128; do
  timeout 1200 yosys -q -l "$BUILD/w${width}.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; chparam -set OUT_TILE $width gatestack_multihead_decoder_projection_top; hierarchy -check -top gatestack_multihead_decoder_projection_top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check; stat -liberty $LIB; write_verilog -noattr $BUILD/w${width}_mapped.v"
done

python3 scripts/summarize_gatestack_supertile_mapping.py \
  --mapping-dir "$BUILD" \
  --cycle-build-dir build_hitflow/gatestack_projection_supertile_sweep \
  --output-dir results/gatestack_supertile_mapping_20260720

echo "PASS: supertile宽度开放目标库逻辑映射及真实RTL周期联合报告完成"
