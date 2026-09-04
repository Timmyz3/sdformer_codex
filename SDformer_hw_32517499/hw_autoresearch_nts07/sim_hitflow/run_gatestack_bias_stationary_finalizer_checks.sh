#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_bias_stationary_finalizer"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

SINGLE_RTL=(
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_single_head_projection_top.sv
)
SINGLE_TB=tb_hitflow/tb_gatestack_single_head_projection_top.sv

iverilog -g2012 -Wall \
  -Ptb_gatestack_single_head_projection_top.BIAS_STATIONARY_ENABLE=1 \
  -s tb_gatestack_single_head_projection_top \
  -o "$BUILD/single_head.vvp" "${SINGLE_RTL[@]}" "$SINGLE_TB" \
  >"$BUILD/single_head_iverilog_build.log" 2>&1
vvp "$BUILD/single_head.vvp" | tee "$BUILD/single_head_iverilog.log"

verilator --binary --timing --assert -Wall \
  "-GBIAS_STATIONARY_ENABLE=1'b1" \
  --top-module tb_gatestack_single_head_projection_top \
  -Mdir "$BUILD/single_head_verilator_obj" "${SINGLE_RTL[@]}" \
  verif_hitflow/gatestack_single_head_projection_assertions.sv \
  verif_hitflow/bind_gatestack_single_head_projection_assertions.sv \
  "$SINGLE_TB" >"$BUILD/single_head_verilator_build.log" 2>&1
"$BUILD/single_head_verilator_obj/Vtb_gatestack_single_head_projection_top" \
  | tee "$BUILD/single_head_verilator.log"

for log in "$BUILD/single_head_iverilog.log" \
           "$BUILD/single_head_verilator.log"; do
  grep -Eq 'PASS: single-head req/rsp projection bsf=1 .*bias=8 req=2 rsp=2 .*final_stalls=[1-9][0-9]*' "$log"
done

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
REAL_RTL=(
  "${BUILDER_RTL[@]}"
  "${EXECUTION_RTL[@]}"
  rtl_hitflow/gatestack_builder_projection_single_context_top.sv
)
REAL_TB=tb_hitflow/tb_gatestack_builder_projection_real_s0.sv

for stage in 0 1 2 3; do
  stage_dir="$BUILD/hatf96_s${stage}"
  mkdir -p "$stage_dir"
  iverilog -g2012 -Wall \
    -Ptb_gatestack_builder_projection_real_s0.BUILDER_C1_ENABLE=0 \
    -Ptb_gatestack_builder_projection_real_s0.STAGE_ID="$stage" \
    -Ptb_gatestack_builder_projection_real_s0.OUT_TILE=96 \
    -Ptb_gatestack_builder_projection_real_s0.BIAS_STATIONARY_ENABLE=1 \
    -s tb_gatestack_builder_projection_real_s0 \
    -o "$stage_dir/tb.vvp" "${REAL_RTL[@]}" "$REAL_TB" \
    >"$stage_dir/iverilog_build.log" 2>&1
  timeout 1200 vvp "$stage_dir/tb.vvp" | tee "$stage_dir/iverilog.log"
  result_line="$(grep '^RESULT ' "$stage_dir/iverilog.log")"
  expected_tiles=$((1 << stage))
  expected_compared=$((162 * (3 << stage) * 32))
  if [[ "$result_line" != *"out_tile=96 bsf=1 status=PASS"* ||
        "$result_line" != *"compared=$expected_compared"* ||
        "$result_line" != *"mismatches=0"* ||
        "$result_line" != *"bias_req_hs=$expected_tiles"* ||
        "$result_line" != *"bias_rsp_hs=$expected_tiles"* ]]; then
    echo "FAIL: HATF96 S${stage} BSF RESULT异常: $result_line" >&2
    exit 1
  fi
done

verilator --lint-only --timing --assert -Wall \
  -GBUILDER_C1_ENABLE=0 -GSTAGE_ID=3 -GOUT_TILE=96 \
  "-GBIAS_STATIONARY_ENABLE=1'b1" \
  --top-module tb_gatestack_builder_projection_real_s0 \
  "${REAL_RTL[@]}" \
  verif_hitflow/gatestack_multihead_tile_projection_assertions.sv \
  verif_hitflow/bind_gatestack_multihead_tile_projection_assertions.sv \
  "$REAL_TB" >"$BUILD/hatf96_s3_verilator_sva.log" 2>&1

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${SINGLE_RTL[*]}; hierarchy -check -top gatestack_single_head_projection_top -chparam BIAS_STATIONARY_ENABLE 1; proc; opt; check; stat"

ERIE_RTL=(
  rtl_hitflow/gatestack_single_head_projection_top.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_routed_single_head_projection_top.sv
  rtl_hitflow/gatestack_decoder_projection_top.sv
  rtl_hitflow/gatestack_routed_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_multihead_decoder_projection_top.sv
  rtl_hitflow/gatestack_direct_raw_multihead_projection_top.sv
  rtl_hitflow/gatestack_single_context_execution_top.sv
  rtl_hitflow/gatestack_builder_projection_single_context_top.sv
)
for source in "${ERIE_RTL[@]}"; do
  name="$(basename "$source" .sv)"
  python3 "$LINTER" "$source" >"$BUILD/erie_${name}.log" 2>&1 || true
  if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_${name}.log"; then
    echo "FAIL: Erie lint MUST error: $source" >&2
    cat "$BUILD/erie_${name}.log" >&2
    exit 1
  fi
done

if grep -Eq '%Warning|%Error' "$BUILD/single_head_verilator_build.log" \
   "$BUILD/hatf96_s3_verilator_sva.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi

echo "RESULT suite=bias_stationary_finalizer status=PASS single_head=PASS hatf96_s0_s3=PASS mismatches=0 requests_per_supertile=1 iverilog=PASS verilator_sva=PASS yosys=PASS erie=PASS"
