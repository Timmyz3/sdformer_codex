#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_direct_raw_physical_baseline"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

DIRECT_RTL=(
  rtl_hitflow/gatestack_raw41_replay_decoder.sv
  rtl_hitflow/gatestack_raw_tail_retimer.sv
  rtl_hitflow/gatestack_raw_issue_adapter.sv
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_direct_raw_multihead_projection_top.sv
)
FULL_RTL=(
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

iverilog -g2012 -Wall \
  -s tb_gatestack_direct_raw_multihead_projection_top \
  -o "$BUILD/tb.vvp" "${DIRECT_RTL[@]}" \
  tb_hitflow/tb_gatestack_direct_raw_multihead_projection_top.sv \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_direct_raw_multihead_projection_top \
  -Mdir "$BUILD/verilator_obj" "${DIRECT_RTL[@]}" \
  verif_hitflow/gatestack_raw_tail_retimer_assertions.sv \
  verif_hitflow/bind_gatestack_raw_tail_retimer_assertions.sv \
  verif_hitflow/gatestack_multihead_tile_projection_assertions.sv \
  verif_hitflow/bind_gatestack_multihead_tile_projection_assertions.sv \
  tb_hitflow/tb_gatestack_direct_raw_multihead_projection_top.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_direct_raw_multihead_projection_top" \
  | tee "$BUILD/verilator.log"

yosys -ql "$BUILD/yosys_direct_fair.log" -p \
  "read_verilog -sv ${DIRECT_RTL[*]}; hierarchy -check -top gatestack_direct_raw_multihead_projection_top; proc; flatten; opt; memory -nomap; stat"
yosys -ql "$BUILD/yosys_ipd_fair.log" -p \
  "read_verilog -sv ${FULL_RTL[*]}; hierarchy -check -top gatestack_multihead_decoder_projection_top; proc; flatten; opt; memory -nomap; stat"
yosys -ql "$BUILD/yosys_adaptive_fair.log" -p \
  "read_verilog -sv ${FULL_RTL[*]}; chparam -set CSR_FORMAT_FADC24 2 gatestack_multihead_decoder_projection_top; hierarchy -check -top gatestack_multihead_decoder_projection_top; proc; flatten; opt; memory -nomap; stat"

python3 "$LINTER" rtl_hitflow/gatestack_direct_raw_multihead_projection_top.sv \
  >"$BUILD/erie_lint.log" 2>&1 || true
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Direct RAW Verilator存在warning/error" >&2
  exit 1
fi
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' "$BUILD/erie_lint.log"; then
  echo "FAIL: Direct RAW Erie lint存在MUST error" >&2
  cat "$BUILD/erie_lint.log" >&2
  exit 1
fi
if grep -Eq 'gatestack_(resident|ipd32w|fadc24|adaptive_csr|replay_mux)' \
    "$BUILD/yosys_direct_fair.log"; then
  echo "FAIL: Direct RAW综合层次仍含CSR/resident/replay mux" >&2
  exit 1
fi

python3 scripts/summarize_gatestack_direct_raw_physical_baseline.py \
  --output-dir results/gatestack_direct_raw_physical_baseline_20260718

echo "PASS: physically-stripped Direct RAW41仿真、SVA、lint与结构综合完成"
