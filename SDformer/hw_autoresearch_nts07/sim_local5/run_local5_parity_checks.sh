#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_local5/parity"
mkdir -p "$BUILD"
cd "$ROOT"

VFLAGS=(--binary --timing -Wall -Wno-fatal
  -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL
  -Wno-UNUSEDPARAM -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-CMPCONST
  -Wno-BLKSEQ -Wno-UNOPTFLAT -Wno-INITIALDLY -Wno-CASEINCOMPLETE
  -Wno-LATCH -Wno-SYNCASYNCNET -Wno-UNDRIVEN)

TARE_RTL=(
  rtl_delta/alpha_xnor_raw32.sv
  rtl_delta/alpha_xnor_delta4.sv
  rtl_delta/delta_bounded_classifier.sv
  rtl_delta/tare4_residual_composite_core.sv
  rtl_delta/local5_tare4_composite_top.sv
  rtl_local5/local5_row_context_tare_engine.sv
)

run_one() {
  local top="$1"; shift
  local dir="$1"; shift
  echo "=== Verilator $top ==="
  verilator "${VFLAGS[@]}" --top-module "$top" -Mdir "$BUILD/$dir" "$@" \
    2>&1 | tee "$BUILD/${dir}_build.log"
  "$BUILD/$dir/V${top}" 2>&1 | tee "$BUILD/${dir}_sim.log"
  grep -q "^PASS " "$BUILD/${dir}_sim.log"
}

python3 scripts/generate_local5_masked_integer_vectors.py \
  --output "$BUILD/local5_score_shiftmax_vectors.txt"
python3 scripts/generate_local5_t450_window_vectors.py \
  --output "$BUILD/local5_t450_window_vectors.txt"

run_one tb_local5_score_shiftmax_vectors numeric \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  tb_local5/tb_local5_score_shiftmax_vectors.sv

run_one tb_local5_row_context row \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  tb_local5/tb_local5_row_context.sv

run_one tb_local5_row_protocol row_protocol \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  tb_local5/tb_local5_row_protocol.sv

run_one tb_local5_row_context_tare row_tare \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/local5_tare4_composite_top.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_row_context_tare_engine.sv \
  tb_local5/tb_local5_row_context_tare.sv

run_one tb_local5_mfep_term_builder mfep \
  rtl_local5/local5_mfep_term_builder.sv \
  tb_local5/tb_local5_mfep_term_builder.sv

run_one tb_local5_mfep_sparse_last mfep_sparse_last \
  rtl_local5/local5_mfep_term_builder.sv \
  tb_local5/tb_local5_mfep_sparse_last.sv

run_one tb_local5_mfep_protocol mfep_protocol \
  rtl_local5/local5_mfep_term_builder.sv \
  tb_local5/tb_local5_mfep_protocol.sv

run_one tb_local5_mfep_t450_counter mfep_t450_counter \
  rtl_local5/local5_mfep_term_builder.sv \
  tb_local5/tb_local5_mfep_t450_counter.sv

run_one tb_local5_dctf_multiset_bridge_protocol bridge_protocol \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  tb_local5/tb_local5_dctf_multiset_bridge_protocol.sv

run_one tb_local5_score_gate_term_top sgt \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  tb_local5/tb_local5_score_gate_term_top.sv

run_one tb_local5_score_gate_term_top sgt_direct \
  -DLOCAL5_DIRECT_BASELINE \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  tb_local5/tb_local5_score_gate_term_top.sv

run_one tb_local5_score_gate_term_top sgt_t450 \
  -DLOCAL5_DEST_W9 \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  tb_local5/tb_local5_score_gate_term_top.sv

run_one tb_local5_line_buffer lbuf \
  rtl_local5/local5_line_buffer_3row.sv \
  tb_local5/tb_local5_line_buffer.sv

# window TB logs as window_sim.log for ledger sniff
echo "=== Verilator tb_local5_window_attention ==="
verilator "${VFLAGS[@]}" --top-module tb_local5_window_attention \
  -Mdir "$BUILD/window" \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_window_attention.sv \
  2>&1 | tee "$BUILD/window_build.log"
"$BUILD/window/Vtb_local5_window_attention" 2>&1 | tee "$BUILD/window_sim.log"
grep -q "^PASS tb_local5_window_attention mode=TARE" "$BUILD/window_sim.log"

echo "=== Verilator tb_local5_window_attention DIRECT ==="
verilator "${VFLAGS[@]}" --top-module tb_local5_window_attention \
  -DLOCAL5_DIRECT_BASELINE \
  -Mdir "$BUILD/window_direct" \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_window_attention.sv \
  2>&1 | tee "$BUILD/window_direct_build.log"
"$BUILD/window_direct/Vtb_local5_window_attention" 2>&1 | tee "$BUILD/window_direct_sim.log"
grep -q "^PASS tb_local5_window_attention mode=DIRECT" "$BUILD/window_direct_sim.log"

run_one tb_local5_zero_term_window zero_term \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_zero_term_window.sv

run_one tb_local5_zero_term_window zero_term_tare \
  -DLOCAL5_TARE_MODE \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_zero_term_window.sv

run_one tb_local5_window_t450 window_t450 \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_window_t450.sv

echo "=== Verilator tb_local5_window16 ==="
verilator "${VFLAGS[@]}" --top-module tb_local5_window16 \
  -Mdir "$BUILD/w16" \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_window16.sv \
  2>&1 | tee "$BUILD/w16_build.log"
"$BUILD/w16/Vtb_local5_window16" 2>&1 | tee "$BUILD/w16_sim.log"
grep -q "^PASS tb_local5_window16 mode=TARE" "$BUILD/w16_sim.log"

echo "=== Verilator tb_local5_window16 DIRECT ==="
verilator "${VFLAGS[@]}" --top-module tb_local5_window16 \
  -DLOCAL5_DIRECT_BASELINE \
  -Mdir "$BUILD/w16_direct" \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_banklocal_projection_top.sv \
  rtl_local5/local5_stt_descriptor.sv \
  rtl_local5/local5_window_attention_top.sv \
  tb_local5/tb_local5_window16.sv \
  2>&1 | tee "$BUILD/w16_direct_build.log"
"$BUILD/w16_direct/Vtb_local5_window16" 2>&1 | tee "$BUILD/w16_direct_sim.log"
grep -q "^PASS tb_local5_window16 mode=DIRECT" "$BUILD/w16_direct_sim.log"

echo "=== Verilator tb_local5_linebuf_window ==="
verilator "${VFLAGS[@]}" --top-module tb_local5_linebuf_window \
  -Mdir "$BUILD/lbw" \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_multibank_projection_top.sv \
  rtl_local5/local5_line_buffer_3row.sv \
  rtl_local5/local5_stencil_linebuf_fetcher.sv \
  rtl_local5/local5_linebuf_window_top.sv \
  tb_local5/tb_local5_linebuf_window.sv \
  2>&1 | tee "$BUILD/lbw_build.log"
"$BUILD/lbw/Vtb_local5_linebuf_window" 2>&1 | tee "$BUILD/lbw_sim.log"
grep -q "^PASS tb_local5_linebuf_window mode=TARE" "$BUILD/lbw_sim.log"

echo "=== Verilator tb_local5_linebuf_window DIRECT ==="
verilator "${VFLAGS[@]}" --top-module tb_local5_linebuf_window \
  -DLOCAL5_DIRECT_BASELINE \
  -Mdir "$BUILD/lbw_direct" \
  "${TARE_RTL[@]}" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_stencil_token.sv \
  rtl_local5/local5_row_context_engine.sv \
  rtl_local5/local5_mfep_term_builder.sv \
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv \
  rtl_local5/local5_score_gate_term_top.sv \
  rtl_local5/local5_dctf_multiset_bridge.sv \
  rtl_local5/local5_multibank_projection_top.sv \
  rtl_local5/local5_line_buffer_3row.sv \
  rtl_local5/local5_stencil_linebuf_fetcher.sv \
  rtl_local5/local5_linebuf_window_top.sv \
  tb_local5/tb_local5_linebuf_window.sv \
  2>&1 | tee "$BUILD/lbw_direct_build.log"
"$BUILD/lbw_direct/Vtb_local5_linebuf_window" 2>&1 | tee "$BUILD/lbw_direct_sim.log"
grep -q "^PASS tb_local5_linebuf_window mode=DIRECT" "$BUILD/lbw_direct_sim.log"

echo "=== Python ledgers ==="
python3 scripts/local5_motion_parity_cycle_model.py
python3 scripts/local5_prosperity_style_simulator.py
python3 scripts/local5_equal_lane_cycle_ledger.py
python3 scripts/local5_collect_verilator_cycles.py

echo "ALL LOCAL5 PARITY+WINDOW CHECKS PASSED"
