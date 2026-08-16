#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_local5/sva"
mkdir -p "$BUILD"
cd "$ROOT"

python3 scripts/generate_local5_t450_window_vectors.py \
  --output "$ROOT/build_local5/parity/local5_t450_window_vectors.txt"

RTL=(
  rtl_delta/alpha_xnor_raw32.sv
  rtl_delta/alpha_xnor_delta4.sv
  rtl_delta/delta_bounded_classifier.sv
  rtl_delta/tare4_residual_composite_core.sv
  rtl_delta/local5_tare4_composite_top.sv
  rtl_local5/local5_row_context_tare_engine.sv
  rtl_local5/local5_axnor_score_q7.sv
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_local5/local5_stencil_token.sv
  rtl_local5/local5_row_context_engine.sv
  rtl_local5/local5_mfep_term_builder.sv
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv
  rtl_local5/local5_score_gate_term_top.sv
)

run_mode() {
  local mode="$1"
  local define=()
  if [[ "$mode" == "direct" ]]; then
    define=(-DLOCAL5_DIRECT_BASELINE)
  fi
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL \
    -Wno-UNUSEDPARAM -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
    -Wno-CMPCONST -Wno-BLKSEQ -Wno-INITIALDLY \
    --top-module tb_local5_score_gate_term_top \
    -Mdir "$BUILD/$mode" \
    "${define[@]}" "${RTL[@]}" \
    verif_local5/local5_score_gate_term_assertions.sv \
    tb_local5/tb_local5_score_gate_term_top.sv \
    > "$BUILD/${mode}_build.log" 2>&1
  "$BUILD/$mode/Vtb_local5_score_gate_term_top" \
    > "$BUILD/${mode}_sim.log" 2>&1
  grep -q "^PASS tb_local5_score_gate_term_top" "$BUILD/${mode}_sim.log"
}

run_mode tare
run_mode direct

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  -Wno-CMPCONST -Wno-BLKSEQ -Wno-INITIALDLY -Wno-UNOPTFLAT \
  --top-module tb_local5_window_t450 \
  -Mdir "$BUILD/t450" \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/local5_tare4_composite_top.sv \
  rtl_local5/local5_row_context_tare_engine.sv \
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
  verif_local5/local5_banklocal_projection_assertions.sv \
  tb_local5/tb_local5_window_t450.sv \
  > "$BUILD/t450_build.log" 2>&1
"$BUILD/t450/Vtb_local5_window_t450" > "$BUILD/t450_sim.log" 2>&1
grep -q "^PASS tb_local5_window_t450" "$BUILD/t450_sim.log"

echo "ALL LOCAL5 SVA CHECKS PASSED"
