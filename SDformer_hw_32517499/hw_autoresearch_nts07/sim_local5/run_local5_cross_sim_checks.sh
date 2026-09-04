#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_local5/crosscheck"
OUT="$ROOT/results/local5_cross_sim_20260729.log"
mkdir -p "$BUILD"
cd "$ROOT"
: > "$OUT"

TARE_RTL=(
  rtl_delta/alpha_xnor_raw32.sv
  rtl_delta/alpha_xnor_delta4.sv
  rtl_delta/delta_bounded_classifier.sv
  rtl_delta/tare4_residual_composite_core.sv
  rtl_delta/local5_tare4_composite_top.sv
  rtl_local5/local5_row_context_tare_engine.sv
)
SGT_RTL=(
  "${TARE_RTL[@]}"
  rtl_local5/local5_axnor_score_q7.sv
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_local5/local5_stencil_token.sv
  rtl_local5/local5_row_context_engine.sv
  rtl_local5/local5_mfep_term_builder.sv
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv
  rtl_local5/local5_score_gate_term_top.sv
)

run_iverilog() {
  local name="$1"; shift
  local top="$1"; shift
  iverilog -g2012 -s "$top" -o "$BUILD/$name.vvp" "$@" \
    2>> "$OUT"
  vvp "$BUILD/$name.vvp" >> "$OUT"
}

run_iverilog row_tare tb_local5_row_context_tare \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/local5_tare4_composite_top.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  rtl_local5/local5_row_context_tare_engine.sv \
  tb_local5/tb_local5_row_context_tare.sv

run_iverilog sgt_tare tb_local5_score_gate_term_top \
  "${SGT_RTL[@]}" tb_local5/tb_local5_score_gate_term_top.sv

run_iverilog sgt_direct tb_local5_score_gate_term_top \
  -DLOCAL5_DIRECT_BASELINE \
  "${SGT_RTL[@]}" tb_local5/tb_local5_score_gate_term_top.sv

grep -q "^PASS tb_local5_row_context_tare" "$OUT"
grep -q "^PASS tb_local5_score_gate_term_top mode=DIRECT" "$OUT"
[[ "$(grep -c '^PASS tb_local5_score_gate_term_top' "$OUT")" -eq 2 ]]
echo "ALL LOCAL5 ICARUS CROSS-SIM CHECKS PASSED" | tee -a "$OUT"
