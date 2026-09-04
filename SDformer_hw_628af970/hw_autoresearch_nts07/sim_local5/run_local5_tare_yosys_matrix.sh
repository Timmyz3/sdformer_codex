#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/local5_tare_yosys_20260729"
mkdir -p "$OUT"
cd "$ROOT"

RTL=(
  rtl_delta/alpha_xnor_raw32.sv
  rtl_delta/alpha_xnor_delta4.sv
  rtl_delta/delta_bounded_classifier.sv
  rtl_delta/tare4_residual_composite_core.sv
  rtl_delta/local5_tare4_composite_top.sv
  rtl_local5/local5_row_context_tare_engine.sv
  rtl_local5/local5_axnor_score_q7.sv
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_local5/local5_row_context_engine.sv
  rtl_local5/local5_mfep_term_builder.sv
  rtl_local5/local5_mfep_dctf_cmd_adapter.sv
  rtl_local5/local5_score_gate_term_top.sv
)

run_point() {
  local name="$1"
  local use_tare="$2"
  local dest_w="$3"
  local read_cmd="read_verilog -sv ${RTL[*]}"

  yosys -q -l "$OUT/${name}.log" -p "
    $read_cmd;
    chparam -set USE_TARE $use_tare -set DEST_W $dest_w \
      local5_score_gate_term_top;
    hierarchy -top local5_score_gate_term_top;
    proc; flatten; opt; memory; opt; techmap; opt;
    abc -fast; clean;
    write_verilog -noattr $OUT/${name}_netlist.v
  "

  yosys -q -p "
    read_verilog $OUT/${name}_netlist.v;
    hierarchy -top local5_score_gate_term_top;
    proc; opt;
    tee -o $OUT/${name}_stat.json stat -json
  "
  python3 -m json.tool "$OUT/${name}_stat.json" >/dev/null
}

run_point tare_dw8 1 8
run_point direct_dw8 0 8
run_point tare_dw9 1 9
run_point direct_dw9 0 9

python3 scripts/report_local5_tare_direct_matrix.py
echo "ALL LOCAL5 TARE YOSYS MATRIX CHECKS PASSED"
