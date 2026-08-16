#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_denominator_certificate_rtl_20260814}"
BUILD="$OUT/build"
LOGS="$OUT/logs"
STD_LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
MACRO_LIB="$ROOT/third_party/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/fakeram45_256x16.lib"
PROFILE="$ROOT/results/h67_static_denominator_certificate_20260814/report.json"
mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_h67/h67_balanced_popcount32.sv
  rtl_h67/h67_row_qmax_denominator_certificate.sv
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_gate_quant_q17.sv
  rtl_h67/h67_certified_gate_quant_q17.sv
)
TB=tb_h67/tb_h67_denominator_certificate.sv
SVA=verif_h67/h67_denominator_certificate_assertions.sv

iverilog -g2012 -Wall -s tb_h67_denominator_certificate \
  -o "$BUILD/cert.vvp" "${RTL[@]}" "$TB" \
  >"$LOGS/iverilog_build.log" 2>&1
vvp "$BUILD/cert.vvp" >"$LOGS/iverilog.log" 2>&1
grep -q '^PASS tb_h67_denominator_certificate rows=18 gates=16200 errors=0$' \
  "$LOGS/iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-WIDTHTRUNC \
  -Wno-WIDTHEXPAND -Wno-DECLFILENAME -Wno-BLKSEQ \
  --top-module tb_h67_denominator_certificate \
  --Mdir "$BUILD/verilator" "${RTL[@]}" "$TB" "$SVA" \
  >"$LOGS/verilator_build.log" 2>&1
"$BUILD/verilator/Vtb_h67_denominator_certificate" \
  >"$LOGS/verilator.log" 2>&1
grep -q '^PASS tb_h67_denominator_certificate rows=18 gates=16200 errors=0$' \
  "$LOGS/verilator.log"

TOPS=(
  h67_row_qmax_denominator_certificate_core
  h67_row_qmax_denominator_certificate
  h67_row_qkm_denominator_certificate_core
  h67_row_qkm_denominator_certificate
  h67_row_qkm_denominator_certificate_reuse_qcounts
  ttx_gate_quant_q17
  h67_certified_gate_quant_q17
)
for top in "${TOPS[@]}"; do
  yosys -l "$LOGS/nangate45_${top}.log" -p "
    read_liberty -lib $STD_LIB;
    read_verilog -sv ${RTL[*]};
    hierarchy -check -top $top;
    synth -top $top;
    dfflibmap -liberty $STD_LIB;
    abc -liberty $STD_LIB;
    clean; check -assert; stat -liberty $STD_LIB;
    write_verilog -noattr $BUILD/${top}_mapped.v
  " >/dev/null

  if [[ "$top" == *gate_quant* ]]; then
    {
      echo "read_liberty $STD_LIB"
      echo "read_verilog $BUILD/${top}_mapped.v"
      echo "link_design $top"
      echo "create_clock -name vclk -period 3"
      echo 'set_input_delay 0.1 -clock vclk [all_inputs]'
      echo 'set_output_delay 0.1 -clock vclk [all_outputs]'
      echo 'report_checks -path_delay max -format full -digits 6'
      echo 'exit'
    } | sta >"$LOGS/sta_${top}.log" 2>&1
  else
    {
      echo "read_liberty $STD_LIB"
      echo "read_verilog $BUILD/${top}_mapped.v"
      echo "link_design $top"
      echo 'create_clock -name clk_core -period 3 [get_ports clk_core]'
      echo 'set_input_delay 0.1 -clock clk_core [remove_from_collection [all_inputs] [get_ports clk_core]]'
      echo 'set_output_delay 0.1 -clock clk_core [all_outputs]'
      echo 'report_checks -path_delay max -format full -digits 6'
      echo 'exit'
    } | sta >"$LOGS/sta_${top}.log" 2>&1
  fi
  grep -q 'data arrival time' "$LOGS/sta_${top}.log"
done

python3 -m unittest scripts.test_report_h67_denominator_certificate_rtl \
  >"$LOGS/python_unit_tests.log" 2>&1
python3 scripts/report_h67_denominator_certificate_rtl.py \
  --output-dir "$OUT" --profile "$PROFILE" --macro-lib "$MACRO_LIB" \
  >"$LOGS/report_stdout.log" 2>&1
grep -q 'FROZEN_LEAF_ONLY_NO_DIRECTORY' \
  "$OUT/report.json"

sha256sum "${RTL[@]}" "$TB" "$SVA" \
  scripts/report_h67_denominator_certificate_rtl.py \
  scripts/test_report_h67_denominator_certificate_rtl.py \
  sim_h67/run_h67_denominator_certificate_checks.sh \
  "$STD_LIB" "$MACRO_LIB" "$PROFILE" >"$OUT/source_input_sha256.txt"
sha256sum "$OUT"/logs/*.log "$OUT"/build/*_mapped.v \
  "$OUT/report.json" "$OUT/report.md" "$OUT/source_input_sha256.txt" \
  >"$OUT/result_sha256.txt"

echo 'PASS H67 load-time denominator certificate RTL screening'
