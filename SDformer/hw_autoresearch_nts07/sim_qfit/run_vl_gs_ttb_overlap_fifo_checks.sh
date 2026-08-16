#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

OUT=results/vl_gs_ttb_overlap_fifo_rtl_20260801
mkdir -p "$OUT"

iverilog -g2012 -s tb_qfit_vl_gs_ttb_motion_dvco \
  -o "$OUT/motion_dvco.vvp" \
  rtl_qfit/qfit_vl_gs_ttb_motion_dvco.sv \
  tb_qfit/tb_qfit_vl_gs_ttb_motion_dvco.sv
vvp "$OUT/motion_dvco.vvp" | tee "$OUT/motion_dvco.log"

iverilog -g2012 -s tb_qfit_vl_gs_ttb_abic_trace \
  -o "$OUT/local_abic.vvp" \
  rtl_qfit/qfit_vl_gs_ttb_abic_decoder.sv \
  tb_qfit/tb_qfit_vl_gs_ttb_abic_trace.sv
vvp "$OUT/local_abic.vvp" | tee "$OUT/local_abic.log"

verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module qfit_vl_gs_ttb_motion_dvco \
  rtl_qfit/qfit_vl_gs_ttb_motion_dvco.sv \
  > "$OUT/motion_dvco_lint.log" 2>&1
verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module qfit_vl_gs_ttb_abic_decoder \
  rtl_qfit/qfit_vl_gs_ttb_abic_decoder.sv \
  > "$OUT/local_abic_lint.log" 2>&1

rm -rf "$OUT/verilator_motion" "$OUT/verilator_abic"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "$OUT/verilator_motion" \
  --top-module tb_qfit_vl_gs_ttb_motion_dvco \
  rtl_qfit/qfit_vl_gs_ttb_motion_dvco.sv \
  tb_qfit/tb_qfit_vl_gs_ttb_motion_dvco.sv \
  verif_qfit/qfit_vl_gs_ttb_overlap_assertions.sv \
  > "$OUT/motion_dvco_verilator_build.log" 2>&1
"$OUT/verilator_motion/Vtb_qfit_vl_gs_ttb_motion_dvco" \
  | tee "$OUT/motion_dvco_verilator_run.log"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "$OUT/verilator_abic" \
  --top-module tb_qfit_vl_gs_ttb_abic_trace \
  rtl_qfit/qfit_vl_gs_ttb_abic_decoder.sv \
  tb_qfit/tb_qfit_vl_gs_ttb_abic_trace.sv \
  verif_qfit/qfit_vl_gs_ttb_overlap_assertions.sv \
  > "$OUT/local_abic_verilator_build.log" 2>&1
"$OUT/verilator_abic/Vtb_qfit_vl_gs_ttb_abic_trace" \
  | tee "$OUT/local_abic_verilator_run.log"

yosys -p \
  'read_verilog -sv rtl_qfit/qfit_vl_gs_ttb_motion_dvco.sv; hierarchy -top qfit_vl_gs_ttb_motion_dvco; proc; opt; check; stat' \
  > "$OUT/motion_dvco_yosys.log" 2>&1
yosys -p \
  'read_verilog -sv rtl_qfit/qfit_vl_gs_ttb_abic_decoder.sv; hierarchy -top qfit_vl_gs_ttb_abic_decoder; proc; opt; check; stat' \
  > "$OUT/local_abic_yosys.log" 2>&1

python -m unittest tests.test_vl_gs_ttb_overlap_fifo \
  | tee "$OUT/model_unittest.log"
python scripts/model_vl_gs_ttb_overlap_fifo.py \
  > "$OUT/model_stdout.json"

sha256sum \
  rtl_qfit/qfit_vl_gs_ttb_motion_dvco.sv \
  rtl_qfit/qfit_vl_gs_ttb_abic_decoder.sv \
  tb_qfit/tb_qfit_vl_gs_ttb_motion_dvco.sv \
  tb_qfit/tb_qfit_vl_gs_ttb_abic_trace.sv \
  verif_qfit/qfit_vl_gs_ttb_overlap_assertions.sv \
  scripts/model_vl_gs_ttb_overlap_fifo.py \
  tests/test_vl_gs_ttb_overlap_fifo.py \
  results/qfit_local5_projection_tile_yosys_20260731/ordered_term_trace.csv \
  "$0" > "$OUT/sha256.txt"

echo "PASS VL-GS-TTB overlap/FIFO RTL checks"
