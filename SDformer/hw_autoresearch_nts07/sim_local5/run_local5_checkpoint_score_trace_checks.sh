#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_DIR="${TRACE_DIR:?TRACE_DIR is required}"
VECTOR_DIR="${VECTOR_DIR:?VECTOR_DIR is required}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"

mkdir -p "$VECTOR_DIR" "$RESULT_DIR"
cd "$ROOT"

"$PYTHON" scripts/generate_local5_checkpoint_score_vectors.py \
  --input-dir "$TRACE_DIR" \
  --output-dir "$VECTOR_DIR" \
  --per-stage 25 \
  > "$RESULT_DIR/vector_generation.log"

expected="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["vector_count"])' "$VECTOR_DIR/manifest.json")"
vectors="$VECTOR_DIR/local5_checkpoint_score_vectors.txt"

iverilog -g2012 -s tb_local5_score_shiftmax_vectors \
  -o "$RESULT_DIR/tb_local5_checkpoint_score.vvp" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  tb_local5/tb_local5_score_shiftmax_vectors.sv
vvp "$RESULT_DIR/tb_local5_checkpoint_score.vvp" \
  "+VECTORS=$vectors" "+EXPECTED=$expected" \
  > "$RESULT_DIR/iverilog.log"
grep -q "PASS tb_local5_score_shiftmax_vectors vectors=$expected" \
  "$RESULT_DIR/iverilog.log"

verilator --binary --timing -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  --top-module tb_local5_score_shiftmax_vectors \
  -Mdir "$RESULT_DIR/verilator_obj" \
  rtl_local5/local5_axnor_score_q7.sv \
  rtl_local5/local5_shiftmax5_q17.sv \
  tb_local5/tb_local5_score_shiftmax_vectors.sv \
  > "$RESULT_DIR/verilator_build.log" 2>&1
"$RESULT_DIR/verilator_obj/Vtb_local5_score_shiftmax_vectors" \
  "+VECTORS=$vectors" "+EXPECTED=$expected" \
  > "$RESULT_DIR/verilator.log"
grep -q "PASS tb_local5_score_shiftmax_vectors vectors=$expected" \
  "$RESULT_DIR/verilator.log"

yosys -p \
  "read_verilog -sv rtl_local5/local5_axnor_score_q7.sv; hierarchy -top local5_axnor_score_q7; proc; opt; check -assert; stat" \
  > "$RESULT_DIR/yosys.log" 2>&1
yosys -p \
  "read_verilog -sv rtl_local5/local5_shiftmax5_q17.sv; hierarchy -top local5_shiftmax5_q17; proc; opt; check -assert; stat" \
  >> "$RESULT_DIR/yosys.log" 2>&1

"$PYTHON" scripts/report_local5_checkpoint_score_rtl.py \
  --vector-dir "$VECTOR_DIR" \
  --result-dir "$RESULT_DIR" \
  --rtl-root "$ROOT"
