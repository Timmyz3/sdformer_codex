#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_MANIFEST="${TRACE_MANIFEST:?TRACE_MANIFEST is required}"
VECTOR_DIR="${VECTOR_DIR:?VECTOR_DIR is required}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"

mkdir -p "$VECTOR_DIR" "$RESULT_DIR"
cd "$ROOT"

"$PYTHON" scripts/generate_h67_checkpoint_row_vectors.py \
  --manifest "$TRACE_MANIFEST" \
  --output-dir "$VECTOR_DIR" \
  --expected-tokens 450 \
  > "$RESULT_DIR/vector_generation.log"

rows="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["row_count"])' "$VECTOR_DIR/manifest.json")"
active="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_active_outputs"])' "$VECTOR_DIR/manifest.json")"
vectors="$VECTOR_DIR/h67_checkpoint_rows.txt"
rtl=(
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_gate_quant_q17.sv
  rtl_h67/h67_temporal_pair_adapter.sv
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_score_class_row_engine.sv
)

iverilog -g2012 -s tb_h67_checkpoint_rows \
  -o "$RESULT_DIR/tb_h67_checkpoint_rows.vvp" \
  "${rtl[@]}" tb_h67/tb_h67_checkpoint_rows.sv
vvp "$RESULT_DIR/tb_h67_checkpoint_rows.vvp" \
  "+VECTORS=$vectors" > "$RESULT_DIR/iverilog.log"
grep -q "PASS tb_h67_checkpoint_rows rows=$rows tokens=450 checked_outputs=$active" \
  "$RESULT_DIR/iverilog.log"

verilator --binary --timing -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  --top-module tb_h67_checkpoint_rows \
  -Mdir "$RESULT_DIR/verilator_obj" \
  "${rtl[@]}" tb_h67/tb_h67_checkpoint_rows.sv \
  > "$RESULT_DIR/verilator_build.log" 2>&1
"$RESULT_DIR/verilator_obj/Vtb_h67_checkpoint_rows" \
  "+VECTORS=$vectors" > "$RESULT_DIR/verilator.log"
grep -q "PASS tb_h67_checkpoint_rows rows=$rows tokens=450 checked_outputs=$active" \
  "$RESULT_DIR/verilator.log"

yosys -p \
  "read_verilog -sv ${rtl[*]}; chparam -set MAX_TOKENS 450 h67_score_class_row_engine; hierarchy -check -top h67_score_class_row_engine; proc; opt; check; stat" \
  > "$RESULT_DIR/yosys.log" 2>&1

"$PYTHON" scripts/report_h67_checkpoint_row_rtl.py \
  --vector-dir "$VECTOR_DIR" \
  --result-dir "$RESULT_DIR" \
  --rtl-root "$ROOT"
