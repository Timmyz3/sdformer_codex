#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_h67/vectors/h67_zkqi_canonical_multisample_20260809}"
OUT="${RESULT_DIR:-$ROOT/results/h67_zkqi_canonical_replay_20260809}"
BUILD="$OUT/build"
LOGS="$OUT/logs"
mkdir -p "$VECTOR_DIR" "$BUILD" "$LOGS"
cd "$ROOT"

python3 -m unittest \
  tests.test_profile_h67_zkqi_multisample_ordered \
  tests.test_generate_h67_zkqi_canonical_vectors -v
python3 scripts/generate_h67_zkqi_canonical_vectors.py \
  --output-dir "$VECTOR_DIR" >"$LOGS/vector_generation.log"

MANIFEST="$VECTOR_DIR/manifest.json"
VECTORS="$VECTOR_DIR/h67_canonical_rows.txt"
ROWS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["coverage"]["rows"])' "$MANIFEST")"
RTL=(
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_sync_qk_row_store.sv
  rtl_h67/h67_fakeram45_qk_row_store.sv
  rtl_h67/h67_ttb8_metadata_builder.sv
  rtl_h67/h67_pair_bitmap_metadata_builder.sv
  rtl_h67/h67_active_bundle_fifo.sv
  rtl_h67/h67_banked_active_descriptor_store.sv
  rtl_h67/h67_temporal_weighted_scs_directory_seed_2s.sv
  rtl_h67/h67_zkqi_row_shiftmax_top.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_gate_quant_q17.sv
)

iverilog -g2012 -Wall \
  -P tb_h67_zkqi_row_miter.CANDIDATE_BUNDLE_SKIP_ENABLE=1 \
  -s tb_h67_zkqi_row_miter -o "$BUILD/canonical.vvp" \
  "${RTL[@]}" tb_h67/tb_h67_zkqi_row_miter.sv \
  >"$LOGS/iverilog_build.log" 2>&1
for mode in 0 3; do
  vvp "$BUILD/canonical.vvp" "+VECTORS=$VECTORS" "+ROW_LIMIT=$ROWS" \
    "+STALL_MODE=$mode" +WATCHDOG_CYCLES=100000000 \
    >"$LOGS/iverilog_mode${mode}.log" 2>&1
  grep -q "^PASS tb_h67_zkqi_row_miter rows=$ROWS stall_mode=$mode bundle_skip=1 " \
    "$LOGS/iverilog_mode${mode}.log"
done

rm -rf "$BUILD/verilator"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  -GCANDIDATE_BUNDLE_SKIP_ENABLE=1 \
  --top-module tb_h67_zkqi_row_miter --Mdir "$BUILD/verilator" \
  "${RTL[@]}" verif_h67/h67_zkqi_assertions.sv \
  tb_h67/tb_h67_zkqi_row_miter.sv >"$LOGS/verilator_build.log" 2>&1
for mode in 0 3; do
  "$BUILD/verilator/Vtb_h67_zkqi_row_miter" "+VECTORS=$VECTORS" \
    "+ROW_LIMIT=$ROWS" "+STALL_MODE=$mode" +WATCHDOG_CYCLES=100000000 \
    >"$LOGS/verilator_mode${mode}.log" 2>&1
  grep -q "^PASS tb_h67_zkqi_row_miter rows=$ROWS stall_mode=$mode bundle_skip=1 " \
    "$LOGS/verilator_mode${mode}.log"
done

python3 scripts/report_h67_zkqi_canonical_replay.py \
  --manifest "$MANIFEST" \
  --iverilog-mode0 "$LOGS/iverilog_mode0.log" \
  --verilator-mode0 "$LOGS/verilator_mode0.log" \
  --iverilog-mode3 "$LOGS/iverilog_mode3.log" \
  --verilator-mode3 "$LOGS/verilator_mode3.log" \
  --output-dir "$OUT"

echo "PASS Motion ZKQI canonical multi-sample replay"
