#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_h67/vectors/h67_multisample_t450}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/h67_rqtb_multisample_real_rtl}"
VECTORS="$VECTOR_DIR/h67_multisample_checkpoint_rows.txt"
VECTOR_MANIFEST="$VECTOR_DIR/manifest.json"
ROW_INDEX="$VECTOR_DIR/row_index.jsonl"
BUILD="$RESULT_DIR/build"
LOGS="$RESULT_DIR/logs"
TOP="tb_h67_temporal_slot_flow_real_trace_2s"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
FROZEN_SOURCE_DIR="${FROZEN_SOURCE_DIR:-}"

for path in "$VECTORS" "$VECTOR_MANIFEST" "$ROW_INDEX"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing pre-generated vector artifact: $path" >&2
    exit 2
  fi
done

python3 - "$VECTOR_MANIFEST" "$MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
samples = manifest.get("sample_count")
maximum = int(sys.argv[2])
if not isinstance(samples, int) or samples < 2:
    raise SystemExit("vector manifest must contain at least two complete samples")
if maximum < 2 or samples > maximum:
    raise SystemExit(
        f"sample_count={samples} exceeds watchdog-safe runner limit={maximum}"
    )
if manifest.get("row_count") != samples * 138:
    raise SystemExit("vector manifest row count is not 138 rows per sample")
PY

if [[ -d "$RESULT_DIR" ]] && find "$RESULT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty result directory: $RESULT_DIR" >&2
  exit 2
fi

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

SOURCE_ARGS=()
if [[ -n "$FROZEN_SOURCE_DIR" ]]; then
  if [[ ! -d "$FROZEN_SOURCE_DIR" ]]; then
    echo "Frozen source directory does not exist: $FROZEN_SOURCE_DIR" >&2
    exit 2
  fi
  FROZEN_SOURCE_DIR="$(cd "$FROZEN_SOURCE_DIR" && pwd)"
  RTL=(
    "$FROZEN_SOURCE_DIR/ttx_ceil_log2_u32.sv"
    "$FROZEN_SOURCE_DIR/ttx_exp2_lut_q8.sv"
    "$FROZEN_SOURCE_DIR/ttx_gate_quant_q17.sv"
    "$FROZEN_SOURCE_DIR/h67_motionxor_score_q7.sv"
    "$FROZEN_SOURCE_DIR/h67_temporal_slot_encoder.sv"
    "$FROZEN_SOURCE_DIR/h67_sync_dual_bank_k_store.sv"
    "$FROZEN_SOURCE_DIR/h67_temporal_slot_fifo_2s.sv"
    "$FROZEN_SOURCE_DIR/h67_temporal_weighted_scs_directory_2s.sv"
    "$FROZEN_SOURCE_DIR/h67_temporal_slot_shiftmax_sync_k_2s_top.sv"
  )
  TB="$FROZEN_SOURCE_DIR/tb_h67_temporal_slot_flow_real_trace_2s.sv"
  SOURCE_ARGS=(--rtl-source-dir "$FROZEN_SOURCE_DIR")
else
  RTL=(
    rtl_ttx/ttx_ceil_log2_u32.sv
    rtl_ttx/ttx_exp2_lut_q8.sv
    rtl_ttx/ttx_gate_quant_q17.sv
    rtl_h67/h67_motionxor_score_q7.sv
    rtl_h67/h67_temporal_slot_encoder.sv
    rtl_h67/h67_sync_dual_bank_k_store.sv
    rtl_h67/h67_temporal_slot_fifo_2s.sv
    rtl_h67/h67_temporal_weighted_scs_directory_2s.sv
    rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv
  )
  TB=tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv
fi
ASSERT=verif_h67/h67_temporal_slot_flow_2s_assertions.sv

for path in "${RTL[@]}" "$TB" "$ASSERT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing RTL replay source: $path" >&2
    exit 2
  fi
done

iverilog -g2012 -Wall -s "$TOP" \
  -o "$BUILD/icarus.vvp" "${RTL[@]}" "$TB" \
  >"$LOGS/icarus_build.log" 2>&1
vvp "$BUILD/icarus.vvp" "+VECTORS=$VECTORS" \
  | tee "$LOGS/icarus_full.log"

if [[ -e "$BUILD/verilator_sva" ]]; then
  echo "Refusing to overwrite existing Verilator build: $BUILD/verilator_sva" >&2
  exit 2
fi
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module "$TOP" --Mdir "$BUILD/verilator_sva" \
  "${RTL[@]}" "$ASSERT" "$TB" \
  >"$LOGS/verilator_sva_build.log" 2>&1
"$BUILD/verilator_sva/V$TOP" "+VECTORS=$VECTORS" \
  | tee "$LOGS/verilator_sva_full.log"

python3 scripts/summarize_h67_rqtb_multisample_real_rtl.py \
  --icarus-log "$LOGS/icarus_full.log" \
  --verilator-log "$LOGS/verilator_sva_full.log" \
  --row-index "$ROW_INDEX" \
  --vector-manifest "$VECTOR_MANIFEST" \
  "${SOURCE_ARGS[@]}" \
  --output "$RESULT_DIR/summary.json"

echo "PASS H67 RQTB multisample real RTL pipeline"
