#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_rqtb_fifo_depth_dse_t450_20260809}"
VECTORS="${VECTORS:-$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt}"
BUILD="$OUT/build"
LOGS="$OUT/logs"

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

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

for depth in 2 4 8 16 32; do
  iverilog -g2012 -Wall \
    -P "tb_h67_temporal_slot_flow_real_trace_2s.SLOT_FIFO_DEPTH=$depth" \
    -s tb_h67_temporal_slot_flow_real_trace_2s \
    -o "$BUILD/depth_${depth}.vvp" "${RTL[@]}" \
    tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv \
    >"$LOGS/depth_${depth}_build.log" 2>&1
  vvp "$BUILD/depth_${depth}.vvp" "+VECTORS=$VECTORS" \
    | tee "$LOGS/depth_${depth}.log"
done

PYTHONPATH=scripts python3 scripts/summarize_h67_rqtb_fifo_depth_dse.py \
  --log-dir "$LOGS" --output-dir "$OUT"

git diff --check -- \
  scripts/summarize_h67_rqtb_fifo_depth_dse.py \
  sim_h67/run_h67_rqtb_fifo_depth_dse.sh

echo "PASS H67 RQTB FIFO depth DSE"
