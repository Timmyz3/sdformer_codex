#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_date_paper_stack_20260813}"
EP35="$ROOT/tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt"
EP30="$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt"
BUILD="$OUT/build"
mkdir -p "$BUILD" "$OUT"
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
  rtl_h67/h67_laws_shared_backend_2s_top.sv
)

iverilog -g2012 -Wall -s tb_h67_laws_shared_backend_2s \
  -o "$BUILD/sb.vvp" "${RTL[@]}" tb_h67/tb_h67_laws_shared_backend_2s.sv \
  >"$OUT/iverilog_build.log" 2>&1

vvp "$BUILD/sb.vvp" "+VECTORS=$EP35" | tee "$OUT/ep35_overlap_acc32.log"
vvp "$BUILD/sb.vvp" "+VECTORS=$EP35" "+SEQUENTIAL=1" | tee "$OUT/ep35_sequential_acc32.log"
vvp "$BUILD/sb.vvp" "+VECTORS=$EP30" | tee "$OUT/ep30_overlap.log"
vvp "$BUILD/sb.vvp" "+VECTORS=$EP30" "+SEQUENTIAL=1" | tee "$OUT/ep30_sequential.log"

python3 scripts/report_h67_date_paper_stack.py --result-dir "$OUT"
echo "PASS H67 DATE paper stack"
