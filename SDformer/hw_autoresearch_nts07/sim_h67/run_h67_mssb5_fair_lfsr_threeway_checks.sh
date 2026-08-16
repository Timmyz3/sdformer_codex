#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_mssb5_fair_ep35_rtl_20260814}"
VECTORS="${VECTORS:-$ROOT/tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt}"
BUILD="$OUT/build"
mkdir -p "$BUILD" "$OUT"
cd "$ROOT"

RTL=(
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_gate_quant_q17.sv
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_temporal_slot_encoder.sv
  rtl_h67/h67_mssb5_score_pair.sv
  rtl_h67/h67_mssb5_temporal_slot_encoder.sv
  rtl_h67/h67_sync_dual_bank_k_store.sv
  rtl_h67/h67_temporal_slot_fifo_2s.sv
  rtl_h67/h67_temporal_weighted_scs_directory_2s.sv
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv
  rtl_h67/h67_laws_shared_backend_2s_top.sv
)

iverilog -g2012 -Wall -DMSSB5_SCORE_FRONT \
  -s tb_h67_laws_fair_lfsr_threeway_2s \
  -o "$BUILD/fair.vvp" "${RTL[@]}" \
  tb_h67/tb_h67_laws_fair_lfsr_threeway_2s.sv \
  >"$OUT/iverilog_build.log" 2>&1
vvp "$BUILD/fair.vvp" "+VECTORS=$VECTORS" \
  ${ROW_LIMIT:+"+ROW_LIMIT=$ROW_LIMIT"} \
  | tee "$OUT/fair_lfsr_threeway_iverilog.log"

python3 scripts/report_h67_laws_fair_lfsr_threeway.py --result-dir "$OUT"
echo "PASS H67 MSSB5 LFSR three-way fair package"
