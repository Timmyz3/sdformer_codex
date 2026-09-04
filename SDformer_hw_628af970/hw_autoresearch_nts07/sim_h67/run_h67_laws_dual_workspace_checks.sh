#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_laws_dual_workspace_rtl_20260813}"
VECTORS="${VECTORS:-$ROOT/tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt}"
PHASE="${PHASE_REPORT:-$ROOT/results/h67_laws_shared_backend_phase_20260813/report.json}"
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
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv
)

iverilog -g2012 -Wall -s tb_h67_laws_dual_workspace_2s \
  -o "$BUILD/dw.vvp" "${RTL[@]}" tb_h67/tb_h67_laws_dual_workspace_2s.sv \
  >"$OUT/iverilog_build.log" 2>&1
vvp "$BUILD/dw.vvp" "+VECTORS=$VECTORS" | tee "$OUT/dual_workspace_iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY -Wno-UNOPTFLAT \
  --top-module tb_h67_laws_dual_workspace_2s \
  --Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" tb_h67/tb_h67_laws_dual_workspace_2s.sv \
  >"$OUT/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_h67_laws_dual_workspace_2s" \
  "+VECTORS=$VECTORS" | tee "$OUT/dual_workspace_verilator.log"

python3 scripts/report_h67_laws_dual_workspace.py \
  --log "$OUT/dual_workspace_verilator.log" \
  --phase-report "$PHASE" \
  --output-dir "$OUT"

echo "PASS H67 dual-workspace RTL flow"
