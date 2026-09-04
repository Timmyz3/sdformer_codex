#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_empty_row_skip_2s_20260813}"
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
  rtl_h67/h67_empty_row_skip_2s.sv
)

iverilog -g2012 -Wall -s tb_h67_empty_row_skip_2s \
  -o "$BUILD/empty_skip.vvp" "${RTL[@]}" tb_h67/tb_h67_empty_row_skip_2s.sv \
  >"$OUT/iverilog_build.log" 2>&1
vvp "$BUILD/empty_skip.vvp" "+VECTORS=$VECTORS" | tee "$OUT/empty_skip_iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY \
  --top-module tb_h67_empty_row_skip_2s \
  --Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" tb_h67/tb_h67_empty_row_skip_2s.sv \
  >"$OUT/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_h67_empty_row_skip_2s" \
  "+VECTORS=$VECTORS" | tee "$OUT/empty_skip_verilator.log"

yosys -q -l "$OUT/yosys.log" -p "
  read_verilog -sv ${RTL[*]};
  hierarchy -check -top h67_empty_row_skip_2s;
  proc; opt; memory_collect; check -assert;
  tee -o $OUT/yosys_stat.json stat -json
"

python3 scripts/report_h67_empty_row_skip.py \
  --log "$OUT/empty_skip_verilator.log" \
  --phase-report "$PHASE" \
  --output-dir "$OUT"

echo "PASS H67 empty-row skip flow"
