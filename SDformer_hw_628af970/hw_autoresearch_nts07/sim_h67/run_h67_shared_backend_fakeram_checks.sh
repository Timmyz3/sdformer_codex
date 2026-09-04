#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_laws_shared_backend_fakeram_20260813}"
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
  rtl_h67/h67_sync_dual_bank_k_store.sv
  rtl_h67/h67_temporal_slot_fifo_2s.sv
  rtl_h67/h67_temporal_weighted_scs_directory_2s.sv
  rtl_h67/h67_laws_shared_backend_2s_top.sv
)
SINGLE=(
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
FAKERAM=tb_h67/fakeram45_256x32_functional_model.sv

iverilog -g2012 -Wall -s tb_h67_laws_shared_backend_2s \
  -Ptb_h67_laws_shared_backend_2s.MEMORY_IMPL=1 \
  -o "$BUILD/sb_fr.vvp" "${RTL[@]}" "$FAKERAM" tb_h67/tb_h67_laws_shared_backend_2s.sv \
  >"$OUT/iverilog_build.log" 2>&1
vvp "$BUILD/sb_fr.vvp" "+VECTORS=$VECTORS" | tee "$OUT/shared_backend_fakeram_iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY \
  --top-module tb_h67_laws_shared_backend_2s \
  -GMEMORY_IMPL=0 \
  --Mdir "$BUILD/verilator_sva" \
  "${RTL[@]}" tb_h67/tb_h67_laws_shared_backend_2s.sv \
  verif_h67/h67_laws_shared_backend_assertions.sv \
  >"$OUT/verilator_sva_build.log" 2>&1
"$BUILD/verilator_sva/Vtb_h67_laws_shared_backend_2s" \
  "+VECTORS=$VECTORS" | tee "$OUT/shared_backend_sva_verilator.log"

yosys -q -l "$OUT/yosys_shared_bb.log" -p "
  read_verilog -sv rtl_h67/fakeram45_256x32_bb.sv ${RTL[*]};
  blackbox fakeram45_256x32;
  chparam -set MEMORY_IMPL 1 h67_laws_shared_backend_2s_top;
  hierarchy -check -top h67_laws_shared_backend_2s_top;
  proc; opt; memory_collect; check -assert;
  tee -o $OUT/yosys_shared_bb_stat.json stat -json
"
yosys -q -l "$OUT/yosys_single_bb.log" -p "
  read_verilog -sv rtl_h67/fakeram45_256x32_bb.sv ${SINGLE[*]};
  blackbox fakeram45_256x32;
  chparam -set MEMORY_IMPL 1 h67_temporal_slot_shiftmax_sync_k_2s_top;
  hierarchy -check -top h67_temporal_slot_shiftmax_sync_k_2s_top;
  proc; opt; memory_collect; check -assert;
  tee -o $OUT/yosys_single_bb_stat.json stat -json
"

python3 scripts/report_h67_shared_backend_fakeram.py --result-dir "$OUT"
echo "PASS H67 shared-backend fakeram + SVA"
