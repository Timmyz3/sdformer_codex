#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_rqtb_physical_flow_t450_20260809}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805}"
VECTORS="$VECTOR_DIR/h67_checkpoint_rows.txt"
MANIFEST="$VECTOR_DIR/manifest.json"
BUILD="$OUT/build"
LOGS="$OUT/logs"
TOP="tb_h67_temporal_slot_flow_real_trace"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_gate_quant_q17.sv
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_temporal_slot_encoder.sv
  rtl_h67/h67_temporal_slot_fifo.sv
  rtl_h67/h67_sync_dual_bank_k_store.sv
  rtl_h67/h67_temporal_weighted_scs_directory.sv
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_top.sv
)
TB=tb_h67/tb_h67_temporal_slot_flow_real_trace.sv
ASSERT=verif_h67/h67_temporal_slot_flow_assertions.sv

iverilog -g2012 -Wall -s "$TOP" -o "$BUILD/rqtb_icarus.vvp" \
  "${RTL[@]}" "$TB" >"$LOGS/iverilog_build.log" 2>&1
vvp "$BUILD/rqtb_icarus.vvp" \
  "+VECTORS=$VECTORS" +ROW_LIMIT=1 "+DUMP=$OUT/rqtb_row0.vcd" \
  | tee "$LOGS/icarus_row1_vcd.log"
python3 scripts/analyze_rqtb_vcd_activity.py \
  --vcd "$OUT/rqtb_row0.vcd" --output "$OUT/vcd_activity.json"

rm -rf "$BUILD/verilator" "$BUILD/verilator_assert"
verilator --binary --timing -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ \
  --top-module "$TOP" --Mdir "$BUILD/verilator" \
  "${RTL[@]}" "$TB" >"$LOGS/verilator_build.log" 2>&1
"$BUILD/verilator/V$TOP" "+VECTORS=$VECTORS" \
  | tee "$LOGS/verilator_full138.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ \
  --top-module "$TOP" --Mdir "$BUILD/verilator_assert" \
  "${RTL[@]}" "$ASSERT" "$TB" >"$LOGS/verilator_assert_build.log" 2>&1
"$BUILD/verilator_assert/V$TOP" "+VECTORS=$VECTORS" \
  | tee "$LOGS/verilator_assert_full138.log"

verilator --lint-only -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  --top-module h67_temporal_slot_shiftmax_sync_k_top \
  "${RTL[@]}" >"$LOGS/verilator_lint.log" 2>&1

for mode in fixed rqtb; do
  quotient=0
  if [[ "$mode" == "rqtb" ]]; then quotient=1; fi
  yosys -q -l "$LOGS/yosys_${mode}.log" -p "
    read_verilog -sv ${RTL[*]};
    chparam -set QUOTIENT_ENABLE $quotient h67_temporal_slot_shiftmax_sync_k_top;
    hierarchy -top h67_temporal_slot_shiftmax_sync_k_top;
    proc; opt; memory_collect; check -assert;
    tee -o $OUT/yosys_${mode}_stat.json stat -json
  "
  yosys -l "$LOGS/nangate45_${mode}.log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${RTL[*]};
    chparam -set QUOTIENT_ENABLE $quotient h67_temporal_slot_shiftmax_sync_k_top;
    hierarchy -check -top h67_temporal_slot_shiftmax_sync_k_top;
    proc; flatten; opt; memory -nomap; opt; techmap; opt;
    dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert;
    stat -liberty $LIB
  " >/dev/null
done

python3 scripts/summarize_h67_rqtb_physical_flow.py \
  --verilator-log "$LOGS/verilator_full138.log" \
  --assert-log "$LOGS/verilator_assert_full138.log" \
  --icarus-log "$LOGS/icarus_row1_vcd.log" \
  --vector-manifest "$MANIFEST" \
  --activity "$OUT/vcd_activity.json" \
  --fixed-yosys "$OUT/yosys_fixed_stat.json" \
  --rqtb-yosys "$OUT/yosys_rqtb_stat.json" \
  --fixed-mapping-log "$LOGS/nangate45_fixed.log" \
  --rqtb-mapping-log "$LOGS/nangate45_rqtb.log" \
  --rtl-source rtl_h67/h67_motionxor_score_q7.sv \
  --rtl-source rtl_h67/h67_temporal_slot_encoder.sv \
  --rtl-source rtl_h67/h67_temporal_slot_fifo.sv \
  --rtl-source rtl_h67/h67_sync_dual_bank_k_store.sv \
  --rtl-source rtl_h67/h67_temporal_weighted_scs_directory.sv \
  --rtl-source rtl_h67/h67_temporal_slot_shiftmax_sync_k_top.sv \
  --rtl-source rtl_ttx/ttx_ceil_log2_u32.sv \
  --rtl-source rtl_ttx/ttx_exp2_lut_q8.sv \
  --rtl-source rtl_ttx/ttx_gate_quant_q17.sv \
  --verification-source "$ASSERT" \
  --verification-source "$TB" \
  --verification-source scripts/analyze_rqtb_vcd_activity.py \
  --verification-source sim_h67/run_h67_rqtb_physical_flow_checks.sh \
  --output-dir "$OUT"

python3 -m unittest \
  tests.test_analyze_rqtb_vcd_activity \
  tests.test_summarize_h67_rqtb_physical_flow \
  tests.test_summarize_rqtb_openroad_proxy

git diff --check -- \
  rtl_h67/h67_temporal_slot_encoder.sv \
  rtl_h67/h67_temporal_slot_fifo.sv \
  rtl_h67/h67_sync_dual_bank_k_store.sv \
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_top.sv \
  verif_h67/h67_temporal_slot_flow_assertions.sv \
  "$TB" \
  sim_h67/run_h67_rqtb_physical_flow_checks.sh \
  scripts/analyze_rqtb_vcd_activity.py \
  scripts/summarize_h67_rqtb_physical_flow.py

echo "PASS H67 RQTB T450 physical-flow RTL checks"
