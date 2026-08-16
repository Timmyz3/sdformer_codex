#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_rqtb_strong_baseline_2s_t450_20260809}"
VECTORS="${VECTORS:-$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt}"
VECTORS_MANIFEST="${VECTORS_MANIFEST:-$(dirname "$VECTORS")/manifest.json}"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
BUILD="$OUT/build"
LOGS="$OUT/logs"

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

COMMON=(
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_gate_quant_q17.sv
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_temporal_slot_encoder.sv
  rtl_h67/h67_sync_dual_bank_k_store.sv
)
RTL_1S=(
  "${COMMON[@]}"
  rtl_h67/h67_temporal_slot_fifo.sv
  rtl_h67/h67_temporal_weighted_scs_directory.sv
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_top.sv
)
RTL_2S=(
  "${COMMON[@]}"
  rtl_h67/h67_temporal_slot_fifo_2s.sv
  rtl_h67/h67_temporal_weighted_scs_directory_2s.sv
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv
)

iverilog -g2012 -Wall -s tb_h67_temporal_slot_flow_real_trace_1s_random \
  -o "$BUILD/one_slot.vvp" "${RTL_1S[@]}" \
  tb_h67/tb_h67_temporal_slot_flow_real_trace_1s_random.sv \
  >"$LOGS/iverilog_1s_build.log" 2>&1
vvp "$BUILD/one_slot.vvp" "+VECTORS=$VECTORS" | tee "$LOGS/one_slot_full138.log"

iverilog -g2012 -Wall -s tb_h67_temporal_slot_flow_real_trace_2s \
  -o "$BUILD/two_slot.vvp" "${RTL_2S[@]}" \
  tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv \
  >"$LOGS/iverilog_2s_build.log" 2>&1
vvp "$BUILD/two_slot.vvp" "+VECTORS=$VECTORS" | tee "$LOGS/two_slot_full138.log"

iverilog -g2012 -Wall -s tb_h67_temporal_slot_restart_reject_2s \
  -o "$BUILD/restart_reject.vvp" "${RTL_2S[@]}" \
  tb_h67/tb_h67_temporal_slot_restart_reject_2s.sv \
  >"$LOGS/iverilog_restart_reject_build.log" 2>&1
vvp "$BUILD/restart_reject.vvp" | tee "$LOGS/restart_reject.log"

iverilog -g2012 -Wall -s tb_h67_temporal_slot_build_restart_reject_2s \
  -o "$BUILD/build_restart_reject.vvp" "${RTL_2S[@]}" \
  tb_h67/tb_h67_temporal_slot_build_restart_reject_2s.sv \
  >"$LOGS/iverilog_build_restart_reject_build.log" 2>&1
vvp "$BUILD/build_restart_reject.vvp" | tee "$LOGS/build_restart_reject.log"

rm -rf "$BUILD/verilator_2s_assert"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module tb_h67_temporal_slot_flow_real_trace_2s \
  --Mdir "$BUILD/verilator_2s_assert" \
  "${RTL_2S[@]}" verif_h67/h67_temporal_slot_flow_2s_assertions.sv \
  tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv \
  >"$LOGS/verilator_2s_assert_build.log" 2>&1
"$BUILD/verilator_2s_assert/Vtb_h67_temporal_slot_flow_real_trace_2s" \
  "+VECTORS=$VECTORS" | tee "$LOGS/two_slot_sva_full138.log"

rm -rf "$BUILD/verilator_restart_reject"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module tb_h67_temporal_slot_restart_reject_2s \
  --Mdir "$BUILD/verilator_restart_reject" \
  "${RTL_2S[@]}" verif_h67/h67_temporal_slot_flow_2s_assertions.sv \
  tb_h67/tb_h67_temporal_slot_restart_reject_2s.sv \
  >"$LOGS/verilator_restart_reject_build.log" 2>&1
"$BUILD/verilator_restart_reject/Vtb_h67_temporal_slot_restart_reject_2s" \
  | tee "$LOGS/restart_reject_sva.log"

rm -rf "$BUILD/verilator_build_restart_reject"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module tb_h67_temporal_slot_build_restart_reject_2s \
  --Mdir "$BUILD/verilator_build_restart_reject" \
  "${RTL_2S[@]}" verif_h67/h67_temporal_slot_flow_2s_assertions.sv \
  tb_h67/tb_h67_temporal_slot_build_restart_reject_2s.sv \
  >"$LOGS/verilator_build_restart_reject_build.log" 2>&1
"$BUILD/verilator_build_restart_reject/Vtb_h67_temporal_slot_build_restart_reject_2s" \
  | tee "$LOGS/build_restart_reject_sva.log"

verilator --lint-only -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-DECLFILENAME \
  --top-module h67_temporal_slot_shiftmax_sync_k_2s_top \
  "${RTL_2S[@]}" >"$LOGS/verilator_2s_lint.log" 2>&1

for width in 1s 2s; do
  top=h67_temporal_slot_shiftmax_sync_k_top
  sources=("${RTL_1S[@]}")
  if [[ "$width" == "2s" ]]; then
    top=h67_temporal_slot_shiftmax_sync_k_2s_top
    sources=("${RTL_2S[@]}")
  fi
  for mode in fixed rqtb; do
    quotient=0
    if [[ "$mode" == "rqtb" ]]; then quotient=1; fi
    yosys -q -l "$LOGS/yosys_${mode}_${width}.log" -p "
      read_verilog -sv ${sources[*]};
      chparam -set QUOTIENT_ENABLE $quotient $top;
      hierarchy -check -top $top;
      proc; opt; memory_collect; check -assert;
      tee -o $OUT/yosys_${mode}_${width}_stat.json stat -json
    "
    yosys -l "$LOGS/nangate45_${mode}_${width}.log" -p "
      read_liberty -lib $LIB;
      read_verilog -sv ${sources[*]};
      chparam -set QUOTIENT_ENABLE $quotient $top;
      hierarchy -check -top $top;
      proc; flatten; opt; memory -nomap; opt; techmap; opt;
      dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert;
      stat -liberty $LIB
    " >/dev/null
  done
done

python3 -m unittest \
  tests.test_summarize_h67_rqtb_strong_baseline \
  tests.test_summarize_h67_rqtb_physical_flow \
  tests.test_summarize_rqtb_openroad_proxy \
  tests.test_summarize_h67_rqtb_fifo_depth_dse \
  2>&1 | tee "$LOGS/python_unit_tests.log"

PYTHONPATH=scripts python3 scripts/summarize_h67_rqtb_strong_baseline.py \
  --log-1s "$LOGS/one_slot_full138.log" \
  --log-2s "$LOGS/two_slot_full138.log" \
  --log-2s-sva "$LOGS/two_slot_sva_full138.log" \
  --restart-log "$LOGS/restart_reject.log" \
  --restart-sva-log "$LOGS/restart_reject_sva.log" \
  --build-restart-log "$LOGS/build_restart_reject.log" \
  --build-restart-sva-log "$LOGS/build_restart_reject_sva.log" \
  --map-fixed-1s "$LOGS/nangate45_fixed_1s.log" \
  --map-rqtb-1s "$LOGS/nangate45_rqtb_1s.log" \
  --map-fixed-2s "$LOGS/nangate45_fixed_2s.log" \
  --map-rqtb-2s "$LOGS/nangate45_rqtb_2s.log" \
  --vectors "$VECTORS" \
  --vector-manifest "$VECTORS_MANIFEST" \
  --output-dir "$OUT"

git diff --check -- \
  rtl_h67/h67_temporal_slot_fifo_2s.sv \
  rtl_h67/h67_temporal_weighted_scs_directory_2s.sv \
  rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv \
  verif_h67/h67_temporal_slot_flow_2s_assertions.sv \
  tb_h67/tb_h67_temporal_slot_flow_real_trace_1s_random.sv \
  tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv \
  tb_h67/tb_h67_temporal_slot_restart_reject_2s.sv \
  tb_h67/tb_h67_temporal_slot_build_restart_reject_2s.sv \
  scripts/summarize_h67_rqtb_strong_baseline.py \
  sim_h67/run_h67_rqtb_strong_baseline_checks.sh

echo "PASS H67 RQTB strong-baseline checks"
