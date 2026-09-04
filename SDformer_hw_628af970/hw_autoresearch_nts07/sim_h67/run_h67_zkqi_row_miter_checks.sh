#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_zkqi_row_miter_20260809}"
VECTORS="${VECTORS:-$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt}"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
BUILD="$OUT/build"
LOGS="$OUT/logs"

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_sync_qk_row_store.sv
  rtl_h67/h67_fakeram45_qk_row_store.sv
  rtl_h67/h67_ttb8_metadata_builder.sv
  rtl_h67/h67_active_bundle_fifo.sv
  rtl_h67/h67_banked_active_descriptor_store.sv
  rtl_h67/h67_temporal_weighted_scs_directory_seed_2s.sv
  rtl_h67/h67_zkqi_row_shiftmax_top.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_gate_quant_q17.sv
)

iverilog -g2012 -Wall -s tb_h67_zkqi_row_miter \
  -o "$BUILD/miter.vvp" "${RTL[@]}" tb_h67/tb_h67_zkqi_row_miter.sv \
  >"$LOGS/iverilog_build.log" 2>&1
for mode in 0 1 2 3; do
  vvp "$BUILD/miter.vvp" "+VECTORS=$VECTORS" +ROW_LIMIT=138 "+STALL_MODE=$mode" \
    >"$LOGS/iverilog_mode${mode}.log" 2>&1
  grep -q "^PASS tb_h67_zkqi_row_miter rows=138 stall_mode=$mode " \
    "$LOGS/iverilog_mode${mode}.log"
done

rm -rf "$BUILD/verilator_assert"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module tb_h67_zkqi_row_miter --Mdir "$BUILD/verilator_assert" \
  "${RTL[@]}" verif_h67/h67_zkqi_assertions.sv \
  tb_h67/tb_h67_zkqi_row_miter.sv >"$LOGS/verilator_build.log" 2>&1
for mode in 0 1 2 3; do
  "$BUILD/verilator_assert/Vtb_h67_zkqi_row_miter" \
    "+VECTORS=$VECTORS" +ROW_LIMIT=138 "+STALL_MODE=$mode" \
    >"$LOGS/verilator_mode${mode}.log" 2>&1
  grep -q "^PASS tb_h67_zkqi_row_miter rows=138 stall_mode=$mode " \
    "$LOGS/verilator_mode${mode}.log"
done

for mode in baseline zkqi; do
  parameter=0
  [[ "$mode" == "zkqi" ]] && parameter=1
  verilator --lint-only -Wall -Wno-fatal \
    -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
    -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-DECLFILENAME \
    -GZK_BYPASS_ENABLE="$parameter" \
    --top-module h67_zkqi_row_shiftmax_top "${RTL[@]}" \
    >"$LOGS/verilator_lint_${mode}.log" 2>&1

  yosys -q -l "$LOGS/yosys_${mode}.log" -p "
    read_verilog -sv ${RTL[*]};
    chparam -set ZK_BYPASS_ENABLE $parameter h67_zkqi_row_shiftmax_top;
    hierarchy -check -top h67_zkqi_row_shiftmax_top;
    proc; flatten; opt; memory_collect; check -assert;
    tee -o $OUT/yosys_${mode}_stat.json stat -json
  "
  yosys -l "$LOGS/nangate45_${mode}.log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${RTL[*]};
    chparam -set ZK_BYPASS_ENABLE $parameter h67_zkqi_row_shiftmax_top;
    hierarchy -check -top h67_zkqi_row_shiftmax_top;
    proc; flatten; opt; memory -nomap; opt; techmap; opt;
    dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert;
    stat -liberty $LIB
  " >/dev/null
done

python3 -m unittest tests.test_report_h67_zkqi_row_miter \
  >"$LOGS/python_unit_tests.log" 2>&1

python3 scripts/report_h67_zkqi_row_miter.py \
  --iverilog-logs \
    "$LOGS/iverilog_mode0.log" "$LOGS/iverilog_mode1.log" \
    "$LOGS/iverilog_mode2.log" "$LOGS/iverilog_mode3.log" \
  --verilator-logs \
    "$LOGS/verilator_mode0.log" "$LOGS/verilator_mode1.log" \
    "$LOGS/verilator_mode2.log" "$LOGS/verilator_mode3.log" \
  --map-baseline "$LOGS/nangate45_baseline.log" \
  --map-zkqi "$LOGS/nangate45_zkqi.log" \
  --yosys-baseline "$OUT/yosys_baseline_stat.json" \
  --yosys-zkqi "$OUT/yosys_zkqi_stat.json" \
  --vector "$VECTORS" \
  --sources "${RTL[@]}" verif_h67/h67_zkqi_assertions.sv \
    tb_h67/tb_h67_zkqi_row_miter.sv \
    scripts/report_h67_zkqi_row_miter.py \
    sim_h67/run_h67_zkqi_row_miter_checks.sh \
  --output-dir "$OUT"

cat >"$OUT/status.tsv" <<'EOF'
check	status
iverilog_mode0	PASS
iverilog_mode1	PASS
iverilog_mode2	PASS
iverilog_mode3	PASS
verilator_sva_mode0	PASS
verilator_sva_mode1	PASS
verilator_sva_mode2	PASS
verilator_sva_mode3	PASS
verilator_lint_baseline	PASS
verilator_lint_zkqi	PASS
yosys_nangate45_baseline	PASS
yosys_nangate45_zkqi	PASS
report_contract	PASS
EOF

git diff --check -- \
  rtl_h67/h67_sync_qk_row_store.sv \
  rtl_h67/h67_ttb8_metadata_builder.sv \
  rtl_h67/h67_active_bundle_fifo.sv \
  rtl_h67/h67_temporal_weighted_scs_directory_seed_2s.sv \
  rtl_h67/h67_zkqi_row_shiftmax_top.sv \
  verif_h67/h67_zkqi_assertions.sv \
  tb_h67/tb_h67_zkqi_row_miter.sv \
  scripts/report_h67_zkqi_row_miter.py \
  tests/test_report_h67_zkqi_row_miter.py \
  sim_h67/run_h67_zkqi_row_miter_checks.sh

echo "PASS Motion TTB8-ZKQI row-miter checks"
