#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_zkqi_threeway_20260809}"
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
  rtl_h67/h67_pair_bitmap_metadata_builder.sv
  rtl_h67/h67_active_bundle_fifo.sv
  rtl_h67/h67_banked_active_descriptor_store.sv
  rtl_h67/h67_temporal_weighted_scs_directory_seed_2s.sv
  rtl_h67/h67_zkqi_row_shiftmax_top.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_gate_quant_q17.sv
)

for family in pairbitmap ttb8; do
  bundle=0
  [[ "$family" == "ttb8" ]] && bundle=1
  iverilog -g2012 -Wall \
    -P tb_h67_zkqi_row_miter.CANDIDATE_BUNDLE_SKIP_ENABLE="$bundle" \
    -s tb_h67_zkqi_row_miter -o "$BUILD/${family}.vvp" \
    "${RTL[@]}" tb_h67/tb_h67_zkqi_row_miter.sv \
    >"$LOGS/iverilog_${family}_build.log" 2>&1
  for mode in 0 1 2 3; do
    vvp "$BUILD/${family}.vvp" "+VECTORS=$VECTORS" +ROW_LIMIT=138 \
      "+STALL_MODE=$mode" >"$LOGS/iverilog_${family}_mode${mode}.log" 2>&1
    grep -q "^PASS tb_h67_zkqi_row_miter rows=138 stall_mode=$mode bundle_skip=$bundle " \
      "$LOGS/iverilog_${family}_mode${mode}.log"
  done

  rm -rf "$BUILD/verilator_${family}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
    -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
    -GCANDIDATE_BUNDLE_SKIP_ENABLE="$bundle" \
    --top-module tb_h67_zkqi_row_miter --Mdir "$BUILD/verilator_${family}" \
    "${RTL[@]}" verif_h67/h67_zkqi_assertions.sv \
    tb_h67/tb_h67_zkqi_row_miter.sv >"$LOGS/verilator_${family}_build.log" 2>&1
  for mode in 0 1 2 3; do
    "$BUILD/verilator_${family}/Vtb_h67_zkqi_row_miter" \
      "+VECTORS=$VECTORS" +ROW_LIMIT=138 "+STALL_MODE=$mode" \
      >"$LOGS/verilator_${family}_mode${mode}.log" 2>&1
    grep -q "^PASS tb_h67_zkqi_row_miter rows=138 stall_mode=$mode bundle_skip=$bundle " \
      "$LOGS/verilator_${family}_mode${mode}.log"
  done
done

for family in baseline pairbitmap ttb8; do
  zk=0
  bundle=0
  if [[ "$family" == "pairbitmap" ]]; then zk=1; fi
  if [[ "$family" == "ttb8" ]]; then zk=1; bundle=1; fi
  verilator --lint-only -Wall -Wno-fatal \
    -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
    -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-DECLFILENAME \
    -GZK_BYPASS_ENABLE="$zk" -GBUNDLE_SKIP_ENABLE="$bundle" \
    --top-module h67_zkqi_row_shiftmax_top "${RTL[@]}" \
    >"$LOGS/verilator_lint_${family}.log" 2>&1

  yosys -q -l "$LOGS/yosys_${family}.log" -p "
    read_verilog -sv ${RTL[*]};
    chparam -set ZK_BYPASS_ENABLE $zk -set BUNDLE_SKIP_ENABLE $bundle h67_zkqi_row_shiftmax_top;
    hierarchy -check -top h67_zkqi_row_shiftmax_top;
    proc; flatten; opt; memory_collect; check -assert;
    tee -o $OUT/yosys_${family}_stat.json stat -json
  "
  yosys -l "$LOGS/nangate45_${family}.log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${RTL[*]};
    chparam -set ZK_BYPASS_ENABLE $zk -set BUNDLE_SKIP_ENABLE $bundle h67_zkqi_row_shiftmax_top;
    hierarchy -check -top h67_zkqi_row_shiftmax_top;
    proc; flatten; opt; memory -nomap; opt; techmap; opt;
    dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert;
    stat -liberty $LIB
  " >/dev/null
done

python3 -m unittest \
  tests.test_report_h67_zkqi_row_miter \
  tests.test_report_h67_zkqi_threeway \
  >"$LOGS/python_unit_tests.log" 2>&1

python3 scripts/report_h67_zkqi_threeway.py \
  --iverilog-pair \
    "$LOGS/iverilog_pairbitmap_mode0.log" "$LOGS/iverilog_pairbitmap_mode1.log" \
    "$LOGS/iverilog_pairbitmap_mode2.log" "$LOGS/iverilog_pairbitmap_mode3.log" \
  --verilator-pair \
    "$LOGS/verilator_pairbitmap_mode0.log" "$LOGS/verilator_pairbitmap_mode1.log" \
    "$LOGS/verilator_pairbitmap_mode2.log" "$LOGS/verilator_pairbitmap_mode3.log" \
  --iverilog-ttb8 \
    "$LOGS/iverilog_ttb8_mode0.log" "$LOGS/iverilog_ttb8_mode1.log" \
    "$LOGS/iverilog_ttb8_mode2.log" "$LOGS/iverilog_ttb8_mode3.log" \
  --verilator-ttb8 \
    "$LOGS/verilator_ttb8_mode0.log" "$LOGS/verilator_ttb8_mode1.log" \
    "$LOGS/verilator_ttb8_mode2.log" "$LOGS/verilator_ttb8_mode3.log" \
  --map-baseline "$LOGS/nangate45_baseline.log" \
  --map-pair "$LOGS/nangate45_pairbitmap.log" \
  --map-ttb8 "$LOGS/nangate45_ttb8.log" \
  --yosys-baseline "$OUT/yosys_baseline_stat.json" \
  --yosys-pair "$OUT/yosys_pairbitmap_stat.json" \
  --yosys-ttb8 "$OUT/yosys_ttb8_stat.json" \
  --vector "$VECTORS" \
  --sources "${RTL[@]}" verif_h67/h67_zkqi_assertions.sv \
    tb_h67/tb_h67_zkqi_row_miter.sv scripts/report_h67_zkqi_row_miter.py \
    scripts/report_h67_zkqi_threeway.py sim_h67/run_h67_zkqi_threeway_checks.sh \
  --output-dir "$OUT"

cat >"$OUT/status.tsv" <<'EOF'
check	status
iverilog_pairbitmap_modes0_3	PASS
iverilog_ttb8_modes0_3	PASS
verilator_sva_pairbitmap_modes0_3	PASS
verilator_sva_ttb8_modes0_3	PASS
verilator_lint_three_modes	PASS
yosys_nangate45_three_modes	PASS
threeway_report_contract	PASS
EOF

git diff --check -- \
  rtl_h67/h67_pair_bitmap_metadata_builder.sv \
  rtl_h67/h67_zkqi_row_shiftmax_top.sv \
  verif_h67/h67_zkqi_assertions.sv \
  tb_h67/tb_h67_zkqi_row_miter.sv \
  scripts/report_h67_zkqi_threeway.py \
  tests/test_report_h67_zkqi_threeway.py \
  sim_h67/run_h67_zkqi_threeway_checks.sh

echo "PASS Motion ZKQI three-way strong-baseline checks"
