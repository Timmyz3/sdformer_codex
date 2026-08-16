#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_tare_zkqi_row_rtl_20260810}"
VECTORS="${VECTORS:-$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt}"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
BUILD="$OUT/build"
LOGS="$OUT/logs"

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_tare_score_pair.sv
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

for width in 8 16; do
  iverilog -g2012 -Wall -s tb_h67_tare_score_pair \
    -Ptb_h67_tare_score_pair.RESIDUAL_W="$width" \
    -o "$BUILD/leaf_w${width}.vvp" \
    rtl_h67/h67_motionxor_score_q7.sv rtl_h67/h67_tare_score_pair.sv \
    tb_h67/tb_h67_tare_score_pair.sv >"$LOGS/iverilog_leaf_w${width}_build.log" 2>&1
  vvp "$BUILD/leaf_w${width}.vvp" >"$LOGS/iverilog_leaf_w${width}.log" 2>&1
  grep -q "^PASS tb_h67_tare_score_pair W=$width " "$LOGS/iverilog_leaf_w${width}.log"

  rm -rf "$BUILD/verilator_leaf_w${width}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
    -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
    -GRESIDUAL_W="$width" --top-module tb_h67_tare_score_pair \
    --Mdir "$BUILD/verilator_leaf_w${width}" \
    rtl_h67/h67_motionxor_score_q7.sv rtl_h67/h67_tare_score_pair.sv \
    verif_h67/h67_tare_score_pair_assertions.sv \
    tb_h67/tb_h67_tare_score_pair.sv >"$LOGS/verilator_leaf_w${width}_build.log" 2>&1
  "$BUILD/verilator_leaf_w${width}/Vtb_h67_tare_score_pair" \
    >"$LOGS/verilator_leaf_w${width}.log" 2>&1
  grep -q "^PASS tb_h67_tare_score_pair W=$width " "$LOGS/verilator_leaf_w${width}.log"
done

for width in 8 16; do
  iverilog -g2012 -Wall -s tb_h67_zkqi_row_miter \
    -Ptb_h67_zkqi_row_miter.BASELINE_ZK_BYPASS_ENABLE=1 \
    -Ptb_h67_zkqi_row_miter.BASELINE_ACTIVE_SCORE_RESIDUAL_W=0 \
    -Ptb_h67_zkqi_row_miter.CANDIDATE_ACTIVE_SCORE_RESIDUAL_W="$width" \
    -o "$BUILD/row_w${width}.vvp" "${RTL[@]}" \
    tb_h67/tb_h67_zkqi_row_miter.sv >"$LOGS/iverilog_w${width}_build.log" 2>&1
  for mode in 0 1 2 3; do
    vvp "$BUILD/row_w${width}.vvp" "+VECTORS=$VECTORS" +ROW_LIMIT=138 \
      "+STALL_MODE=$mode" >"$LOGS/iverilog_w${width}_mode${mode}.log" 2>&1
    grep -q "^PASS tb_h67_zkqi_row_miter rows=138 stall_mode=$mode " \
      "$LOGS/iverilog_w${width}_mode${mode}.log"
  done
done

rm -rf "$BUILD/verilator_row_w16"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  -Wno-PINCONNECTEMPTY \
  -GBASELINE_ZK_BYPASS_ENABLE=1 \
  -GBASELINE_ACTIVE_SCORE_RESIDUAL_W=0 \
  -GCANDIDATE_ACTIVE_SCORE_RESIDUAL_W=16 \
  --top-module tb_h67_zkqi_row_miter --Mdir "$BUILD/verilator_row_w16" \
  "${RTL[@]}" verif_h67/h67_zkqi_assertions.sv \
  verif_h67/h67_tare_score_pair_assertions.sv \
  tb_h67/tb_h67_zkqi_row_miter.sv >"$LOGS/verilator_w16_build.log" 2>&1
for mode in 0 1 2 3; do
  "$BUILD/verilator_row_w16/Vtb_h67_zkqi_row_miter" \
    "+VECTORS=$VECTORS" +ROW_LIMIT=138 "+STALL_MODE=$mode" \
    >"$LOGS/verilator_w16_mode${mode}.log" 2>&1
  grep -q "^PASS tb_h67_zkqi_row_miter rows=138 stall_mode=$mode " \
    "$LOGS/verilator_w16_mode${mode}.log"
done

for width in 0 8 16; do
  verilator --lint-only -Wall -Wno-fatal -Wno-TIMESCALEMOD \
    -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-WIDTHTRUNC \
    -Wno-WIDTHEXPAND -Wno-DECLFILENAME -Wno-PINCONNECTEMPTY \
    -GZK_BYPASS_ENABLE=1 -GACTIVE_SCORE_RESIDUAL_W="$width" \
    --top-module h67_zkqi_row_shiftmax_top "${RTL[@]}" \
    >"$LOGS/verilator_lint_w${width}.log" 2>&1
  yosys -q -l "$LOGS/yosys_check_w${width}.log" -p "
    read_verilog -sv ${RTL[*]};
    chparam -set ZK_BYPASS_ENABLE 1 -set ACTIVE_SCORE_RESIDUAL_W $width h67_zkqi_row_shiftmax_top;
    hierarchy -check -top h67_zkqi_row_shiftmax_top;
    proc; flatten; opt; memory_collect; check -assert
  "
done

for item in direct:0 w16:16; do
  name="${item%%:*}"
  width="${item##*:}"
  yosys -l "$LOGS/nangate45_fast_${name}.log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${RTL[*]};
    chparam -set ZK_BYPASS_ENABLE 1 -set BUNDLE_SKIP_ENABLE 1 -set ACTIVE_SCORE_RESIDUAL_W $width h67_zkqi_row_shiftmax_top;
    hierarchy -check -top h67_zkqi_row_shiftmax_top;
    proc; flatten; opt; memory -nomap; opt; techmap; opt;
    dfflibmap -liberty $LIB; abc -fast -liberty $LIB;
    clean; check -assert; stat -liberty $LIB
  " >/dev/null
done

python3 -m unittest tests.test_report_h67_tare_zkqi_row_rtl \
  >"$LOGS/python_unit_tests.log" 2>&1
python3 scripts/report_h67_tare_zkqi_row_rtl.py --output-dir "$OUT"

cat >"$OUT/status.tsv" <<'EOF'
check	status
leaf_w8_iverilog	PASS
leaf_w16_iverilog	PASS
leaf_w8_verilator_sva	PASS
leaf_w16_verilator_sva	PASS
row_w8_iverilog_four_modes	PASS_REJECT_CYCLE
row_w16_iverilog_four_modes	PASS
row_w16_verilator_sva_four_modes	PASS
lint_yosys_w0_w8_w16	PASS
nangate45_fast_direct_w16	PASS_PROXY
final_decision	REJECT_TARE
EOF

git diff --check -- \
  rtl_h67/h67_tare_score_pair.sv \
  rtl_h67/h67_zkqi_row_shiftmax_top.sv \
  verif_h67/h67_tare_score_pair_assertions.sv \
  tb_h67/tb_h67_tare_score_pair.sv \
  tb_h67/tb_h67_zkqi_row_miter.sv \
  scripts/report_h67_tare_zkqi_row_rtl.py \
  tests/test_report_h67_tare_zkqi_row_rtl.py \
  sim_h67/run_h67_tare_zkqi_row_checks.sh

echo "PASS Motion TARE/ZKQI row RTL screening: REJECT_TARE"
