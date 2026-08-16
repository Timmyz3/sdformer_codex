#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_tesc_gated_k_miter_20260805}"
BUILD="$OUT/build"
LOGS="$OUT/logs"
mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_ttx/ttx_ceil_log2_u32.sv
  rtl_ttx/ttx_exp2_lut_q8.sv
  rtl_ttx/ttx_gate_quant_q17.sv
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_temporal_pair_adapter.sv
  rtl_h67/h67_score_class_row_engine.sv
  rtl_h67/h67_temporal_score_quotient.sv
  rtl_h67/h67_temporal_weighted_scs_directory.sv
  rtl_h67/h67_temporal_quotient_scs_frontend.sv
  rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv
)
TB=tb_h67/tb_h67_temporal_quotient_shiftmax_gate_miter.sv
TOP=tb_h67_temporal_quotient_shiftmax_gate_miter

for seed in 1 17 103 4099 65537; do
  name="small_s${seed}"
  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.PAIRS=40" -P"$TOP.SEED=$seed" \
    -o "$BUILD/$name.vvp" "${RTL[@]}" "$TB" \
    >"$LOGS/iverilog_${name}_build.log" 2>&1
  vvp "$BUILD/$name.vvp" | tee "$LOGS/icarus_${name}.log"
done

for preserve in 0 1; do
  name="t450_p${preserve}"
  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.PAIRS=225" -P"$TOP.SEED=1740701678" \
    -P"$TOP.PRESERVE_MEAN=$preserve" \
    -o "$BUILD/$name.vvp" "${RTL[@]}" "$TB" \
    >"$LOGS/iverilog_${name}_build.log" 2>&1
  vvp "$BUILD/$name.vvp" | tee "$LOGS/icarus_${name}.log"
done

rm -rf "$BUILD/verilator_t450"
verilator --binary --timing --assert -Wall \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY -Wno-BLKSEQ \
  --top-module "$TOP" -GPAIRS=225 -GSEED=1740701678 \
  -GPRESERVE_MEAN=0 \
  --Mdir "$BUILD/verilator_t450" \
  "${RTL[@]}" \
  verif_h67/h67_temporal_score_quotient_assertions.sv \
  verif_h67/h67_temporal_quotient_scs_assertions.sv \
  verif_h67/h67_temporal_quotient_shiftmax_gate_assertions.sv \
  "$TB" >"$LOGS/verilator_t450_build.log" 2>&1
"$BUILD/verilator_t450/V$TOP" | tee "$LOGS/verilator_t450.log"

verilator --lint-only -Wall \
  -Wno-TIMESCALEMOD -Wno-PINCONNECTEMPTY \
  --top-module h67_temporal_quotient_shiftmax_gate_top \
  "${RTL[@]:0:3}" \
  rtl_h67/h67_motionxor_score_q7.sv \
  rtl_h67/h67_temporal_score_quotient.sv \
  rtl_h67/h67_temporal_weighted_scs_directory.sv \
  rtl_h67/h67_temporal_quotient_scs_frontend.sv \
  rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv \
  >"$LOGS/focused_lint.log" 2>&1

yosys -q -l "$LOGS/yosys_candidate.log" -p "
  read_verilog -sv \
    rtl_ttx/ttx_ceil_log2_u32.sv \
    rtl_ttx/ttx_exp2_lut_q8.sv \
    rtl_ttx/ttx_gate_quant_q17.sv \
    rtl_h67/h67_motionxor_score_q7.sv \
    rtl_h67/h67_temporal_score_quotient.sv \
    rtl_h67/h67_temporal_weighted_scs_directory.sv \
    rtl_h67/h67_temporal_quotient_scs_frontend.sv \
    rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv;
  chparam -set PAIRS 225 h67_temporal_quotient_shiftmax_gate_top;
  hierarchy -top h67_temporal_quotient_shiftmax_gate_top;
  proc; opt; memory_collect; check -assert;
  tee -o $OUT/yosys_candidate_stat.json stat -json
"

yosys -q -l "$LOGS/yosys_baseline.log" -p "
  read_verilog -sv \
    rtl_ttx/ttx_ceil_log2_u32.sv \
    rtl_ttx/ttx_exp2_lut_q8.sv \
    rtl_ttx/ttx_gate_quant_q17.sv \
    rtl_h67/h67_temporal_pair_adapter.sv \
    rtl_h67/h67_motionxor_score_q7.sv \
    rtl_h67/h67_score_class_row_engine.sv;
  chparam -set MAX_TOKENS 450 -set ACTIVE_MEM_DEPTH 450 \
    -set TOKEN_W 9 h67_score_class_row_engine;
  hierarchy -top h67_score_class_row_engine;
  proc; opt; memory_collect; check -assert;
  tee -o $OUT/yosys_baseline_stat.json stat -json
"

cat >"$OUT/source_files.txt" <<'EOF'
rtl_ttx/ttx_ceil_log2_u32.sv
rtl_ttx/ttx_exp2_lut_q8.sv
rtl_ttx/ttx_gate_quant_q17.sv
rtl_h67/h67_motionxor_score_q7.sv
rtl_h67/h67_temporal_pair_adapter.sv
rtl_h67/h67_score_class_row_engine.sv
rtl_h67/h67_temporal_score_quotient.sv
rtl_h67/h67_temporal_weighted_scs_directory.sv
rtl_h67/h67_temporal_quotient_scs_frontend.sv
rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv
verif_h67/h67_temporal_score_quotient_assertions.sv
verif_h67/h67_temporal_quotient_scs_assertions.sv
verif_h67/h67_temporal_quotient_shiftmax_gate_assertions.sv
tb_h67/tb_h67_temporal_quotient_shiftmax_gate_miter.sv
sim_h67/run_h67_tesc_gated_k_miter_checks.sh
scripts/summarize_h67_tesc_gated_k_miter.py
EOF

python scripts/summarize_h67_tesc_gated_k_miter.py \
  --result-dir "$OUT" --source-list "$OUT/source_files.txt"
jq -e '.status == "PASS" and .cross_simulator_t450_equal == true' \
  "$OUT/report.json" >/dev/null
git diff --check -- \
  rtl_h67/h67_temporal_weighted_scs_directory.sv \
  rtl_h67/h67_temporal_quotient_scs_frontend.sv \
  rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv \
  verif_h67/h67_temporal_quotient_shiftmax_gate_assertions.sv \
  tb_h67/tb_h67_temporal_quotient_shiftmax_gate_miter.sv \
  sim_h67/run_h67_tesc_gated_k_miter_checks.sh \
  scripts/summarize_h67_tesc_gated_k_miter.py

echo "PASS H67 TESC gated-K miter full flow"
