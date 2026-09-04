#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VEC="${VECTOR_DIR:-$ROOT/tb_qfit/vectors/local5_qsilent_s3b0_window_proj_20260813}"
OUT="${RESULT_DIR:-$ROOT/results/local5_qsilent_s3_window_acc32_20260813}"
BUILD="${BUILD_DIR:-$ROOT/build_new_arch/local5_qsilent_s3_window}"
mkdir -p "$VEC" "$OUT" "$BUILD"
cd "$ROOT"

python3 scripts/generate_local5_window_score_projection.py \
  --output-dir "$VEC" --sample 0 --stage 3 --block 0

python3 scripts/analyze_local5_s3_residual_fastpath.py \
  --vector-dir "$VEC" --output-dir "$OUT/residual_decision"

RTL=(
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_qfit/qfit_tagged_compactor4.sv
  rtl_qfit/qfit_xorbank_compactor4.sv
  rtl_qfit/qfit_local5_score_leaf.sv
  rtl_qfit/qfit_local5_qsilent_score_leaf.sv
  rtl_qfit/qfit_dual_color_word_skipper_index.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_tcfm5_acc_bank.sv
  rtl_qfit/qfit_tcfm5_projection_top.sv
  rtl_qfit/qfit_linear5_projection_top.sv
  rtl_qfit/qfit_local5_active_projection_tile.sv
  rtl_qfit/qfit_local5_score_active_projection_tile.sv
)
TB=tb_qfit/tb_qfit_local5_score_projection_postg0.sv
HEADS=$(python3 -c "import json; print(json.load(open('$VEC/manifest.json'))['selection']['groups'])")

for spec in "residual:0" "qsilent:1"; do
  name="${spec%%:*}"
  qsilent="${spec##*:}"
  obj="$BUILD/${name}_obj"
  rm -rf "$obj"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
    --top-module tb_qfit_local5_score_projection_postg0 \
    -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
    -GARCH_QSILENT="$qsilent" -GGROUPS="$HEADS" -GRUN_GROUPS="$HEADS" \
    --Mdir "$obj" "${RTL[@]}" "$TB"
  "$obj/Vtb_qfit_local5_score_projection_postg0" \
    "+VECTOR_DIR=$VEC" \
    "+ACTUAL_ACC_FILE=$OUT/${name}_actual_acc32.memh" \
    | tee "$OUT/${name}_verilator.log"
done

python3 scripts/report_local5_qsilent_s3_window_acc32.py --result-dir "$OUT" --vector-dir "$VEC"
echo "PASS Local5 S3 window Acc32"
