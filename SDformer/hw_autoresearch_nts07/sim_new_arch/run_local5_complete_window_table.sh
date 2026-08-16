#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/local5_complete_window_table_20260813}"
BUILD="${BUILD_DIR:-$ROOT/build_new_arch/local5_complete_window_table}"
mkdir -p "$OUT" "$BUILD"
cd "$ROOT"

generate_stage() {
  local stage="$1"
  local vec="$ROOT/tb_qfit/vectors/local5_qsilent_s${stage}b0_window_proj_20260813"
  if [[ ! -f "$vec/manifest.json" ]]; then
    python3 scripts/generate_local5_window_score_projection.py \
      --output-dir "$vec" --sample 0 --stage "$stage" --block 0 >&2
  fi
  printf '%s\n' "$vec"
}

S0="$ROOT/tb_qfit/vectors/local5_qsilent_window_proj_20260813"
S1=$(generate_stage 1 6)
S2=$(generate_stage 2 12)
S3="$ROOT/tb_qfit/vectors/local5_qsilent_s3b0_window_proj_20260813"
if [[ ! -f "$S3/manifest.json" ]]; then
  S3=$(generate_stage 3 24)
fi

python3 scripts/analyze_local5_residual_leftover.py \
  --vector-dirs "$S1" "$S2" "$S3" \
  --output-dir "$OUT/residual_leftover"

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

run_pair() {
  local tag="$1" vec="$2" heads="$3"
  for spec in "residual:0" "qsilent:1"; do
    local name="${spec%%:*}"
    local qsilent="${spec##*:}"
    local obj="$BUILD/${tag}_${name}_obj"
    rm -rf "$obj"
    verilator --binary --timing --assert -Wall -Wno-fatal \
      -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
      --top-module tb_qfit_local5_score_projection_postg0 \
      -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
      -GARCH_QSILENT="$qsilent" -GGROUPS="$heads" -GRUN_GROUPS="$heads" \
      --Mdir "$obj" "${RTL[@]}" "$TB"
    "$obj/Vtb_qfit_local5_score_projection_postg0" \
      "+VECTOR_DIR=$vec" \
      "+ACTUAL_ACC_FILE=$OUT/${tag}_${name}_actual_acc32.memh" \
      | tee "$OUT/${tag}_${name}_verilator.log"
  done
}

run_pair s1 "$S1" 6
run_pair s2 "$S2" 12
# S3 already has residual/qsilent from GOAL 355; re-run qsilent only for RTL counters.
obj="$BUILD/s3_qsilent_obj"
rm -rf "$obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
  --top-module tb_qfit_local5_score_projection_postg0 \
  -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=1 -GGROUPS=24 -GRUN_GROUPS=24 \
  --Mdir "$obj" "${RTL[@]}" "$TB"
"$obj/Vtb_qfit_local5_score_projection_postg0" \
  "+VECTOR_DIR=$S3" \
  "+ACTUAL_ACC_FILE=$OUT/s3_qsilent_actual_acc32.memh" \
  | tee "$OUT/s3_qsilent_verilator.log"
cp "$ROOT/results/local5_qsilent_s3_window_acc32_20260813/residual_verilator.log" \
  "$OUT/s3_residual_verilator.log"

python3 scripts/report_local5_complete_window_table.py \
  --result-dir "$OUT" \
  --s0-residual "$ROOT/results/local5_qsilent_window_acc32_20260813" \
  --s3-old "$ROOT/results/local5_qsilent_s3_window_acc32_20260813"
echo "PASS Local5 complete-window table"
