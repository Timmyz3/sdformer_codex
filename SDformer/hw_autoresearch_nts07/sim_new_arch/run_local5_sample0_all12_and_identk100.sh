#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/local5_sample0_all12_identk100_20260813}"
BUILD="${BUILD_DIR:-$ROOT/build_new_arch/local5_sample0_all12}"
VEC100="${VECTOR100:-$ROOT/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
mkdir -p "$OUT" "$BUILD"
cd "$ROOT"

python3 scripts/generate_local5_sample0_all12_windows.py

python3 scripts/analyze_local5_residual_leftover.py \
  --vector-dirs \
    tb_qfit/vectors/local5_s0b0_window_proj_20260813 \
    tb_qfit/vectors/local5_s0b1_window_proj_20260813 \
    tb_qfit/vectors/local5_s1b0_window_proj_20260813 \
    tb_qfit/vectors/local5_s1b1_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b0_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b1_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b2_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b3_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b4_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b5_window_proj_20260813 \
    tb_qfit/vectors/local5_s3b0_window_proj_20260813 \
    tb_qfit/vectors/local5_s3b1_window_proj_20260813 \
  --output-dir "$OUT/leftover_all12" || python3 scripts/analyze_local5_residual_leftover.py \
  --vector-dirs \
    tb_qfit/vectors/local5_qsilent_window_proj_20260813 \
    tb_qfit/vectors/local5_qsilent_s1b0_window_proj_20260813 \
    tb_qfit/vectors/local5_qsilent_s2b0_window_proj_20260813 \
    tb_qfit/vectors/local5_qsilent_s3b0_window_proj_20260813 \
    tb_qfit/vectors/local5_s0b1_window_proj_20260813 \
    tb_qfit/vectors/local5_s1b1_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b1_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b2_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b3_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b4_window_proj_20260813 \
    tb_qfit/vectors/local5_s2b5_window_proj_20260813 \
    tb_qfit/vectors/local5_s3b1_window_proj_20260813 \
  --output-dir "$OUT/leftover_all12"

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

compile() {
  local heads="$1" qsilent="$2" name="$3"
  local obj="$BUILD/${name}_obj"
  if [[ -x "$obj/Vtb_qfit_local5_score_projection_postg0" ]]; then
    return
  fi
  rm -rf "$obj"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
    --top-module tb_qfit_local5_score_projection_postg0 \
    -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
    -GARCH_QSILENT="$qsilent" -GGROUPS="$heads" -GRUN_GROUPS="$heads" \
    --Mdir "$obj" "${RTL[@]}" "$TB"
}

run_win() {
  local tag="$1" vec="$2" heads="$3"
  compile "$heads" 0 "${heads}_res"
  compile "$heads" 1 "${heads}_qs"
  "$BUILD/${heads}_res_obj/Vtb_qfit_local5_score_projection_postg0" \
    "+VECTOR_DIR=$vec" | tee "$OUT/${tag}_residual.log"
  "$BUILD/${heads}_qs_obj/Vtb_qfit_local5_score_projection_postg0" \
    "+VECTOR_DIR=$vec" | tee "$OUT/${tag}_qsilent.log"
}

# Prefer newly generated names; fall back to earlier B0 aliases.
vec_for() {
  local st="$1" bl="$2"
  local a="$ROOT/tb_qfit/vectors/local5_s${st}b${bl}_window_proj_20260813"
  if [[ -f "$a/manifest.json" ]]; then echo "$a"; return; fi
  if [[ "$st" == 0 && "$bl" == 0 ]]; then
    echo "$ROOT/tb_qfit/vectors/local5_qsilent_window_proj_20260813"; return
  fi
  echo "$ROOT/tb_qfit/vectors/local5_qsilent_s${st}b${bl}_window_proj_20260813"
}

run_win s0b0 "$(vec_for 0 0)" 3
run_win s0b1 "$(vec_for 0 1)" 3
run_win s1b0 "$(vec_for 1 0)" 6
run_win s1b1 "$(vec_for 1 1)" 6
run_win s2b0 "$(vec_for 2 0)" 12
run_win s2b1 "$(vec_for 2 1)" 12
run_win s2b2 "$(vec_for 2 2)" 12
run_win s2b3 "$(vec_for 2 3)" 12
run_win s2b4 "$(vec_for 2 4)" 12
run_win s2b5 "$(vec_for 2 5)" 12
run_win s3b0 "$(vec_for 3 0)" 24
run_win s3b1 "$(vec_for 3 1)" 24

# S3 Linear5
obj="$BUILD/s3_lin_res_obj"
rm -rf "$obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
  --top-module tb_qfit_local5_score_projection_postg0 \
  -GBACKEND_KIND=1 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=0 -GGROUPS=24 -GRUN_GROUPS=24 \
  --Mdir "$obj" "${RTL[@]}" "$TB"
"$obj/Vtb_qfit_local5_score_projection_postg0" \
  "+VECTOR_DIR=$(vec_for 3 0)" | tee "$OUT/s3b0_linear5_residual.log"
obj="$BUILD/s3_lin_qs_obj"
rm -rf "$obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
  --top-module tb_qfit_local5_score_projection_postg0 \
  -GBACKEND_KIND=1 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=1 -GGROUPS=24 -GRUN_GROUPS=24 \
  --Mdir "$obj" "${RTL[@]}" "$TB"
"$obj/Vtb_qfit_local5_score_projection_postg0" \
  "+VECTOR_DIR=$(vec_for 3 0)" | tee "$OUT/s3b0_linear5_qsilent.log"

# 100-group ident-K TCFM5 L1
obj="$BUILD/g100_qs_obj"
rm -rf "$obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
  --top-module tb_qfit_local5_score_projection_postg0 \
  -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=1 -GGROUPS=100 -GRUN_GROUPS=100 \
  --Mdir "$obj" "${RTL[@]}" "$TB"
"$obj/Vtb_qfit_local5_score_projection_postg0" \
  "+VECTOR_DIR=$VEC100" | tee "$OUT/identk100_tcfm5_l1.log"

python3 scripts/report_local5_sample0_all12_identk100.py --result-dir "$OUT"
echo "PASS Local5 sample0-all12 + ident-K 100-group"
