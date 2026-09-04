#!/usr/bin/env bash
# VCS/SVA replay of sim_new_arch/run_local5_sample0_all12_and_identk100.sh.
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYN_ROOT="${SYN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SYNOPSYS_ENV:-/home/zhumd/work/synopsys_date_dual/env.sh}"
export PATH="/opt/synopsys/vcs/V-2023.12-SP1/bin:${PATH}"
export VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
OUT="${OUTPUT_DIR:-${SYN_ROOT}/runs/local5_all12_vcs_sva_20260821}"
VEC100="${VECTOR100:-${SOURCE_ROOT}/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
mkdir -p "${OUT}/build"

RTL=(
  "${SOURCE_ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_dual_color_word_skipper_index.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_sync_relation_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_active_projection_tile.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_score_active_projection_tile.sv"
)
COMMON_ASSERTIONS=(
  "${SOURCE_ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_local5_score_active_projection_assertions.sv"
)
TB="${SOURCE_ROOT}/tb_qfit/tb_qfit_local5_score_projection_postg0.sv"

compile() {
  local backend="$1" heads="$2" qsilent="$3" kind assertion name build
  if [[ "${backend}" == tcfm5 ]]; then
    kind=0
    assertion="${SOURCE_ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  else
    kind=1
    assertion="${SOURCE_ROOT}/verif_qfit/qfit_linear5_assertions.sv"
  fi
  name="${backend}_h${heads}_q${qsilent}"
  build="${OUT}/build/${name}"
  if [[ -x "${build}/simv" ]]; then return; fi
  mkdir -p "${build}"
  cd "${build}"
  vcs -full64 -sverilog +v2k -timescale=1ns/1ps -assert svaext \
    -pvalue+tb_qfit_local5_score_projection_postg0.BACKEND_KIND="${kind}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.RELATION_READ_LATENCY=1 \
    -pvalue+tb_qfit_local5_score_projection_postg0.ARCH_QSILENT="${qsilent}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.GROUPS="${heads}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.RUN_GROUPS="${heads}" \
    -top tb_qfit_local5_score_projection_postg0 -o simv \
    "${RTL[@]}" "${COMMON_ASSERTIONS[@]}" "${assertion}" "${TB}" \
    2>&1 | tee compile.log
}

sim() {
  local backend="$1" heads="$2" qsilent="$3" vec="$4" log="$5"
  compile "${backend}" "${heads}" "${qsilent}"
  "${OUT}/build/${backend}_h${heads}_q${qsilent}/simv" +VECTOR_DIR="${vec}" \
    2>&1 | tee "${OUT}/${log}"
  grep -q "PASS Local5 score-to-projection" "${OUT}/${log}"
  if grep -Eq 'Error-|Assertion.*failed|\$error|\$fatal' "${OUT}/${log}"; then
    echo "Unexpected VCS/SVA error in ${log}" >&2
    return 1
  fi
}

vec_for() {
  local stage="$1" block="$2"
  local primary="${SOURCE_ROOT}/tb_qfit/vectors/local5_s${stage}b${block}_window_proj_20260813"
  if [[ -f "${primary}/manifest.json" ]]; then echo "${primary}"; return; fi
  if [[ "${stage}" == 0 && "${block}" == 0 ]]; then
    echo "${SOURCE_ROOT}/tb_qfit/vectors/local5_qsilent_window_proj_20260813"
  else
    echo "${SOURCE_ROOT}/tb_qfit/vectors/local5_qsilent_s${stage}b${block}_window_proj_20260813"
  fi
}

run_win() {
  local tag="$1" stage="$2" block="$3" heads="$4" vec
  vec="$(vec_for "${stage}" "${block}")"
  sim tcfm5 "${heads}" 0 "${vec}" "${tag}_residual.log"
  sim tcfm5 "${heads}" 1 "${vec}" "${tag}_qsilent.log"
}

run_win s0b0 0 0 3
run_win s0b1 0 1 3
run_win s1b0 1 0 6
run_win s1b1 1 1 6
run_win s2b0 2 0 12
run_win s2b1 2 1 12
run_win s2b2 2 2 12
run_win s2b3 2 3 12
run_win s2b4 2 4 12
run_win s2b5 2 5 12
run_win s3b0 3 0 24
run_win s3b1 3 1 24
sim linear5 24 0 "$(vec_for 3 0)" s3b0_linear5_residual.log
sim linear5 24 1 "$(vec_for 3 0)" s3b0_linear5_qsilent.log
sim tcfm5 100 1 "${VEC100}" identk100_tcfm5_l1.log
python3 "${SOURCE_ROOT}/scripts/report_local5_sample0_all12_identk100.py" --result-dir "${OUT}"
echo "PASS VCS/SVA Local5 sample0 all12 + ident-K 100-group"
