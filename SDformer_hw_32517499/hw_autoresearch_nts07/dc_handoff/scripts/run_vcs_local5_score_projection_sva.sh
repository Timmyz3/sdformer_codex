#!/usr/bin/env bash
# VCS/SVA replay of sim_new_arch/run_local5_score_projection_checks.sh.
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYN_ROOT="${SYN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SYNOPSYS_ENV:-/home/zhumd/work/synopsys_date_dual/env.sh}"
export PATH="/opt/synopsys/vcs/V-2023.12-SP1/bin:${PATH}"
export VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
OUT="${OUTPUT_DIR:-${SYN_ROOT}/runs/local5_score_projection_vcs_sva_20260821}"
VECTOR_DIR="${VECTOR_DIR:-${SOURCE_ROOT}/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
mkdir -p "${OUT}"
test -d "${VECTOR_DIR}"

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

run_one() {
  local backend="$1" latency="$2" groups="$3" random="$4"
  local kind assertion tag run_dir
  if [[ "${backend}" == tcfm5 ]]; then
    kind=0
    assertion="${SOURCE_ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  else
    kind=1
    assertion="${SOURCE_ROOT}/verif_qfit/qfit_linear5_assertions.sv"
  fi
  if [[ "${random}" == 1 ]]; then tag="${backend}_l${latency}_random8"; else tag="${backend}_l${latency}_full100"; fi
  run_dir="${OUT}/${tag}"
  mkdir -p "${run_dir}"
  cd "${run_dir}"
  vcs -full64 -sverilog +v2k -timescale=1ns/1ps -assert svaext \
    -pvalue+tb_qfit_local5_score_projection_postg0.BACKEND_KIND="${kind}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.RELATION_READ_LATENCY="${latency}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.RUN_GROUPS="${groups}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.RANDOM_INPUT_GAPS="${random}" \
    -pvalue+tb_qfit_local5_score_projection_postg0.RANDOM_READ_GAPS="${random}" \
    -top tb_qfit_local5_score_projection_postg0 -o simv \
    "${RTL[@]}" "${COMMON_ASSERTIONS[@]}" "${assertion}" "${TB}" \
    2>&1 | tee compile.log
  ./simv +VECTOR_DIR="${VECTOR_DIR}" 2>&1 | tee simulation.log
  grep -q "PASS Local5 score-to-projection backend=${kind} latency=${latency} groups=${groups}" simulation.log
  if grep -Eq 'Error-|Assertion.*failed|\$error|\$fatal' simulation.log; then
    echo "Unexpected VCS/SVA error in ${tag}" >&2
    return 1
  fi
}

for latency in 1 2; do
  for backend in tcfm5 linear5; do
    run_one "${backend}" "${latency}" 100 0
    run_one "${backend}" "${latency}" 8 1
  done
done
echo "PASS VCS/SVA Local5 full100 + random8 matrix"
