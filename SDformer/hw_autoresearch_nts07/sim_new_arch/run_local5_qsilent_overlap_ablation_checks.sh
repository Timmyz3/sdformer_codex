#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build_new_arch/local5_qsilent_overlap_ablation}"
OUT="${RESULT_DIR:-${ROOT}/results/local5_qsilent_overlap_ablation_20260813}"
VECTOR_DIR="${VECTOR_DIR:-${ROOT}/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
mkdir -p "${BUILD}" "${OUT}"

RTL=(
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_dual_color_word_skipper_index.sv"
  "${ROOT}/rtl_qfit/qfit_sync_relation_bank.sv"
  "${ROOT}/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_active_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_active_projection_tile.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_score_leaf_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_score_active_projection_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
)
TB="${ROOT}/tb_qfit/tb_qfit_local5_score_projection_postg0.sv"

run_config() {
  local name="$1"
  local qsilent="$2"
  local identk="$3"
  local overlap="$4"
  local obj="${BUILD}/${name}_obj"

  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
    --top-module tb_qfit_local5_score_projection_postg0 \
    -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
    -GARCH_QSILENT="${qsilent}" -GARCH_IDENTK="${identk}" \
    -GARCH_QSILENT_OVERLAP="${overlap}" \
    --Mdir "${obj}" "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
    >"${OUT}/${name}_build.log" 2>&1

  "${obj}/Vtb_qfit_local5_score_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" \
    "+ACTUAL_ACC_FILE=${OUT}/${name}_actual_acc32.memh" \
    | tee "${OUT}/${name}.log"
}

# OUT_DIM=2 projection tile ablation. These are not encoder cycles and are not
# extrapolated to the 21600 head-window population.
run_config residual 0 0 0
run_config q0_serial 1 0 0
run_config q0_overlap 1 0 1
run_config q0_ident_serial 1 1 0
run_config q0_ident_overlap 1 1 1

python3 "${ROOT}/scripts/report_local5_qsilent_overlap_ablation.py" \
  --result-dir "${OUT}" --vector-dir "${VECTOR_DIR}"

sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
  "${VECTOR_DIR}/manifest.json" \
  "${ROOT}/scripts/report_local5_qsilent_overlap_ablation.py" \
  "${BASH_SOURCE[0]}" >"${OUT}/source_sha256.txt"

echo "PASS Local5 Q-silent overlap ablation"
