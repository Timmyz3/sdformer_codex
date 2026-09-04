#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build_new_arch/local5_fullout32_stage_highload}"
OUT="${RESULT_DIR:-${ROOT}/results/local5_fullout32_stage_highload_rtl_20260813}"
VECTOR_DIR="${VECTOR_DIR:-${ROOT}/tb_qfit/vectors/local5_joint_ep29_fullout32_stage_highload_v1_20260813}"
mkdir -p "${BUILD}" "${OUT}"

RTL=(
  "${ROOT}/rtl_qfit/qfit_dual_color_word_skipper_index.sv"
  "${ROOT}/rtl_qfit/qfit_sync_relation_bank.sv"
  "${ROOT}/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_active_projection_tile.sv"
)
COMMON_ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
)
TB="${ROOT}/tb_qfit/tb_qfit_local5_active_projection_postg0.sv"

for latency in 1 2; do
 for backend in tcfm5 linear5; do
  if [[ "${backend}" == "tcfm5" ]]; then
   kind=0
   BACKEND_ASSERTION="${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  else
   kind=1
   BACKEND_ASSERTION="${ROOT}/verif_qfit/qfit_linear5_assertions.sv"
  fi
  OBJ="${BUILD}/${backend}_l${latency}_obj"
  rm -rf "${OBJ}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    "-GBACKEND_KIND=${kind}" "-GRELATION_READ_LATENCY=${latency}" \
    -GGROUPS=4 -GRUN_GROUPS=4 -GOUT_DIM=32 \
    --Mdir "${OBJ}" "${RTL[@]}" "${COMMON_ASSERTIONS[@]}" \
    "${BACKEND_ASSERTION}" "${TB}"
  "${OBJ}/Vtb_qfit_local5_active_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" +CHECKPOINT_WEIGHTS \
    "+ACTUAL_ACC_FILE=${OUT}/${backend}_l${latency}_actual_acc32.memh" \
    | tee "${OUT}/${backend}_l${latency}_verilator.log"
 done
done

sha256sum "${RTL[@]}" "${COMMON_ASSERTIONS[@]}" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_linear5_assertions.sv" "${TB}" \
  "${ROOT}/scripts/generate_local5_active_projection_postg0_vectors.py" \
  "${ROOT}/scripts/report_local5_fullout32_stage_highload_rtl.py" \
  "${BASH_SOURCE[0]}" "${VECTOR_DIR}/manifest.json" \
  > "${OUT}/source_sha256.txt"
python3 "${ROOT}/scripts/report_local5_fullout32_stage_highload_rtl.py" \
  --result-dir "${OUT}" --vector-dir "${VECTOR_DIR}"
echo "PASS Local5 fullout32 stage-highload checks"
