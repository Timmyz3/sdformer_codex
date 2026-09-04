#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_new_arch/local5_relation_macro"
OUT="${ROOT}/results/local5_relation_macro_postg0_rtl_20260803"
VECTOR_DIR="${ROOT}/tb_qfit/vectors/local5_active_projection_postg0_100"
mkdir -p "${BUILD}" "${OUT}"

RTL=(
  "${ROOT}/tb_qfit/fakeram45_relation_models.sv"
  "${ROOT}/rtl_qfit/qfit_dual_color_word_skipper_index.sv"
  "${ROOT}/rtl_qfit/qfit_sync_relation_bank.sv"
  "${ROOT}/rtl_qfit/qfit_fakeram45_relation_bank_450.sv"
  "${ROOT}/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_active_projection_tile.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_fakeram45_relation_bank_assertions.sv"
)

iverilog -g2012 -Wall -s tb_qfit_fakeram45_relation_bank_450 \
  -o "${BUILD}/relation_miter.vvp" \
  "${ROOT}/tb_qfit/fakeram45_relation_models.sv" \
  "${ROOT}/rtl_qfit/qfit_sync_relation_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_fakeram45_relation_bank_450.sv" \
  "${ROOT}/tb_qfit/tb_qfit_fakeram45_relation_bank_450.sv"
vvp "${BUILD}/relation_miter.vvp" | tee "${OUT}/relation_miter.log"

for backend in tcfm5 linear5; do
  if [[ "${backend}" == "tcfm5" ]]; then
    kind=0
    backend_assert="${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  else
    kind=1
    backend_assert="${ROOT}/verif_qfit/qfit_linear5_assertions.sv"
  fi
  rm -rf "${BUILD}/${backend}_obj"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    "-GBACKEND_KIND=${kind}" -GRELATION_READ_LATENCY=1 \
    -GRELATION_MEMORY_IMPL=1 \
    --Mdir "${BUILD}/${backend}_obj" \
    "${RTL[@]}" "${ASSERTIONS[@]}" "${backend_assert}" \
    "${ROOT}/tb_qfit/tb_qfit_local5_active_projection_postg0.sv"
  "${BUILD}/${backend}_obj/Vtb_qfit_local5_active_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" \
    | tee "${OUT}/${backend}_macro_verilator.log"
done

sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_fakeram45_relation_bank_450.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_active_projection_postg0.sv" \
  "${VECTOR_DIR}/manifest.json" > "${OUT}/source_sha256.txt"
printf '450-depth generic-to-macro miter\tPASS\n' > "${OUT}/status.tsv"
printf 'TCFM5 real post-G0 macro replay + SVA\tPASS\n' >> "${OUT}/status.tsv"
printf 'Linear5 real post-G0 macro replay + SVA\tPASS\n' >> "${OUT}/status.tsv"
echo "PASS Local5 relation macro checks"
