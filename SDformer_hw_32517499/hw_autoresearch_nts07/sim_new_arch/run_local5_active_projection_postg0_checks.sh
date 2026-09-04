#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build_new_arch/local5_active_projection_postg0}"
OUT="${RESULT_DIR:-${ROOT}/results/local5_active_projection_sync_sram_postg0_rtl_20260803}"
VECTOR_DIR="${VECTOR_DIR:-${ROOT}/tb_qfit/vectors/local5_active_projection_postg0_100}"
mkdir -p "${BUILD}" "${OUT}"

WEIGHT_ARGS=()
if [[ "${CHECKPOINT_WEIGHTS:-0}" == "1" ]]; then
  WEIGHT_ARGS+=(+CHECKPOINT_WEIGHTS)
fi

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
TB_SMALL="${ROOT}/tb_qfit/tb_qfit_local5_active_projection_tile.sv"
TB_REAL="${ROOT}/tb_qfit/tb_qfit_local5_active_projection_postg0.sv"
ASSERTION_SOURCES=(
  "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${ROOT}/verif_qfit/qfit_linear5_assertions.sv"
)

if [[ "${VECTOR_PREGENERATED:-0}" != "1" ]]; then
  python3 "${ROOT}/scripts/generate_local5_active_projection_postg0_vectors.py"
fi

for latency in 1 2; do
 for backend in tcfm5 linear5; do
  if [[ "${backend}" == "tcfm5" ]]; then kind=0; else kind=1; fi
  iverilog -g2012 -s tb_qfit_local5_active_projection_tile \
    -P "tb_qfit_local5_active_projection_tile.BACKEND_KIND=${kind}" \
    -P "tb_qfit_local5_active_projection_tile.RELATION_READ_LATENCY=${latency}" \
    -o "${BUILD}/${backend}_l${latency}_small_iv" "${RTL[@]}" "${TB_SMALL}"
  vvp "${BUILD}/${backend}_l${latency}_small_iv" \
    | tee "${OUT}/${backend}_l${latency}_small_iverilog.log"

  rm -rf "${BUILD}/${backend}_l${latency}_obj"
  ASSERTIONS=(
    "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
    "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
    "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
    "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  )
  if [[ "${backend}" == "tcfm5" ]]; then
    ASSERTIONS+=("${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv")
  else
    ASSERTIONS+=("${ROOT}/verif_qfit/qfit_linear5_assertions.sv")
  fi
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    "-GBACKEND_KIND=${kind}" \
    "-GRELATION_READ_LATENCY=${latency}" \
    --Mdir "${BUILD}/${backend}_l${latency}_obj" \
    "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_REAL}"
  "${BUILD}/${backend}_l${latency}_obj/Vtb_qfit_local5_active_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" "${WEIGHT_ARGS[@]}" \
    "+ACTUAL_ACC_FILE=${OUT}/${backend}_l${latency}_actual_acc32.memh" \
    | tee "${OUT}/${backend}_l${latency}_verilator.log"

  if [[ "${latency}" == "1" ]]; then
   yosys -q -l "${OUT}/${backend}_yosys.log" -p "
    read_verilog -sv ${RTL[*]};
    chparam -set BACKEND_KIND ${kind} -set RELATION_READ_LATENCY 1 qfit_local5_active_projection_tile;
    hierarchy -top qfit_local5_active_projection_tile;
    proc; opt; memory_collect; check -assert;
    tee -o ${OUT}/${backend}_stat.json stat -json;
    flatten; opt; memory_collect; check -assert;
    tee -o ${OUT}/${backend}_flat_stat.json stat -json
  "
  fi
 done
done

for latency in 1 2; do
 for backend in tcfm5 linear5; do
  if [[ "${backend}" == "tcfm5" ]]; then
   kind=0
   BACKEND_ASSERTION="${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  else
   kind=1
   BACKEND_ASSERTION="${ROOT}/verif_qfit/qfit_linear5_assertions.sv"
  fi
  STRESS_BUILD="${BUILD}/${backend}_l${latency}_random_stress_obj"
  rm -rf "${STRESS_BUILD}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    "-GBACKEND_KIND=${kind}" "-GRELATION_READ_LATENCY=${latency}" \
    -GRANDOM_INPUT_GAPS=1 -GRANDOM_READ_GAPS=1 \
    --Mdir "${STRESS_BUILD}" "${RTL[@]}" \
    "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv" \
    "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv" \
    "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
    "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
    "${BACKEND_ASSERTION}" "${TB_REAL}"
  "${STRESS_BUILD}/Vtb_qfit_local5_active_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" "${WEIGHT_ARGS[@]}" \
    | tee "${OUT}/${backend}_l${latency}_random_stress_verilator.log"
 done
done

sha256sum "${RTL[@]}" "${ASSERTION_SOURCES[@]}" "${TB_SMALL}" "${TB_REAL}" \
  "${ROOT}/scripts/report_local5_active_projection_postg0_rtl.py" \
  "${ROOT}/scripts/generate_local5_active_projection_postg0_vectors.py" \
  "${BASH_SOURCE[0]}" "${VECTOR_DIR}/manifest.json" \
  > "${OUT}/source_sha256.txt"
printf 'Icarus small exact L1/L2 TCFM5/Linear5\tPASS\n' > "${OUT}/status.tsv"
printf 'Verilator/SVA post-G0 L1/L2 TCFM5\tPASS\n' >> "${OUT}/status.tsv"
printf 'Verilator/SVA post-G0 L1/L2 Linear5\tPASS\n' >> "${OUT}/status.tsv"
printf 'Yosys T450 structural TCFM5/Linear5\tPASS\n' >> "${OUT}/status.tsv"
printf 'Verilator/SVA L1/L2 TCFM5/Linear5 random input/read gaps\tPASS\n' >> "${OUT}/status.tsv"
python3 "${ROOT}/scripts/report_local5_active_projection_postg0_rtl.py" \
  --result-dir "${OUT}" --vector-dir "${VECTOR_DIR}"
echo "PASS Local5 active projection post-G0 checks"
