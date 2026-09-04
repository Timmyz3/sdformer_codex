#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build_new_arch/local5_score_projection}"
OUT="${RESULT_DIR:-${ROOT}/results/local5_score_projection_rtl_20260813}"
VECTOR_DIR="${VECTOR_DIR:-${ROOT}/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
POSTSCORE_REPORT="${POSTSCORE_REPORT:-${ROOT}/results/local5_joint_ep29_tcfm5_linear5_realw_sample100_population_rtl_v5_final_20260813/report.json}"
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
TB="${ROOT}/tb_qfit/tb_qfit_local5_score_projection_postg0.sv"
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_score_active_projection_assertions.sv"
)

(
  cd "${ROOT}"
  python3 -m unittest \
    scripts.test_generate_local5_score_projection_vectors \
    scripts.test_report_local5_score_projection_rtl
)

# Fast independent Icarus smoke on both architectural backends.
for backend in tcfm5 linear5; do
  if [[ "${backend}" == "tcfm5" ]]; then kind=0; else kind=1; fi
  iverilog -g2012 -s tb_qfit_local5_score_projection_postg0 \
    -Ptb_qfit_local5_score_projection_postg0.BACKEND_KIND="${kind}" \
    -Ptb_qfit_local5_score_projection_postg0.RUN_GROUPS=1 \
    -o "${BUILD}/${backend}_smoke_iv" "${RTL[@]}" "${TB}"
  vvp "${BUILD}/${backend}_smoke_iv" "+VECTOR_DIR=${VECTOR_DIR}" \
    | tee "${OUT}/${backend}_smoke_iverilog.log"
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
    key="${backend}_l${latency}"
    obj="${BUILD}/${key}_obj"
    rm -rf "${obj}"
    verilator --binary --timing --assert -Wall -Wno-fatal \
      -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
      --top-module tb_qfit_local5_score_projection_postg0 \
      "-GBACKEND_KIND=${kind}" "-GRELATION_READ_LATENCY=${latency}" \
      --Mdir "${obj}" "${RTL[@]}" "${ASSERTIONS[@]}" \
      "${BACKEND_ASSERTION}" "${TB}"
    "${obj}/Vtb_qfit_local5_score_projection_postg0" \
      "+VECTOR_DIR=${VECTOR_DIR}" \
      "+ACTUAL_ACC_FILE=${OUT}/${key}_actual_acc32.memh" \
      | tee "${OUT}/${key}_verilator.log"

    stress_obj="${BUILD}/${key}_stress_obj"
    rm -rf "${stress_obj}"
    verilator --binary --timing --assert -Wall -Wno-fatal \
      -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
      --top-module tb_qfit_local5_score_projection_postg0 \
      "-GBACKEND_KIND=${kind}" "-GRELATION_READ_LATENCY=${latency}" \
      -GRUN_GROUPS=8 -GRANDOM_INPUT_GAPS=1 -GRANDOM_READ_GAPS=1 \
      --Mdir "${stress_obj}" "${RTL[@]}" "${ASSERTIONS[@]}" \
      "${BACKEND_ASSERTION}" "${TB}"
    "${stress_obj}/Vtb_qfit_local5_score_projection_postg0" \
      "+VECTOR_DIR=${VECTOR_DIR}" \
      | tee "${OUT}/${key}_random_stress_verilator.log"

    if [[ "${latency}" == "1" ]]; then
      yosys -q -l "${OUT}/${backend}_yosys.log" -p "
        read_verilog -sv ${RTL[*]};
        chparam -set BACKEND_KIND ${kind} -set RELATION_READ_LATENCY 1 qfit_local5_score_active_projection_tile;
        hierarchy -check -top qfit_local5_score_active_projection_tile;
        proc; opt; memory_collect; check -assert;
        tee -o ${OUT}/${backend}_stat.json stat -json
      "
    fi
  done
done

printf 'Icarus one-group TCFM5/Linear5\tPASS\n' > "${OUT}/status.tsv"
printf 'Verilator/SVA 100-group TCFM5/Linear5 L1/L2\tPASS\n' >> "${OUT}/status.tsv"
printf 'Verilator/SVA 8-group random input/read gaps all four configs\tPASS\n' >> "${OUT}/status.tsv"
printf 'Yosys structural TCFM5/Linear5\tPASS\n' >> "${OUT}/status.tsv"

python3 "${ROOT}/scripts/report_local5_score_projection_rtl.py" \
  --result-dir "${OUT}" --vector-dir "${VECTOR_DIR}" \
  --postscore-report "${POSTSCORE_REPORT}" \
  $(printf -- ' --source %q' "${RTL[@]}" "${ASSERTIONS[@]}" \
      "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
      "${ROOT}/verif_qfit/qfit_linear5_assertions.sv" "${TB}" \
      "${ROOT}/scripts/generate_local5_score_projection_vectors.py" \
      "${ROOT}/scripts/generate_local5_checkpoint_score_vectors.py" \
      "${ROOT}/scripts/generate_local5_active_projection_postg0_vectors.py" \
      "${ROOT}/scripts/generate_local5_masked_integer_vectors.py" \
      "${ROOT}/scripts/report_local5_score_projection_rtl.py" \
      "${BASH_SOURCE[0]}")

echo "PASS Local5 score/Shiftmax5-to-Acc checks"
