#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build_new_arch/local5_qsilent_score}"
OUT="${RESULT_DIR:-${ROOT}/results/local5_qsilent_score_rtl_20260813}"
VECTOR_DIR="${VECTOR_DIR:-${ROOT}/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
BASELINE_REPORT="${BASELINE_REPORT:-${ROOT}/results/local5_score_projection_rtl_20260813/report.json}"
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
LEAF_TB="${ROOT}/tb_qfit/tb_qfit_local5_qsilent_score_leaf.sv"
PROJ_TB="${ROOT}/tb_qfit/tb_qfit_local5_score_projection_postg0.sv"
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_score_leaf_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_word_skipper_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_score_active_projection_assertions.sv"
)

iverilog -g2012 -s tb_qfit_local5_qsilent_score_leaf \
  -o "${BUILD}/qsilent_miter_iv" \
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv" \
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv" \
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv" \
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv" \
  "${ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv" \
  "${LEAF_TB}"
vvp "${BUILD}/qsilent_miter_iv" | tee "${OUT}/qsilent_miter_iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
  --top-module tb_qfit_local5_qsilent_score_leaf \
  --Mdir "${BUILD}/qsilent_miter_obj" \
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv" \
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv" \
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv" \
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv" \
  "${ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv" \
  "${ROOT}/verif_qfit/qfit_score_leaf_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv" \
  "${LEAF_TB}"
"${BUILD}/qsilent_miter_obj/Vtb_qfit_local5_qsilent_score_leaf" \
  | tee "${OUT}/qsilent_miter_verilator.log"

# One-group smoke, then 100-group TCFM5/Linear5 L1 plus TCFM5 L2.
for backend in tcfm5 linear5; do
  if [[ "${backend}" == "tcfm5" ]]; then kind=0; else kind=1; fi
  iverilog -g2012 -s tb_qfit_local5_score_projection_postg0 \
    -Ptb_qfit_local5_score_projection_postg0.BACKEND_KIND="${kind}" \
    -Ptb_qfit_local5_score_projection_postg0.ARCH_QSILENT=1 \
    -Ptb_qfit_local5_score_projection_postg0.RUN_GROUPS=1 \
    -o "${BUILD}/${backend}_smoke_iv" "${RTL[@]}" "${PROJ_TB}"
  vvp "${BUILD}/${backend}_smoke_iv" "+VECTOR_DIR=${VECTOR_DIR}" \
    | tee "${OUT}/${backend}_smoke_iverilog.log"
done

for spec in "tcfm5:0:1" "linear5:1:1" "tcfm5:0:2"; do
  backend="${spec%%:*}"
  rest="${spec#*:}"
  kind="${rest%%:*}"
  latency="${rest##*:}"
  key="${backend}_l${latency}"
  obj="${BUILD}/${key}_obj"
  rm -rf "${obj}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
    --top-module tb_qfit_local5_score_projection_postg0 \
    "-GBACKEND_KIND=${kind}" "-GRELATION_READ_LATENCY=${latency}" \
    "-GARCH_QSILENT=1" \
    --Mdir "${obj}" "${RTL[@]}" "${ASSERTIONS[@]}" \
    "${ROOT}/verif_qfit/qfit_${backend}_assertions.sv" "${PROJ_TB}"
  "${obj}/Vtb_qfit_local5_score_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" \
    "+ACTUAL_ACC_FILE=${OUT}/${key}_actual_acc32.memh" \
    | tee "${OUT}/${key}_verilator.log"
done

yosys -q -l "${OUT}/qsilent_yosys.log" -p "
  read_verilog -sv ${RTL[*]};
  chparam -set BACKEND_KIND 0 -set RELATION_READ_LATENCY 1 -set ARCH_QSILENT 1 \
    qfit_local5_score_active_projection_tile;
  hierarchy -check -top qfit_local5_score_active_projection_tile;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/qsilent_stat.json stat -json
"

python3 "${ROOT}/scripts/report_local5_qsilent_score_rtl.py" \
  --result-dir "${OUT}" --vector-dir "${VECTOR_DIR}" \
  --baseline-report "${BASELINE_REPORT}" \
  $(printf -- ' --source %q' "${RTL[@]}" "${ASSERTIONS[@]}" \
      "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
      "${ROOT}/verif_qfit/qfit_linear5_assertions.sv" \
      "${LEAF_TB}" "${PROJ_TB}" \
      "${ROOT}/scripts/report_local5_qsilent_score_rtl.py" \
      "${BASH_SOURCE[0]}")

echo "PASS Local5 Q-silent exact score checks"
