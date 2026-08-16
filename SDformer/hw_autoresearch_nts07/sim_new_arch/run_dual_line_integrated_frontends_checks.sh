#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_new_arch/integrated_frontends"
OUT="${ROOT}/results/dual_line_integrated_frontends_rtl_20260803"
mkdir -p "${BUILD}" "${OUT}"

LOCAL_RTL=(
  "${ROOT}/rtl_qfit/qfit_dual_color_word_skipper_index.sv"
  "${ROOT}/rtl_qfit/qfit_dual_color_relation_frontier.sv"
)
MOTION_RTL=(
  "${ROOT}/rtl_h67/h67_motionxor_score_q7.sv"
  "${ROOT}/rtl_h67/h67_temporal_score_quotient.sv"
  "${ROOT}/rtl_h67/h67_temporal_weighted_scs_directory.sv"
  "${ROOT}/rtl_h67/h67_temporal_quotient_scs_frontend.sv"
)

iverilog -g2012 -s tb_qfit_dual_color_relation_frontier \
  -o "${BUILD}/local_iv" \
  "${LOCAL_RTL[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_dual_color_relation_frontier.sv"
vvp "${BUILD}/local_iv" | tee "${OUT}/local_iverilog.log"

iverilog -g2012 -s tb_qfit_dual_color_index_equivalence \
  -o "${BUILD}/local_equiv_iv" \
  "${ROOT}/rtl_qfit/qfit_dual_color_active_source_index.sv" \
  "${ROOT}/rtl_qfit/qfit_dual_color_word_skipper_index.sv" \
  "${ROOT}/tb_qfit/tb_qfit_dual_color_index_equivalence.sv"
vvp "${BUILD}/local_equiv_iv" | tee "${OUT}/local_equivalence_iverilog.log"

iverilog -g2012 -s tb_h67_temporal_quotient_scs_frontend \
  -o "${BUILD}/motion_iv" \
  "${MOTION_RTL[@]}" \
  "${ROOT}/tb_h67/tb_h67_temporal_quotient_scs_frontend.sv"
vvp "${BUILD}/motion_iv" | tee "${OUT}/motion_iverilog.log"

rm -rf "${BUILD}/local_obj" "${BUILD}/motion_obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_dual_color_relation_frontier \
  --Mdir "${BUILD}/local_obj" \
  "${LOCAL_RTL[@]}" \
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_assertions.sv" \
  "${ROOT}/tb_qfit/tb_qfit_dual_color_relation_frontier.sv"
"${BUILD}/local_obj/Vtb_qfit_dual_color_relation_frontier" \
  | tee "${OUT}/local_verilator.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_h67_temporal_quotient_scs_frontend \
  --Mdir "${BUILD}/motion_obj" \
  "${MOTION_RTL[@]}" \
  "${ROOT}/verif_h67/h67_temporal_score_quotient_assertions.sv" \
  "${ROOT}/verif_h67/h67_temporal_quotient_scs_assertions.sv" \
  "${ROOT}/tb_h67/tb_h67_temporal_quotient_scs_frontend.sv"
"${BUILD}/motion_obj/Vtb_h67_temporal_quotient_scs_frontend" \
  | tee "${OUT}/motion_verilator.log"

yosys -q -l "${OUT}/local_yosys.log" -p "
  read_verilog -sv ${LOCAL_RTL[*]};
  hierarchy -top qfit_dual_color_relation_frontier;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/local_stat.json stat -json
"
yosys -q -l "${OUT}/motion_yosys.log" -p "
  read_verilog -sv ${MOTION_RTL[*]};
  hierarchy -top h67_temporal_quotient_scs_frontend;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/motion_stat.json stat -json
"

printf 'Local5 relation-frontier Icarus\tPASS\n' > "${OUT}/status.tsv"
printf 'Local5 T450 old/new index equivalence\tPASS\n' >> "${OUT}/status.tsv"
printf 'Motion weighted-SCS Icarus\tPASS\n' >> "${OUT}/status.tsv"
printf 'Local5 relation-frontier Verilator/SVA\tPASS\n' >> "${OUT}/status.tsv"
printf 'Motion weighted-SCS Verilator/SVA\tPASS\n' >> "${OUT}/status.tsv"
printf 'Local5 relation-frontier Yosys\tPASS\n' >> "${OUT}/status.tsv"
printf 'Motion weighted-SCS Yosys\tPASS\n' >> "${OUT}/status.tsv"
echo "PASS dual-line integrated frontend checks"
