#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_new_arch/frontier_quotient"
OUT="${ROOT}/results/dual_line_frontier_quotient_rtl_20260803"
mkdir -p "${BUILD}" "${OUT}"

iverilog -g2012 -s tb_qfit_dual_color_active_source_index \
  -o "${BUILD}/local_iv" \
  "${ROOT}/rtl_qfit/qfit_dual_color_active_source_index.sv" \
  "${ROOT}/tb_qfit/tb_qfit_dual_color_active_source_index.sv"
vvp "${BUILD}/local_iv" | tee "${OUT}/local_iverilog.log"

iverilog -g2012 -s tb_h67_temporal_score_quotient \
  -o "${BUILD}/motion_iv" \
  "${ROOT}/rtl_h67/h67_motionxor_score_q7.sv" \
  "${ROOT}/rtl_h67/h67_temporal_score_quotient.sv" \
  "${ROOT}/tb_h67/tb_h67_temporal_score_quotient.sv"
vvp "${BUILD}/motion_iv" | tee "${OUT}/motion_iverilog.log"

rm -rf "${BUILD}/local_obj" "${BUILD}/motion_obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_dual_color_active_source_index \
  --Mdir "${BUILD}/local_obj" \
  "${ROOT}/rtl_qfit/qfit_dual_color_active_source_index.sv" \
  "${ROOT}/verif_qfit/qfit_dual_color_active_source_assertions.sv" \
  "${ROOT}/tb_qfit/tb_qfit_dual_color_active_source_index.sv"
"${BUILD}/local_obj/Vtb_qfit_dual_color_active_source_index" \
  | tee "${OUT}/local_verilator.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_h67_temporal_score_quotient \
  --Mdir "${BUILD}/motion_obj" \
  "${ROOT}/rtl_h67/h67_motionxor_score_q7.sv" \
  "${ROOT}/rtl_h67/h67_temporal_score_quotient.sv" \
  "${ROOT}/verif_h67/h67_temporal_score_quotient_assertions.sv" \
  "${ROOT}/tb_h67/tb_h67_temporal_score_quotient.sv"
"${BUILD}/motion_obj/Vtb_h67_temporal_score_quotient" \
  | tee "${OUT}/motion_verilator.log"

yosys -q -l "${OUT}/local_yosys.log" -p "
  read_verilog -sv ${ROOT}/rtl_qfit/qfit_dual_color_active_source_index.sv;
  hierarchy -top qfit_dual_color_active_source_index;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/local_stat.json stat -json
"
yosys -q -l "${OUT}/motion_yosys.log" -p "
  read_verilog -sv ${ROOT}/rtl_h67/h67_motionxor_score_q7.sv ${ROOT}/rtl_h67/h67_temporal_score_quotient.sv;
  hierarchy -top h67_temporal_score_quotient;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/motion_stat.json stat -json
"

printf 'Local5 Icarus exact set\tPASS\n' > "${OUT}/status.tsv"
printf 'Motion Icarus quotient inverse\tPASS\n' >> "${OUT}/status.tsv"
printf 'Local5 Verilator/SVA\tPASS\n' >> "${OUT}/status.tsv"
printf 'Motion Verilator/SVA\tPASS\n' >> "${OUT}/status.tsv"
printf 'Local5 Yosys check\tPASS\n' >> "${OUT}/status.tsv"
printf 'Motion Yosys check\tPASS\n' >> "${OUT}/status.tsv"
echo "PASS dual-line frontier/quotient RTL checks"
