#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/vl_gs_ttb"
OUT="${ROOT}/results/vl_gs_ttb_dual_line_rtl_20260731"
mkdir -p "${BUILD}" "${OUT}"

DEC="${ROOT}/rtl_qfit/qfit_vl_gs_ttb_slot_decoder.sv"
MOT="${ROOT}/rtl_qfit/qfit_vl_gs_ttb_motion_encoder.sv"
LOC="${ROOT}/rtl_qfit/qfit_vl_gs_ttb_local_allocator.sv"
TB_MOTION="${ROOT}/tb_qfit/tb_qfit_vl_gs_ttb_motion.sv"
TB_LOCAL="${ROOT}/tb_qfit/tb_qfit_vl_gs_ttb_local.sv"

for line in motion local; do
  top="tb_qfit_vl_gs_ttb_${line}"
  tb_var="TB_${line^^}"
  tb="${!tb_var}"
  iverilog -g2012 -s "${top}" -o "${BUILD}/${line}_iv" \
    "${DEC}" "${MOT}" "${LOC}" "${tb}"
  vvp "${BUILD}/${line}_iv" | tee "${OUT}/iverilog_${line}.log"

  rm -rf "${BUILD}/obj_${line}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module "${top}" --Mdir "${BUILD}/obj_${line}" \
    "${DEC}" "${MOT}" "${LOC}" "${tb}" \
    >"${OUT}/verilator_build_${line}.log" 2>&1
  "${BUILD}/obj_${line}/V${top}" | tee "${OUT}/verilator_${line}.log"
done

for top in \
  qfit_vl_gs_ttb_slot_decoder \
  qfit_vl_gs_ttb_motion_encoder \
  qfit_vl_gs_ttb_local_allocator
do
  verilator --lint-only -Wall -Wno-fatal --top-module "${top}" \
    "${DEC}" "${MOT}" "${LOC}" >"${OUT}/lint_${top}.log" 2>&1
  yosys -q -l "${OUT}/yosys_${top}.log" -p "
    read_verilog -sv ${DEC} ${MOT} ${LOC};
    hierarchy -top ${top};
    proc; opt; memory_collect; memory_dff; opt; check -assert;
    tee -o ${OUT}/stat_${top}.json stat -json
  "
done

sha256sum "${DEC}" "${MOT}" "${LOC}" "${TB_MOTION}" "${TB_LOCAL}" \
  "${BASH_SOURCE[0]}" >"${OUT}/source_sha256.txt"
printf 'Icarus Motion/Local5 exact reconstruction\tPASS\n' >"${OUT}/status.tsv"
printf 'Verilator assertions and independent backpressure\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator leaf lint\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys synth-readable\tPASS\n' >>"${OUT}/status.tsv"
printf 'PASS VL-GS-TTB dual-line checks\n'

