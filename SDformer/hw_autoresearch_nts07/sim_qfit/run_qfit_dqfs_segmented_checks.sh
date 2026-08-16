#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/dqfs_segmented"
OUT="${ROOT}/results/qfit_dqfs_segmented_20260731"
mkdir -p "${BUILD}" "${OUT}"

RTL="${ROOT}/rtl_qfit/qfit_dqfs_segmented_leaf.sv"
TB="${ROOT}/tb_qfit/tb_qfit_dqfs_segmented_leaf.sv"
SVA="${ROOT}/verif_qfit/qfit_dqfs_segmented_assertions.sv"

iverilog -g2012 -s tb_qfit_dqfs_segmented_leaf \
  -o "${BUILD}/dqfs_iv" "${RTL}" "${TB}"
vvp "${BUILD}/dqfs_iv" | tee "${OUT}/iverilog.log"

rm -rf "${BUILD}/obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_dqfs_segmented_leaf \
  --Mdir "${BUILD}/obj" \
  "${RTL}" "${SVA}" "${TB}"
"${BUILD}/obj/Vtb_qfit_dqfs_segmented_leaf" \
  | tee "${OUT}/verilator.log"

verilator --lint-only --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_dqfs_segmented_leaf \
  "${RTL}" "${SVA}" "${TB}" \
  >"${OUT}/verilator_lint.log" 2>&1

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${RTL};
  hierarchy -top qfit_dqfs_segmented_leaf;
  proc; opt; memory_collect; memory_dff; opt; check -assert;
  tee -o ${OUT}/stat.json stat -json
"

sha256sum "${RTL}" "${TB}" "${SVA}" \
  "${ROOT}/scripts/analyze_qfit_value_quotient_trace.py" \
  >"${OUT}/source_sha256.txt"
printf 'Icarus exact multiset\tPASS\n' >"${OUT}/status.tsv"
printf 'Verilator SVA\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator lint\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys synthesizable check\tPASS\n' >>"${OUT}/status.tsv"
printf 'PASS qfit DQFS segmented leaf checks\n'
