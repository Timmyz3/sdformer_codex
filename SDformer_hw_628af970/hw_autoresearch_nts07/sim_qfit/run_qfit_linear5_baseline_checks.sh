#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/linear5_baseline"
OUT="${ROOT}/results/qfit_linear5_baseline_20260731"
mkdir -p "${BUILD}" "${OUT}"

COMMON=(
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/tb_qfit/tb_qfit_tcfm5_projection_top.sv"
)

iverilog -g2012 \
  -DQFIT_PROJECTION_DUT=qfit_linear5_projection_top \
  -DQFIT_EXHAUSTIVE_MASKS \
  -s tb_qfit_tcfm5_projection_top \
  -o "${BUILD}/linear5.vvp" \
  "${COMMON[@]}"
vvp "${BUILD}/linear5.vvp" | tee "${OUT}/iverilog.log"

rm -rf "${BUILD}/obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -DQFIT_PROJECTION_DUT=qfit_linear5_projection_top \
  -DQFIT_EXHAUSTIVE_MASKS \
  --top-module tb_qfit_tcfm5_projection_top \
  --Mdir "${BUILD}/obj" \
  "${COMMON[@]}" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_linear5_assertions.sv"
"${BUILD}/obj/Vtb_qfit_tcfm5_projection_top" \
  | tee "${OUT}/verilator.log"

verilator --lint-only -Wall -Wno-fatal \
  --top-module qfit_linear5_projection_top \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv" \
  >"${OUT}/lint.log" 2>&1

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv ${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv;
  hierarchy -top qfit_linear5_projection_top;
  proc; opt; memory_collect; memory_dff; flatten; opt; memory_collect;
  check -assert;
  tee -o ${OUT}/stat.json stat -json
"

sha256sum \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv" \
  "${ROOT}/tb_qfit/tb_qfit_tcfm5_projection_top.sv" \
  "${ROOT}/verif_qfit/qfit_linear5_assertions.sv" \
  >"${OUT}/source_sha256.txt"

printf 'Icarus exhaustive masks\tPASS\n' >"${OUT}/status.tsv"
printf 'Verilator/SVA\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys check\tPASS\n' >>"${OUT}/status.tsv"
python3 "${ROOT}/scripts/report_qfit_linear5_baseline.py"
printf 'PASS qfit Linear-5 baseline checks\n'
