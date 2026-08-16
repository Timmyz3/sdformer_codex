#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_qfit/retirement"
mkdir -p "${BUILD_DIR}"

iverilog -g2012 \
  -s tb_qfit_retirement_scheduler \
  -o "${BUILD_DIR}/tb_retirement.vvp" \
  "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
  "${ROOT_DIR}/tb_qfit/tb_qfit_retirement_scheduler.sv"
vvp "${BUILD_DIR}/tb_retirement.vvp"

for mode in 0 1 2; do
  verilator --lint-only --timing -Wall -Wno-fatal \
    --top-module qfit_retirement_scheduler \
    -GMODE="${mode}" \
    -GHEIGHT=15 -GWIDTH=15 -GTIME_PLANES=2 \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv"

  iverilog -g2012 \
    -Ptb_qfit_retirement_scheduler_single.MODE="${mode}" \
    -s tb_qfit_retirement_scheduler_single \
    -o "${BUILD_DIR}/tb_single_mode${mode}.vvp" \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
    "${ROOT_DIR}/tb_qfit/tb_qfit_retirement_scheduler_single.sv"
  vvp "${BUILD_DIR}/tb_single_mode${mode}.vvp"

  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_retirement_scheduler_single \
    -GMODE="${mode}" \
    --Mdir "${BUILD_DIR}/obj_assert_mode${mode}" \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
    "${ROOT_DIR}/tb_qfit/tb_qfit_retirement_scheduler_single.sv" \
    "${ROOT_DIR}/verif_qfit/qfit_retirement_scheduler_assertions.sv" \
    --exe
  "${BUILD_DIR}/obj_assert_mode${mode}/Vtb_qfit_retirement_scheduler_single"
done
