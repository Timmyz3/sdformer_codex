#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_qfit/local5_tile"
mkdir -p "${BUILD_DIR}"

RTL_SOURCES=(
  "${ROOT_DIR}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT_DIR}/rtl_local5/local5_axnor_score_q7.sv"
  "${ROOT_DIR}/rtl_local5/local5_stencil_token.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT_DIR}/rtl_qfit/qfit_local5_tile.sv"
)
TB_SOURCE="${ROOT_DIR}/tb_qfit/tb_qfit_local5_tile.sv"

iverilog -g2012 \
  -s tb_qfit_local5_tile \
  -o "${BUILD_DIR}/tile.vvp" \
  "${RTL_SOURCES[@]}" \
  "${TB_SOURCE}"
vvp "${BUILD_DIR}/tile.vvp"

verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module qfit_local5_tile \
  -GHEIGHT=15 -GWIDTH=15 -GTIME_PLANES=2 \
  "${RTL_SOURCES[@]}"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_local5_tile \
  --Mdir "${BUILD_DIR}/obj_assert" \
  "${RTL_SOURCES[@]}" \
  "${TB_SOURCE}" \
  "${ROOT_DIR}/verif_qfit/qfit_score_leaf_assertions.sv" \
  "${ROOT_DIR}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT_DIR}/verif_qfit/qfit_sync_bank_assertions.sv" \
  --exe
"${BUILD_DIR}/obj_assert/Vtb_qfit_local5_tile"
