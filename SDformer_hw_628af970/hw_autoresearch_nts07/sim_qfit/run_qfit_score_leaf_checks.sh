#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/iverilog"
mkdir -p "${BUILD}"

COMMON=(
  "${ROOT}/rtl_local5/local5_axnor_score_q7.sv"
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_local5/local5_stencil_token.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
)

iverilog -g2012 -s tb_qfit_local5_score_leaf \
  -o "${BUILD}/tb_qfit_local5_score_leaf.vvp" \
  "${COMMON[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_score_leaf.sv"
vvp "${BUILD}/tb_qfit_local5_score_leaf.vvp" \
  | tee "${BUILD}/tb_qfit_local5_score_leaf.log"
grep -q "PASS tb_qfit_local5_score_leaf" \
  "${BUILD}/tb_qfit_local5_score_leaf.log"

lint_variant() {
  local name="$1"
  shift
  verilator --lint-only --timing -Wall -Wno-fatal \
    --top-module qfit_local5_score_leaf \
    "$@" \
    "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv" \
    "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv" \
    "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv" \
    "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv" \
    >"${BUILD}/lint_${name}.log" 2>&1
  printf 'PASS lint %s\n' "${name}"
}

lint_variant w1 \
  -GARCH_QFSA=0
lint_variant global_qfsa_pipe \
  -GARCH_QFSA=1 -GPIPE_COMPACTOR=1
lint_variant xbf_t8 \
  -GARCH_QFSA=1 -GPIPE_COMPACTOR=1 -GXBF_BANKED=1 \
  -GUSE_THRESHOLD_ROUTE=1 -GROUTE_THRESHOLD=8
lint_variant xbf_t8b2 \
  -GARCH_QFSA=1 -GPIPE_COMPACTOR=1 -GXBF_BANKED=1 \
  -GUSE_THRESHOLD_ROUTE=1 -GROUTE_THRESHOLD=8 \
  -GUSE_BANK_PRESSURE_ROUTE=1 -GBANK_PRESSURE_THRESHOLD=2

ASSERT_BUILD="${ROOT}/build_qfit/verilator_assert"
rm -rf "${ASSERT_BUILD}"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_local5_score_leaf \
  --Mdir "${ASSERT_BUILD}" \
  "${COMMON[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_score_leaf.sv" \
  "${ROOT}/verif_qfit/qfit_score_leaf_assertions.sv" \
  >"${BUILD}/verilator_assert_build.log" 2>&1
"${ASSERT_BUILD}/Vtb_qfit_local5_score_leaf" \
  | tee "${BUILD}/verilator_assert_run.log"
grep -q "PASS tb_qfit_local5_score_leaf" \
  "${BUILD}/verilator_assert_run.log"
printf 'PASS Verilator SVA simulation\n'

{
  printf '功能仿真\tPASS\t%s\n' \
    "${BUILD}/tb_qfit_local5_score_leaf.log"
  printf 'Verilator参数化lint\tPASS\tw1/global_qfsa_pipe/xbf_t8/xbf_t8b2\n'
  printf 'Verilator_SVA仿真\tPASS\tDBDR上界/反压稳定\n'
} >"${BUILD}/verification_status.tsv"
