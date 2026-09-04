#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_qfit/relation_transpose"
mkdir -p "${BUILD_DIR}"

iverilog -g2012 \
  -s tb_qfit_sync_1r1w_bank \
  -o "${BUILD_DIR}/bank.vvp" \
  "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
  "${ROOT_DIR}/tb_qfit/tb_qfit_sync_1r1w_bank.sv"
vvp "${BUILD_DIR}/bank.vvp"

for mode in 0 1 2; do
  iverilog -g2012 \
    -Ptb_qfit_relation_transpose_leaf.SCHED_MODE="${mode}" \
    -s tb_qfit_relation_transpose_leaf \
    -o "${BUILD_DIR}/mode${mode}.vvp" \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv" \
    "${ROOT_DIR}/tb_qfit/tb_qfit_relation_transpose_leaf.sv"
  vvp "${BUILD_DIR}/mode${mode}.vvp"

  verilator --lint-only --timing -Wall -Wno-fatal \
    --top-module qfit_relation_transpose_leaf \
    -GSCHED_MODE="${mode}" \
    -GHEIGHT=15 -GWIDTH=15 -GTIME_PLANES=2 \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv"

  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_relation_transpose_leaf \
    -GSCHED_MODE="${mode}" \
    --Mdir "${BUILD_DIR}/obj_mode${mode}" \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv" \
    "${ROOT_DIR}/tb_qfit/tb_qfit_relation_transpose_leaf.sv" \
    "${ROOT_DIR}/verif_qfit/qfit_relation_transpose_assertions.sv" \
    "${ROOT_DIR}/verif_qfit/qfit_sync_bank_assertions.sv" \
    --exe
  "${BUILD_DIR}/obj_mode${mode}/Vtb_qfit_relation_transpose_leaf"

  iverilog -g2012 \
    -Ptb_qfit_relation_transpose_perf.SCHED_MODE="${mode}" \
    -s tb_qfit_relation_transpose_perf \
    -o "${BUILD_DIR}/perf_mode${mode}.vvp" \
    "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
    "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv" \
    "${ROOT_DIR}/tb_qfit/tb_qfit_relation_transpose_perf.sv"
  vvp "${BUILD_DIR}/perf_mode${mode}.vvp"
done

# Fair safe three-row Stripe Pareto point: row-start admission, no early fill.
iverilog -g2012 \
  -Ptb_qfit_relation_transpose_leaf.SCHED_MODE=2 \
  -Ptb_qfit_relation_transpose_leaf.STRIPE_RING_ROWS=3 \
  -s tb_qfit_relation_transpose_leaf \
  -o "${BUILD_DIR}/mode2_rows3.vvp" \
  "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
  "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
  "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv" \
  "${ROOT_DIR}/tb_qfit/tb_qfit_relation_transpose_leaf.sv"
vvp "${BUILD_DIR}/mode2_rows3.vvp"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_relation_transpose_leaf \
  -GSCHED_MODE=2 -GSTRIPE_RING_ROWS=3 \
  --Mdir "${BUILD_DIR}/obj_mode2_rows3" \
  "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
  "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
  "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv" \
  "${ROOT_DIR}/tb_qfit/tb_qfit_relation_transpose_leaf.sv" \
  "${ROOT_DIR}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT_DIR}/verif_qfit/qfit_sync_bank_assertions.sv" \
  --exe
"${BUILD_DIR}/obj_mode2_rows3/Vtb_qfit_relation_transpose_leaf"

iverilog -g2012 \
  -Ptb_qfit_relation_transpose_perf.SCHED_MODE=2 \
  -Ptb_qfit_relation_transpose_perf.STRIPE_RING_ROWS=3 \
  -s tb_qfit_relation_transpose_perf \
  -o "${BUILD_DIR}/perf_mode2_rows3.vvp" \
  "${ROOT_DIR}/rtl_qfit/qfit_retirement_scheduler.sv" \
  "${ROOT_DIR}/rtl_qfit/qfit_sync_1r1w_bank.sv" \
  "${ROOT_DIR}/rtl_qfit/qfit_relation_transpose_leaf.sv" \
  "${ROOT_DIR}/tb_qfit/tb_qfit_relation_transpose_perf.sv"
vvp "${BUILD_DIR}/perf_mode2_rows3.vvp"
