#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_score_active_cross_head_20260815"
OUT="${ROOT}/results/local5_score_active_cross_head_20260815"
mkdir -p "${BUILD}" "${OUT}"

RTL=(
  "${ROOT}/rtl_hitflow/gatestack_output_tile_scheduler.sv"
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT}/rtl_qfit/qfit_banked_dynamic_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT}/rtl_qfit/sidecar/qfit_dual_color_relation_frontier_sync.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_narrow_gate_weight_mul.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1rw_bank.sv"
  "${ROOT}/rtl_qfit/qfit_lane_product_cache_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_direct_1rw_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_cached_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_active_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_active_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
  "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv"
  "${ROOT}/rtl_qfit/qfit_fakeram45_acc_memory_90x1024.sv"
  "${ROOT}/rtl_qfit/qfit_local5_cross_head_tile_executor.sv"
)
SHELL_RTL=(
  "${RTL[@]}"
  "${ROOT}/rtl_qfit/qfit_local5_encoder_job_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_local5_encoder_t450_numeric_shell.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv"
  "${ROOT}/verif_qfit/qfit_retirement_scheduler_assertions.sv"
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_score_active_projection_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv"
)

for head in 0 1 2; do
  python3 "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
    --seed "$((0x510000 + head))" --out-dim 32 \
    --inputs "${BUILD}/h${head}_inputs.txt" \
    --expected "${BUILD}/h${head}_expected.txt" \
    >"${OUT}/oracle_head_${head}.log"
done
ORACLE_ARGS=(
  "+PY_INPUTS_H0=${BUILD}/h0_inputs.txt"
  "+PY_EXPECTED_H0=${BUILD}/h0_expected.txt"
  "+PY_INPUTS_H1=${BUILD}/h1_inputs.txt"
  "+PY_EXPECTED_H1=${BUILD}/h1_expected.txt"
  "+PY_INPUTS_H2=${BUILD}/h2_inputs.txt"
  "+PY_EXPECTED_H2=${BUILD}/h2_expected.txt"
)

iverilog -g2012 -DQFIT_SCORE_ACTIVE_FRONT \
  -s tb_qfit_local5_cross_head_tile_executor \
  -o "${BUILD}/main_iv" "${RTL[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv" \
  >"${OUT}/iverilog_build.log" 2>&1
vvp "${BUILD}/main_iv" "${ORACLE_ARGS[@]}" +SERVICE_SEED=17717 \
  | tee "${OUT}/iverilog_seed_17717.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -DQFIT_SCORE_ACTIVE_FRONT \
  --top-module tb_qfit_local5_cross_head_tile_executor \
  --Mdir "${BUILD}/obj_prod" "${RTL[@]}" "${ASSERTIONS[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv" \
  >"${OUT}/verilator_build.log" 2>&1
"${BUILD}/obj_prod/Vtb_qfit_local5_cross_head_tile_executor" \
  "${ORACLE_ARGS[@]}" +SERVICE_SEED=17717 \
  | tee "${OUT}/verilator_seed_17717.log"

verilator --lint-only --timing -Wall -Wno-fatal \
  -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  -GUSE_SCORE_ACTIVE_FRONT=1 \
  --top-module qfit_local5_encoder_t450_numeric_shell \
  "${SHELL_RTL[@]}" >"${OUT}/verilator_shell_prod_lint.log" 2>&1

yosys -q -l "${OUT}/yosys_shell_prod.log" -p "
  read_verilog -sv ${SHELL_RTL[*]};
  chparam -set USE_SCORE_ACTIVE_FRONT 1 qfit_local5_encoder_t450_numeric_shell;
  hierarchy -check -top qfit_local5_encoder_t450_numeric_shell;
  proc; opt; memory_collect; check -assert; stat
"

sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" \
  "${ROOT}/rtl_qfit/qfit_local5_encoder_t450_numeric_shell.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv" \
  "${ROOT}/scripts/report_local5_score_active_cross_head.py" \
  "${ROOT}/sim_qfit/run_local5_score_active_cross_head_checks.sh" \
  >"${OUT}/source_sha256.txt"

python3 "${ROOT}/scripts/report_local5_score_active_cross_head.py" \
  --root "${ROOT}" --evidence "${OUT}"
