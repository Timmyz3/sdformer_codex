#!/usr/bin/env bash
# VCS/SVA replay of sim_qfit/run_local5_score_active_cross_head_checks.sh.
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYN_ROOT="${SYN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SYNOPSYS_ENV:-/home/zhumd/work/synopsys_date_dual/env.sh}"
export PATH="/opt/anaconda3/bin:/opt/synopsys/vcs/V-2023.12-SP1/bin:${PATH}"
export VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
OUT="${OUTPUT_DIR:-${SYN_ROOT}/runs/local5_crosshead_vcs_sva_20260821}"
mkdir -p "${OUT}"

RTL=(
  "${SOURCE_ROOT}/rtl_hitflow/gatestack_output_tile_scheduler.sv"
  "${SOURCE_ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_sync_relation_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_banked_dynamic_retirement_scheduler.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${SOURCE_ROOT}/rtl_qfit/sidecar/qfit_dual_color_relation_frontier_sync.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_narrow_gate_weight_mul.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_sync_1rw_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_lane_product_cache_leaf.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_direct_1rw_acc_bank.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_cached_tcfm5_projection_top.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_active_projection_tile.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_score_active_projection_tile.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_fakeram45_acc_memory_90x1024.sv"
  "${SOURCE_ROOT}/rtl_qfit/qfit_local5_cross_head_tile_executor.sv"
)
ASSERTIONS=(
  "${SOURCE_ROOT}/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_retirement_scheduler_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_sync_bank_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_local5_score_active_projection_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
  "${SOURCE_ROOT}/verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv"
)

for head in 0 1 2; do
  /opt/anaconda3/bin/python3.12 "${SOURCE_ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
    --seed "$((0x510000 + head))" --out-dim 32 \
    --inputs "${OUT}/h${head}_inputs.txt" --expected "${OUT}/h${head}_expected.txt" \
    >"${OUT}/oracle_head_${head}.log"
done

cd "${OUT}"
vcs -full64 -sverilog +v2k -timescale=1ns/1ps -assert svaext \
  +define+QFIT_SCORE_ACTIVE_FRONT \
  -top tb_qfit_local5_cross_head_tile_executor -o simv_crosshead \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  "${SOURCE_ROOT}/tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv" \
  2>&1 | tee compile.log

./simv_crosshead \
  +PY_INPUTS_H0="${OUT}/h0_inputs.txt" +PY_EXPECTED_H0="${OUT}/h0_expected.txt" \
  +PY_INPUTS_H1="${OUT}/h1_inputs.txt" +PY_EXPECTED_H1="${OUT}/h1_expected.txt" \
  +PY_INPUTS_H2="${OUT}/h2_inputs.txt" +PY_EXPECTED_H2="${OUT}/h2_expected.txt" \
  +SERVICE_SEED=17717 2>&1 | tee simulation.log
grep -q "PASS Local5 cross-head OUT32 seed=17717 cycles=263583" simulation.log
if grep -Eq 'Error-|Assertion.*failed|\$error|\$fatal' simulation.log; then
  echo "Unexpected VCS/SVA error in Local5 cross-head" >&2
  exit 1
fi
echo "PASS VCS/SVA exact Local5 cross-head OUT32"
