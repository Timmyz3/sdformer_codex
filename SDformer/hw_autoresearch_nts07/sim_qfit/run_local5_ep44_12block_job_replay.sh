#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build_qfit/local5_ep44_12block_job_replay_20260815}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/local5_ep44_12block_job_replay_20260815}"
VECTOR_DIR="${VECTOR_DIR:-${ROOT}/tb_qfit/vectors/local5_ep44_hardware_rebind_20260815_score_projection100}"
PLAN_DIR="${PLAN_DIR:-${ROOT}/tb_qfit/vectors/local5_ep44_12block_job_plan_v2_20260815}"

if [[ -e "${BUILD_DIR}" || -e "${RESULT_DIR}" ]]; then
  echo "build/result directory already exists" >&2
  exit 2
fi
mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"

RTL=(
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
)
TB="${ROOT}/tb_qfit/tb_qfit_local5_ep44_12block_job_replay.sv"
RUN_ARGS=(
  "+VECTOR_DIR=${VECTOR_DIR}"
  "+PLAN_DIR=${PLAN_DIR}"
  "+SERVICE_SEED=23133"
)

iverilog -g2012 -s tb_qfit_local5_ep44_12block_job_replay \
  -o "${BUILD_DIR}/replay_iv" "${RTL[@]}" "${TB}" \
  >"${RESULT_DIR}/iverilog_build.log" 2>&1
vvp "${BUILD_DIR}/replay_iv" "${RUN_ARGS[@]}" \
  | tee "${RESULT_DIR}/iverilog_seed_23133.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_ep44_12block_job_replay \
  --Mdir "${BUILD_DIR}/obj" "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
  >"${RESULT_DIR}/verilator_build.log" 2>&1
"${BUILD_DIR}/obj/Vtb_qfit_local5_ep44_12block_job_replay" \
  "${RUN_ARGS[@]}" | tee "${RESULT_DIR}/verilator_seed_23133.log"

yosys -q -l "${RESULT_DIR}/yosys.log" -p "
  read_verilog -sv ${RTL[*]};
  chparam -set USE_SCORE_ACTIVE_FRONT 1 qfit_local5_tagged_t450_job_engine;
  hierarchy -check -top qfit_local5_tagged_t450_job_engine;
  proc; opt; memory_collect; check -assert; stat
"

sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
  "${ROOT}/scripts/generate_local5_ep44_12block_job_plan.py" \
  "${ROOT}/scripts/report_local5_ep44_12block_job_replay.py" \
  "${ROOT}/sim_qfit/run_local5_ep44_12block_job_replay.sh" \
  "${PLAN_DIR}/plan.json" "${VECTOR_DIR}/manifest.json" \
  >"${RESULT_DIR}/source_sha256.txt"

python3 "${ROOT}/scripts/report_local5_ep44_12block_job_replay.py" \
  --root "${ROOT}" --result-dir "${RESULT_DIR}" \
  --plan "${PLAN_DIR}/plan.json"
