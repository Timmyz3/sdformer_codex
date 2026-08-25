#!/usr/bin/env bash
# Exact VCS/SVA replay of the former Verilator H67 multisample pipeline.
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYN_ROOT="${SYN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SYNOPSYS_ENV:-/home/zhumd/work/synopsys_date_dual/env.sh}"
export PATH="/opt/synopsys/vcs/V-2023.12-SP1/bin:${PATH}"
export VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

VECTOR_DIR="${VECTOR_DIR:-${SOURCE_ROOT}/tb_h67/vectors/h67_ep35_multisample10_t450_real_rtl}"
VECTORS="${VECTORS:-${VECTOR_DIR}/h67_multisample_checkpoint_rows.txt}"
ROW_LIMIT="${ROW_LIMIT:-1380}"
OUTPUT_DIR="${OUTPUT_DIR:-${SYN_ROOT}/runs/h67_multisample10_vcs_sva_20260821}"
test -s "${VECTORS}"
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

RTL=(
  "${SOURCE_ROOT}/rtl_ttx/ttx_ceil_log2_u32.sv"
  "${SOURCE_ROOT}/rtl_ttx/ttx_exp2_lut_q8.sv"
  "${SOURCE_ROOT}/rtl_ttx/ttx_gate_quant_q17.sv"
  "${SOURCE_ROOT}/rtl_h67/h67_motionxor_score_q7.sv"
  "${SOURCE_ROOT}/rtl_h67/h67_temporal_slot_encoder.sv"
  "${SOURCE_ROOT}/rtl_h67/h67_sync_dual_bank_k_store.sv"
  "${SOURCE_ROOT}/rtl_h67/h67_temporal_slot_fifo_2s.sv"
  "${SOURCE_ROOT}/rtl_h67/h67_temporal_weighted_scs_directory_2s.sv"
  "${SOURCE_ROOT}/rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv"
)

vcs -full64 -sverilog +v2k -timescale=1ns/1ps -assert svaext \
  -top tb_h67_temporal_slot_flow_real_trace_2s \
  -o simv_h67_multisample_sva \
  "${RTL[@]}" \
  "${SOURCE_ROOT}/verif_h67/h67_temporal_slot_flow_2s_assertions.sv" \
  "${SOURCE_ROOT}/tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv" \
  2>&1 | tee compile.log

./simv_h67_multisample_sva +VECTORS="${VECTORS}" +ROW_LIMIT="${ROW_LIMIT}" \
  2>&1 | tee simulation.log
grep -q "^PASS H67 RQTB 2S physical flow rows=${ROW_LIMIT} " simulation.log
echo "PASS VCS/SVA exact H67 multisample replay rows=${ROW_LIMIT}"
