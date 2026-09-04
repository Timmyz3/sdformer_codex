#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BUILD_ROOT="${BUILD_ROOT:?BUILD_ROOT must name a fresh ranked-checkpoint build directory}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
VECTOR_DIR="${VECTOR_DIR:?VECTOR_DIR is required}"

if [[ -e "${BUILD_ROOT}" ]]; then
  echo "ERROR: ranked-checkpoint build directory already exists: ${BUILD_ROOT}" >&2
  exit 2
fi
mkdir -p "${BUILD_ROOT}" "${RESULT_DIR}"

WEIGHT_ARGS=()
if [[ "${CHECKPOINT_WEIGHTS:-0}" == "1" ]]; then
  WEIGHT_ARGS+=(+CHECKPOINT_WEIGHTS)
fi

RTL=(
  tb_qfit/tb_qfit_local5_active_projection_postg0.sv
  rtl_qfit/qfit_local5_1rw_active_projection_tile.sv
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv
  rtl_qfit/qfit_dual_color_word_skipper_index.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_fakeram45_relation_bank_450.sv
  rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_local5_1rw_projection_backend.sv
  rtl_qfit/qfit_local5_color_map.sv
  rtl_qfit/qfit_direct_1rw_acc_bank.sv
  rtl_qfit/qfit_gasr2c_acc_bank.sv
  rtl_qfit/qfit_single_port_acc_memory.sv
)

SVA=(
  verif_qfit/qfit_local5_1rw_active_projection_assertions.sv
  verif_qfit/qfit_gasr2c_acc_bank_assertions.sv
  verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv
  verif_qfit/qfit_single_port_acc_memory_assertions.sv
  verif_qfit/qfit_dual_color_word_skipper_assertions.sv
  verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv
  verif_qfit/qfit_source_multicast_assertions.sv
)

SYNTH_RTL="${RTL[*]:1}"

for mode in 0 1; do
  name="direct"
  if [[ "${mode}" == "1" ]]; then
    name="qgasr"
  fi

  obj="${BUILD_ROOT}/${name}_obj"
  verilator --binary --timing -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    -Mdir "${obj}" -GNEW_1RW_BACKEND=1 -GMODE="${mode}" \
    -GGROUPS=100 -GRUN_GROUPS=100 "${RTL[@]}" \
    >"${RESULT_DIR}/${name}_compile.log" 2>&1
  "${obj}/Vtb_qfit_local5_active_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" "${WEIGHT_ARGS[@]}" \
    >"${RESULT_DIR}/${name}_profile100.log" 2>&1

  obj_sva="${BUILD_ROOT}/${name}_sva_obj"
  verilator --binary --timing --assert -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    -Mdir "${obj_sva}" -GNEW_1RW_BACKEND=1 -GMODE="${mode}" \
    -GGROUPS=100 -GRUN_GROUPS=100 \
    -GRANDOM_INPUT_GAPS=1 -GRANDOM_READ_GAPS=1 \
    "${RTL[@]}" "${SVA[@]}" \
    >"${RESULT_DIR}/${name}_random_sva_compile.log" 2>&1
  "${obj_sva}/Vtb_qfit_local5_active_projection_postg0" \
    "+VECTOR_DIR=${VECTOR_DIR}" "${WEIGHT_ARGS[@]}" \
    >"${RESULT_DIR}/${name}_random_sva.log" 2>&1

  verilator --lint-only --timing -Wno-fatal \
    --top-module qfit_local5_1rw_active_projection_tile -GMODE="${mode}" \
    ${SYNTH_RTL} >"${RESULT_DIR}/${name}_lint.log" 2>&1

  yosys -p "read_verilog -sv ${SYNTH_RTL}; chparam -set MODE ${mode} qfit_local5_1rw_active_projection_tile; hierarchy -check -top qfit_local5_1rw_active_projection_tile; proc; memory_dff; memory_collect; opt; check -assert; stat" \
    >"${RESULT_DIR}/${name}_yosys_memory_collect.log" 2>&1
done

sha256sum "${RTL[@]}" "${SVA[@]}" "${VECTOR_DIR}/manifest.json" \
  "${BASH_SOURCE[0]}" >"${RESULT_DIR}/source_sha256.txt"

python3 scripts/summarize_local5_gasr2c_fivebank_rtl.py \
  --manifest "${VECTOR_DIR}/manifest.json" \
  --source-manifest "${RESULT_DIR}/source_sha256.txt" \
  --direct-log "${RESULT_DIR}/direct_profile100.log" \
  --gasr-log "${RESULT_DIR}/qgasr_profile100.log" \
  --output-dir "${RESULT_DIR}" --variant descriptor_synchronized \
  >"${RESULT_DIR}/summary_stdout.json"

echo "PASS Local5 ranked-checkpoint DS-GASR-2C five-bank full flow"
