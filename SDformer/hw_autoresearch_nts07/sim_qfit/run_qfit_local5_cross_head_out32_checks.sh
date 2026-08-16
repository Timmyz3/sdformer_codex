#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_cross_head_tile_executor"
OUT="${ROOT}/results/qfit_local5_cross_head_out32_20260809"
mkdir -p "${BUILD}" "${OUT}"
rm -rf "${BUILD}/obj_main" "${BUILD}/obj_error"
rm -f "${OUT}"/*.log "${OUT}"/*.json "${OUT}"/*.txt \
  "${OUT}"/*.tsv "${OUT}"/*.md

RTL_CORE=(
  "${ROOT}/rtl_hitflow/gatestack_output_tile_scheduler.sv"
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
  "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv"
  "${ROOT}/rtl_qfit/qfit_fakeram45_acc_memory_90x1024.sv"
  "${ROOT}/rtl_qfit/qfit_local5_cross_head_tile_executor.sv"
)
RTL_SHELL=(
  "${RTL_CORE[@]}"
  "${ROOT}/rtl_qfit/qfit_local5_encoder_job_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_local5_encoder_t450_numeric_shell.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv"
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv"
)
TB_MAIN="${ROOT}/tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv"
TB_ERROR="${ROOT}/tb_qfit/tb_qfit_local5_cross_head_tile_executor_error.sv"
SEEDS=(17717 44257 48879)

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

{
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  yosys -V
  uname -srvmo
} >"${OUT}/tool_versions.txt"

iverilog -g2012 -s tb_qfit_local5_cross_head_tile_executor \
  -o "${BUILD}/main_iv" "${RTL_CORE[@]}" "${TB_MAIN}" \
  >"${OUT}/iverilog_main_build.log" 2>&1
for seed in "${SEEDS[@]}"; do
  vvp "${BUILD}/main_iv" "${ORACLE_ARGS[@]}" \
    "+SERVICE_SEED=${seed}" \
    | tee "${OUT}/main_seed_${seed}_iverilog.log"
done

iverilog -g2012 -s tb_qfit_local5_cross_head_tile_executor_error \
  -o "${BUILD}/error_iv" "${RTL_CORE[@]}" "${TB_ERROR}" \
  >"${OUT}/iverilog_error_build.log" 2>&1
for mode in 0 1; do
  vvp "${BUILD}/error_iv" "+ERROR_MODE=${mode}" \
    | tee "${OUT}/error_mode_${mode}_iverilog.log"
done

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_cross_head_tile_executor \
  --Mdir "${BUILD}/obj_main" \
  "${RTL_CORE[@]}" "${ASSERTIONS[@]}" "${TB_MAIN}" \
  >"${OUT}/verilator_main_build.log" 2>&1
for seed in "${SEEDS[@]}"; do
  "${BUILD}/obj_main/Vtb_qfit_local5_cross_head_tile_executor" \
    "${ORACLE_ARGS[@]}" "+SERVICE_SEED=${seed}" \
    | tee "${OUT}/main_seed_${seed}_verilator_sva.log"
done

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_cross_head_tile_executor_error \
  --Mdir "${BUILD}/obj_error" \
  "${RTL_CORE[@]}" "${ASSERTIONS[@]}" "${TB_ERROR}" \
  >"${OUT}/verilator_error_build.log" 2>&1
for mode in 0 1; do
  "${BUILD}/obj_error/Vtb_qfit_local5_cross_head_tile_executor_error" \
    "+ERROR_MODE=${mode}" \
    | tee "${OUT}/error_mode_${mode}_verilator_sva.log"
done

verilator --lint-only --timing -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module qfit_local5_cross_head_tile_executor \
  "${RTL_CORE[@]}" >"${OUT}/verilator_executor_lint.log" 2>&1
verilator --lint-only --timing -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module qfit_local5_encoder_t450_numeric_shell \
  "${RTL_SHELL[@]}" >"${OUT}/verilator_shell_lint.log" 2>&1

yosys -q -l "${OUT}/yosys_executor.log" -p "
  read_verilog -sv ${RTL_CORE[*]};
  hierarchy -check -top qfit_local5_cross_head_tile_executor;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/executor_stat.json stat -json;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/executor_flat_stat.json stat -json
"
yosys -q -l "${OUT}/yosys_shell.log" -p "
  read_verilog -sv ${RTL_SHELL[*]};
  hierarchy -check -top qfit_local5_encoder_t450_numeric_shell;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/shell_stat.json stat -json;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/shell_flat_stat.json stat -json
"

sha256sum "${BUILD}"/h?_inputs.txt "${BUILD}"/h?_expected.txt \
  >"${OUT}/oracle_sha256.txt"
sha256sum "${RTL_SHELL[@]}" "${ASSERTIONS[@]}" "${TB_MAIN}" \
  "${TB_ERROR}" "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
  "${ROOT}/scripts/report_qfit_local5_cross_head_out32.py" \
  "${ROOT}/sim_qfit/run_qfit_local5_cross_head_out32_checks.sh" \
  >"${OUT}/source_sha256.txt"

printf 'Icarus 三种时序 scheduler-to-OUT32 Acc32 miter\tPASS\n' >"${OUT}/status.tsv"
printf 'Verilator/SVA 三种时序 scheduler-to-OUT32 Acc32 miter\tPASS\n' >>"${OUT}/status.tsv"
printf '错 tag/head 零 partial write fail-closed\tPASS\n' >>"${OUT}/status.tsv"
printf 'executor 与 12-block shell lint exit-code PASS（含已审阅警告）\tPASS\n' >>"${OUT}/status.tsv"
printf 'executor 与 12-block shell Yosys hierarchy/memory check\tPASS\n' >>"${OUT}/status.tsv"

python3 "${ROOT}/scripts/report_qfit_local5_cross_head_out32.py"
printf 'PASS Local5 cross-head OUT32 checks\n'
