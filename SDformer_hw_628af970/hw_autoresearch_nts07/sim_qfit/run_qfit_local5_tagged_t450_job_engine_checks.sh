#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_tagged_t450_job_engine"
OUT="${ROOT}/results/qfit_local5_tagged_t450_job_engine_20260809"
mkdir -p "${BUILD}" "${OUT}"
rm -rf "${BUILD}/obj_main" "${BUILD}/obj_error"
rm -f "${OUT}"/*.log "${OUT}"/*.json "${OUT}"/*.tsv \
  "${OUT}"/*.txt "${OUT}"/*.md

RTL=(
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
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv"
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
)
TB_MAIN="${ROOT}/tb_qfit/tb_qfit_local5_tagged_t450_job_engine.sv"
TB_ERROR="${ROOT}/tb_qfit/tb_qfit_local5_tagged_t450_job_engine_error.sv"
SEEDS=(1 17717 44257 48879)
ERROR_MODES=(0 1 2 3 4 5 6 7 8 9 10 11)

python3 "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
  --seed 0x45052026 --out-dim 2 \
  --inputs "${BUILD}/t450_inputs.txt" \
  --expected "${BUILD}/t450_expected.txt" \
  >"${OUT}/oracle_generation.log"
ORACLE_ARGS=(
  "+PY_INPUTS=${BUILD}/t450_inputs.txt"
  "+PY_EXPECTED=${BUILD}/t450_expected.txt"
)

{
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  yosys -V
  uname -srvmo
} >"${OUT}/tool_versions.txt"

iverilog -g2012 -s tb_qfit_local5_tagged_t450_job_engine \
  -o "${BUILD}/main_iv" "${RTL[@]}" "${TB_MAIN}" \
  >"${OUT}/iverilog_main_build.log" 2>&1
for seed in "${SEEDS[@]}"; do
  vvp "${BUILD}/main_iv" "${ORACLE_ARGS[@]}" \
    "+SERVICE_SEED=${seed}" \
    | tee "${OUT}/main_seed_${seed}_iverilog.log"
done

iverilog -g2012 -s tb_qfit_local5_tagged_t450_job_engine_error \
  -o "${BUILD}/error_iv" "${RTL[@]}" "${TB_ERROR}" \
  >"${OUT}/iverilog_error_build.log" 2>&1
for mode in "${ERROR_MODES[@]}"; do
  vvp "${BUILD}/error_iv" "+ERROR_MODE=${mode}" \
    | tee "${OUT}/error_mode_${mode}_iverilog.log"
done

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_tagged_t450_job_engine \
  --Mdir "${BUILD}/obj_main" \
  "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_MAIN}" \
  >"${OUT}/verilator_main_build.log" 2>&1
for seed in "${SEEDS[@]}"; do
  "${BUILD}/obj_main/Vtb_qfit_local5_tagged_t450_job_engine" \
    "${ORACLE_ARGS[@]}" "+SERVICE_SEED=${seed}" \
    | tee "${OUT}/main_seed_${seed}_verilator_sva.log"
done

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_tagged_t450_job_engine_error \
  --Mdir "${BUILD}/obj_error" \
  "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_ERROR}" \
  >"${OUT}/verilator_error_build.log" 2>&1
for mode in "${ERROR_MODES[@]}"; do
  "${BUILD}/obj_error/Vtb_qfit_local5_tagged_t450_job_engine_error" \
    "+ERROR_MODE=${mode}" \
    | tee "${OUT}/error_mode_${mode}_verilator_sva.log"
done

verilator --lint-only --timing -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module qfit_local5_tagged_t450_job_engine \
  "${RTL[@]}" >"${OUT}/verilator_rtl_lint.log" 2>&1

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${RTL[*]};
  hierarchy -check -top qfit_local5_tagged_t450_job_engine;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/hier_stat.json stat -json;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/flat_stat.json stat -json
"

sha256sum "${BUILD}/t450_inputs.txt" "${BUILD}/t450_expected.txt" \
  >"${OUT}/oracle_sha256.txt"
sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_MAIN}" "${TB_ERROR}" \
  "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
  "${ROOT}/scripts/report_qfit_local5_tagged_t450_job_engine.py" \
  "${ROOT}/sim_qfit/run_qfit_local5_tagged_t450_job_engine_checks.sh" \
  >"${OUT}/source_sha256.txt"

printf 'Icarus 四种服务时序种子 Python-to-Acc32\tPASS\n' >"${OUT}/status.tsv"
printf 'Icarus 12类 response 故障矩阵 fail-closed\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator/SVA 四种服务时序种子 Python-to-Acc32\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator/SVA 12类 response 故障矩阵 fail-closed\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator lint exit-code PASS（含已审阅警告）\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys hierarchy/flatten check\tPASS\n' >>"${OUT}/status.tsv"

python3 "${ROOT}/scripts/report_qfit_local5_tagged_t450_job_engine.py"
printf 'PASS Local5 tagged T450 job engine checks\n'
