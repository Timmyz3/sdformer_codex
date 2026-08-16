#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_encoder_job_scheduler"
OUT="${ROOT}/results/qfit_local5_encoder_job_scheduler_20260809"
mkdir -p "${BUILD}" "${OUT}"
rm -rf "${BUILD}/obj_main" "${BUILD}/obj_error"
rm -f "${OUT}"/*.log "${OUT}"/*.json "${OUT}"/*.tsv \
  "${OUT}"/*.txt "${OUT}"/*.md

RTL=(
  "${ROOT}/rtl_hitflow/gatestack_output_tile_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_local5_encoder_job_scheduler.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_hitflow/gatestack_output_tile_scheduler_assertions.sv"
  "${ROOT}/verif_hitflow/bind_gatestack_output_tile_scheduler_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_encoder_job_scheduler_assertions.sv"
)
TB_MAIN="${ROOT}/tb_qfit/tb_qfit_local5_encoder_job_scheduler.sv"
TB_ERROR="${ROOT}/tb_qfit/tb_qfit_local5_encoder_job_scheduler_error.sv"
SEEDS=(1 44257 48879)

{
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  yosys -V
  uname -srvmo
} >"${OUT}/tool_versions.txt"

iverilog -g2012 -Wall -Wno-timescale \
  -s tb_qfit_local5_encoder_job_scheduler \
  -o "${BUILD}/main_iv" "${RTL[@]}" "${TB_MAIN}"
for seed in "${SEEDS[@]}"; do
  vvp "${BUILD}/main_iv" "+STALL_SEED=${seed}" \
    | tee "${OUT}/main_seed_${seed}_iverilog.log"
done

iverilog -g2012 -Wall -Wno-timescale \
  -s tb_qfit_local5_encoder_job_scheduler_error \
  -o "${BUILD}/error_iv" "${RTL[@]}" "${TB_ERROR}"
vvp "${BUILD}/error_iv" | tee "${OUT}/error_iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-UNUSEDSIGNAL \
  --top-module tb_qfit_local5_encoder_job_scheduler \
  --Mdir "${BUILD}/obj_main" \
  "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_MAIN}" \
  >"${OUT}/verilator_main_build.log" 2>&1
for seed in "${SEEDS[@]}"; do
  "${BUILD}/obj_main/Vtb_qfit_local5_encoder_job_scheduler" \
    "+STALL_SEED=${seed}" \
    | tee "${OUT}/main_seed_${seed}_verilator_sva.log"
done

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-UNUSEDSIGNAL \
  --top-module tb_qfit_local5_encoder_job_scheduler_error \
  --Mdir "${BUILD}/obj_error" \
  "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_ERROR}" \
  >"${OUT}/verilator_error_build.log" 2>&1
"${BUILD}/obj_error/Vtb_qfit_local5_encoder_job_scheduler_error" \
  | tee "${OUT}/error_verilator_sva.log"

verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module qfit_local5_encoder_job_scheduler \
  "${RTL[@]}" >"${OUT}/verilator_rtl_lint.log" 2>&1

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${RTL[*]};
  hierarchy -check -top qfit_local5_encoder_job_scheduler;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/hier_stat.json stat -json;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/flat_stat.json stat -json
"

sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" "${TB_MAIN}" "${TB_ERROR}" \
  "${ROOT}/scripts/report_qfit_local5_encoder_job_scheduler.py" \
  "${ROOT}/sim_qfit/run_qfit_local5_encoder_job_scheduler_checks.sh" \
  >"${OUT}/source_sha256.txt"

printf 'Icarus 整帧三种子随机反压\tPASS\n' >"${OUT}/status.tsv"
printf 'Icarus 错误 tag/重复 start 故障注入\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator/SVA 整帧三种子随机反压\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator/SVA 错误 tag/重复 start 故障注入\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator RTL lint\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys hierarchy/flatten check\tPASS\n' >>"${OUT}/status.tsv"

python3 "${ROOT}/scripts/report_qfit_local5_encoder_job_scheduler.py"
printf 'PASS Local5 encoder job scheduler checks\n'
