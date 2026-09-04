#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${TMPDIR:-/tmp}/qfit_affine4_projection_build"
OUT="${ROOT}/results/qfit_affine4_projection_20260731"

rm -rf "${BUILD}"
mkdir -p "${BUILD}" "${OUT}"

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Icarus\t%s\n' "$(iverilog -V 2>&1 | head -n 1)"
  printf 'Verilator\t%s\n' "$(verilator --version)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '证据边界\tRTL仿真、动态SVA、lint和开放结构综合；非PPA\n'
} >"${OUT}/tool_versions.tsv"

iverilog -g2012 \
  -s tb_qfit_affine4_projection_top \
  -o "${BUILD}/affine4.vvp" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv" \
  "${ROOT}/tb_qfit/tb_qfit_affine4_projection_top.sv" \
  >"${OUT}/icarus_build.log" 2>&1
vvp "${BUILD}/affine4.vvp" | tee "${OUT}/icarus_run.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_affine4_projection_top \
  --Mdir "${BUILD}/obj_affine4" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv" \
  "${ROOT}/tb_qfit/tb_qfit_affine4_projection_top.sv" \
  "${ROOT}/verif_qfit/qfit_affine4_projection_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  >"${OUT}/verilator_build.log" 2>&1
"${BUILD}/obj_affine4/Vtb_qfit_affine4_projection_top" \
  | tee "${OUT}/verilator_run.log"

verilator --lint-only --Wall -Wno-fatal \
  --top-module qfit_affine4_projection_top \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv" \
  >"${OUT}/verilator_lint.log" 2>&1

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv \
    ${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv \
    ${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv;
  hierarchy -top qfit_affine4_projection_top;
  proc; opt; memory_collect; memory_dff; opt;
  check -assert;
  tee -o ${OUT}/yosys_stat.json stat -json;
  write_json ${OUT}/yosys_netlist.json
"

grep -q '^PASS qfit_affine4_projection masks=32 ' \
  "${OUT}/icarus_run.log"
grep -q '^PASS qfit_affine4_projection masks=32 ' \
  "${OUT}/verilator_run.log"
if grep -Eq '^%Warning|^%Error|Assertion failed|FAIL|FATAL' \
  "${OUT}/verilator_build.log" \
  "${OUT}/verilator_run.log"; then
  printf 'Verilator warning/error/assertion marker found\n' >&2
  exit 1
fi
if [[ -s "${OUT}/verilator_lint.log" ]]; then
  printf 'Focused Verilator lint log is not empty\n' >&2
  exit 1
fi
grep -q 'Found and reported 0 problems.' "${OUT}/yosys.log"

sha256sum \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv" \
  "${ROOT}/tb_qfit/tb_qfit_affine4_projection_top.sv" \
  "${ROOT}/verif_qfit/qfit_affine4_projection_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/sim_qfit/run_qfit_affine4_projection_checks.sh" \
  >"${OUT}/input_sha256.txt"

{
  printf 'Icarus exact TB\tPASS\n'
  printf 'Verilator dynamic SVA exact TB\tPASS\n'
  printf 'Verilator focused lint zero-warning\tPASS\n'
  printf 'Yosys hierarchy/check/stat\tPASS\n'
} >"${OUT}/status.tsv"

printf 'PASS qfit Affine-4 exact-replay checks\n'
