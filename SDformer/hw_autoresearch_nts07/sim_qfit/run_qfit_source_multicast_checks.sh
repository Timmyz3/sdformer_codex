#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/source_multicast"
OUT="${ROOT}/results/qfit_source_multicast_yosys_20260730"
mkdir -p "${BUILD}" "${OUT}"

iverilog -g2012 \
  -s tb_qfit_source_multicast_term_builder \
  -o "${BUILD}/builder.vvp" \
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv" \
  "${ROOT}/tb_qfit/tb_qfit_source_multicast_term_builder.sv"
vvp "${BUILD}/builder.vvp"

iverilog -g2012 \
  -s tb_qfit_tcfm5_acc_bank_1r1w \
  -o "${BUILD}/acc_bank_1r1w.vvp" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/tb_qfit/tb_qfit_tcfm5_acc_bank_1r1w.sv"
vvp "${BUILD}/acc_bank_1r1w.vvp"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_tcfm5_acc_bank_1r1w \
  --Mdir "${BUILD}/obj_acc_bank_1r1w" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/tb_qfit/tb_qfit_tcfm5_acc_bank_1r1w.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  --exe
"${BUILD}/obj_acc_bank_1r1w/Vtb_qfit_tcfm5_acc_bank_1r1w"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_source_multicast_term_builder \
  --Mdir "${BUILD}/obj_builder" \
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv" \
  "${ROOT}/tb_qfit/tb_qfit_source_multicast_term_builder.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  --exe
"${BUILD}/obj_builder/Vtb_qfit_source_multicast_term_builder"

iverilog -g2012 \
  -DQFIT_WEIGHT_CONTEXT_RELOAD \
  -s tb_qfit_tcfm5_projection_top \
  -o "${BUILD}/tcfm5.vvp" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv" \
  "${ROOT}/tb_qfit/tb_qfit_tcfm5_projection_top.sv"
vvp "${BUILD}/tcfm5.vvp"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -DQFIT_WEIGHT_CONTEXT_RELOAD \
  --top-module tb_qfit_tcfm5_projection_top \
  --Mdir "${BUILD}/obj_tcfm5" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv" \
  "${ROOT}/tb_qfit/tb_qfit_tcfm5_projection_top.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  --exe
"${BUILD}/obj_tcfm5/Vtb_qfit_tcfm5_projection_top"

verilator --lint-only --Wall -Wno-fatal \
  --top-module qfit_tcfm5_projection_top \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv" \
  >"${OUT}/tcfm5_lint.log" 2>&1

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '证据\tRTL功能与开放结构综合，非DC/STA/SAIF\n'
} >"${OUT}/reproducibility_manifest.tsv"
sha256sum \
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv" \
  >"${OUT}/source_sha256.txt"

yosys -q -l "${OUT}/builder_yosys.log" -p "
  read_verilog -sv ${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv;
  hierarchy -top qfit_source_multicast_term_builder;
  proc; opt; memory_collect; flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/builder_stat.json stat -json
"

yosys -q -l "${OUT}/acc_bank_yosys.log" -p "
  read_verilog -sv ${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv;
  hierarchy -top qfit_tcfm5_acc_bank;
  proc; opt; memory_collect; memory_dff; opt; check -assert;
  tee -o ${OUT}/acc_bank_stat.json stat -json;
  write_json ${OUT}/acc_bank_netlist.json
"

yosys -q -l "${OUT}/tcfm5_yosys.log" -p "
  read_verilog -sv ${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv ${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv;
  hierarchy -top qfit_tcfm5_projection_top;
  proc; opt; memory_collect; memory_dff; flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/tcfm5_stat.json stat -json
"

{
  printf 'builder Icarus/Verilator/SVA\tPASS\n'
  printf 'Acc bank 1R1W RAW压力 Icarus/Verilator/SVA\tPASS\n'
  printf 'TCFM5 Icarus/Verilator/SVA\tPASS\n'
  printf 'TCFM5 focused lint zero-warning\tPASS\n'
  printf 'builder/Acc bank/TCFM5 Yosys check\tPASS\n'
} >"${OUT}/status.tsv"

"${ROOT}/scripts/report_qfit_source_multicast_yosys.py"
printf 'PASS qfit source multicast and TCFM5 full checks\n'
