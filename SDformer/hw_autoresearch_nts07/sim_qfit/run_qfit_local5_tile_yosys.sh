#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/qfit_local5_tile_yosys_20260730"
SOURCES=(
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
)
mkdir -p "${OUT}"

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '参数\tHEIGHT=15 WIDTH=15 TIME_PLANES=2 GATE_W=9\n'
  printf '架构\tXBF-DBDR score + FCSR-RX synchronous relation transpose\n'
  printf '约束\t无SDC、无Liberty，不是DC/STA/SAIF\n'
} >"${OUT}/reproducibility_manifest.tsv"
sha256sum "${SOURCES[@]}" >"${OUT}/source_sha256.txt"

yosys -q -l "${OUT}/tile_yosys.log" -p "
  read_verilog -sv ${SOURCES[*]};
  chparam -set HEIGHT 15 -set WIDTH 15 -set TIME_PLANES 2 -set GATE_W 9 qfit_local5_tile;
  hierarchy -top qfit_local5_tile;
  proc;
  flatten;
  opt;
  memory_collect;
  check -assert;
  tee -o ${OUT}/tile_memory_stat.json stat -json;
  memory_map;
  setundef -zero;
  opt;
  techmap;
  opt;
  tee -o ${OUT}/tile_flat_stat.json stat -json
"

{
  printf 'tile宏阶段check_assert\tPASS\n'
  printf 'tile打平结构统计\tPASS\n'
} >"${OUT}/yosys_status.tsv"
printf 'PASS yosys qfit_local5_tile\n'
