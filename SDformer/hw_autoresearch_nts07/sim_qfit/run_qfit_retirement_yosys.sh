#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/qfit_retirement_yosys_20260730"
SOURCE="${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
mkdir -p "${OUT}"

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '参数\tHEIGHT=15 WIDTH=15 TIME_PLANES=2\n'
  printf '流程\tproc; flatten; opt; memory; opt; techmap; opt\n'
  printf '约束\t无SDC、无Liberty、generic cell趋势，不是DC/STA\n'
  printf 'MODE0\tFCSR闭式逐源退休\n'
  printf 'MODE1\tDynamic Frontier逐源计数\n'
  printf 'MODE2\tNonblocking Stripe双行上下文\n'
} >"${OUT}/reproducibility_manifest.tsv"
sha256sum "${SOURCE}" >"${OUT}/source_sha256.txt"

for mode in 0 1 2; do
  case "${mode}" in
    0) name="fcsr" ;;
    1) name="dynamic_frontier" ;;
    2) name="nonblocking_stripe" ;;
  esac
  yosys -q -l "${OUT}/${name}_yosys.log" -p "
    read_verilog -sv ${SOURCE};
    chparam -set MODE ${mode} -set HEIGHT 15 -set WIDTH 15 -set TIME_PLANES 2 qfit_retirement_scheduler;
    hierarchy -top qfit_retirement_scheduler;
    proc;
    flatten;
    opt;
    memory;
    opt;
    techmap;
    opt;
    check -assert;
    tee -o ${OUT}/${name}_stat.json stat -json
  "
  printf 'PASS yosys %s\n' "${name}"
done

{
  printf '三模式综合\tPASS\n'
  printf 'check_assert\tPASS\n'
} >"${OUT}/yosys_status.tsv"
