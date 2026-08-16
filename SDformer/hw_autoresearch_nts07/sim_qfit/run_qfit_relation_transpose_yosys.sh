#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/qfit_relation_transpose_yosys_20260730"
SCHED="${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
BANK="${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
LEAF="${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
mkdir -p "${OUT}"

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '参数\tHEIGHT=15 WIDTH=15 TIME_PLANES=2 K_W=32 GATE_W=9，gate bank另含1-bit valid\n'
  printf '宏口径\tproc; flatten; opt; memory_collect\n'
  printf '打平口径\tmemory_map; setundef -zero; opt; techmap; opt\n'
  printf '约束\t无SDC、无Liberty，不是DC/STA/SAIF\n'
} >"${OUT}/reproducibility_manifest.tsv"
sha256sum "${SCHED}" "${BANK}" "${LEAF}" >"${OUT}/source_sha256.txt"

for config in \
  "0 fcsr 3" \
  "1 dynamic_frontier 3" \
  "2 stripe3 3" \
  "2 stripe4 4"; do
  read -r mode name ring_rows <<<"${config}"
  yosys -q -l "${OUT}/${name}_yosys.log" -p "
    read_verilog -sv ${SCHED} ${BANK} ${LEAF};
    chparam -set SCHED_MODE ${mode} -set STRIPE_RING_ROWS ${ring_rows} -set HEIGHT 15 -set WIDTH 15 -set TIME_PLANES 2 -set K_W 32 -set GATE_W 9 qfit_relation_transpose_leaf;
    hierarchy -top qfit_relation_transpose_leaf;
    proc;
    flatten;
    opt;
    memory_collect;
    check -assert;
    tee -o ${OUT}/${name}_memory_stat.json stat -json;
    memory_map;
    setundef -zero;
    opt;
    techmap;
    opt;
    tee -o ${OUT}/${name}_flat_stat.json stat -json
  "
  printf 'PASS yosys %s\n' "${name}"
done

{
  printf '四候选宏与打平综合\tPASS\n'
  printf 'macro阶段check_assert\tPASS\n'
  printf '打平阶段\t仅结构统计，未对未复位SRAM执行check_assert\n'
} >"${OUT}/yosys_status.tsv"
