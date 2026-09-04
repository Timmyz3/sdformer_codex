#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/qfit_score_leaf_yosys_20260730"
mkdir -p "${OUT}"

SOURCES=(
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
)

synth_leaf() {
  local name="$1"
  local arch="$2"
  local pipe="$3"
  local xbf="$4"
  local use_threshold="$5"
  local threshold="$6"
  local bank_pressure="$7"
  local json="${OUT}/${name}_flat_stat.json"
  local log="${OUT}/${name}_flat_yosys.log"

  yosys -q -l "${log}" -p "
    read_verilog -sv ${SOURCES[*]};
    chparam -set ARCH_QFSA ${arch} -set PIPE_COMPACTOR ${pipe} -set XBF_BANKED ${xbf} -set USE_THRESHOLD_ROUTE ${use_threshold} -set ROUTE_THRESHOLD ${threshold} -set USE_BANK_PRESSURE_ROUTE ${bank_pressure} -set BANK_PRESSURE_THRESHOLD 2 qfit_local5_score_leaf;
    hierarchy -top qfit_local5_score_leaf;
    proc;
    flatten;
    opt;
    memory;
    opt;
    techmap;
    opt;
    check -assert;
    tee -o ${json} stat -json
  "
  printf 'PASS yosys %s\n' "${name}"
}

synth_standalone() {
  local name="$1"
  local top="$2"
  local source="$3"
  local json="${OUT}/${name}_stat.json"
  local log="${OUT}/${name}_yosys.log"

  yosys -q -l "${log}" -p "
    read_verilog -sv ${source};
    hierarchy -top ${top};
    proc;
    flatten;
    opt;
    memory;
    opt;
    techmap;
    opt;
    check -assert;
    tee -o ${json} stat -json
  "
  printf 'PASS yosys %s\n' "${name}"
}

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '流程\tproc; flatten; opt; memory; opt; techmap; opt\n'
  printf '约束\t无SDC、无Liberty、generic cell趋势，不是DC/STA\n'
  printf '变体\tARCH_QFSA\tPIPE_COMPACTOR\tXBF_BANKED\tUSE_T\tT\tB2\n'
  printf 'w1_exact\t0\t0\t0\t0\t8\t0\n'
  printf 'global_qfsa_1c\t1\t0\t0\t0\t8\t0\n'
  printf 'global_qfsa_2c\t1\t1\t0\t0\t8\t0\n'
  printf 'xbf_exact\t1\t1\t1\t0\t8\t0\n'
  printf 'w1_t8\t0\t0\t0\t1\t8\t0\n'
  printf 'xbf_t8\t1\t1\t1\t1\t8\t0\n'
  printf 'xbf_t8b2\t1\t1\t1\t1\t8\t1\n'
} >"${OUT}/reproducibility_manifest.tsv"

sha256sum "${SOURCES[@]}" >"${OUT}/source_sha256.txt"

synth_leaf w1_exact 0 0 0 0 8 0
synth_leaf global_qfsa_1c 1 0 0 0 8 0
synth_leaf global_qfsa_2c 1 1 0 0 8 0
synth_leaf xbf_exact 1 1 1 0 8 0
synth_leaf w1_t8 0 0 0 1 8 0
synth_leaf xbf_t8 1 1 1 1 8 0
synth_leaf xbf_t8b2 1 1 1 1 8 1
synth_standalone \
  qfit_tagged_compactor4 \
  qfit_tagged_compactor4 \
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
synth_standalone \
  qfit_xorbank_compactor4 \
  qfit_xorbank_compactor4 \
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"

{
  printf 'Yosys七变体\tPASS\n'
  printf 'standalone_compactor\tPASS\n'
  printf 'check_assert\tPASS\n'
} >"${OUT}/yosys_status.tsv"
