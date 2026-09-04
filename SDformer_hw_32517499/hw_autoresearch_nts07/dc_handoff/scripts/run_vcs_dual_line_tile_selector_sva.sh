#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYN_ROOT="${SYN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SYNOPSYS_ENV:-/home/zhumd/work/synopsys_date_dual/env.sh}"
export PATH="/opt/synopsys/vcs/V-2023.12-SP1/bin:${PATH}"
export VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
OUT="${OUTPUT_DIR:-${SYN_ROOT}/runs/dual_line_tile_selector_vcs_sva_20260821}"
mkdir -p "${OUT}"

cd "${OUT}"
vcs -full64 -sverilog +v2k -timescale=1ns/1ps -assert svaext \
  -top tb_qfit_dual_line_tile_selector -o simv \
  "${SOURCE_ROOT}/rtl_qfit/qfit_dual_line_tile_selector.sv" \
  "${SOURCE_ROOT}/verif_qfit/qfit_dual_line_tile_selector_assertions.sv" \
  "${SOURCE_ROOT}/tb_qfit/tb_qfit_dual_line_tile_selector.sv" \
  2>&1 | tee compile.log

./simv 2>&1 | tee simulation.log
grep -q 'PASS dual-line selector requests=20000' simulation.log
if grep -Eq 'Error-|Assertion.*failed|\$error|\$fatal' simulation.log; then
  echo "Unexpected VCS/SVA error in dual-line selector" >&2
  exit 1
fi
echo "PASS VCS/SVA dual-line tile selector"
