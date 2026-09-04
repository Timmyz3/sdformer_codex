#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/openroad_hifp/work"
LIB_BASE="$WORK/objects/nangate45"

run_check() {
  local design="$1"
  local variant="$2"
  local result_stage="$3"
  local object_stage="$4"
  local log_dir="$WORK/logs/nangate45/$design/$variant"

  ODB_FILE="$WORK/results/nangate45/$design/$variant/$result_stage.odb" \
  SDC_FILE="$WORK/results/nangate45/$design/$variant/$object_stage.sdc" \
  LIB_FILE="$LIB_BASE/$design/$variant/lib/NangateOpenCellLibrary_typical.lib" \
    openroad -exit -no_init "$ROOT/openroad_hifp/check_setup_verbose.tcl" \
    2>&1 | tee "$log_dir/check_setup_verbose.log"
}

case "${1:-all}" in
  datapath_scalar)
    run_check hifp_dctf96_datapath_t6 scalar 6_1_fill 6_1_fill
    ;;
  datapath_ppdi)
    run_check hifp_dctf96_datapath_t6 ppdi 6_1_fill 6_1_fill
    ;;
  accumulator_rmw)
    run_check hifp_accumulator_t6 rmw 6_1_fill 6_1_fill
    ;;
  accumulator_ibf)
    run_check hifp_accumulator_t6 ibf 6_1_fill 6_1_fill
    ;;
  all)
    "$0" datapath_scalar
    "$0" datapath_ppdi
    "$0" accumulator_rmw
    "$0" accumulator_ibf
    ;;
  *)
    echo "usage: $0 [datapath_scalar|datapath_ppdi|accumulator_rmw|accumulator_ibf|all]" >&2
    exit 2
    ;;
esac
