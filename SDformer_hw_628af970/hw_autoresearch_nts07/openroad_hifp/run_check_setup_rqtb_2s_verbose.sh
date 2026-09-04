#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/openroad_hifp/work"
DESIGN="h67_rqtb_2s_t450_flopmem_proxy"
CHECK_TCL="$ROOT/openroad_hifp/check_setup_verbose.tcl"

run_check() {
  local mode="$1"
  local base="$WORK/results/nangate45/$DESIGN/$mode"
  local object="$WORK/objects/nangate45/$DESIGN/$mode"
  local log="$WORK/logs/nangate45/$DESIGN/$mode/check_setup_verbose.log"

  ODB_FILE="$base/6_1_fill.odb" \
  SDC_FILE="$base/6_1_fill.sdc" \
  LIB_FILE="$object/lib/NangateOpenCellLibrary_typical.lib" \
    openroad -exit -no_init "$CHECK_TCL" 2>&1 | tee "$log"
}

case "${1:-all}" in
  fixed2) run_check fixed2 ;;
  rqtb2) run_check rqtb2 ;;
  all)
    "$0" fixed2
    "$0" rqtb2
    ;;
  *)
    echo "usage: $0 [fixed2|rqtb2|all]" >&2
    exit 2
    ;;
esac
