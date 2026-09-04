#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS="$ROOT/third_party/OpenROAD-flow-scripts"
FLOW="$ORFS/flow"
WORK="$ROOT/openroad_hifp/work"
CONFIG="$ROOT/openroad_hifp/config_rqtb_2s_t450.mk"

expected_commit="$(sed -n 's/^ORFS_COMMIT=//p' "$ROOT/openroad_hifp/ORFS_VERSION.lock")"
if [[ "$(git -C "$ORFS" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "ERROR: ORFS commit does not match ORFS_VERSION.lock" >&2
  exit 2
fi

run_mode() {
  local name="$1"
  local quotient="$2"
  local report_log="$WORK/logs/nangate45/h67_rqtb_2s_t450_flopmem_proxy/$name/6_report.log"
  make -C "$FLOW" \
    DESIGN_CONFIG="$CONFIG" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$name" \
    VERILOG_TOP_PARAMS="QUOTIENT_ENABLE $quotient" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    route

  make -C "$FLOW" \
    DESIGN_CONFIG="$CONFIG" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$name" \
    VERILOG_TOP_PARAMS="QUOTIENT_ENABLE $quotient" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    "$report_log"
}

case "${1:-all}" in
  fixed2) run_mode fixed2 0 ;;
  rqtb2) run_mode rqtb2 1 ;;
  all)
    "$0" fixed2
    "$0" rqtb2
    ;;
  *)
    echo "usage: $0 [fixed2|rqtb2|all]" >&2
    exit 2
    ;;
esac
