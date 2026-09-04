#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS="$ROOT/third_party/OpenROAD-flow-scripts"
FLOW="$ORFS/flow"
WORK="$ROOT/openroad_hifp/work"
CONFIG="$ROOT/openroad_hifp/config_h67_zkqi_threeway_macro.mk"

expected_commit="$(sed -n 's/^ORFS_COMMIT=//p' "$ROOT/openroad_hifp/ORFS_VERSION.lock")"
if [[ "$(git -C "$ORFS" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "ERROR: ORFS commit does not match ORFS_VERSION.lock" >&2
  exit 2
fi

run_mode() {
  local name="$1"
  local zk="$2"
  local bundle="$3"
  local params="ZK_BYPASS_ENABLE $zk BUNDLE_SKIP_ENABLE $bundle ROW_MEMORY_IMPL 1 DIRECTORY_MEMORY_IMPL 1"
  local report_log="$WORK/logs/nangate45/h67_zkqi_threeway_manualmacro_proxy/$name/6_report.log"

  make -C "$FLOW" \
    DESIGN_CONFIG="$CONFIG" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$name" \
    VERILOG_TOP_PARAMS="$params" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    route

  make -C "$FLOW" \
    DESIGN_CONFIG="$CONFIG" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$name" \
    VERILOG_TOP_PARAMS="$params" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    "$report_log"
}

case "${1:-all}" in
  baseline) run_mode baseline 0 0 ;;
  pairbitmap) run_mode pairbitmap 1 0 ;;
  ttb8) run_mode ttb8 1 1 ;;
  all)
    "$0" baseline
    "$0" pairbitmap
    "$0" ttb8
    ;;
  *)
    echo "usage: $0 [baseline|pairbitmap|ttb8|all]" >&2
    exit 2
    ;;
esac
