#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS="$ROOT/third_party/OpenROAD-flow-scripts"
FLOW="$ORFS/flow"
CONFIG="$ROOT/openroad_hifp/config.mk"
WORK="$ROOT/openroad_hifp/work"
DESIGN="hifp_projection_t6_proxy"

if [[ ! -x /usr/bin/openroad ]]; then
  echo "ERROR: /usr/bin/openroad is unavailable" >&2
  exit 2
fi
if [[ "$(git -C "$ORFS" rev-parse HEAD)" != "3a0a1efd1d8d7891de1c4961487eaf6288adf7df" ]]; then
  echo "ERROR: ORFS commit does not match ORFS_VERSION.lock" >&2
  exit 2
fi

run_mode() {
  local name="$1"
  local ppdi="$2"
  local ibf="$3"
  local params
  local report_target

  params="TOKENS 6 ADAPTER_CONTEXTS 2 PPDI_ENABLE $ppdi IMPLICIT_BIAS_FINALIZE_ENABLE $ibf"
  report_target="$WORK/logs/nangate45/$DESIGN/$name/6_report.log"

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
    "$report_target"
}

case "${1:-all}" in
  scalar_rmw) run_mode scalar_rmw 0 0 ;;
  ppdi_rmw) run_mode ppdi_rmw 1 0 ;;
  scalar_ibf) run_mode scalar_ibf 0 1 ;;
  ppdi_ibf) run_mode ppdi_ibf 1 1 ;;
  all)
    run_mode scalar_rmw 0 0
    run_mode ppdi_rmw 1 0
    run_mode scalar_ibf 0 1
    run_mode ppdi_ibf 1 1
    ;;
  *)
    echo "usage: $0 [scalar_rmw|ppdi_rmw|scalar_ibf|ppdi_ibf|all]" >&2
    exit 2
    ;;
esac
