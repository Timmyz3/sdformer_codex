#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS="$ROOT/third_party/OpenROAD-flow-scripts"
FLOW="$ORFS/flow"
WORK="$ROOT/openroad_hifp/work"

if [[ "$(git -C "$ORFS" rev-parse HEAD)" != "3a0a1efd1d8d7891de1c4961487eaf6288adf7df" ]]; then
  echo "ERROR: ORFS commit does not match ORFS_VERSION.lock" >&2
  exit 2
fi

run_block() {
  local config="$1"
  local design="$2"
  local variant="$3"
  local params="$4"
  local report_target

  report_target="$WORK/logs/nangate45/$design/$variant/6_report.log"
  make -C "$FLOW" \
    DESIGN_CONFIG="$ROOT/openroad_hifp/$config" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$variant" \
    VERILOG_TOP_PARAMS="$params" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    route
  make -C "$FLOW" \
    DESIGN_CONFIG="$ROOT/openroad_hifp/$config" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$variant" \
    VERILOG_TOP_PARAMS="$params" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    "$report_target"
}

case "${1:-all}" in
  datapath_scalar)
    run_block config_datapath.mk hifp_dctf96_datapath_t6 \
      scalar "TOKENS 6 ADAPTER_CONTEXTS 2 PPDI_ENABLE 0"
    ;;
  datapath_ppdi)
    run_block config_datapath.mk hifp_dctf96_datapath_t6 \
      ppdi "TOKENS 6 ADAPTER_CONTEXTS 2 PPDI_ENABLE 1"
    ;;
  accumulator_rmw)
    run_block config_acc_rmw.mk hifp_accumulator_t6 \
      rmw "TOKENS 6 BANKS 2 OUT_TILE 32"
    ;;
  accumulator_ibf)
    run_block config_acc_ibf.mk hifp_accumulator_t6 \
      ibf "TOKENS 6 BANKS 2 OUT_TILE 32"
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
