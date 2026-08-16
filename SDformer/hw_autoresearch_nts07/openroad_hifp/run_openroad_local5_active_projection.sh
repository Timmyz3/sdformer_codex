#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS="$ROOT/third_party/OpenROAD-flow-scripts"
FLOW="$ORFS/flow"
CONFIG="$ROOT/openroad_hifp/config_local5_active_projection.mk"
WORK="$ROOT/openroad_hifp/work"
DESIGN="local5_active_projection_t450_proxy"

if [[ "$(git -C "$ORFS" rev-parse HEAD)" != "3a0a1efd1d8d7891de1c4961487eaf6288adf7df" ]]; then
  echo "ERROR: ORFS commit does not match ORFS_VERSION.lock" >&2
  exit 2
fi

run_mode() {
  local mode="$1"
  local kind="$2"
  local target="${3:-route}"
  local params
  params="HEIGHT 15 WIDTH 15 TIME_PLANES 2 HEAD_DIM 32 OUT_DIM 2 BACKEND_KIND $kind RELATION_READ_LATENCY 1"
  make -C "$FLOW" \
    DESIGN_CONFIG="$CONFIG" \
    WORK_HOME="$WORK" \
    FLOW_VARIANT="$mode" \
    VERILOG_TOP_PARAMS="$params" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    "$target"
}

case "${1:-all}" in
  tcfm5) run_mode tcfm5 0 "${2:-route}" ;;
  linear5) run_mode linear5 1 "${2:-route}" ;;
  all)
    run_mode tcfm5 0 "${2:-route}"
    run_mode linear5 1 "${2:-route}"
    ;;
  *)
    echo "usage: $0 [tcfm5|linear5|all] [synth|route|report]" >&2
    exit 2
    ;;
esac
