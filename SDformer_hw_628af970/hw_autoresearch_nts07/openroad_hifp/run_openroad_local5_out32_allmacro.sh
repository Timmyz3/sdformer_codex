#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS="${ROOT}/third_party/OpenROAD-flow-scripts"
FLOW="${ORFS}/flow"
CONFIG="${ROOT}/openroad_hifp/config_local5_out32_allmacro.mk"
WORK="${ROOT}/openroad_hifp/work"

if [[ "$(git -C "${ORFS}" rev-parse HEAD)" != "3a0a1efd1d8d7891de1c4961487eaf6288adf7df" ]]; then
  echo "ERROR: ORFS commit does not match ORFS_VERSION.lock" >&2
  exit 2
fi

run_mode() {
  local name="$1"
  local mode="$2"
  local sync_mode="$3"
  local target="${4:-route}"
  local params
  params="MODE ${mode} GEOMETRY_SYNC_MODE ${sync_mode} HEIGHT 15 WIDTH 15 TIME_PLANES 2 HEAD_DIM 32 OUT_DIM 32 GATE_W 9 W_W 8 ACC_W 32 RELATION_READ_LATENCY 1 RELATION_MEMORY_IMPL 1 ACC_MEMORY_IMPL 1"
  mkdir -p "${WORK}/logs/nangate45/local5_out32_allmacro_proxy/${name}"
  make -C "${FLOW}" \
    DESIGN_CONFIG="${CONFIG}" \
    WORK_HOME="${WORK}" \
    FLOW_VARIANT="${name}" \
    VERILOG_TOP_PARAMS="${params}" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    "${target}"
}

run_diagnose() {
  local name="$1"
  local mode="$2"
  local sync_mode="$3"
  local params
  local diagnostic_tcl="${ROOT}/openroad_hifp/diagnose_local5_constraints.tcl"
  params="MODE ${mode} GEOMETRY_SYNC_MODE ${sync_mode} HEIGHT 15 WIDTH 15 TIME_PLANES 2 HEAD_DIM 32 OUT_DIM 32 GATE_W 9 W_W 8 ACC_W 32 RELATION_READ_LATENCY 1 RELATION_MEMORY_IMPL 1 ACC_MEMORY_IMPL 1"
  make -C "${FLOW}" \
    DESIGN_CONFIG="${CONFIG}" \
    WORK_HOME="${WORK}" \
    FLOW_VARIANT="${name}" \
    VERILOG_TOP_PARAMS="${params}" \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    --eval=".PHONY: local5-diagnose" \
    --eval="local5-diagnose: ; \$(OPENROAD_CMD) ${diagnostic_tcl}" \
    local5-diagnose 2>&1 \
    | tee "${WORK}/logs/nangate45/local5_out32_allmacro_proxy/${name}/constraint_audit.log"
}

run_macro_diagnose() {
  local diagnostic_tcl="${ROOT}/openroad_hifp/diagnose_local5_macro_orient.tcl"
  make -C "${FLOW}" \
    DESIGN_CONFIG="${CONFIG}" \
    WORK_HOME="${WORK}" \
    FLOW_VARIANT=direct \
    OPENROAD_EXE=/usr/bin/openroad \
    YOSYS_EXE=/usr/bin/yosys \
    --eval=".PHONY: local5-macro-diagnose" \
    --eval="local5-macro-diagnose: ; \$(OPENROAD_CMD) ${diagnostic_tcl}" \
    local5-macro-diagnose 2>&1 \
    | tee "${WORK}/logs/nangate45/local5_out32_allmacro_proxy/direct/macro_orient_audit.log"
}

case "${1:-all}" in
  direct) run_mode direct 0 1 "${2:-route}" ;;
  issue) run_mode issue 1 0 "${2:-route}" ;;
  ds) run_mode ds 1 1 "${2:-route}" ;;
  diagnose-direct) run_diagnose direct 0 1 ;;
  diagnose-macro) run_macro_diagnose ;;
  all)
    run_mode direct 0 1 "${2:-route}"
    run_mode issue 1 0 "${2:-route}"
    run_mode ds 1 1 "${2:-route}"
    ;;
  *)
    echo "usage: $0 [direct|issue|ds|all] [synth|route|report] | diagnose-direct | diagnose-macro" >&2
    exit 2
    ;;
esac
