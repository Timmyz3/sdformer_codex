#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESIGN_NAME="gatestack_single_context_execution_top"
CSR_MODE="${CSR_FORMAT_FADC24:-0}"
RESIDENCY="${ENABLE_RESIDENCY:-1}"
FILELIST="$ROOT/rtl_hitflow/filelist_single_context_execution.f"
OUT="$ROOT/dc_handoff/runs/yosys_structure/${DESIGN_NAME}_csr${CSR_MODE}_res${RESIDENCY}"
NETLIST="$OUT/${DESIGN_NAME}_structure.v"
LEC_TIMEOUT_SECONDS="${LEC_TIMEOUT_SECONDS:-600}"
test -s "$NETLIST"
RTL_FILES="$(tr '\n' ' ' < "$FILELIST")"

set +e
timeout "$LEC_TIMEOUT_SECONDS" yosys -Q -p "read_verilog -sv ${RTL_FILES}; chparam -set CSR_FORMAT_FADC24 ${CSR_MODE} -set ENABLE_RESIDENCY ${RESIDENCY} ${DESIGN_NAME}; prep -top ${DESIGN_NAME}; memory_collect; opt_clean; design -stash gold; read_verilog -sv ${NETLIST}; prep -top ${DESIGN_NAME}; memory_collect; opt_clean; design -stash gate; design -copy-from gold -as gold ${DESIGN_NAME}; design -copy-from gate -as gate ${DESIGN_NAME}; equiv_make -inames gold gate equiv; hierarchy -top equiv; equiv_simple -short -seq 1; equiv_induct -seq 4; equiv_status -assert" \
  >"$OUT/lec.log" 2>&1
status=$?
set -e
if [[ $status -eq 124 ]]; then
  echo "GateStack Yosys LEC在${LEC_TIMEOUT_SECONDS}s超时；不得标记等价通过。" >&2
  exit 124
fi
if [[ $status -ne 0 ]] || ! rg -q "Equivalence successfully proven|Equivalence successfully proved" "$OUT/lec.log"; then
  echo "GateStack Yosys LEC未通过；Formality仍是正式门槛。" >&2
  tail -80 "$OUT/lec.log" >&2
  exit 1
fi
echo "$OUT/lec.log"
