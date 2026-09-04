#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESIGN_NAME="gatestack_single_context_execution_top"
CSR_MODE="${CSR_FORMAT_FADC24:-0}"
RESIDENCY="${ENABLE_RESIDENCY:-1}"
FILELIST="$ROOT/rtl_hitflow/filelist_single_context_execution.f"
OUT="$ROOT/dc_handoff/runs/yosys_structure/${DESIGN_NAME}_csr${CSR_MODE}_res${RESIDENCY}"
mkdir -p "$OUT"
RTL_FILES="$(tr '\n' ' ' < "$FILELIST")"

# 保留逻辑memory，避免把head-slot/cache/accumulator错误展开成DFF和大MUX。
yosys -Q -p "read_verilog -sv ${RTL_FILES}; chparam -set CSR_FORMAT_FADC24 ${CSR_MODE} -set ENABLE_RESIDENCY ${RESIDENCY} ${DESIGN_NAME}; hierarchy -check -top ${DESIGN_NAME}; proc; opt; memory_collect; opt_clean; check -assert; tee -o ${OUT}/stat.json stat -json; write_verilog -noattr ${OUT}/${DESIGN_NAME}_structure.v" \
  >"$OUT/yosys.log"
test -s "$OUT/${DESIGN_NAME}_structure.v"
if rg -n "ERROR:|Found and reported [1-9][0-9]* problems" "$OUT/yosys.log"; then
  echo "GateStack Yosys结构综合失败" >&2
  exit 1
fi
echo "$OUT"
