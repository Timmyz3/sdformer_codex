#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_et3"
RTL=(
  "${ROOT}/rtl_et3/et3_bounded_term_directory.sv"
  "${ROOT}/rtl_et3/et3_native_multiset_executor.sv"
  "${ROOT}/rtl_et3/et3_native_slice_top.sv"
)
BASELINE_RTL="${ROOT}/rtl_et3/et3_native_m_queue_baseline.sv"
TB="${ROOT}/tb_et3/tb_et3_native_slice.sv"
BASELINE_TB="${ROOT}/tb_et3/tb_et3_native_m_queue_baseline.sv"
SVA="${ROOT}/verif_et3/et3_native_slice_assertions.sv"
TORCH_PYTHON="/opt/conda/envs/sdformerflow/bin/python"

mkdir -p "${BUILD}/iverilog" "${BUILD}/verilator"

echo "[1/9] Python evidence/replay tests"
if [[ -x "${TORCH_PYTHON}" ]]; then
  "${TORCH_PYTHON}" -m unittest \
    "${ROOT}/tests/test_reconsider_rejected_dual_line_ideas.py" \
    "${ROOT}/tests/test_et3_ordered_trace_replay.py" \
    "${ROOT}/tests/test_local5_ordered_trace_sink.py" -v
else
  echo "ERROR: missing torch Python at ${TORCH_PYTHON}" >&2
  exit 1
fi

echo "[2/9] Icarus ET3 functional simulation"
iverilog -g2012 -Wall -s tb_et3_native_slice \
  -o "${BUILD}/iverilog/tb_et3_native_slice.vvp" \
  "${RTL[@]}" "${TB}"
vvp "${BUILD}/iverilog/tb_et3_native_slice.vvp"

echo "[3/9] Icarus native-m baseline simulation"
iverilog -g2012 -Wall -s tb_et3_native_m_queue_baseline \
  -o "${BUILD}/iverilog/tb_et3_native_m_queue_baseline.vvp" \
  "${ROOT}/rtl_et3/et3_native_multiset_executor.sv" \
  "${BASELINE_RTL}" "${BASELINE_TB}"
vvp "${BUILD}/iverilog/tb_et3_native_m_queue_baseline.vvp"

echo "[4/9] Verilator ET3 lint"
verilator --lint-only --timing --assert -Wall -Wno-fatal \
  --top-module et3_native_slice_top "${RTL[@]}"

echo "[5/9] Verilator native-m baseline lint"
verilator --lint-only --timing --assert -Wall -Wno-fatal \
  --top-module et3_native_m_queue_baseline \
  "${ROOT}/rtl_et3/et3_native_multiset_executor.sv" "${BASELINE_RTL}"

echo "[6/9] Verilator ET3 functional simulation with SVA"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "${BUILD}/verilator" \
  --top-module tb_et3_native_slice "${RTL[@]}" "${SVA}" "${TB}"
"${BUILD}/verilator/Vtb_et3_native_slice"

echo "[7/9] Verilator native-m functional simulation with SVA"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "${BUILD}/verilator_baseline" \
  --top-module tb_et3_native_m_queue_baseline \
  "${ROOT}/rtl_et3/et3_native_multiset_executor.sv" \
  "${BASELINE_RTL}" "${SVA}" "${BASELINE_TB}"
"${BUILD}/verilator_baseline/Vtb_et3_native_m_queue_baseline"

echo "[8/9] Yosys ET3 synthesis-readiness check"
yosys -q -p "
  read_verilog -sv ${RTL[*]};
  hierarchy -check -top et3_native_slice_top;
  proc;
  opt;
  memory;
  opt;
  check;
  stat;
" | tee "${BUILD}/yosys_et3_native_slice.log"

echo "[9/9] Yosys native-m baseline synthesis-readiness check"
yosys -q -p "
  read_verilog -sv \
    ${ROOT}/rtl_et3/et3_native_multiset_executor.sv \
    ${BASELINE_RTL};
  hierarchy -check -top et3_native_m_queue_baseline;
  proc;
  opt;
  memory;
  opt;
  check;
  stat;
" | tee "${BUILD}/yosys_et3_native_m_queue_baseline.log"

echo "PASS: ET3 native SET/MULTISET slice full checks"
