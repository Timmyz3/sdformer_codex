#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/lane_product_cache"
OUT="${ROOT}/results/qfit_lane_product_cache_20260731"
mkdir -p "${BUILD}" "${OUT}"

RTL="${ROOT}/rtl_qfit/qfit_lane_product_cache_leaf.sv"
BANK="${ROOT}/rtl_qfit/qfit_sync_1rw_bank.sv"
MUL="${ROOT}/rtl_qfit/qfit_narrow_gate_weight_mul.sv"
TB="${ROOT}/tb_qfit/tb_qfit_lane_product_cache_leaf.sv"
TRACE_TB="${ROOT}/tb_qfit/tb_qfit_lane_product_cache_trace.sv"
SVA="${ROOT}/verif_qfit/qfit_lane_product_cache_assertions.sv"
STRUCTURE_CHECK="${ROOT}/scripts/check_qfit_lane_product_cache_structure.py"
POLICY_MODEL="${ROOT}/scripts/evaluate_qfit_product_cache_policies.py"
TRACE_CSV="${ROOT}/results/qfit_local5_projection_tile_yosys_20260731/ordered_term_trace.csv"

policy_value() {
  local ways="$1"
  local policy="$2"
  local key="$3"
  python3 - "${TRACE_CSV}" "${ways}" "${policy}" "${key}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[2]))
from scripts.evaluate_qfit_product_cache_policies import (
    load_rows,
    simulate_lru,
    simulate_no_replace,
)

rows = load_rows(Path(sys.argv[1]))
result = (
    simulate_lru(rows, int(sys.argv[2]))
    if sys.argv[3] == "lru"
    else simulate_no_replace(rows, int(sys.argv[2]))
)
if sys.argv[4] == "writes":
    print(result.get("fills", result["product_starts"]))
else:
    print(result["product_starts"])
PY
}

for ways in 4 6 8; do
  iverilog -g2012 -s tb_qfit_lane_product_cache_leaf \
    -Ptb_qfit_lane_product_cache_leaf.WAYS="${ways}" \
    -o "${BUILD}/cache_w${ways}_iv" "${BANK}" "${MUL}" "${RTL}" "${TB}"
  vvp "${BUILD}/cache_w${ways}_iv" \
    | tee "${OUT}/iverilog_w${ways}.log"
done

for ways in 4 6 8; do
  iverilog -g2012 -s tb_qfit_lane_product_cache_trace \
    -Ptb_qfit_lane_product_cache_trace.WAYS="${ways}" \
    -o "${BUILD}/cache_trace_w${ways}_iv" \
    "${BANK}" "${MUL}" "${RTL}" "${TRACE_TB}"
  vvp "${BUILD}/cache_trace_w${ways}_iv" \
    "+TRACE_CSV=${TRACE_CSV}" \
    "+EXPECTED_MISSES=$(policy_value "${ways}" lru misses)" \
    "+EXPECTED_WRITES=$(policy_value "${ways}" lru writes)" \
    | tee "${OUT}/trace_w${ways}.log"
done

for ways in 4 6 8; do
  iverilog -g2012 -s tb_qfit_lane_product_cache_trace \
    -Ptb_qfit_lane_product_cache_trace.WAYS="${ways}" \
    -Ptb_qfit_lane_product_cache_trace.NO_REPLACE=1 \
    -o "${BUILD}/first_bind_trace_w${ways}_iv" \
    "${BANK}" "${MUL}" "${RTL}" "${TRACE_TB}"
  vvp "${BUILD}/first_bind_trace_w${ways}_iv" \
    "+TRACE_CSV=${TRACE_CSV}" \
    "+EXPECTED_MISSES=$(policy_value "${ways}" first_bind misses)" \
    "+EXPECTED_WRITES=$(policy_value "${ways}" first_bind writes)" \
    | tee "${OUT}/first_bind_trace_w${ways}.log"
done

for ways in 4 6 8; do
  rm -rf "${BUILD}/obj_w${ways}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_lane_product_cache_leaf \
    -GWAYS="${ways}" \
    --Mdir "${BUILD}/obj_w${ways}" \
    "${BANK}" "${MUL}" "${RTL}" "${SVA}" "${TB}"
  "${BUILD}/obj_w${ways}/Vtb_qfit_lane_product_cache_leaf" \
    | tee "${OUT}/verilator_w${ways}.log"

  verilator --lint-only --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_lane_product_cache_leaf \
    -GWAYS="${ways}" \
    "${BANK}" "${MUL}" "${RTL}" "${SVA}" "${TB}" \
    >"${OUT}/verilator_lint_w${ways}.log" 2>&1
done

rm -rf "${BUILD}/obj_trace_w6"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_lane_product_cache_trace \
  -GWAYS=6 \
  --Mdir "${BUILD}/obj_trace_w6" \
  "${BANK}" "${MUL}" "${RTL}" "${SVA}" "${TRACE_TB}"
"${BUILD}/obj_trace_w6/Vtb_qfit_lane_product_cache_trace" \
  "+TRACE_CSV=${TRACE_CSV}" \
  "+EXPECTED_MISSES=$(policy_value 6 lru misses)" \
  "+EXPECTED_WRITES=$(policy_value 6 lru writes)" \
  | tee "${OUT}/verilator_trace_w6.log"

for ways in 4 6; do
  rm -rf "${BUILD}/obj_first_bind_w${ways}"
  verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_qfit_lane_product_cache_trace \
    -GWAYS="${ways}" "-GNO_REPLACE=1'b1" \
    --Mdir "${BUILD}/obj_first_bind_w${ways}" \
    "${BANK}" "${MUL}" "${RTL}" "${SVA}" "${TRACE_TB}"
  "${BUILD}/obj_first_bind_w${ways}/Vtb_qfit_lane_product_cache_trace" \
    "+TRACE_CSV=${TRACE_CSV}" \
    "+EXPECTED_MISSES=$(policy_value "${ways}" first_bind misses)" \
    "+EXPECTED_WRITES=$(policy_value "${ways}" first_bind writes)" \
    | tee "${OUT}/verilator_first_bind_w${ways}.log"

  verilator --lint-only -Wall -Wno-fatal \
    --top-module qfit_lane_product_cache_leaf \
    -GWAYS="${ways}" "-GNO_REPLACE=1'b1" \
    "${BANK}" "${MUL}" "${RTL}" \
    >"${OUT}/verilator_lint_first_bind_w${ways}.log" 2>&1
done

for ways in 4 6 8; do
  yosys -q -l "${OUT}/yosys_w${ways}.log" -p "
    read_verilog -sv ${BANK} ${MUL} ${RTL};
    chparam -set WAYS ${ways} qfit_lane_product_cache_leaf;
    hierarchy -top qfit_lane_product_cache_leaf;
    proc; opt; flatten; memory_collect; memory_dff; opt; check -assert;
    tee -o ${OUT}/stat_w${ways}.json stat -json;
    write_json ${OUT}/netlist_w${ways}.json
  "
done

for ways in 4 6; do
  yosys -q -l "${OUT}/yosys_first_bind_w${ways}.log" -p "
    read_verilog -sv ${BANK} ${MUL} ${RTL};
    chparam -set WAYS ${ways} -set NO_REPLACE 1 qfit_lane_product_cache_leaf;
    hierarchy -top qfit_lane_product_cache_leaf;
    proc; opt; flatten; memory_collect; memory_dff; opt; check -assert;
    tee -o ${OUT}/stat_first_bind_w${ways}.json stat -json
  "
done

python3 "${STRUCTURE_CHECK}" \
  --netlist-pattern "${OUT}/netlist_w{ways}.json" \
  --output "${OUT}/structure_contract.json"

sha256sum \
  "${BANK}" "${MUL}" "${RTL}" "${TB}" "${TRACE_TB}" "${SVA}" \
  "${STRUCTURE_CHECK}" "${BASH_SOURCE[0]}" "${TRACE_CSV}" \
  "${POLICY_MODEL}" \
  >"${OUT}/source_sha256.txt"
printf 'Icarus W4/W6/W8 exact ordered product\tPASS\n' >"${OUT}/status.tsv"
printf 'Icarus real W6 trace W4/W6/W8 replay\tPASS\n' >>"${OUT}/status.tsv"
printf 'Icarus first-bind W4/W6/W8 same-boundary replay\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator W4/W6/W8 SVA\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator W6 real-trace SVA\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator first-bind W4/W6 real-trace SVA\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator W4/W6/W8 lint\tPASS\n' >>"${OUT}/status.tsv"
printf 'Verilator first-bind W4/W6 DUT lint\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys W4/W6/W8 synth-readable\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys product-bank/multiplier contract\tPASS\n' >>"${OUT}/status.tsv"
printf 'PASS qfit lane product cache checks\n'
