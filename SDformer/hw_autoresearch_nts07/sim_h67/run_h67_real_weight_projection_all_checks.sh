#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_ep35_real_weight_projection_all_rtl_20260814}"
VEC_DIR="${VECTOR_DIR:-$ROOT/tb_h67/vectors/h67_ep35_real_weight_projection_all_20260814}"
BASE="$ROOT/tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt"
TRACE="$ROOT/results/h67_fullres_ep35_postconvergence_t450_20260805_all12_bit_trace/manifest.json"
FAIR_REPORT="$ROOT/results/h67_fair_row_descriptor_bound_20260814/report.json"
JOINT_REPORT="$ROOT/results/h67_ep35_real_weight_projection2_rtl_20260813/report.json"

if [[ -e "$OUT" || -e "$VEC_DIR" ]]; then
  echo "ERROR: refusing to overwrite result/vector directory" >&2
  exit 2
fi
mkdir -p "$OUT/build" "$OUT/logs"

python3 "$ROOT/scripts/generate_h67_real_weight_projection_all_vectors.py" \
  --trace-manifest "$TRACE" --base-vector "$BASE" --output-dir "$VEC_DIR"

RTL=(
  "$ROOT/rtl_h67/h67_gated_k_projection16_acc.sv"
)
TB=(
  "$ROOT/tb_h67/tb_h67_real_weight_projection_all_sidecar.sv"
)
TOP=tb_h67_real_weight_projection_all_sidecar

iverilog -g2012 -Wall -s "$TOP" -o "$OUT/build/icarus.vvp" \
  "${RTL[@]}" "${TB[@]}" >"$OUT/logs/icarus_build.log" 2>&1
for batch in $(seq -w 0 47); do
  vvp "$OUT/build/icarus.vvp" \
    "+VECTORS=$BASE" "+REALW_BATCH=$VEC_DIR/batch_${batch}.txt" \
    >"$OUT/logs/icarus_batch_${batch}.log" 2>&1
done

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module "$TOP" --Mdir "$OUT/build/verilator_sva" \
  "${RTL[@]}" "${TB[@]}" >"$OUT/logs/verilator_sva_build.log" 2>&1
for batch in $(seq -w 0 47); do
  "$OUT/build/verilator_sva/V$TOP" \
    "+VECTORS=$BASE" "+REALW_BATCH=$VEC_DIR/batch_${batch}.txt" \
    >"$OUT/logs/verilator_sva_batch_${batch}.log" 2>&1
done

python3 -m unittest \
  tests.test_generate_h67_real_weight_projection_all_vectors \
  tests.test_report_h67_real_weight_projection_all_rtl \
  >"$OUT/logs/python_tests.log" 2>&1

SOURCES=(
  "${RTL[@]}" "${TB[@]}"
  "$ROOT/scripts/generate_h67_real_weight_projection_all_vectors.py"
  "$ROOT/scripts/report_h67_real_weight_projection_all_rtl.py"
  "$ROOT/sim_h67/run_h67_real_weight_projection_all_checks.sh"
  "$ROOT/tests/test_generate_h67_real_weight_projection_all_vectors.py"
  "$ROOT/tests/test_report_h67_real_weight_projection_all_rtl.py"
)
ARGS=()
for source in "${SOURCES[@]}"; do ARGS+=(--source "$source"); done
python3 "$ROOT/scripts/report_h67_real_weight_projection_all_rtl.py" \
  --result-dir "$OUT" --vector-manifest "$VEC_DIR/manifest.json" \
  --fair-report "$FAIR_REPORT" --joint-report "$JOINT_REPORT" \
  "${ARGS[@]}"

echo "PASS H67 ep35 real-weight all-output full flow"
