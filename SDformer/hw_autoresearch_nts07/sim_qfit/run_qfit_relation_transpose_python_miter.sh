#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_qfit/relation_python_miter"
VECTORS="$ROOT/tb_qfit/vectors/local5_relation_t450"
RESULT="$ROOT/results/qfit_relation_python_miter_20260731"
PYTHON="/opt/conda/envs/sdformerflow/bin/python"

mkdir -p "$BUILD" "$RESULT"
cd "$ROOT"

"$PYTHON" scripts/generate_local5_relation_transpose_vectors.py \
  --output-dir "$VECTORS"

iverilog -g2012 -Wall \
  -s tb_qfit_relation_transpose_python_miter \
  -o "$BUILD/miter.vvp" \
  rtl_qfit/qfit_retirement_scheduler.sv \
  rtl_qfit/qfit_sync_1r1w_bank.sv \
  rtl_qfit/qfit_relation_transpose_leaf.sv \
  tb_qfit/tb_qfit_relation_transpose_python_miter.sv \
  >"$RESULT/iverilog_build.log" 2>&1
vvp "$BUILD/miter.vvp" "+VECTOR_DIR=$VECTORS" \
  | tee "$RESULT/iverilog_run.log"

rm -rf "$BUILD/obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_relation_transpose_python_miter \
  --Mdir "$BUILD/obj" \
  rtl_qfit/qfit_retirement_scheduler.sv \
  rtl_qfit/qfit_sync_1r1w_bank.sv \
  rtl_qfit/qfit_relation_transpose_leaf.sv \
  verif_qfit/qfit_relation_transpose_assertions.sv \
  verif_qfit/qfit_sync_bank_assertions.sv \
  tb_qfit/tb_qfit_relation_transpose_python_miter.sv \
  >"$RESULT/verilator_build.log" 2>&1
"$BUILD/obj/Vtb_qfit_relation_transpose_python_miter" \
  "+VECTOR_DIR=$VECTORS" \
  | tee "$RESULT/verilator_run.log"

sha256sum \
  sim_qfit/run_qfit_relation_transpose_python_miter.sh \
  scripts/generate_local5_relation_transpose_vectors.py \
  scripts/profile_local5_hardware_features.py \
  rtl_qfit/qfit_retirement_scheduler.sv \
  rtl_qfit/qfit_sync_1r1w_bank.sv \
  rtl_qfit/qfit_relation_transpose_leaf.sv \
  verif_qfit/qfit_relation_transpose_assertions.sv \
  verif_qfit/qfit_sync_bank_assertions.sv \
  tb_qfit/tb_qfit_relation_transpose_python_miter.sv \
  "$VECTORS/manifest.json" \
  >"$RESULT/source_sha256.txt"

printf 'stage\tstatus\n' >"$RESULT/status.tsv"
printf 'python_vector_generation\tPASS\n' >>"$RESULT/status.tsv"
printf 'iverilog_t450_miter\tPASS\n' >>"$RESULT/status.tsv"
printf 'verilator_sva_t450_miter\tPASS\n' >>"$RESULT/status.tsv"
