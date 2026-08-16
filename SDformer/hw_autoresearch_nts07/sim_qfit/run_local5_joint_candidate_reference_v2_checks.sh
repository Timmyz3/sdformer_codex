#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_qfit/local5_joint_candidate_reference_v2"
mkdir -p "$BUILD"

/opt/conda/envs/sdformerflow/bin/python -m unittest \
  tests.test_local5_joint_candidate_reference_v2 \
  tests.test_evaluate_local5_joint_candidates_v2 \
  tests.test_analyze_local5_joint_hardware_contract

iverilog -g2012 -Wall -s tb_qfit_direct_1rw_reference_timing \
  -o "$BUILD/direct_1rw_reference_timing.vvp" \
  "$ROOT/rtl_qfit/qfit_single_port_acc_memory.sv" \
  "$ROOT/rtl_qfit/qfit_direct_1rw_acc_bank.sv" \
  "$ROOT/tb_qfit/tb_qfit_direct_1rw_reference_timing.sv"
vvp "$BUILD/direct_1rw_reference_timing.vvp" \
  | tee "$BUILD/direct_1rw_reference_timing.log"
grep -q '^PASS DIRECT_1RW_REFERENCE_TIMING' \
  "$BUILD/direct_1rw_reference_timing.log"

echo "PASS Local5 joint candidate reference v2 checks"
