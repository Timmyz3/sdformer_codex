#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="$ROOT/rtl_allbinary"

run_lint() {
  local top="$1"
  shift
  echo "[Verilator] lint top=$top"
  verilator --lint-only -Wall -Wno-fatal \
    -I"$RTL" \
    --top-module "$top" \
    "$@"
}

run_lint binary_atlif_unit "$RTL/binary_atlif_unit.v"
run_lint binary_atlif_state_unit "$RTL/binary_atlif_state_unit.v"
run_lint binary_popcount_consensus "$RTL/binary_popcount_consensus.v"
run_lint ttb_skip_unit "$RTL/ttb_skip_unit.v"
run_lint shiftmax_int8_unit "$RTL/shiftmax_int8_unit.v"
run_lint gated_k_unit "$RTL/gated_k_unit.v"
run_lint unibin_h60_token_core \
  "$RTL/binary_popcount_consensus.v" \
  "$RTL/gated_k_unit.v" \
  "$RTL/unibin_h60_token_core.v"

echo "PASS: Verilator lint completed for all UniBin-H60 tops"
