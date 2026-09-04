#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="$ROOT/rtl_allbinary"
OUT="$ROOT/sim_allbinary/build/yosys"
mkdir -p "$OUT"

run_yosys() {
  local top="$1"
  shift
  local log="$OUT/${top}.log"
  local netlist="$OUT/${top}_synth.v"
  echo "[Yosys] synth/check top=$top"
  yosys -q -l "$log" -p "
    read_verilog -sv -I $RTL $*
    hierarchy -check -top $top
    synth -top $top
    check
    stat
    write_verilog -noattr $netlist
  "
  test -s "$netlist"
  if grep -E '(^|[[:space:]])(ERROR|Error):' "$log" >/dev/null; then
    echo "FAIL: Yosys reported an error for $top. See $log"
    exit 1
  fi
}

run_yosys binary_atlif_unit "$RTL/binary_atlif_unit.v"
run_yosys binary_atlif_state_unit "$RTL/binary_atlif_state_unit.v"
run_yosys binary_popcount_consensus "$RTL/binary_popcount_consensus.v"
run_yosys ttb_skip_unit "$RTL/ttb_skip_unit.v"
run_yosys shiftmax_int8_unit "$RTL/shiftmax_int8_unit.v"
run_yosys gated_k_unit "$RTL/gated_k_unit.v"
run_yosys unibin_h60_token_core \
  "$RTL/binary_popcount_consensus.v" \
  "$RTL/gated_k_unit.v" \
  "$RTL/unibin_h60_token_core.v"

echo "PASS: Yosys synthesis/check completed for all UniBin-H60 tops"
