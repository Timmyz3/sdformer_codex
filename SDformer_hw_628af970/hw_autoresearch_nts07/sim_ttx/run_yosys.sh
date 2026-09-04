#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_ttx/build/yosys"
mkdir -p "$BUILD"
cd "$ROOT"
rtl_files="$(tr '\n' ' ' < rtl_ttx/filelist.f)"

tops=(
  ttx_tx_score_q7
  ttx_late_gate_accum
  ttx_gate_quant_q17
  ttx_row_engine
  ttx_descriptor_scheduler
  ttx_attention_top
)

for top in "${tops[@]}"; do
  yosys -Q -p "read_verilog -sv ${rtl_files}; hierarchy -check -top ${top}; proc; opt; memory; setundef -undriven -zero; opt; check; stat; write_verilog -noattr ${BUILD}/${top}_synth.v" \
    > "${BUILD}/${top}.log"
  if rg -n "ERROR:|Found and reported [1-9][0-9]* problems" "${BUILD}/${top}.log"; then
    echo "FAIL: Yosys check failed for ${top}" >&2
    exit 1
  fi
  test -s "${BUILD}/${top}_synth.v"
done

echo "PASS: Yosys synthesis/check completed for TTX tops"
