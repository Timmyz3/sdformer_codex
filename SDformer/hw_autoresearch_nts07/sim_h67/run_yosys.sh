#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_h67/build/yosys"
mkdir -p "$BUILD"
cd "$ROOT"
rtl_files="$(tr '\n' ' ' < rtl_h67/filelist.f)"

tops=(
  h67_temporal_pair_adapter
  h67_motionxor_score_q7
  h67_score_class_row_engine
  h67_attention_top
)

for top in "${tops[@]}"; do
  yosys -Q -p "read_verilog -sv ${rtl_files}; hierarchy -check -top ${top}; proc; opt; memory; opt; check -assert; stat; write_verilog -noattr ${BUILD}/${top}_synth.v" \
    > "${BUILD}/${top}.log"
  if rg -n "ERROR:|Found and reported [1-9][0-9]* problems" "${BUILD}/${top}.log"; then
    echo "FAIL: Yosys check failed for ${top}" >&2
    exit 1
  fi
  test -s "${BUILD}/${top}_synth.v"
done

echo "PASS: H67 Yosys synthesis/check completed"
