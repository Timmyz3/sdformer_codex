#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/dptme"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -DSIMULATOR_ICARUS -s tb_hitflow_dptme_array \
  -o "$BUILD/tb_dptme.vvp" \
  rtl_hitflow/hitflow_dptme_array.sv \
  tb_hitflow/tb_hitflow_dptme_array.sv
vvp "$BUILD/tb_dptme.vvp"

verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL \
  --top-module hitflow_dptme_array \
  rtl_hitflow/hitflow_dptme_array.sv

ASSERT_BUILD="$BUILD/verilator_assertions"
rm -rf "$ASSERT_BUILD"
verilator --binary --timing --assert -DSIMULATOR_VERILATOR \
  -DSVA_RUNTIME_ENABLED -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-BLKSEQ -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  --Mdir "$ASSERT_BUILD" --top-module tb_hitflow_dptme_array \
  rtl_hitflow/hitflow_dptme_array.sv \
  verif_hitflow/hitflow_dptme_assertions.sv \
  verif_hitflow/bind_hitflow_dptme_assertions.sv \
  tb_hitflow/tb_hitflow_dptme_array.sv
"$ASSERT_BUILD/Vtb_hitflow_dptme_array"

yosys -q -l "$BUILD/yosys.log" \
  -p 'read_verilog -sv rtl_hitflow/hitflow_dptme_array.sv; hierarchy -check -top hitflow_dptme_array; proc; opt; check; stat'
if grep -q '^Warning:' "$BUILD/yosys.log"; then
  grep '^Warning:' "$BUILD/yosys.log"
  exit 1
fi
grep -E 'Found and reported|Number of cells' "$BUILD/yosys.log" | tail -8

PYTHONPATH=scripts python -m unittest scripts/test_analyze_dptme_port_contract.py
python scripts/analyze_dptme_port_contract.py
