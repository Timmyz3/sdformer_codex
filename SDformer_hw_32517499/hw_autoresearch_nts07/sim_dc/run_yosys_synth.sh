#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/sim_dc/build"
mkdir -p "$OUT"

yosys -p "read_verilog -sv $ROOT/rtl_dc/unibin_h60_core_dc.sv; synth -top unibin_h60_core_dc; stat" \
  > "$OUT/yosys_unibin_h60_core_dc.rpt"

cat "$OUT/yosys_unibin_h60_core_dc.rpt"
