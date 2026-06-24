#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/sim_dc/build"
mkdir -p "$OUT"

iverilog -g2012 \
  -o "$OUT/tb_unibin_h60_core_dc.vvp" \
  "$ROOT/rtl_dc/unibin_h60_core_dc.sv" \
  "$ROOT/tb_dc/tb_unibin_h60_core_dc.sv"

(cd "$OUT" && vvp tb_unibin_h60_core_dc.vvp)
