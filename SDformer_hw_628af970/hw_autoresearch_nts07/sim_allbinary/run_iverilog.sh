#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/sim_allbinary/build"
mkdir -p "$OUT"

iverilog -g2012 \
  -I "$ROOT/rtl_allbinary" \
  -o "$OUT/tb_unibin_h60_modules.vvp" \
  "$ROOT/rtl_allbinary/binary_atlif_unit.v" \
  "$ROOT/rtl_allbinary/binary_atlif_state_unit.v" \
  "$ROOT/rtl_allbinary/binary_popcount_consensus.v" \
  "$ROOT/rtl_allbinary/ttb_skip_unit.v" \
  "$ROOT/rtl_allbinary/shiftmax_int8_unit.v" \
  "$ROOT/rtl_allbinary/gated_k_unit.v" \
  "$ROOT/rtl_allbinary/unibin_h60_token_core.v" \
  "$ROOT/tb_allbinary/tb_unibin_h60_modules.v"

vvp "$OUT/tb_unibin_h60_modules.vvp"
