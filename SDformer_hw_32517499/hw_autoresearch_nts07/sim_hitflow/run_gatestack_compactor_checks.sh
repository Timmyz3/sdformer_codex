#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_compactor"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_active_token_iterator \
  -o "$BUILD/tb_active_token.vvp" \
  rtl_hitflow/gatestack_obi_iterator.sv \
  rtl_hitflow/gatestack_active_token_iterator.sv \
  tb_hitflow/tb_gatestack_active_token_iterator.sv
vvp "$BUILD/tb_active_token.vvp" | tee "$BUILD/active_token_iverilog.log"

for ways in 2 4; do
  iverilog -g2012 -Wall -DTB_WAYS="$ways" \
    -s tb_gatestack_r4_event_compactor \
    -o "$BUILD/tb_r${ways}_compactor.vvp" \
    rtl_hitflow/gatestack_event_compactor.sv \
    tb_hitflow/tb_gatestack_r4_event_compactor.sv
  vvp "$BUILD/tb_r${ways}_compactor.vvp" \
    | tee "$BUILD/r${ways}_iverilog.log"
done

verilator --lint-only --timing -Wall \
  --top-module gatestack_active_token_iterator \
  rtl_hitflow/gatestack_obi_iterator.sv \
  rtl_hitflow/gatestack_active_token_iterator.sv \
  >"$BUILD/active_token_lint.log" 2>&1

for ways in 2 4; do
  verilator --binary --timing --assert -Wall -DTB_WAYS="$ways" \
    --top-module tb_gatestack_r4_event_compactor \
    -Mdir "$BUILD/r${ways}_obj" \
    rtl_hitflow/gatestack_event_compactor.sv \
    verif_hitflow/gatestack_event_compactor_assertions.sv \
    verif_hitflow/bind_gatestack_event_compactor_assertions.sv \
    tb_hitflow/tb_gatestack_r4_event_compactor.sv \
    >"$BUILD/r${ways}_build.log" 2>&1
  "$BUILD/r${ways}_obj/Vtb_gatestack_r4_event_compactor" \
    | tee "$BUILD/r${ways}_assert.log"
done

yosys -q -l "$BUILD/active_token_yosys.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_obi_iterator.sv rtl_hitflow/gatestack_active_token_iterator.sv; hierarchy -check -top gatestack_active_token_iterator; proc; opt; memory -nomap; check; stat"
for ways in 2 4; do
  yosys -q -l "$BUILD/r${ways}_yosys.log" -p \
    "read_verilog -sv rtl_hitflow/gatestack_event_compactor.sv; chparam -set WAYS $ways gatestack_event_compactor; hierarchy -check -top gatestack_event_compactor; proc; opt; memory -nomap; check; stat"
done

if grep -Eq '%Warning|%Error' \
  "$BUILD/active_token_lint.log" "$BUILD/r2_build.log" "$BUILD/r4_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: GateStack active-token + R2/R4 compactor；Verilator 0 warning/error"
