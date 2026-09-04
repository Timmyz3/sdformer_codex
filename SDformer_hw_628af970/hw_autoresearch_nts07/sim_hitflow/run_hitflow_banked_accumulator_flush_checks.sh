#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/banked_accumulator_flush"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
RTL=rtl_hitflow/hitflow_banked_accumulator.sv
SVA=verif_hitflow/hitflow_banked_accumulator_assertions.sv
BIND=verif_hitflow/bind_hitflow_banked_accumulator_assertions.sv
mkdir -p "$BUILD"
cd "$ROOT"

run_iverilog() {
  local top="$1"
  local tb="$2"
  local stem="$3"
  iverilog -g2012 -Wall -s "$top" -o "$BUILD/${stem}.vvp" \
    "$RTL" "$tb" >"$BUILD/${stem}_iverilog_build.log" 2>&1
  vvp "$BUILD/${stem}.vvp" | tee "$BUILD/${stem}_iverilog.log"
}

run_verilator_assert() {
  local top="$1"
  local tb="$2"
  local stem="$3"
  local obj="$BUILD/${stem}_verilator_obj"
  rm -rf "$obj"
  verilator --binary --assert --timing --sv -Wall \
    --top-module "$top" --Mdir "$obj" \
    "$RTL" "$SVA" "$BIND" "$tb" \
    >"$BUILD/${stem}_verilator_build.log" 2>&1
  "$obj/V${top}" | tee "$BUILD/${stem}_verilator.log"
  if grep -Eq '%Error|%Warning' "$BUILD/${stem}_verilator_build.log"; then
    cat "$BUILD/${stem}_verilator_build.log" >&2
    return 1
  fi
}

run_erie() {
  local source="$1"
  local mode="$2"
  local stem
  stem="$(basename "$source" .sv)"
  python3 "$LINTER" "$source" --mode "$mode" \
    >"$BUILD/erie_${stem}.log" 2>&1
  if [[ "$(grep -Fc 'Summary: 0 error(s), 0 warning(s)' \
      "$BUILD/erie_${stem}.log")" -ne 1 ]]; then
    cat "$BUILD/erie_${stem}.log" >&2
    return 1
  fi
}

run_iverilog tb_hitflow_banked_accumulator \
  tb_hitflow/tb_hitflow_banked_accumulator.sv accumulator
run_iverilog tb_hitflow_banked_accumulator_overflow_atomic \
  tb_hitflow/tb_hitflow_banked_accumulator_overflow_atomic.sv overflow_atomic

run_verilator_assert tb_hitflow_banked_accumulator \
  tb_hitflow/tb_hitflow_banked_accumulator.sv accumulator
run_verilator_assert tb_hitflow_banked_accumulator_overflow_atomic \
  tb_hitflow/tb_hitflow_banked_accumulator_overflow_atomic.sv overflow_atomic

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv -defer $RTL; hierarchy -check -top hitflow_banked_accumulator; proc; opt; memory -nomap; opt; check -assert; stat"
if grep -Eq '^(Warning:|ERROR:)' "$BUILD/yosys.log"; then
  cat "$BUILD/yosys.log" >&2
  exit 1
fi

run_erie rtl_hitflow/hitflow_banked_accumulator.sv rtl
run_erie rtl_hitflow/gatestack_multihead_tile_projection_top.sv rtl
run_erie rtl_hitflow/gatestack_single_head_projection_top.sv rtl
run_erie rtl_hitflow/hitflow_g1_projection_top.sv rtl
run_erie tb_hitflow/tb_hitflow_banked_accumulator.sv tb
run_erie tb_hitflow/tb_hitflow_banked_accumulator_overflow_atomic.sv tb
run_erie verif_hitflow/hitflow_banked_accumulator_assertions.sv tb
run_erie verif_hitflow/bind_hitflow_banked_accumulator_assertions.sv tb

echo "PASS: accumulator synchronous flush Icarus/Verilator-SVA/Yosys/Erie"
