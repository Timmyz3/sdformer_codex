#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_dctf32_bank_executor"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
ENGINE="rtl_hitflow/gatestack_decoupled_product_engine.sv"
RTL="rtl_hitflow/gatestack_dctf32_bank_executor.sv"
TB="tb_hitflow/tb_gatestack_dctf32_bank_executor.sv"
SVA="verif_hitflow/gatestack_dctf32_bank_executor_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf32_bank_executor_assertions.sv"

mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_dctf32_bank_executor \
  -o "$BUILD/tb.vvp" "$ENGINE" "$RTL" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
if grep -Eiq 'warning:|error:' "$BUILD/iverilog_build.log"; then
  cat "$BUILD/iverilog_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
if ! grep -q '^PASS DCTF32 BANK EXECUTOR ' "$BUILD/iverilog.log"; then
  exit 1
fi

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf32_bank_executor \
  -Mdir "$BUILD/verilator_obj" \
  "$ENGINE" "$RTL" "$TB" "$SVA" "$BIND" \
  >"$BUILD/verilator_build.log" 2>&1
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/Vtb_gatestack_dctf32_bank_executor" \
  | tee "$BUILD/verilator.log"
if ! grep -q '^PASS DCTF32 BANK EXECUTOR ' "$BUILD/verilator.log"; then
  exit 1
fi

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $ENGINE $RTL; hierarchy -check -top gatestack_dctf32_bank_executor; proc; opt; check; stat"
if grep -Eq '^Warning:|ERROR:' "$BUILD/yosys.log"; then
  grep -E '^Warning:|ERROR:' "$BUILD/yosys.log" >&2
  exit 1
fi

python "$LINTER" "$RTL" >"$BUILD/erie_rtl.log" 2>&1
python "$LINTER" --mode tb "$TB" >"$BUILD/erie_tb.log" 2>&1
if ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_rtl.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_tb.log"; then
  cat "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log" >&2
  exit 1
fi

echo "PASS: DCTF32 bank executor Icarus、Verilator动态SVA、Yosys、Erie 0 error/warning"
