#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_dctf_term_event_adapter"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
RTL="rtl_hitflow/gatestack_dctf_term_event_adapter.sv"
TB="tb_hitflow/tb_gatestack_dctf_term_event_adapter.sv"
SVA="verif_hitflow/gatestack_dctf_term_event_adapter_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf_term_event_adapter_assertions.sv"

mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_dctf_term_event_adapter \
  -o "$BUILD/tb.vvp" "$RTL" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
if grep -Eq '(^|[^A-Za-z])(ERROR|FATAL)([^A-Za-z]|$)' \
    "$BUILD/iverilog.log" ||
   ! grep -q '^PASS DCTF ADAPTER ' "$BUILD/iverilog.log"; then
  exit 1
fi

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf_term_event_adapter \
  -Mdir "$BUILD/verilator_obj" \
  "$RTL" "$TB" "$SVA" "$BIND" \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_dctf_term_event_adapter" \
  | tee "$BUILD/verilator.log"
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log" ||
   grep -Eq '(^|[^A-Za-z])(ERROR|FATAL)([^A-Za-z]|$)' \
     "$BUILD/verilator.log" ||
   ! grep -q '^PASS DCTF ADAPTER ' "$BUILD/verilator.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top gatestack_dctf_term_event_adapter; proc; opt; check; stat"
yosys -q -l "$BUILD/yosys_event_ways2.log" -p \
  "read_verilog -sv $RTL; chparam -set EVENT_WAYS 2 gatestack_dctf_term_event_adapter; hierarchy -check -top gatestack_dctf_term_event_adapter; proc; opt; check; stat"
grep -E '^Warning:|ERROR:' "$BUILD/yosys.log" \
  | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/yosys_unexpected.log" || true
grep -E '^Warning:|ERROR:' "$BUILD/yosys_event_ways2.log" \
  | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >>"$BUILD/yosys_unexpected.log" || true
if [[ -s "$BUILD/yosys_unexpected.log" ]]; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi

python "$LINTER" "$RTL" >"$BUILD/erie_lint.log" 2>&1 || true
if grep -Eq '\[ERROR\]|(^|[[:space:]])ERROR([:[:space:]]|$)' \
    "$BUILD/erie_lint.log"; then
  cat "$BUILD/erie_lint.log" >&2
  exit 1
fi

echo "PASS: DCTF term/event adapter multi-beat exact validation passed Icarus, Verilator+SVA, Yosys and Erie checks"
