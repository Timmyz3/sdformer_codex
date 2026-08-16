#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_dctf96_banklocal_projection_top"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
TB="tb_hitflow/tb_gatestack_dctf96_banklocal_projection_top.sv"
SVA="verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv"
RTL=(
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv
  rtl_hitflow/gatestack_ppdi_token_bank.sv
  rtl_hitflow/gatestack_ppdi_term_event_adapter_2c.sv
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_ppdi_dctf_term_fabric.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/hitflow_implicit_bias_finalizer_accumulator.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)

mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_dctf96_banklocal_projection_top \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
    "$BUILD/iverilog_build.log"; then
  cat "$BUILD/iverilog_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
if grep -Eiq '(^|[^[:alpha:]])(error|fatal|assertion failed)([^[:alpha:]]|$)' \
      "$BUILD/iverilog.log" ||
   ! grep -q '^PASS DCTF96 BANKLOCAL PROJECTION ' "$BUILD/iverilog.log"; then
  exit 1
fi

rm -rf "$BUILD/verilator_obj"
verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf96_banklocal_projection_top \
  -Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" "$TB" "$SVA" "$BIND" \
  >"$BUILD/verilator_build.log" 2>&1
if grep -Eiq '(%Warning|%Error|warning:|error:|fatal:)' \
    "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/Vtb_gatestack_dctf96_banklocal_projection_top" \
  | tee "$BUILD/verilator.log"
if grep -Eiq '(^|[^[:alpha:]])(error|fatal|assertion failed)([^[:alpha:]]|$)' \
      "$BUILD/verilator.log" ||
   ! grep -q '^PASS DCTF96 BANKLOCAL PROJECTION ' "$BUILD/verilator.log"; then
  exit 1
fi

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_dctf96_banklocal_projection_top; proc; opt; check; stat"
grep -Ei '(^Warning:|ERROR:|fatal|assert)' "$BUILD/yosys.log" \
  | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/yosys_unexpected.log" || true
if [[ -s "$BUILD/yosys_unexpected.log" ]]; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi

python3 "$LINTER" rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv \
  >"$BUILD/erie_rtl.log" 2>&1
python3 "$LINTER" --mode tb "$TB" >"$BUILD/erie_tb.log" 2>&1
if ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_rtl.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_tb.log" ||
   grep -Eiq '(\[ERROR\]|\[WARNING\]|fatal|traceback)' \
       "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log"; then
  cat "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log" >&2
  exit 1
fi

echo "PASS: DCTF96 bank-local projection Icarus、Verilator --assert、Yosys hierarchy/check/stat、Erie RTL+TB 0 error/warning"
