#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_dctf_term_fabric"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
RTL="rtl_hitflow/gatestack_dctf_term_fabric.sv"
TB="tb_hitflow/tb_gatestack_dctf_term_fabric.sv"
SVA="verif_hitflow/gatestack_dctf_term_fabric_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf_term_fabric_assertions.sv"

mkdir -p "$BUILD"
cd "$ROOT"

for q in 2 3 4; do
  iverilog -g2012 -Wall -s tb_gatestack_dctf_term_fabric \
    -Ptb_gatestack_dctf_term_fabric.Q="$q" \
    -o "$BUILD/tb_q${q}.vvp" "$RTL" "$TB" \
    >"$BUILD/iverilog_q${q}_build.log" 2>&1
  if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
      "$BUILD/iverilog_q${q}_build.log"; then
    cat "$BUILD/iverilog_q${q}_build.log" >&2
    exit 1
  fi
  vvp "$BUILD/tb_q${q}.vvp" | tee "$BUILD/iverilog_q${q}.log"
  if grep -Eq '(^|[^A-Za-z])(ERROR|FATAL)([^A-Za-z]|$)' \
      "$BUILD/iverilog_q${q}.log" ||
     ! grep -q '^PASS DCTF ' "$BUILD/iverilog_q${q}.log"; then
    exit 1
  fi
done

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf_term_fabric \
  -Mdir "$BUILD/verilator_obj" \
  "$RTL" "$TB" "$SVA" "$BIND" \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_dctf_term_fabric" \
  | tee "$BUILD/verilator.log"
if grep -Eiq '(%Warning|%Error|warning:|error:|fatal:)' \
     "$BUILD/verilator_build.log" ||
   grep -Eiq '(^|[^A-Za-z])(ERROR|FATAL|ASSERTION FAILED)([^A-Za-z]|$)' \
     "$BUILD/verilator.log" ||
   ! grep -q '^PASS DCTF ' "$BUILD/verilator.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top gatestack_dctf_term_fabric; proc; opt; check; stat"
grep -E '^Warning:|ERROR:' "$BUILD/yosys.log" \
  | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/yosys_unexpected.log" || true
if [[ -s "$BUILD/yosys_unexpected.log" ]]; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi

python "$LINTER" "$RTL" >"$BUILD/erie_rtl.log" 2>&1
python "$LINTER" --mode tb "$TB" >"$BUILD/erie_tb.log" 2>&1
if ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_rtl.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_tb.log" ||
   grep -Eiq '(\[ERROR\]|\[WARNING\]|fatal|traceback)' \
       "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log"; then
  cat "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log" >&2
  exit 1
fi

echo "PASS: DCTF窄命令三消费者fabric通过Q=2/3/4 Icarus、Verilator+SVA、Yosys与Erie严格检查"
