#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ppdi_dctf_term_fabric"
RTL="rtl_hitflow/gatestack_ppdi_dctf_term_fabric.sv"
TB="tb_hitflow/tb_gatestack_ppdi_dctf_term_fabric.sv"
SVA="verif_hitflow/gatestack_ppdi_dctf_term_fabric_assertions.sv"
BIND="verif_hitflow/bind_gatestack_ppdi_dctf_term_fabric_assertions.sv"

mkdir -p "$BUILD"
cd "$ROOT"

for q in 2 3 4; do
  iverilog -g2012 -Wall -s tb_gatestack_ppdi_dctf_term_fabric \
    -Ptb_gatestack_ppdi_dctf_term_fabric.Q="$q" \
    -o "$BUILD/tb_q${q}.vvp" "$RTL" "$TB" \
    >"$BUILD/iverilog_q${q}_build.log" 2>&1
  vvp "$BUILD/tb_q${q}.vvp" | tee "$BUILD/iverilog_q${q}.log"
  grep -q '^PASS PPDI DCTF FABRIC ' "$BUILD/iverilog_q${q}.log"
done

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ppdi_dctf_term_fabric \
  -Mdir "$BUILD/verilator_obj" "$RTL" "$TB" "$SVA" "$BIND" \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_ppdi_dctf_term_fabric" \
  | tee "$BUILD/verilator.log"
grep -q '^PASS PPDI DCTF FABRIC ' "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top gatestack_ppdi_dctf_term_fabric; proc; opt; check; stat"
if grep -E '^Warning:|ERROR:' "$BUILD/yosys.log" \
   | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
   >"$BUILD/yosys_unexpected.log"; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi

echo "PASS: PPDI双目的三消费者有序fabric通过Q=2/3/4仿真、动态SVA和Yosys检查"
