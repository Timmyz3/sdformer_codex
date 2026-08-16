#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_active_descriptor_store_macro_miter_20260809}"
mkdir -p "$OUT/build" "$OUT/logs"
cd "$ROOT"

RTL=(
  tb_h67/fakeram45_256x32_functional_model.sv
  rtl_h67/h67_banked_active_descriptor_store.sv
  tb_h67/tb_h67_active_descriptor_store_macro_miter.sv
)

iverilog -g2012 -Wall -s tb_h67_active_descriptor_store_macro_miter \
  -o "$OUT/build/miter.vvp" "${RTL[@]}" \
  >"$OUT/logs/iverilog_build.log" 2>&1
vvp "$OUT/build/miter.vvp" >"$OUT/logs/iverilog_run.log" 2>&1
grep -q '^PASS tb_h67_active_descriptor_store_macro_miter ' \
  "$OUT/logs/iverilog_run.log"

rm -rf "$OUT/build/verilator"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND \
  -Wno-WIDTHTRUNC -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module tb_h67_active_descriptor_store_macro_miter \
  --Mdir "$OUT/build/verilator" "${RTL[@]}" \
  >"$OUT/logs/verilator_build.log" 2>&1
"$OUT/build/verilator/Vtb_h67_active_descriptor_store_macro_miter" \
  >"$OUT/logs/verilator_run.log" 2>&1
grep -q '^PASS tb_h67_active_descriptor_store_macro_miter ' \
  "$OUT/logs/verilator_run.log"

git diff --check -- \
  rtl_h67/h67_banked_active_descriptor_store.sv \
  tb_h67/fakeram45_256x32_functional_model.sv \
  tb_h67/tb_h67_active_descriptor_store_macro_miter.sv \
  sim_h67/run_h67_active_descriptor_store_macro_miter.sh

cat >"$OUT/status.tsv" <<'EOF'
check	status
iverilog_macro_behavior_miter	PASS
verilator_macro_behavior_miter	PASS
padding_contract	PASS
EOF

echo "PASS H67 active descriptor macro/behavior miter"
