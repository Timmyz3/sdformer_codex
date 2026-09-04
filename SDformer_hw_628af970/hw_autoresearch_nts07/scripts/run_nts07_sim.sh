#!/usr/bin/env bash
# Quick syntax check for NTS-07 RTL (iverilog optional).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RTL="${ROOT}/rtl"

if ! command -v iverilog >/dev/null 2>&1; then
  echo "[skip] iverilog not installed; listing RTL files only"
  ls -la "${RTL}"
  exit 0
fi

iverilog -g2012 -I"${RTL}" -o /tmp/nts07_sim.vvp \
  "${RTL}/nts07_pkg.vh" \
  "${RTL}/atlif_unified_encode_unit.v" \
  "${RTL}/ternary_encode_unit.v" \
  "${RTL}/tx_sc_score_unit.v" \
  "${RTL}/shiftmax_unit.v" \
  "${RTL}/h60_attention_engine.v" \
  "${RTL}/sparse_mac_pe.v" \
  "${RTL}/nts07_controller.v" \
  "${RTL}/nts07_top.v"

echo "[ok] iverilog compile succeeded -> /tmp/nts07_sim.vvp"