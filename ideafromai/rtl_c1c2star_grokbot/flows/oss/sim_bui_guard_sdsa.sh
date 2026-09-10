#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate C2* BUI-GuardSDSA (Card H4)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c2s_bui_guard_sdsa.vvp"
LOG="out/sim/c2s_bui_guard_sdsa_sim.log"

echo "[sim_bui_guard_sdsa] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c2star/c2s_bui_guard_sdsa.sv \
  tb_c2star/tb_c2s_bui_guard_sdsa.sv

echo "[sim_bui_guard_sdsa] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all BUI-GuardSDSA cases" "$LOG"; then
  echo "[sim_bui_guard_sdsa] PASS"
  exit 0
else
  echo "[sim_bui_guard_sdsa] FAIL — see $LOG"
  exit 1
fi
