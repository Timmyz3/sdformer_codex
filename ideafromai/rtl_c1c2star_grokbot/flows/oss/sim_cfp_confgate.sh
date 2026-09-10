#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate C1* CFP-ConfGate (Card H2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_cfp_confgate.vvp"
LOG="out/sim/c1s_cfp_confgate_sim.log"

echo "[sim_cfp_confgate] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_cfp_confgate.sv \
  tb_c1star/tb_c1s_cfp_confgate.sv

echo "[sim_cfp_confgate] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all CFP-ConfGate cases" "$LOG"; then
  echo "[sim_cfp_confgate] PASS"
  exit 0
else
  echo "[sim_cfp_confgate] FAIL — see $LOG"
  exit 1
fi
