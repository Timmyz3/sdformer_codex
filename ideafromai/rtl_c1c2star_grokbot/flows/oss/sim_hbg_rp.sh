#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Simulate Card B HBG-RP with iverilog+vvp; dump VCD + log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c2s_hbg_rp.vvp"
LOG="out/sim/c2s_hbg_rp_sim.log"
VCD="out/waves/c2s_hbg_rp.vcd"

echo "[sim_hbg_rp] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c2star/c2s_hbg_rp_packetizer.sv \
  tb_c2star/tb_c2s_hbg_rp_packetizer.sv

echo "[sim_hbg_rp] run → $VCD"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all 4 HBG-RP cases" "$LOG"; then
  echo "[sim_hbg_rp] PASS"
  exit 0
else
  echo "[sim_hbg_rp] FAIL — see $LOG"
  exit 1
fi
