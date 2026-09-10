#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim
OUT_VVP="out/sim/c1s_cfp_sci_exact_glue.vvp"
LOG="out/sim/c1s_cfp_sci_exact_glue_sim.log"

echo "[sim_cfp_sci_exact_glue] compile..."
iverilog -g2012 -o "$OUT_VVP" \
  rtl_c1star/c1s_cfp_confgate.sv \
  rtl_c1star/c1s_sci_cleanexit.sv \
  rtl_c1star/c1s_ogec_gate.sv \
  rtl_c1star/c1s_prrc_ledger.sv \
  rtl_c1star/c1s_exact_capture_wrap.sv \
  rtl_c1star/c1s_cfp_sci_exact_glue.sv \
  tb_c1star/tb_c1s_cfp_sci_exact_glue.sv

echo "[sim_cfp_sci_exact_glue] run"
vvp "$OUT_VVP" +DUMP_VCD=1 2>&1 | tee "$LOG"

if grep -q "PASS: all CFP/SCI exact glue cases" "$LOG"; then
  echo "[sim_cfp_sci_exact_glue] PASS"
  exit 0
else
  echo "[sim_cfp_sci_exact_glue] FAIL — see $LOG"
  exit 1
fi
