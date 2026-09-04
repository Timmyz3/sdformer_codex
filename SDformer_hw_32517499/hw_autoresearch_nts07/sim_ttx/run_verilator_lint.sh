#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

verilator --lint-only -Wall -Wno-fatal --timing \
  --top-module ttx_attention_top -f rtl_ttx/filelist.f
verilator --lint-only -Wall -Wno-fatal --timing \
  -Wno-TIMESCALEMOD -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL -Wno-BLKSEQ \
  --top-module tb_ttx_row_engine -f rtl_ttx/filelist.f tb_ttx/tb_ttx_row_engine.sv
verilator --lint-only -Wall -Wno-fatal --timing \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-BLKSEQ \
  --top-module tb_ttx_scheduler rtl_ttx/ttx_descriptor_scheduler.sv tb_ttx/tb_ttx_scheduler.sv
verilator --lint-only -Wall -Wno-fatal --timing \
  -Wno-TIMESCALEMOD -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL -Wno-BLKSEQ \
  --top-module tb_ttx_gate_quant_q17 \
  rtl_ttx/ttx_ceil_log2_u32.sv rtl_ttx/ttx_gate_quant_q17.sv \
  tb_ttx/tb_ttx_gate_quant_q17.sv
