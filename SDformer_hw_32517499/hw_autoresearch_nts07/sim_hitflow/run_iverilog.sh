#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p build_hitflow
iverilog -g2012 -Wall -s tb_hitflow_event_lifetime_router \
  -o build_hitflow/tb_event_router.vvp \
  rtl_hitflow/hitflow_single_event_buffer.sv \
  rtl_hitflow/hitflow_fanout_event_buffer.sv \
  rtl_hitflow/hitflow_qk_pair_assembler.sv \
  rtl_hitflow/hitflow_event_lifetime_router.sv \
  tb_hitflow/tb_hitflow_event_lifetime_router.sv
vvp build_hitflow/tb_event_router.vvp
