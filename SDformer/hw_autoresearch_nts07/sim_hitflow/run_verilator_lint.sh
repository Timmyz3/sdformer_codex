#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module hitflow_event_lifetime_router \
  rtl_hitflow/hitflow_single_event_buffer.sv \
  rtl_hitflow/hitflow_fanout_event_buffer.sv \
  rtl_hitflow/hitflow_qk_pair_assembler.sv \
  rtl_hitflow/hitflow_event_lifetime_router.sv
