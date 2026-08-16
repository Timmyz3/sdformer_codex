#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p build_hitflow
yosys -p 'read_verilog -sv rtl_hitflow/hitflow_single_event_buffer.sv rtl_hitflow/hitflow_fanout_event_buffer.sv rtl_hitflow/hitflow_qk_pair_assembler.sv rtl_hitflow/hitflow_event_lifetime_router.sv; hierarchy -check -top hitflow_event_lifetime_router; proc; opt; memory; opt; check; stat' \
  | tee build_hitflow/yosys_event_router.log
