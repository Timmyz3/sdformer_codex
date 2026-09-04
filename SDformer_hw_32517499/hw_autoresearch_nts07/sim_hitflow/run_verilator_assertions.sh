#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/verilator_assertions"
rm -rf "$BUILD"
cd "$ROOT"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-BLKSEQ \
  --Mdir "$BUILD" --top-module tb_hitflow_event_lifetime_router \
  rtl_hitflow/hitflow_single_event_buffer.sv \
  rtl_hitflow/hitflow_fanout_event_buffer.sv \
  rtl_hitflow/hitflow_qk_pair_assembler.sv \
  rtl_hitflow/hitflow_event_lifetime_router.sv \
  verif_hitflow/hitflow_event_router_assertions.sv \
  verif_hitflow/bind_hitflow_event_router_assertions.sv \
  tb_hitflow/tb_hitflow_event_lifetime_router.sv
"$BUILD/Vtb_hitflow_event_lifetime_router"
