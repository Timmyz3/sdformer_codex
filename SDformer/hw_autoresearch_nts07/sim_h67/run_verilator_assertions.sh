#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_h67/build/verilator_assertions"
rm -rf "$BUILD"
cd "$ROOT"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL -Wno-BLKSEQ \
  --Mdir "$BUILD" --top-module tb_h67_score_class_row_engine \
  -f rtl_h67/filelist.f \
  verif_h67_h68/h67_h68_protocol_assertions.sv \
  verif_h67_h68/bind_row_engine_assertions.sv \
  tb_h67/tb_h67_score_class_row_engine.sv
"$BUILD/Vtb_h67_score_class_row_engine"
echo "PASS：H67行引擎SVA绑定仿真完成"
