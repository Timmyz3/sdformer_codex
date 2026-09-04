#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_h67/build"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -s tb_h67_motionxor_score \
  -o "$BUILD/tb_h67_motionxor_score.vvp" \
  rtl_h67/h67_motionxor_score_q7.sv tb_h67/tb_h67_motionxor_score.sv
vvp "$BUILD/tb_h67_motionxor_score.vvp"

iverilog -g2012 -s tb_h67_score_class_row_engine \
  -o "$BUILD/tb_h67_score_class_row_engine.vvp" \
  -f rtl_h67/filelist.f tb_h67/tb_h67_score_class_row_engine.sv
vvp "$BUILD/tb_h67_score_class_row_engine.vvp"

iverilog -g2012 -s tb_h67_score_class_row_engine \
  -P tb_h67_score_class_row_engine.MAX_TOKENS=162 \
  -o "$BUILD/tb_h67_score_class_row_engine_max162.vvp" \
  -f rtl_h67/filelist.f tb_h67/tb_h67_score_class_row_engine.sv
vvp "$BUILD/tb_h67_score_class_row_engine_max162.vvp"
