#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_h68/build/gatelevel"
mkdir -p "$BUILD"
cd "$ROOT"
RTL_FILES="$(tr '\n' ' ' < rtl_h68/filelist.f)"
yosys -Q -p "read_verilog -sv ${RTL_FILES}; chparam -set MAX_TOKENS 8 -set ENABLE_MOTION_XOR 0 h67_score_class_row_engine; hierarchy -check -top h67_score_class_row_engine; synth -flatten -top h67_score_class_row_engine; check -assert; dffunmap; opt_clean; check -assert; write_verilog -noattr ${BUILD}/h68_row_mapped.v" \
  > "$BUILD/yosys.log"
iverilog -g2012 -DGATE_LEVEL_NETLIST -s tb_h67_score_class_row_engine \
  -P tb_h67_score_class_row_engine.ENABLE_MOTION_XOR=0 \
  -o "$BUILD/tb_h68_row_gate.vvp" \
  "$BUILD/h68_row_mapped.v" tb_h67/tb_h67_score_class_row_engine.sv
vvp "$BUILD/tb_h68_row_gate.vvp"
echo "PASS：H68部署映射网表回灌行级自检完成"
