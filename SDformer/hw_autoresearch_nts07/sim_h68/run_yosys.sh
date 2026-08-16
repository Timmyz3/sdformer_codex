#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sim_h68/build/yosys"
mkdir -p "$BUILD"
cd "$ROOT"
rtl_files="$(tr '\n' ' ' < rtl_h68/filelist.f)"

yosys -Q -p "read_verilog -sv ${rtl_files}; hierarchy -check -top h68_castling_deploy_top; proc; opt; memory; opt; check -assert; stat; write_verilog -noattr ${BUILD}/h68_castling_deploy_top_synth.v" \
  > "$BUILD/h68_castling_deploy_top.log"
if rg -n "ERROR:|Found and reported [1-9][0-9]* problems" "$BUILD/h68_castling_deploy_top.log"; then
  echo "FAIL: H68 Yosys检查失败" >&2
  exit 1
fi
test -s "$BUILD/h68_castling_deploy_top_synth.v"
echo "PASS: H68 Yosys综合与结构检查完成"
