#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESIGN_NAME="${DESIGN_NAME:-h67_attention_top}"
case "$DESIGN_NAME" in
  h67_attention_top) FILELIST="$ROOT/rtl_h67/filelist.f" ;;
  h68_castling_deploy_top) FILELIST="$ROOT/rtl_h68/filelist.f" ;;
  *) echo "不支持的DESIGN_NAME: $DESIGN_NAME" >&2; exit 2 ;;
esac

OUT="$ROOT/dc_handoff/runs/yosys_generic/$DESIGN_NAME"
mkdir -p "$OUT"
RTL_FILES="$(tr '\n' ' ' < "$FILELIST")"
yosys -Q -p "read_verilog -sv ${RTL_FILES}; hierarchy -check -top ${DESIGN_NAME}; synth -flatten -top ${DESIGN_NAME}; opt_clean; check -assert; tee -o ${OUT}/stat.json stat -json; dffunmap; opt_clean; check -assert; write_verilog -noattr ${OUT}/${DESIGN_NAME}_generic.v" \
  > "$OUT/yosys.log"
test -s "$OUT/${DESIGN_NAME}_generic.v"
if rg -n "ERROR:|Found and reported [1-9][0-9]* problems" "$OUT/yosys.log"; then
  echo "Yosys通用综合检查失败" >&2
  exit 1
fi
echo "$OUT"
