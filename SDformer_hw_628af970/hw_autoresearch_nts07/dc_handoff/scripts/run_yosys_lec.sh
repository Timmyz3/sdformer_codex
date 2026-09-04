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
NETLIST="$OUT/${DESIGN_NAME}_generic.v"
test -s "$NETLIST"
RTL_FILES="$(tr '\n' ' ' < "$FILELIST")"
LEC_TIMEOUT_SECONDS="${LEC_TIMEOUT_SECONDS:-600}"
timeout "$LEC_TIMEOUT_SECONDS" yosys -Q -p "read_verilog -sv ${RTL_FILES}; prep -flatten -top ${DESIGN_NAME}; opt_clean; check -assert; design -stash gold; read_verilog ${NETLIST}; prep -top ${DESIGN_NAME}; check -assert; design -stash gate; design -copy-from gold -as gold ${DESIGN_NAME}; design -copy-from gate -as gate ${DESIGN_NAME}; equiv_make -inames gold gate equiv; hierarchy -top equiv; equiv_simple -short -seq 1; equiv_induct -seq 4; equiv_status -assert" \
  > "$OUT/lec.log"
if ! rg -q "Equivalence successfully proven|Equivalence successfully proved" "$OUT/lec.log"; then
  echo "Yosys LEC未找到成功标记" >&2
  tail -80 "$OUT/lec.log" >&2
  exit 1
fi
echo "$OUT/lec.log"
