#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
MODE="${MODE:-}"
VECTORS="${VECTORS:-}"
ROW_LIMIT="${ROW_LIMIT:-1}"
DUMP_START_ROW="${DUMP_START_ROW:-0}"
DUMP_ROWS="${DUMP_ROWS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/motion_${MODE}_activity}"
ACTIVITY_PURPOSE="${ACTIVITY_PURPOSE:-identity_smoke}"

case "$MODE" in
  fixed) RQTB=0; DESIGN_NAME=h67_fixed2s_mssb5_dc_top; WRAPPER_SCOPE=g_fixed ;;
  rqtb) RQTB=1; DESIGN_NAME=h67_rqtb2s_mssb5_dc_top; WRAPPER_SCOPE=g_rqtb ;;
  *) echo "MODE必须是fixed或rqtb。" >&2; exit 2 ;;
esac
if ! command -v verilator >/dev/null 2>&1; then
  echo "未找到Verilator。" >&2
  exit 3
fi
if [[ -z "$VECTORS" || ! -s "$VECTORS" ]]; then
  echo "必须通过VECTORS指定真实Motion行向量。" >&2
  exit 4
fi
VECTORS="$(cd "$(dirname "$VECTORS")" && pwd)/$(basename "$VECTORS")"
if (( ROW_LIMIT < 1 || ROW_LIMIT > 138 || DUMP_START_ROW < 0 \
      || DUMP_ROWS < 1 || DUMP_START_ROW + DUMP_ROWS > ROW_LIMIT )); then
  echo "ROW范围非法。" >&2
  exit 5
fi

mkdir -p "$OUTPUT_DIR"
OBJ_DIR="$OUTPUT_DIR/obj"
VCD_FILE="$OUTPUT_DIR/${DESIGN_NAME}.vcd"
SIM_LOG="$OUTPUT_DIR/simulation.log"
CONTRACT="$OUTPUT_DIR/activity_contract.json"
mapfile -t RTL_FILES < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' \
  "$ROOT/filelists/date_motion_2s.f")
cd "$HW_ROOT"

verilator --binary --timing --assert --trace -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDPARAM -Wno-UNDRIVEN \
  --top-module tb_h67_motion_dc_activity \
  -GRQTB="$RQTB" -GMAX_ROWS=138 -GROW_LIMIT="$ROW_LIMIT" \
  --Mdir "$OBJ_DIR" \
  "${RTL_FILES[@]}" \
  "$HW_ROOT/tb_h67/tb_h67_motion_dc_activity.sv" \
  "$HW_ROOT/verif_h67/h67_temporal_slot_flow_2s_assertions.sv" \
  2>&1 | tee "$OUTPUT_DIR/compile.log"

"$OBJ_DIR/Vtb_h67_motion_dc_activity" \
  +VECTORS="$VECTORS" \
  +DUMP_FILE="$VCD_FILE" \
  +DUMP_START_ROW="$DUMP_START_ROW" \
  +DUMP_ROWS="$DUMP_ROWS" \
  | tee "$SIM_LOG"

python3 "$ROOT/scripts/report_activity_vcd.py" \
  --design "$DESIGN_NAME" \
  --vcd "$VCD_FILE" \
  --log "$SIM_LOG" \
  --trace-root "$VECTORS" \
  --strip-path "TOP/tb_h67_motion_dc_activity/$WRAPPER_SCOPE/dut" \
  --purpose "$ACTIVITY_PURPOSE" \
  --measurement-scope fair_lfsr_row_execution \
  --output "$CONTRACT"
echo "$CONTRACT"
