#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
DESIGN_NAME="${DESIGN_NAME:?DESIGN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
LIB_DB="${LIB_DB:?LIB_DB is required}"
MIN_LIB_DB="${MIN_LIB_DB:-${HOLD_LIB_DB:-}}"
OPERATING_CONDITION="${OPERATING_CONDITION:?OPERATING_CONDITION is required}"
DC_HOLD_UNCERTAINTY_NS="${DC_HOLD_UNCERTAINTY_NS:-0.100}"
DC_HOLD_REPORT_UNCERTAINTY_NS="${DC_HOLD_REPORT_UNCERTAINTY_NS:-0.090}"
DDC_FILE="$OUTPUT_DIR/netlist/${DESIGN_NAME}.ddc"

test -s "$LIB_DB"
test -s "$MIN_LIB_DB"
test -s "$DDC_FILE"
command -v dc_shell >/dev/null 2>&1

case "$DESIGN_NAME" in
  qfit_local_banked_multisource_p1_l96_top|qfit_local_banked_multisource_p2_l96_top|qfit_local_banked_multisource_p4_l96_top|qfit_local_banked_multisource_p8_l96_top)
    RTL_FILELIST="$ROOT/filelists/date_local_banked_multisource_l96.f" ;;
  qfit_dual_line_multicontext_engine)
    RTL_FILELIST="$ROOT/filelists/date_dual_line_multicontext.f" ;;
  qfit_dual_line_descriptor_resident_engine)
    RTL_FILELIST="$ROOT/filelists/date_dual_line_descriptor_resident.f" ;;
  *) echo "hold-report refresh is not admitted for $DESIGN_NAME" >&2; exit 2 ;;
esac
SDC_FILE="$ROOT/constraints/date_dual_core.sdc"
export DESIGN_NAME OUTPUT_DIR LIB_DB MIN_LIB_DB OPERATING_CONDITION
export DC_HOLD_UNCERTAINTY_NS DC_HOLD_REPORT_UNCERTAINTY_NS DDC_FILE
export HW_ROOT RTL_FILELIST SDC_FILE

dc_shell -f "$ROOT/scripts/report_dc_hold_guard.tcl" \
  | tee "$OUTPUT_DIR/dc_hold_report_refresh.log"
python3 "$ROOT/scripts/write_synopsys_run_manifest.py" \
  --mode dc --design "$DESIGN_NAME" --root "$HW_ROOT" \
  --output "$OUTPUT_DIR/dc_run_manifest.json"
python3 "$ROOT/scripts/audit_dc_artifacts.py" \
  --design "$DESIGN_NAME" --run-dir "$OUTPUT_DIR"
