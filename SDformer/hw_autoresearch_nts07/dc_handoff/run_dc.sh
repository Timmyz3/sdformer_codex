#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
DESIGN_NAME="${DESIGN_NAME:-h67_attention_top}"
LIB_DB="${LIB_DB:-}"
MIN_LIB_DB="${MIN_LIB_DB:-${HOLD_LIB_DB:-}}"
DC_HOLD_UNCERTAINTY_NS="${DC_HOLD_UNCERTAINTY_NS:-0.100}"
DC_HOLD_REPORT_UNCERTAINTY_NS="${DC_HOLD_REPORT_UNCERTAINTY_NS:-0.090}"
MACRO_DBS="${MACRO_DBS:-}"
OPERATING_CONDITION="${OPERATING_CONDITION:-}"
PPA_ADMISSION="${PPA_ADMISSION:-0}"
EXPECTED_MACRO_REFS="${EXPECTED_MACRO_REFS:-}"
SAIF_FILE="${SAIF_FILE:-}"
SAIF_INSTANCE="${SAIF_INSTANCE:-}"
SAIF_MANIFEST="${SAIF_MANIFEST:-}"
ELAB_PARAMETERS="${ELAB_PARAMETERS:-}"
PREMACRO_LOGICAL_SHELL_PATTERN="${PREMACRO_LOGICAL_SHELL_PATTERN:-}"
PREMACRO_LOGICAL_SHELL_COUNT="${PREMACRO_LOGICAL_SHELL_COUNT:-}"

case "$DESIGN_NAME" in
  h67_attention_top) RTL_FILELIST="$HW_ROOT/rtl_h67/filelist.f" ;;
  h68_castling_deploy_top) RTL_FILELIST="$HW_ROOT/rtl_h68/filelist.f" ;;
  h67_fixed2s_mssb5_dc_top|h67_rqtb2s_mssb5_dc_top)
    RTL_FILELIST="$ROOT/filelists/date_motion_2s.f" ;;
  local5_unified_out2_dc_top)
    RTL_FILELIST="$ROOT/filelists/date_local5_out2.f" ;;
  local5_unified_out2_1rw_dc_top)
    RTL_FILELIST="$ROOT/filelists/date_local5_out2_1rw.f" ;;
  qfit_dual_line_stateful_tile_top)
    RTL_FILELIST="$ROOT/filelists/date_dual_line_stateful_tile.f" ;;
  qfit_local_banked_multisource_p1_top|qfit_local_banked_multisource_p2_top|qfit_local_banked_multisource_p4_top|qfit_local_banked_multisource_p8_top)
    RTL_FILELIST="$ROOT/filelists/date_local_banked_multisource.f" ;;
  qfit_local_banked_multisource_p1_l96_top|qfit_local_banked_multisource_p2_l96_top|qfit_local_banked_multisource_p4_l96_top|qfit_local_banked_multisource_p8_l96_top)
    RTL_FILELIST="$ROOT/filelists/date_local_banked_multisource_l96.f" ;;
  qfit_dual_line_multicontext_engine)
    RTL_FILELIST="$ROOT/filelists/date_dual_line_multicontext.f" ;;
  qfit_dual_line_descriptor_resident_engine)
    RTL_FILELIST="$ROOT/filelists/date_dual_line_descriptor_resident.f" ;;
  qfit_dual_line_descriptor_stateful_engine)
    RTL_FILELIST="$ROOT/filelists/date_m4_descriptor_stateful_premacro.f" ;;
  qfit_dual_granularity_temporal_state_engine)
    RTL_FILELIST="$ROOT/filelists/date_m9_dual_granularity_state.f" ;;
  hitflow_dptme_paper_top)
    RTL_FILELIST="$ROOT/filelists/date_m7_atlif_dptme.f" ;;
  gatestack_single_context_execution_top)
    RTL_FILELIST="$HW_ROOT/rtl_hitflow/filelist_single_context_execution.f" ;;
  *) echo "不支持的DESIGN_NAME: $DESIGN_NAME" >&2; exit 2 ;;
esac

if ! command -v dc_shell >/dev/null 2>&1; then
  echo "未找到dc_shell；交付包已生成，但本机不能执行Design Compiler。" >&2
  exit 3
fi
if [[ -z "$LIB_DB" || ! -f "$LIB_DB" ]]; then
  echo "必须通过LIB_DB指定有效的Synopsys .db标准单元库。" >&2
  exit 4
fi
if [[ -n "$MIN_LIB_DB" && ! -f "$MIN_LIB_DB" ]]; then
  echo "MIN_LIB_DB/HOLD_LIB_DB不是有效的Synopsys .db: $MIN_LIB_DB" >&2
  exit 8
fi
if [[ -n "$MACRO_DBS" ]]; then
  IFS=: read -r -a macro_db_paths <<< "$MACRO_DBS"
  for macro_db in "${macro_db_paths[@]}"; do
    if [[ -z "$macro_db" || ! -f "$macro_db" ]]; then
      echo "MACRO_DBS包含无效.db: $macro_db" >&2
      exit 5
    fi
  done
fi
if [[ "$PPA_ADMISSION" == "1" ]]; then
  if [[ -z "$OPERATING_CONDITION" || -z "$MACRO_DBS" || -z "$EXPECTED_MACRO_REFS" ]]; then
    echo "PPA_ADMISSION=1要求OPERATING_CONDITION、MACRO_DBS和EXPECTED_MACRO_REFS。" >&2
    exit 6
  fi
fi
if [[ -n "$SAIF_FILE" ]]; then
  if [[ ! -s "$SAIF_FILE" || -z "$SAIF_INSTANCE" || -z "$SAIF_MANIFEST" || ! -s "$SAIF_MANIFEST" ]]; then
    echo "DC读取活动时必须同时提供有效SAIF_FILE、SAIF_INSTANCE和SAIF_MANIFEST。" >&2
    exit 7
  fi
  python3 "$ROOT/scripts/audit_saif_manifest.py" \
    --design "$DESIGN_NAME" --saif "$SAIF_FILE" \
    --strip-path "$SAIF_INSTANCE" --manifest "$SAIF_MANIFEST"
  if [[ "$PPA_ADMISSION" == "1" ]]; then
    python3 "$ROOT/scripts/audit_saif_manifest.py" \
      --design "$DESIGN_NAME" --saif "$SAIF_FILE" \
      --strip-path "$SAIF_INSTANCE" --manifest "$SAIF_MANIFEST" \
      --require-paper-power-eligible
  fi
fi

export DESIGN_NAME HW_ROOT RTL_FILELIST LIB_DB MIN_LIB_DB MACRO_DBS OPERATING_CONDITION
export DC_HOLD_UNCERTAINTY_NS
export DC_HOLD_REPORT_UNCERTAINTY_NS
export PPA_ADMISSION EXPECTED_MACRO_REFS
export SAIF_FILE SAIF_INSTANCE SAIF_MANIFEST
export ELAB_PARAMETERS
export PREMACRO_LOGICAL_SHELL_PATTERN PREMACRO_LOGICAL_SHELL_COUNT
if [[ "$DESIGN_NAME" == "gatestack_single_context_execution_top" ]]; then
  export SDC_FILE="$ROOT/constraints/gatestack_single_context_500mhz.sdc"
elif [[ "$DESIGN_NAME" == "h67_fixed2s_mssb5_dc_top" \
     || "$DESIGN_NAME" == "h67_rqtb2s_mssb5_dc_top" \
     || "$DESIGN_NAME" == "local5_unified_out2_dc_top" \
     || "$DESIGN_NAME" == "local5_unified_out2_1rw_dc_top" \
     || "$DESIGN_NAME" == "qfit_dual_line_stateful_tile_top" \
     || "$DESIGN_NAME" == "qfit_dual_line_multicontext_engine" \
     || "$DESIGN_NAME" == "qfit_dual_line_descriptor_resident_engine" \
     || "$DESIGN_NAME" == "qfit_dual_line_descriptor_stateful_engine" \
     || "$DESIGN_NAME" == "qfit_dual_granularity_temporal_state_engine" \
     || "$DESIGN_NAME" == "hitflow_dptme_paper_top" \
     || "$DESIGN_NAME" == qfit_local_banked_multisource_p*_top ]]; then
  export SDC_FILE="$ROOT/constraints/date_dual_core.sdc"
  export CLOCK_PERIOD_NS="${CLOCK_PERIOD_NS:-3.000}"
else
  export SDC_FILE="$ROOT/constraints/h67_h68_500mhz.sdc"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/${DESIGN_NAME}}"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/reports/power.rpt" "$OUTPUT_DIR/reports/power_hierarchy.rpt"
# A new mapped netlist invalidates every older equivalence receipt.  Remove
# those receipts before DC starts so an interrupted rerun cannot look like a
# DC+Formality PASS by inheriting stale files.
rm -f "$OUTPUT_DIR/formality.log" "$OUTPUT_DIR/formality_run_manifest.json" \
  "$OUTPUT_DIR/reports/formality_status.txt" \
  "$OUTPUT_DIR/reports/formality_unmatched.rpt" \
  "$OUTPUT_DIR/reports/formality_verify.rpt"
dc_shell -f "$ROOT/scripts/run_dc.tcl" | tee "$OUTPUT_DIR/dc.log"
python3 "$ROOT/scripts/write_synopsys_run_manifest.py" \
  --mode dc --design "$DESIGN_NAME" --root "$HW_ROOT" \
  --output "$OUTPUT_DIR/dc_run_manifest.json"
python3 "$ROOT/scripts/audit_dc_artifacts.py" \
  --design "$DESIGN_NAME" \
  --run-dir "$OUTPUT_DIR"
if [[ -n "$EXPECTED_MACRO_REFS" ]]; then
  python3 "$ROOT/scripts/audit_expected_macro_refs.py" \
    --report "$OUTPUT_DIR/reports/references.rpt" \
    --expected "$EXPECTED_MACRO_REFS" \
    --output "$OUTPUT_DIR/reports/macro_reference_audit.json"
fi
