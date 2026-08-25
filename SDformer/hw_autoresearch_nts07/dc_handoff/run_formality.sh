#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
DESIGN_NAME="${DESIGN_NAME:-h67_attention_top}"
LIB_DB="${LIB_DB:-}"
MACRO_DBS="${MACRO_DBS:-}"
DC_RUN_DIR="${DC_RUN_DIR:-$ROOT/runs/$DESIGN_NAME}"
ELAB_PARAMETERS="${ELAB_PARAMETERS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
  qfit_atlif_unified_t10_t2_stream_core)
    RTL_FILELIST="$ROOT/filelists/date_m31_unified_t10_t2_dc.f" ;;
  qfit_complement_csd8_late_scale)
    RTL_FILELIST="$ROOT/filelists/date_m35_complement_csd8_dc.f" ;;
  qfit_threshold_late_scale_uq0p24_radix20x4)
    RTL_FILELIST="$ROOT/filelists/date_m33_threshold_late_scale_uq0p24_dc.f" ;;
  gatestack_single_context_execution_top)
    RTL_FILELIST="$HW_ROOT/rtl_hitflow/filelist_single_context_execution.f" ;;
  *) echo "不支持的DESIGN_NAME: $DESIGN_NAME" >&2; exit 2 ;;
esac

if ! command -v fm_shell >/dev/null 2>&1; then
  echo "未找到fm_shell，不能执行正式RTL到DC网表等价验证。" >&2
  exit 3
fi
if [[ -z "$LIB_DB" || ! -f "$LIB_DB" ]]; then
  echo "必须通过LIB_DB指定与DC一致的Synopsys .db库。" >&2
  exit 4
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

export DESIGN_NAME HW_ROOT RTL_FILELIST LIB_DB MACRO_DBS ELAB_PARAMETERS
export MAPPED_NETLIST="$DC_RUN_DIR/netlist/${DESIGN_NAME}_mapped.v"
export SVF_FILE="$DC_RUN_DIR/netlist/${DESIGN_NAME}.svf"
export OUTPUT_DIR="$DC_RUN_DIR"
test -s "$MAPPED_NETLIST"
test -s "$SVF_FILE"
IMPLEMENTATION_TOP="$DESIGN_NAME"
if [[ -n "$ELAB_PARAMETERS" ]]; then
  IMPLEMENTATION_TOP="$("$PYTHON_BIN" - "$MAPPED_NETLIST" "$DESIGN_NAME" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
design = sys.argv[2]
pattern = re.compile(
    r"^\s*module\s+(" + re.escape(design) + r"_[A-Za-z0-9_$]+)\s*\(",
    re.MULTILINE,
)
matches = sorted(set(pattern.findall(path.read_text(encoding="utf-8"))))
if len(matches) != 1:
    raise SystemExit(
        "expected exactly one parameterized implementation top for {}: {}".format(
            design, matches
        )
    )
print(matches[0])
PY
)"
fi
export IMPLEMENTATION_TOP
FORMALITY_ATTEMPT_TAG="${FORMALITY_ATTEMPT_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$FORMALITY_ATTEMPT_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "FORMALITY_ATTEMPT_TAG包含不安全字符" >&2
  exit 6
fi
FORMALITY_ATTEMPT_LOG="$DC_RUN_DIR/formality_${FORMALITY_ATTEMPT_TAG}.log"
FORMALITY_ATTEMPT_STATUS="$DC_RUN_DIR/formality_${FORMALITY_ATTEMPT_TAG}.exit_status"
FORMALITY_STATUS="$DC_RUN_DIR/reports/formality_status.txt"
if [[ -e "$FORMALITY_ATTEMPT_LOG" || -e "$FORMALITY_ATTEMPT_STATUS" ]]; then
  echo "拒绝覆盖既有Formality attempt: $FORMALITY_ATTEMPT_TAG" >&2
  exit 7
fi
# A prior PASS/FAIL file must never survive a tool crash and be mistaken for
# this attempt's result.  The attempt-specific log preserves every SIGSEGV or
# license failure even though the canonical log is refreshed for manifest use.
rm -f "$FORMALITY_STATUS"
set +e
fm_shell -f "$ROOT/scripts/run_formality.tcl" \
  | tee "$FORMALITY_ATTEMPT_LOG" "$DC_RUN_DIR/formality.log"
FORMALITY_RC=${PIPESTATUS[0]}
set -e
echo "$FORMALITY_RC" > "$FORMALITY_ATTEMPT_STATUS"
if [[ "$FORMALITY_RC" -ne 0 ]]; then
  echo "Formality attempt失败，完整日志: $FORMALITY_ATTEMPT_LOG" >&2
  exit "$FORMALITY_RC"
fi
if [[ ! -s "$FORMALITY_STATUS" ]] \
    || [[ "$(tr -d '[:space:]' < "$FORMALITY_STATUS")" != "PASS" ]]; then
  echo "Formality未生成当前attempt的PASS状态" >&2
  exit 8
fi
"$PYTHON_BIN" "$ROOT/scripts/write_synopsys_run_manifest.py" \
  --mode formality --design "$DESIGN_NAME" --root "$HW_ROOT" \
  --output "$DC_RUN_DIR/formality_run_manifest.json"
