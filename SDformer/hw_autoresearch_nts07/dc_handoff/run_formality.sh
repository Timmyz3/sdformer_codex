#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
DESIGN_NAME="${DESIGN_NAME:-h67_attention_top}"
LIB_DB="${LIB_DB:-}"
MACRO_DBS="${MACRO_DBS:-}"
DC_RUN_DIR="${DC_RUN_DIR:-$ROOT/runs/$DESIGN_NAME}"

case "$DESIGN_NAME" in
  h67_attention_top) RTL_FILELIST="$HW_ROOT/rtl_h67/filelist.f" ;;
  h68_castling_deploy_top) RTL_FILELIST="$HW_ROOT/rtl_h68/filelist.f" ;;
  h67_fixed2s_mssb5_dc_top|h67_rqtb2s_mssb5_dc_top)
    RTL_FILELIST="$ROOT/filelists/date_motion_2s.f" ;;
  local5_unified_out2_dc_top)
    RTL_FILELIST="$ROOT/filelists/date_local5_out2.f" ;;
  local5_unified_out2_1rw_dc_top)
    RTL_FILELIST="$ROOT/filelists/date_local5_out2_1rw.f" ;;
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

export DESIGN_NAME HW_ROOT RTL_FILELIST LIB_DB MACRO_DBS
export MAPPED_NETLIST="$DC_RUN_DIR/netlist/${DESIGN_NAME}_mapped.v"
export SVF_FILE="$DC_RUN_DIR/netlist/${DESIGN_NAME}.svf"
export OUTPUT_DIR="$DC_RUN_DIR"
test -s "$MAPPED_NETLIST"
test -s "$SVF_FILE"
fm_shell -f "$ROOT/scripts/run_formality.tcl" | tee "$DC_RUN_DIR/formality.log"
test "$(tr -d '[:space:]' < "$DC_RUN_DIR/reports/formality_status.txt")" = "PASS"
python3 "$ROOT/scripts/write_synopsys_run_manifest.py" \
  --mode formality --design "$DESIGN_NAME" --root "$HW_ROOT" \
  --output "$DC_RUN_DIR/formality_run_manifest.json"
