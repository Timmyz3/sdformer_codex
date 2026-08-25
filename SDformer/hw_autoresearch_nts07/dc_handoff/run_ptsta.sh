#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_NAME="${DESIGN_NAME:-}"
LIB_DB="${LIB_DB:-}"
MACRO_DBS="${MACRO_DBS:-}"
OPERATING_CONDITION="${OPERATING_CONDITION:-}"
CORNER_ROLE="${CORNER_ROLE:-}"
DC_RUN_DIR="${DC_RUN_DIR:-$ROOT/runs/$DESIGN_NAME}"
PT_RUN_DIR="${PT_RUN_DIR:-$DC_RUN_DIR}"
NETLIST_FILE="${NETLIST_FILE:-}"
SDC_FILE="${SDC_FILE:-}"
SPEF_FILE="${SPEF_FILE:-}"

case "$DESIGN_NAME" in
  h67_fixed2s_mssb5_dc_top|h67_rqtb2s_mssb5_dc_top|local5_unified_out2_dc_top|local5_unified_out2_1rw_dc_top|qfit_local_banked_multisource_p1_top|qfit_local_banked_multisource_p2_top|qfit_local_banked_multisource_p4_top|qfit_local_banked_multisource_p8_top|qfit_local_banked_multisource_p1_l96_top|qfit_local_banked_multisource_p2_l96_top|qfit_local_banked_multisource_p4_l96_top|qfit_local_banked_multisource_p8_l96_top|qfit_dual_line_multicontext_engine|qfit_dual_line_descriptor_resident_engine|qfit_dual_granularity_temporal_state_engine|hitflow_dptme_paper_top) ;;
  *) echo "PrimeTime STA仅接受当前DATE双线冻结顶层: $DESIGN_NAME" >&2; exit 2 ;;
esac

if ! command -v pt_shell >/dev/null 2>&1; then
  echo "未找到pt_shell，不能执行PrimeTime STA。" >&2
  exit 3
fi
if [[ -z "$LIB_DB" || ! -f "$LIB_DB" ]]; then
  echo "必须通过LIB_DB指定与DC一致的Synopsys .db库。" >&2
  exit 4
fi
if [[ -z "$OPERATING_CONDITION" || -z "$CORNER_ROLE" ]]; then
  echo "PrimeTime STA要求OPERATING_CONDITION和CORNER_ROLE明确记录角点。" >&2
  exit 5
fi
if [[ -n "$MACRO_DBS" ]]; then
  IFS=: read -r -a macro_db_paths <<< "$MACRO_DBS"
  for macro_db in "${macro_db_paths[@]}"; do
    [[ -n "$macro_db" && -f "$macro_db" ]] || { echo "无效MACRO_DBS: $macro_db" >&2; exit 6; }
  done
fi
if [[ -n "$SPEF_FILE" && -z "$NETLIST_FILE" ]]; then
  echo "读取SPEF时必须通过NETLIST_FILE显式提供产生该SPEF的P&R网表。" >&2
  exit 7
fi

export DESIGN_NAME LIB_DB MACRO_DBS OPERATING_CONDITION CORNER_ROLE
export MAPPED_NETLIST="${NETLIST_FILE:-$DC_RUN_DIR/netlist/${DESIGN_NAME}_mapped.v}"
export OUTPUT_DIR="$PT_RUN_DIR"
mkdir -p "$PT_RUN_DIR/netlist"
test -s "$MAPPED_NETLIST"
export MAPPED_SDC_SOURCE="${SDC_FILE:-$DC_RUN_DIR/netlist/${DESIGN_NAME}_mapped.sdc}"
test -s "$MAPPED_SDC_SOURCE"
export MAPPED_SDC="$PT_RUN_DIR/netlist/${DESIGN_NAME}_${CORNER_ROLE}_effective.sdc"
prepare_sdc_args=(
  --source "$MAPPED_SDC_SOURCE" --output "$MAPPED_SDC"
  --operating-condition "$OPERATING_CONDITION"
)
if [[ -n "$SDC_FILE" ]]; then
  prepare_sdc_args+=(--allow-corner-neutral-source)
fi
python3 "$ROOT/scripts/prepare_pt_sdc.py" "${prepare_sdc_args[@]}"
if [[ -n "$SPEF_FILE" ]]; then
  test -s "$SPEF_FILE"
  export SPEF_FILE
fi
pt_shell -f "$ROOT/scripts/run_ptsta.tcl" | tee "$PT_RUN_DIR/ptsta.log"
python3 "$ROOT/scripts/write_synopsys_run_manifest.py" \
  --mode ptsta --design "$DESIGN_NAME" --root "$ROOT/.." \
  --output "$PT_RUN_DIR/ptsta_run_manifest.json"
python3 "$ROOT/scripts/audit_synopsys_postrun.py" \
  --mode ptsta \
  --run-dir "$PT_RUN_DIR"
