#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_NAME="${DESIGN_NAME:-}"
LIB_DB="${LIB_DB:-}"
MACRO_DBS="${MACRO_DBS:-}"
OPERATING_CONDITION="${OPERATING_CONDITION:-}"
CORNER_ROLE="${CORNER_ROLE:-power}"
DC_RUN_DIR="${DC_RUN_DIR:-$ROOT/runs/$DESIGN_NAME}"
PTPX_RUN_DIR="${PTPX_RUN_DIR:-$DC_RUN_DIR}"
SAIF_FILE="${SAIF_FILE:-}"
SAIF_INSTANCE="${SAIF_INSTANCE:-}"
SAIF_MANIFEST="${SAIF_MANIFEST:-}"
MIN_SAIF_COVERAGE_PCT="${MIN_SAIF_COVERAGE_PCT:-95.0}"
PTPX_REQUIRE_PAPER_POWER_ELIGIBLE="${PTPX_REQUIRE_PAPER_POWER_ELIGIBLE:-1}"
NETLIST_FILE="${NETLIST_FILE:-}"
SDC_FILE="${SDC_FILE:-}"
SPEF_FILE="${SPEF_FILE:-}"
RTL_GATE_MAP_TCL="${RTL_GATE_MAP_TCL:-}"

case "$DESIGN_NAME" in
  h67_fixed2s_mssb5_dc_top|h67_rqtb2s_mssb5_dc_top|local5_unified_out2_dc_top|local5_unified_out2_1rw_dc_top|qfit_local_banked_multisource_p1_top|qfit_local_banked_multisource_p2_top|qfit_local_banked_multisource_p4_top|qfit_local_banked_multisource_p8_top|qfit_local_banked_multisource_p1_l96_top|qfit_local_banked_multisource_p2_l96_top|qfit_local_banked_multisource_p4_l96_top|qfit_local_banked_multisource_p8_l96_top|qfit_dual_line_descriptor_resident_engine|qfit_dual_granularity_temporal_state_engine|hitflow_dptme_paper_top) ;;
  *) echo "PTPX仅接受当前DATE双线冻结顶层: $DESIGN_NAME" >&2; exit 2 ;;
esac

if ! command -v pt_shell >/dev/null 2>&1; then
  echo "未找到pt_shell，不能执行PrimeTime PX功耗分析。" >&2
  exit 3
fi
if [[ -z "$LIB_DB" || ! -f "$LIB_DB" ]]; then
  echo "必须通过LIB_DB指定与DC一致的Synopsys .db库。" >&2
  exit 4
fi
if [[ -z "$SAIF_FILE" || ! -s "$SAIF_FILE" ]]; then
  echo "必须通过SAIF_FILE提供非空的真实trace活动文件。" >&2
  exit 5
fi
if [[ -z "$SAIF_INSTANCE" ]]; then
  echo "必须通过SAIF_INSTANCE指定SAIF相对综合网表的层次前缀。" >&2
  exit 6
fi
if [[ -z "$OPERATING_CONDITION" ]]; then
  echo "PTPX要求OPERATING_CONDITION明确记录功耗角。" >&2
  exit 7
fi
if [[ -z "$SAIF_MANIFEST" || ! -s "$SAIF_MANIFEST" ]]; then
  echo "必须通过SAIF_MANIFEST提供trace、测量区间和SAIF身份合同。" >&2
  exit 8
fi
if [[ -n "$MACRO_DBS" ]]; then
  IFS=: read -r -a macro_db_paths <<< "$MACRO_DBS"
  for macro_db in "${macro_db_paths[@]}"; do
    [[ -n "$macro_db" && -f "$macro_db" ]] || { echo "无效MACRO_DBS: $macro_db" >&2; exit 9; }
  done
fi
if [[ -n "$SPEF_FILE" && -z "$NETLIST_FILE" ]]; then
  echo "读取SPEF时必须通过NETLIST_FILE显式提供产生该SPEF的P&R网表。" >&2
  exit 10
fi
if [[ -n "$RTL_GATE_MAP_TCL" && ! -s "$RTL_GATE_MAP_TCL" ]]; then
  echo "RTL_GATE_MAP_TCL不是有效的PrimeTime映射文件: $RTL_GATE_MAP_TCL" >&2
  exit 9
fi
audit_args=(
  --design "$DESIGN_NAME" --saif "$SAIF_FILE"
  --strip-path "$SAIF_INSTANCE" --manifest "$SAIF_MANIFEST"
)
if [[ "$PTPX_REQUIRE_PAPER_POWER_ELIGIBLE" == "1" ]]; then
  audit_args+=(--require-paper-power-eligible)
elif [[ "$PTPX_REQUIRE_PAPER_POWER_ELIGIBLE" != "0" ]]; then
  echo "PTPX_REQUIRE_PAPER_POWER_ELIGIBLE必须是0或1。" >&2
  exit 11
fi
python3 "$ROOT/scripts/audit_saif_manifest.py" "${audit_args[@]}"

export DESIGN_NAME LIB_DB MACRO_DBS OPERATING_CONDITION CORNER_ROLE
export RTL_GATE_MAP_TCL
export SAIF_FILE SAIF_INSTANCE SAIF_MANIFEST MIN_SAIF_COVERAGE_PCT
export MAPPED_NETLIST="${NETLIST_FILE:-$DC_RUN_DIR/netlist/${DESIGN_NAME}_mapped.v}"
export OUTPUT_DIR="$PTPX_RUN_DIR"
mkdir -p "$PTPX_RUN_DIR/netlist" "$PTPX_RUN_DIR/reports"
test -s "$MAPPED_NETLIST"
export MAPPED_SDC_SOURCE="${SDC_FILE:-$DC_RUN_DIR/netlist/${DESIGN_NAME}_mapped.sdc}"
test -s "$MAPPED_SDC_SOURCE"
export MAPPED_SDC="$PTPX_RUN_DIR/netlist/${DESIGN_NAME}_ptpx_effective.sdc"
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
pt_shell -f "$ROOT/scripts/run_ptpx.tcl" | tee "$PTPX_RUN_DIR/ptpx.log"
python3 "$ROOT/scripts/write_synopsys_run_manifest.py" \
  --mode ptpx --design "$DESIGN_NAME" --root "$ROOT/.." \
  --output "$PTPX_RUN_DIR/ptpx_run_manifest.json"
python3 "$ROOT/scripts/audit_synopsys_postrun.py" \
  --mode ptpx \
  --run-dir "$PTPX_RUN_DIR" \
  --min-saif-coverage-pct "$MIN_SAIF_COVERAGE_PCT"
