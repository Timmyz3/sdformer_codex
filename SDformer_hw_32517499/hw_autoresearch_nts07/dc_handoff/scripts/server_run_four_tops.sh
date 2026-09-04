#!/usr/bin/env bash
# Four frozen DATE tops on a Synopsys server. This machine must not run it
# unless dc_shell/fm_shell/pt_shell and LIB_DB are actually present.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$here/run_dc.sh" ]]; then
  DC="$here"
elif [[ -x "$here/../run_dc.sh" ]]; then
  DC="$(cd "$here/.." && pwd)"
elif [[ -x "$here/../dc_handoff/run_dc.sh" ]]; then
  DC="$(cd "$here/../dc_handoff" && pwd)"
else
  echo "找不到 dc_handoff/run_dc.sh。请在解包后的 hw_autoresearch_nts07 下运行，或把本脚本放到 dc_handoff/scripts/。" >&2
  exit 2
fi
HW="$(cd "$DC/.." && pwd)"
cd "$HW"

PHASE="${1:-preflight}"
CLOCK_PERIOD_NS="${CLOCK_PERIOD_NS:-3.0}"
export CLOCK_PERIOD_NS
export PPA_ADMISSION="${PPA_ADMISSION:-0}"

if [[ "$PPA_ADMISSION" == "1" ]]; then
  echo "当前包 MEMORY_IMPL=0，禁止默认 PPA_ADMISSION=1。只有你自备宏 .db 和 EXPECTED_MACRO_REFS 时才允许。" >&2
  exit 6
fi

declare -a DESIGNS=(
  h67_fixed2s_mssb5_dc_top
  h67_rqtb2s_mssb5_dc_top
  local5_unified_out2_dc_top
  local5_unified_out2_1rw_dc_top
)

vcd_for() {
  case "$1" in
    h67_fixed2s_mssb5_dc_top)
      echo dc_handoff/runs/motion_fixed_dc_activity_population138_fair/h67_fixed2s_mssb5_dc_top.vcd
      ;;
    h67_rqtb2s_mssb5_dc_top)
      echo dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/h67_rqtb2s_mssb5_dc_top.vcd
      ;;
    local5_unified_out2_dc_top)
      echo dc_handoff/runs/local5_dc_activity_full_population100/local5_unified_out2_dc_top.vcd
      ;;
    local5_unified_out2_1rw_dc_top)
      echo dc_handoff/runs/local5_1rw_activity_population100_full/local5_unified_out2_1rw_dc_top.vcd
      ;;
  esac
}

contract_for() {
  case "$1" in
    h67_fixed2s_mssb5_dc_top)
      echo dc_handoff/runs/motion_fixed_dc_activity_population138_fair/activity_contract.json
      ;;
    h67_rqtb2s_mssb5_dc_top)
      echo dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/activity_contract.json
      ;;
    local5_unified_out2_dc_top)
      echo dc_handoff/runs/local5_dc_activity_full_population100/activity_contract.json
      ;;
    local5_unified_out2_1rw_dc_top)
      echo dc_handoff/runs/local5_1rw_activity_population100_full/activity_contract.json
      ;;
  esac
}

saif_instance_for() {
  case "$1" in
    h67_fixed2s_mssb5_dc_top) echo TOP/tb_h67_motion_dc_activity/g_fixed/dut ;;
    h67_rqtb2s_mssb5_dc_top) echo TOP/tb_h67_motion_dc_activity/g_rqtb/dut ;;
    local5_unified_out2_dc_top|local5_unified_out2_1rw_dc_top)
      echo TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut
      ;;
  esac
}

need_lib() {
  if [[ -z "${LIB_DB:-}" || ! -f "$LIB_DB" ]]; then
    echo "必须设置 LIB_DB 为服务器上的标准单元 .db。" >&2
    exit 4
  fi
}

preflight() {
  python3 dc_handoff/scripts/audit_date_dual_handoff.py \
    --root . \
    --output dc_handoff/runs/date_dual_handoff_audit_server.json
  python3 scripts/audit_three_line_predc_gate.py \
    --root . \
    --output results/grok_codex_collab/three_line_predc_gate_server.json
  echo "preflight PASS（仍是 READY_PREMACRO，不是 paper PPA）"
}

run_dc_all() {
  need_lib
  command -v dc_shell >/dev/null 2>&1 || {
    echo "未找到 dc_shell。本脚本只应在新思服务器上跑 DC。" >&2
    exit 3
  }
  for design in "${DESIGNS[@]}"; do
    echo "=== DC $design ==="
    DESIGN_NAME="$design" \
    DC_RUN_DIR="dc_handoff/runs/$design" \
    "$DC/run_dc.sh"
  done
}

run_fm_all() {
  need_lib
  command -v fm_shell >/dev/null 2>&1 || {
    echo "未找到 fm_shell。" >&2
    exit 3
  }
  for design in "${DESIGNS[@]}"; do
    echo "=== Formality $design ==="
    DESIGN_NAME="$design" \
    DC_RUN_DIR="dc_handoff/runs/$design" \
    "$DC/run_formality.sh"
  done
}

run_vcd2saif_all() {
  command -v vcd2saif >/dev/null 2>&1 || {
    echo "未找到 vcd2saif。在服务器上转换 VCD，本机不要转。" >&2
    exit 3
  }
  for design in "${DESIGNS[@]}"; do
    local run_dir vcd saif contract
    run_dir="dc_handoff/runs/$design"
    mkdir -p "$run_dir"
    vcd="$(vcd_for "$design")"
    saif="$run_dir/${design}.saif"
    contract="$(contract_for "$design")"
    echo "=== vcd2saif $design ==="
    vcd2saif -input "$vcd" -output "$saif"
    python3 dc_handoff/scripts/make_saif_manifest.py \
      --root . \
      --activity-contract "$contract" \
      --saif "$saif" \
      --output "$run_dir/${design}_saif_manifest.json"
  done
}

run_ptpx_all() {
  need_lib
  if [[ -z "${OPERATING_CONDITION:-}" ]]; then
    echo "PTPX 需要 OPERATING_CONDITION。" >&2
    exit 5
  fi
  command -v pt_shell >/dev/null 2>&1 || {
    echo "未找到 pt_shell。" >&2
    exit 3
  }
  for design in "${DESIGNS[@]}"; do
    echo "=== PTPX $design ==="
    DESIGN_NAME="$design" \
    DC_RUN_DIR="dc_handoff/runs/$design" \
    SAIF_FILE="dc_handoff/runs/$design/${design}.saif" \
    SAIF_INSTANCE="$(saif_instance_for "$design")" \
    SAIF_MANIFEST="dc_handoff/runs/$design/${design}_saif_manifest.json" \
    CORNER_ROLE=power \
    MIN_SAIF_COVERAGE_PCT=95.0 \
    "$DC/run_ptpx.sh"
  done
}

run_ptsta_all() {
  need_lib
  command -v pt_shell >/dev/null 2>&1 || {
    echo "未找到 pt_shell。" >&2
    exit 3
  }
  if [[ -z "${SETUP_LIB_DB:-}" || -z "${SETUP_OPERATING_CONDITION:-}" || \
        -z "${HOLD_LIB_DB:-}" || -z "${HOLD_OPERATING_CONDITION:-}" ]]; then
    echo "PTSTA 需要 SETUP_LIB_DB/SETUP_OPERATING_CONDITION 和 HOLD_LIB_DB/HOLD_OPERATING_CONDITION。" >&2
    exit 5
  fi
  for design in "${DESIGNS[@]}"; do
    echo "=== PTSTA setup $design ==="
    DESIGN_NAME="$design" \
    DC_RUN_DIR="dc_handoff/runs/$design" \
    PT_RUN_DIR="dc_handoff/runs/$design/pt_setup" \
    LIB_DB="$SETUP_LIB_DB" \
    OPERATING_CONDITION="$SETUP_OPERATING_CONDITION" \
    CORNER_ROLE=setup \
    "$DC/run_ptsta.sh"
    echo "=== PTSTA hold $design ==="
    DESIGN_NAME="$design" \
    DC_RUN_DIR="dc_handoff/runs/$design" \
    PT_RUN_DIR="dc_handoff/runs/$design/pt_hold" \
    LIB_DB="$HOLD_LIB_DB" \
    OPERATING_CONDITION="$HOLD_OPERATING_CONDITION" \
    CORNER_ROLE=hold \
    "$DC/run_ptsta.sh"
  done
}

case "$PHASE" in
  preflight) preflight ;;
  dc) run_dc_all ;;
  formality) run_fm_all ;;
  vcd2saif) run_vcd2saif_all ;;
  ptpx) run_ptpx_all ;;
  ptsta) run_ptsta_all ;;
  all)
    preflight
    run_dc_all
    run_fm_all
    if command -v vcd2saif >/dev/null 2>&1; then
      run_vcd2saif_all
    else
      echo "跳过 vcd2saif：服务器上没有该命令。"
    fi
    if command -v pt_shell >/dev/null 2>&1 && [[ -n "${OPERATING_CONDITION:-}" ]]; then
      run_ptpx_all
    fi
    if command -v pt_shell >/dev/null 2>&1 && \
       [[ -n "${SETUP_LIB_DB:-}" && -n "${HOLD_LIB_DB:-}" ]]; then
      run_ptsta_all
    fi
    ;;
  *)
    echo "用法: $0 preflight|dc|formality|vcd2saif|ptpx|ptsta|all" >&2
    exit 2
    ;;
esac
