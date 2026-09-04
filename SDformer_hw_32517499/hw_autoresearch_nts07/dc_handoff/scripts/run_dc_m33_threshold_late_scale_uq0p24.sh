#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_design=qfit_threshold_late_scale_uq0p24_radix20x4
task_filelist="${task_dc_root}/filelists/date_m33_threshold_late_scale_uq0p24_dc.f"
task_sdc="${task_dc_root}/constraints/date_m33_threshold_late_scale_uq0p24.sdc"
task_period="${CLOCK_PERIOD_NS:-3.000}"
task_period_tag="${task_period//./p}ns"
task_output="${OUTPUT_DIR:-${task_dc_root}/runs/m33_uq_dc_${task_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"

task_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$task_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$task_fast_default}"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M33b DC requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$LIB_DB" || ! -f "$MIN_LIB_DB" ]]; then
    echo "M33b DC slow/fast library is missing" >&2
    exit 3
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
    || pgrep -f 'common_shell_exec.*-shell[[:space:]]+dc_shell|/dc_shell([[:space:]]|$)' >/dev/null; then
    echo "refusing M33b DC because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$task_output/dc.log" ]]; then
    echo "refusing to overwrite M33b DC evidence: $task_output" >&2
    exit 5
fi

mkdir -p "$task_output"
export DESIGN_NAME="$task_design"
export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_filelist"
export SDC_FILE="$task_sdc"
export OUTPUT_DIR="$task_output"
export CLOCK_PERIOD_NS="$task_period"
export LIB_DB MIN_LIB_DB
export OPERATING_CONDITION="${OPERATING_CONDITION:-ssg0p9v125c}"

dc_shell -f "${task_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24.tcl" \
    2>&1 | tee "$task_output/dc.log"

if grep -q '^Error:' "$task_output/dc.log"; then
    echo "M33b DC log contains a Tcl/DC error" >&2
    exit 9
fi
for task_report in \
    reports/m33_uq_resource_precompile.rpt \
    reports/m33_uq_resource_postcompile.rpt \
    reports/references_postcompile.rpt reports/qor.rpt reports/area.rpt \
    reports/timing_setup.rpt reports/timing_hold.rpt \
    reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt \
    netlist/${task_design}_mapped.v netlist/${task_design}.ddc; do
    if [[ ! -s "$task_output/$task_report" ]]; then
        echo "M33b DC missing evidence: $task_report" >&2
        exit 6
    fi
done
if [[ "$(grep -c '^status=PASS_ONE_POOL_96_LOGICAL_80_MAPPED_16_CONSTANT_SPARES_REMOVED$' \
        "$task_output/reports/m33_uq_resource_postcompile.rpt")" -ne 1 ]]; then
    echo "M33b exact resource audit did not pass" >&2
    exit 7
fi
if grep -Eiq '(^|[^A-Za-z])(GTECH|DW_[A-Za-z0-9_]*mult|unresolved)([^A-Za-z]|$)' \
        "$task_output/reports/references_postcompile.rpt"; then
    echo "M33b postcompile references contain unresolved arithmetic" >&2
    exit 8
fi

task_timing_status=NOT_MET
if grep -q 'slack (MET)' "$task_output/reports/timing_setup.rpt" \
    && grep -q 'slack (MET)' "$task_output/reports/timing_hold.rpt" \
    && ! grep -q 'slack (VIOLATED)' "$task_output/reports/timing_setup.rpt" \
    && ! grep -q 'slack (VIOLATED)' "$task_output/reports/timing_hold.rpt"; then
    task_timing_status=MET
fi
task_warning_count="$(grep -c '^Warning:' \
    "$task_output/reports/check_design_postcompile.rpt" || true)"
{
    echo "status=PASS_STANDALONE_UQ_CLIENT_DC"
    echo "paper_ppa_ready=false"
    echo "system_unique_pool_admitted=false"
    echo "clock_period_ns=$task_period"
    echo "timing_status=$task_timing_status"
    echo "declared_pool_count=1"
    echo "declared_multiplier_leaf_count=96"
    echo "mapped_active_leaf_count=80"
    echo "mapped_spare_leaf_count=0"
    echo "constant_spare_leaf_count_removed=16"
    echo "check_design_warning_count=$task_warning_count"
    echo "macro_db=NONE"
    echo "library_slow=$LIB_DB"
    echo "library_fast=$MIN_LIB_DB"
} > "$task_output/admission.txt"

sha256sum \
    "$task_hw_root/rtl_m31/qfit_signed_int8_mul96_pool.sv" \
    "$task_hw_root/rtl_m33/qfit_threshold_late_scale_uq0p24_radix20x4.sv" \
    "$task_filelist" "$task_sdc" \
    "${task_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24.tcl" \
    "${task_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24.sh" \
    "$LIB_DB" "$MIN_LIB_DB" "$task_output/dc.log" \
    "$task_output/admission.txt" \
    "$task_output/reports/m33_uq_resource_precompile.rpt" \
    "$task_output/reports/m33_uq_resource_postcompile.rpt" \
    "$task_output/reports/qor.rpt" "$task_output/reports/area.rpt" \
    "$task_output/reports/timing_setup.rpt" \
    "$task_output/reports/timing_hold.rpt" \
    "$task_output/netlist/${task_design}_mapped.v" \
    "$task_output/netlist/${task_design}.ddc" \
    > "$task_output/evidence.sha256"
echo "M33_UQ_DC_COMPLETE timing=$task_timing_status run=$task_output"
