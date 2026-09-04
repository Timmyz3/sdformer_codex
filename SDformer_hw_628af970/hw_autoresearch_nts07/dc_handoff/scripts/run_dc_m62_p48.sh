#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_design=qfit_head_p48_signed_lane_fold
task_filelist="${task_dc_root}/filelists/date_m62_p48_dc.f"
task_sdc="${task_dc_root}/constraints/date_m62_p48.sdc"
task_period="${CLOCK_PERIOD_NS:-3.000}"
task_period_tag="${task_period//./p}ns"
task_output="${OUTPUT_DIR:-${task_dc_root}/runs/m62_p48_dc_${task_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"

task_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$task_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$task_fast_default}"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M62 DC requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$LIB_DB" || ! -f "$MIN_LIB_DB" ]]; then
    echo "M62 DC slow/fast library is missing" >&2
    exit 3
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    echo "refusing M62 DC because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$task_output/dc.log" ]]; then
    echo "refusing to overwrite M62 DC evidence: $task_output" >&2
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

{
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "headline_admitted=false"
    echo "clock_period_ns=$task_period"
    echo "macros=0"
    echo "identity=M62_P48x2_S8_SIGNED13_PREMACRO_LOGIC_ONLY"
} > "$task_output/admission.txt"

dc_shell -f "${task_dc_root}/scripts/run_dc_m62_p48.tcl" \
    2>&1 | tee "$task_output/dc.log"

if grep -q '^Error:' "$task_output/dc.log"; then
    echo "M62 DC log contains a Tcl/DC error" >&2
    exit 9
fi
for task_report in \
    reports/qor.rpt reports/area.rpt \
    reports/timing_setup.rpt reports/timing_hold.rpt \
    netlist/${task_design}_mapped.v; do
    if [[ ! -s "$task_output/$task_report" ]]; then
        echo "M62 DC missing evidence: $task_report" >&2
        exit 6
    fi
done
echo "M62 DC completed at $task_output"
