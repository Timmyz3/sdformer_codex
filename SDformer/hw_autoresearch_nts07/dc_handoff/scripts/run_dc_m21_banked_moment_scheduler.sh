#!/usr/bin/env bash
set -euo pipefail

# Fail closed: this large inferred-register experiment must not contend with
# an active legacy synthesis.  Both explicit acknowledgement and a clean
# process check are required before a new DC process is launched.
if [[ "${M21_START_DC:-0}" != "1" \
      || "${M21_CONFIRM_NO_ACTIVE_LEGACY_DC:-}" != "YES" ]]; then
    echo "M21 DC is disabled; after legacy DC finishes set M21_START_DC=1 and M21_CONFIRM_NO_ACTIVE_LEGACY_DC=YES." >&2
    exit 2
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
    || pgrep -f 'common_shell_exec.*-shell[[:space:]]+dc_shell|/dc_shell([[:space:]]|$)' >/dev/null; then
    echo "refusing M21 DC: an existing dc_shell process is active" >&2
    exit 3
fi

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_design=date_m21_banked_moment_scheduler_dc_top
task_filelist="${task_dc_root}/filelists/date_m21_banked_moment_scheduler.f"
task_sdc="${task_dc_root}/constraints/date_m21_banked_moment_scheduler_3ns.sdc"
task_output="${OUTPUT_DIR:-${task_dc_root}/runs/m21_banked_moment_scheduler_dc_3ns}"

if [[ -z "${LIB_DB:-}" || ! -f "${LIB_DB}" ]]; then
    echo "M21 DC requires LIB_DB to name an existing Synopsys standard-cell .db." >&2
    exit 4
fi
if [[ -e "${task_output}/dc.log" || -e "${task_output}/netlist/${task_design}.ddc" ]]; then
    echo "refusing to overwrite existing M21 DC evidence: ${task_output}" >&2
    exit 5
fi
mkdir -p "${task_output}"

export DESIGN_NAME="${task_design}"
export HW_ROOT="${task_hw_root}"
export RTL_FILELIST="${task_filelist}"
export SDC_FILE="${task_sdc}"
export OUTPUT_DIR="${task_output}"
export CLOCK_PERIOD_NS="${CLOCK_PERIOD_NS:-3.000}"
export LIB_DB
export MIN_LIB_DB="${MIN_LIB_DB:-}"
export MACRO_DBS=""
export OPERATING_CONDITION="${OPERATING_CONDITION:-}"
export ELAB_PARAMETERS="${ELAB_PARAMETERS:-IN_W=32,TAG_W=48,MAX_REDUCTION_POPULATION=4194304,MAX_LANE_TILES=16}"

dc_shell -f "${task_dc_root}/scripts/run_dc.tcl" 2>&1 | tee "${task_output}/dc.log"

for task_report in \
    reports/qor.rpt reports/area.rpt reports/resources.rpt \
    reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt \
    reports/timing_setup.rpt reports/timing_hold.rpt \
    reports/constraint_violators.rpt; do
    if [[ ! -s "${task_output}/${task_report}" ]]; then
        echo "M21 DC missing required report: ${task_report}" >&2
        exit 6
    fi
done
if grep -Eq "^Error:|^Fatal:" "${task_output}/dc.log"; then
    echo "M21 DC log contains a fatal tool diagnostic" >&2
    exit 7
fi
if ! grep -q "slack (MET)" "${task_output}/reports/timing_setup.rpt" \
    || ! grep -q "slack (MET)" "${task_output}/reports/timing_hold.rpt" \
    || grep -q "slack (VIOLATED)" "${task_output}/reports/timing_setup.rpt" \
    || grep -q "slack (VIOLATED)" "${task_output}/reports/timing_hold.rpt"; then
    echo "M21 DC timing reports do not close" >&2
    exit 8
fi
if ! grep -Eq "Total Negative Slack:[[:space:]]+0\.00" "${task_output}/reports/qor.rpt" \
    || ! grep -Eq "No\. of Violating Paths:[[:space:]]+0\.00" "${task_output}/reports/qor.rpt"; then
    echo "M21 DC QoR reports negative slack or violating paths" >&2
    exit 9
fi
task_check_timing_result="$(awk 'NF{last=$0} END{gsub(/[[:space:]]/, "", last); print last}' \
    "${task_output}/reports/check_timing_postcompile.rpt")"
task_check_design_result="$(awk 'NF{last=$0} END{gsub(/[[:space:]]/, "", last); print last}' \
    "${task_output}/reports/check_design_postcompile.rpt")"
if ! grep -q "Checking unconstrained_endpoints" "${task_output}/reports/check_timing_postcompile.rpt" \
    || [[ "${task_check_timing_result}" != "1" ]] \
    || [[ "${task_check_design_result}" != "1" ]] \
    || grep -Eqi "unconstrained endpoint.*(found|exist|[1-9])|unresolved|black[ -]?box|multiply driven|multiple driver" \
        "${task_output}/reports/check_timing_postcompile.rpt" \
        "${task_output}/reports/check_design_postcompile.rpt"; then
    echo "M21 DC design/timing checks are not clean" >&2
    exit 10
fi
if [[ "$(grep -c "This design has no violated constraints\." \
        "${task_output}/reports/constraint_violators.rpt")" -ne 6 ]]; then
    echo "M21 DC constraint report contains an unexpected violation" >&2
    exit 11
fi
if ! awk '/Combinational area:/{if ($3 + 0 > 0) pass=1} END{exit !pass}' \
        "${task_output}/reports/area.rpt" \
    || ! awk '/Noncombinational area:/{if ($3 + 0 > 0) pass=1} END{exit !pass}' \
        "${task_output}/reports/area.rpt"; then
    echo "M21 DC produced a zero combinational or sequential implementation" >&2
    exit 12
fi
task_mapped="${task_output}/netlist/${task_design}_mapped.v"
if ! grep -Eq '^[[:space:]]*module[[:space:]]+date_m21_banked_moment_scheduler_dc_top' "${task_mapped}" \
    || ! grep -Eq '(input|wire)[[:space:]]+\[3071:0\][[:space:]]+packet_values' "${task_mapped}" \
    || ! grep -Eq '(output|wire)[[:space:]]+\[863:0\][[:space:]]+result_sum' "${task_mapped}" \
    || ! grep -Eq '(output|wire)[[:space:]]+\[1359:0\][[:space:]]+result_sumsq' "${task_mapped}"; then
    echo "M21 DC mapped top lost the default 96x32/MAX width contract" >&2
    exit 13
fi
{
    echo "status=PASS_FAIL_CLOSED_LOGIC_ONLY_NOT_PAPER_PPA"
    echo "claim=BANKED_RAW_MOMENTS_ONLY_NOT_COMPLETE_BN_OR_SYSTEM_SPEEDUP"
    echo "timing_caveat=LAST_PACKET_REGISTERED_RESULT_RETIREMENT_BUBBLES_NOT_IN_PYTHON_DSE"
    echo "clock_period_ns=${CLOCK_PERIOD_NS:-3.000}"
    echo "elab_parameters=${ELAB_PARAMETERS}"
    echo "library_db=${LIB_DB}"
    echo "min_library_db=${MIN_LIB_DB:-NONE}"
} > "${task_output}/admission.txt"
sha256sum \
    "${task_hw_root}/rtl_m21/qfit_dynamic_bn_banked_moment_scheduler.sv" \
    "${task_dc_root}/rtl/date_m21_banked_moment_scheduler_dc_top.sv" \
    "${task_filelist}" "${task_sdc}" \
    "${task_dc_root}/scripts/run_dc.tcl" \
    "${task_dc_root}/scripts/run_dc_m21_banked_moment_scheduler.sh" \
    "${LIB_DB}" "${task_output}/dc.log" "${task_output}/admission.txt" \
    "${task_output}/reports/qor.rpt" "${task_output}/reports/area.rpt" \
    "${task_output}/reports/check_design_postcompile.rpt" \
    "${task_output}/reports/check_timing_postcompile.rpt" \
    "${task_output}/reports/timing_setup.rpt" \
    "${task_output}/reports/timing_hold.rpt" \
    "${task_output}/netlist/${task_design}.ddc" "${task_mapped}" \
    > "${task_output}/evidence.sha256"
echo "PASS Synopsys DC M21 banked raw-moment scheduler 3ns logic-only run"
