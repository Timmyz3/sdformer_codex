#!/usr/bin/env bash
set -euo pipefail

m35_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m35_hw_root="$(cd "${m35_dc_root}/.." && pwd)"
m35_design=qfit_complement_csd8_late_scale
m35_filelist="${m35_dc_root}/filelists/date_m35_complement_csd8_dc.f"
m35_sdc="${m35_dc_root}/constraints/date_m35_complement_csd8.sdc"
m35_period="${CLOCK_PERIOD_NS:-2.000}"
m35_period_tag="${m35_period//./p}ns"
m35_output="${OUTPUT_DIR:-${m35_dc_root}/runs/m35_complement_csd8_dc_${m35_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"

m35_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m35_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$m35_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$m35_fast_default}"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M35 DC requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$LIB_DB" || ! -f "$MIN_LIB_DB" ]]; then
    echo "M35 DC slow/fast library is missing" >&2
    exit 3
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
    || pgrep -f 'common_shell_exec.*-shell[[:space:]]+dc_shell|/dc_shell([[:space:]]|$)' >/dev/null; then
    echo "refusing M35 DC because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$m35_output/dc.log" ]]; then
    echo "refusing to overwrite M35 DC evidence: $m35_output" >&2
    exit 5
fi

mkdir -p "$m35_output"
export DESIGN_NAME="$m35_design"
export HW_ROOT="$m35_hw_root"
export RTL_FILELIST="$m35_filelist"
export SDC_FILE="$m35_sdc"
export OUTPUT_DIR="$m35_output"
export CLOCK_PERIOD_NS="$m35_period"
export LIB_DB MIN_LIB_DB
export OPERATING_CONDITION="${OPERATING_CONDITION:-ssg0p9v125c}"

dc_shell -f "${m35_dc_root}/scripts/run_dc_m35_complement_csd8.tcl" \
    2>&1 | tee "$m35_output/dc.log"

if grep -q '^Error:' "$m35_output/dc.log"; then
    echo "M35 DC log contains a Tcl/DC error" >&2
    exit 9
fi
for m35_report in \
    reports/references_precompile.rpt reports/resources_precompile.rpt \
    reports/references_postcompile.rpt reports/resources_postcompile.rpt \
    reports/qor.rpt reports/area.rpt reports/timing_setup.rpt \
    reports/timing_hold.rpt reports/check_design_postcompile.rpt \
    reports/check_timing_postcompile.rpt reports/constraint_violators.rpt \
    netlist/${m35_design}_mapped.v netlist/${m35_design}_mapped.sdc \
    netlist/${m35_design}.ddc netlist/${m35_design}.svf; do
    if [[ ! -s "$m35_output/$m35_report" ]]; then
        echo "M35 DC missing evidence: $m35_report" >&2
        exit 6
    fi
done
if grep -Eiq '(^|[^A-Za-z0-9_])(DW[0-9]*_?mult|DW_mult|GTECH_MULT|MULT_OP|mult_x_[0-9]+|mul_[0-9]+)([^A-Za-z0-9_]|$)' \
        "$m35_output/reports/references_postcompile.rpt" \
        "$m35_output/reports/resources_postcompile.rpt" \
        "$m35_output/netlist/${m35_design}_mapped.v"; then
    echo "M35 postcompile evidence contains a multiplier implementation" >&2
    exit 7
fi
m35_resource_modules="$m35_output/resource_modules.txt"
awk -F'|' '
    BEGIN { in_resource_table=0; seen_header=0; seen_module=0 }
    /^\| Cell[[:space:]]+\| Module[[:space:]]+\| Parameters[[:space:]]+\| Contained Operations/ {
        in_resource_table=1
        seen_header=1
        next
    }
    in_resource_table && /^=+$/ {
        if (seen_module) exit 0
        next
    }
    in_resource_table && NF >= 5 {
        module=$3
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", module)
        if (module != "") {
            print module
            seen_module=1
        }
    }
    END { if (!seen_header || !seen_module) exit 42 }
' "$m35_output/reports/resources_postcompile.rpt" | sort -u \
    > "$m35_resource_modules"
if [[ ! -s "$m35_resource_modules" ]]; then
    echo "M35 postcompile resource-module audit produced no entries" >&2
    exit 10
fi
if grep -Evq '^(DW_cmp|DW01_add|DW01_sub|DW_leftsh|DP_OP_[A-Za-z0-9_]+)$' \
        "$m35_resource_modules"; then
    echo "M35 postcompile resources contain a non-whitelisted arithmetic operator" >&2
    grep -Ev '^(DW_cmp|DW01_add|DW01_sub|DW_leftsh|DP_OP_[A-Za-z0-9_]+)$' \
        "$m35_resource_modules" >&2
    exit 10
fi
if ! grep -q 'slack (MET)' "$m35_output/reports/timing_setup.rpt" \
    || ! grep -q 'slack (MET)' "$m35_output/reports/timing_hold.rpt" \
    || grep -q 'slack (VIOLATED)' "$m35_output/reports/timing_setup.rpt" \
    || grep -q 'slack (VIOLATED)' "$m35_output/reports/timing_hold.rpt"; then
    echo "M35 DC timing did not meet the requested ${m35_period}ns constraint" >&2
    exit 8
fi

m35_warning_count="$(grep -c '^Warning:' \
    "$m35_output/reports/check_design_postcompile.rpt" || true)"
m35_log_warning_count="$(grep -c '^Warning:' "$m35_output/dc.log" || true)"
m35_uisn40_count="$(grep -c '^Warning: DesignWare synthetic library dw_foundation\.sldb is added to the synthetic_library in the current command\. (UISN-40)$' \
    "$m35_output/dc.log" || true)"
m35_tim134_one_net_count="$(grep -c "^Warning: Design '$m35_design' contains 1 high-fanout nets\. A fanout number of 1000 will be used for delay calculations involving these nets\. (TIM-134)$" \
    "$m35_output/dc.log" || true)"
m35_tim134_two_net_count="$(grep -c "^Warning: Design '$m35_design' contains 2 high-fanout nets\. A fanout number of 1000 will be used for delay calculations involving these nets\. (TIM-134)$" \
    "$m35_output/dc.log" || true)"
if [[ "$m35_warning_count" -ne 0 ]]; then
    echo "M35 check_design contains a non-whitelisted warning" >&2
    grep '^Warning:' "$m35_output/reports/check_design_postcompile.rpt" >&2 || true
    exit 11
fi
if [[ "$m35_log_warning_count" -ne 8 || "$m35_uisn40_count" -ne 4 \
        || "$m35_tim134_one_net_count" -ne 1 \
        || "$m35_tim134_two_net_count" -ne 3 ]]; then
    echo "M35 DC warning population is outside the exact UISN-40/TIM-134 whitelist" >&2
    grep '^Warning:' "$m35_output/dc.log" >&2 || true
    exit 12
fi
if grep '^Warning:' "$m35_output/dc.log" \
        | grep -Ev "^(Warning: DesignWare synthetic library dw_foundation\\.sldb is added to the synthetic_library in the current command\\. \\(UISN-40\\)|Warning: Design '$m35_design' contains (1|2) high-fanout nets\\. A fanout number of 1000 will be used for delay calculations involving these nets\\. \\(TIM-134\\))$" \
        >/dev/null; then
    echo "M35 DC log contains a non-whitelisted warning" >&2
    grep '^Warning:' "$m35_output/dc.log" >&2 || true
    exit 13
fi
{
    echo "status=PASS_EXACT_WARNING_WHITELIST"
    echo "check_design_warning_count=$m35_warning_count"
    echo "dc_log_warning_count=$m35_log_warning_count"
    echo "uisn40_exact_count=$m35_uisn40_count"
    echo "tim134_one_net_exact_count=$m35_tim134_one_net_count"
    echo "tim134_two_net_exact_count=$m35_tim134_two_net_count"
} > "$m35_output/reports/m35_warning_audit.rpt"
m35_setup_wns="$(awk '/slack \((MET|VIOLATED)\)/ {value=$NF+0; if (!seen || value < minimum) minimum=value; seen=1} END {if (!seen) exit 1; printf "%.4f", minimum}' "$m35_output/reports/timing_setup.rpt")"
m35_hold_wns="$(awk '/slack \((MET|VIOLATED)\)/ {value=$NF+0; if (!seen || value < minimum) minimum=value; seen=1} END {if (!seen) exit 1; printf "%.4f", minimum}' "$m35_output/reports/timing_hold.rpt")"
{
    echo "status=PASS_STANDALONE_COMPLEMENT_CSD8_DC"
    echo "paper_ppa_ready=false"
    echo "system_cycle_performance_admitted=false"
    echo "clock_period_ns=$m35_period"
    echo "timing_status=MET"
    echo "outputs_per_accepted_packet=8"
    echo "pipeline_initiation_interval_when_unstalled=1"
    echo "integer_multiplier_count=0"
    echo "resource_operator_whitelist=DW_cmp,DW01_add,DW01_sub,DW_leftsh,DP_OP"
    echo "check_design_warning_count=$m35_warning_count"
    echo "dc_log_warning_count=$m35_log_warning_count"
    echo "setup_wns_reported_ns=$m35_setup_wns"
    echo "hold_wns_reported_ns=$m35_hold_wns"
    echo "robust_500mhz_margin_admitted=false"
    echo "macro_db=NONE"
    echo "library_slow=$LIB_DB"
    echo "library_fast=$MIN_LIB_DB"
} > "$m35_output/admission.txt"

sha256sum \
    "$m35_hw_root/rtl_m35/qfit_complement_csd8_late_scale.sv" \
    "$m35_filelist" "$m35_sdc" \
    "${m35_dc_root}/scripts/run_dc_m35_complement_csd8.tcl" \
    "${m35_dc_root}/scripts/run_dc_m35_complement_csd8.sh" \
    "$LIB_DB" "$MIN_LIB_DB" "$m35_output/dc.log" \
    "$m35_output/admission.txt" \
    "$m35_output/reports/check_design_precompile.rpt" \
    "$m35_output/reports/check_timing_precompile.rpt" \
    "$m35_output/reports/references_precompile.rpt" \
    "$m35_output/reports/resources_precompile.rpt" \
    "$m35_output/reports/check_design_postcompile.rpt" \
    "$m35_output/reports/check_timing_postcompile.rpt" \
    "$m35_output/reports/m35_warning_audit.rpt" \
    "$m35_output/reports/hierarchy_postcompile.rpt" \
    "$m35_output/reports/clocks.rpt" \
    "$m35_output/reports/references_postcompile.rpt" \
    "$m35_output/reports/resources_postcompile.rpt" \
    "$m35_output/resource_modules.txt" \
    "$m35_output/reports/constraint_violators.rpt" \
    "$m35_output/reports/qor.rpt" "$m35_output/reports/area.rpt" \
    "$m35_output/reports/timing_setup.rpt" \
    "$m35_output/reports/timing_hold.rpt" \
    "$m35_output/netlist/${m35_design}_mapped.v" \
    "$m35_output/netlist/${m35_design}_mapped.sdc" \
    "$m35_output/netlist/${m35_design}.ddc" \
    "$m35_output/netlist/${m35_design}.svf" \
    > "$m35_output/evidence.sha256"
echo "M35_DC_COMPLETE timing=MET run=$m35_output"
