#!/usr/bin/env bash
set -euo pipefail

flat_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
flat_hw_root="$(cd "${flat_dc_root}/.." && pwd)"
flat_design=qfit_threshold_late_scale_uq0p24_radix20x4
flat_filelist="${flat_dc_root}/filelists/date_m33_threshold_late_scale_uq0p24_dc.f"
flat_sdc="${flat_dc_root}/constraints/date_m33_threshold_late_scale_uq0p24.sdc"
flat_period="${CLOCK_PERIOD_NS:-2.000}"
flat_period_tag="${flat_period//./p}ns"
flat_output="${OUTPUT_DIR:-${flat_dc_root}/runs/m33_uq_flat_dc_${flat_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"

flat_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
flat_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$flat_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$flat_fast_default}"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M33 flat DC requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$LIB_DB" || ! -f "$MIN_LIB_DB" ]]; then
    echo "M33 flat DC slow/fast library is missing" >&2
    exit 3
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
    || pgrep -f 'common_shell_exec.*-shell[[:space:]]+dc_shell|/dc_shell([[:space:]]|$)' >/dev/null; then
    echo "refusing M33 flat DC because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$flat_output/dc.log" ]]; then
    echo "refusing to overwrite M33 flat DC evidence: $flat_output" >&2
    exit 5
fi

mkdir -p "$flat_output"
export DESIGN_NAME="$flat_design"
export HW_ROOT="$flat_hw_root"
export RTL_FILELIST="$flat_filelist"
export SDC_FILE="$flat_sdc"
export OUTPUT_DIR="$flat_output"
export CLOCK_PERIOD_NS="$flat_period"
export LIB_DB MIN_LIB_DB
export OPERATING_CONDITION="${OPERATING_CONDITION:-ssg0p9v125c}"

dc_shell -f "${flat_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24_flat.tcl" \
    2>&1 | tee "$flat_output/dc.log"

if grep -q '^Error:' "$flat_output/dc.log"; then
    echo "M33 flat DC log contains a Tcl/DC error" >&2
    exit 9
fi
for flat_report in \
    reports/check_design_precompile.rpt reports/check_timing_precompile.rpt \
    reports/hierarchy_precompile.rpt reports/resources_precompile.rpt \
    reports/references_precompile.rpt reports/check_design_postcompile.rpt \
    reports/check_timing_postcompile.rpt reports/hierarchy_postcompile.rpt \
    reports/resources_postcompile.rpt reports/references_postcompile.rpt \
    reports/qor.rpt reports/area.rpt reports/clocks.rpt \
    reports/timing_setup.rpt reports/timing_hold.rpt \
    reports/constraint_violators.rpt \
    netlist/${flat_design}_mapped.v netlist/${flat_design}_mapped.sdc \
    netlist/${flat_design}.ddc netlist/${flat_design}.svf; do
    if [[ ! -s "$flat_output/$flat_report" ]]; then
        echo "M33 flat DC missing evidence: $flat_report" >&2
        exit 6
    fi
done
if ! grep -q 'slack (MET)' "$flat_output/reports/timing_setup.rpt" \
    || ! grep -q 'slack (MET)' "$flat_output/reports/timing_hold.rpt" \
    || grep -q 'slack (VIOLATED)' "$flat_output/reports/timing_setup.rpt" \
    || grep -q 'slack (VIOLATED)' "$flat_output/reports/timing_hold.rpt"; then
    echo "M33 flat DC timing did not meet ${flat_period}ns" >&2
    exit 7
fi

flat_check_warning_count="$(grep -c '^Warning:' \
    "$flat_output/reports/check_design_postcompile.rpt" || true)"
flat_log_warning_count="$(grep -c '^Warning:' "$flat_output/dc.log" || true)"
flat_uisn40_count="$(grep -c '^Warning: DesignWare synthetic library dw_foundation\.sldb is added to the synthetic_library in the current command\. (UISN-40)$' \
    "$flat_output/dc.log" || true)"
flat_ver318_unsigned_count="$(grep -Ec '^Warning:  .*/qfit_threshold_late_scale_uq0p24_radix20x4\.sv:(69|93|115): unsigned to signed assignment occurs\. \(VER-318\)$' \
    "$flat_output/dc.log" || true)"
flat_ver318_part_count="$(grep -Ec '^Warning:  .*/qfit_threshold_late_scale_uq0p24_radix20x4\.sv:(99|121|201|228): signed to unsigned part selection occurs\. \(VER-318\)$' \
    "$flat_output/dc.log" || true)"
if [[ "$flat_check_warning_count" -ne 0 ]]; then
    echo "M33 flat check_design contains a non-whitelisted warning" >&2
    grep '^Warning:' "$flat_output/reports/check_design_postcompile.rpt" >&2 || true
    exit 10
fi
if [[ "$flat_log_warning_count" -ne 11 || "$flat_uisn40_count" -ne 4 \
        || "$flat_ver318_unsigned_count" -ne 3 \
        || "$flat_ver318_part_count" -ne 4 ]]; then
    echo "M33 flat DC warning population is outside the exact whitelist" >&2
    grep '^Warning:' "$flat_output/dc.log" >&2 || true
    exit 11
fi
if grep '^Warning:' "$flat_output/dc.log" \
        | grep -Ev '^(Warning: DesignWare synthetic library dw_foundation\.sldb is added to the synthetic_library in the current command\. \(UISN-40\)|Warning:  .*/qfit_threshold_late_scale_uq0p24_radix20x4\.sv:(69|93|115): unsigned to signed assignment occurs\. \(VER-318\)|Warning:  .*/qfit_threshold_late_scale_uq0p24_radix20x4\.sv:(99|121|201|228): signed to unsigned part selection occurs\. \(VER-318\))$' \
        >/dev/null; then
    echo "M33 flat DC log contains a non-whitelisted warning" >&2
    grep '^Warning:' "$flat_output/dc.log" >&2 || true
    exit 12
fi
flat_multiplier_count="$(grep -Ec '^\| mult_x_[^|]*\| DW_mult_tc[[:space:]]*\| a_width=8[[:space:]]*\|' \
    "$flat_output/reports/resources_postcompile.rpt" || true)"
if [[ "$flat_multiplier_count" -ne 80 ]] \
    || grep -E '^\| mult_x_[^|]*\| DW_mult_tc[[:space:]]*\| a_width=([^8]|8[0-9])' \
        "$flat_output/reports/resources_postcompile.rpt" >/dev/null; then
    echo "M33 flat postcompile resource report is not exact 80 signed-INT8 multipliers" >&2
    exit 13
fi
if grep -q 'qfit_signed_int8_mul96_pool' \
        "$flat_output/reports/hierarchy_postcompile.rpt"; then
    echo "M33 flat run retained the multiplier-pool hierarchy" >&2
    exit 14
fi
if ! cmp -s \
        <(grep -E '^compile(_ultra| -incremental_mapping)' \
            "${flat_dc_root}/scripts/run_dc_m35_complement_csd8.tcl") \
        <(grep -E '^compile(_ultra| -incremental_mapping)' \
            "${flat_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24_flat.tcl"); then
    echo "M33 flat and M35 compile directive sequences differ" >&2
    exit 15
fi
{
    echo "status=PASS_EXACT_FLAT_WARNING_AND_MULTIPLIER_AUDIT"
    echo "check_design_warning_count=$flat_check_warning_count"
    echo "uisn40_exact_count=$flat_uisn40_count"
    echo "ver318_unsigned_to_signed_exact_count=$flat_ver318_unsigned_count"
    echo "ver318_signed_part_select_exact_count=$flat_ver318_part_count"
    echo "postcompile_signed_int8_multiplier_resource_count=$flat_multiplier_count"
    echo "postcompile_multiplier_pool_hierarchy_count=0"
    echo "compile_directive_sequence_matches_m35=true"
} > "$flat_output/reports/m33_flat_audit.rpt"
flat_setup_wns="$(awk '/slack \((MET|VIOLATED)\)/ {value=$NF+0; if (!seen || value < minimum) minimum=value; seen=1} END {if (!seen) exit 1; printf "%.4f", minimum}' "$flat_output/reports/timing_setup.rpt")"
flat_hold_wns="$(awk '/slack \((MET|VIOLATED)\)/ {value=$NF+0; if (!seen || value < minimum) minimum=value; seen=1} END {if (!seen) exit 1; printf "%.4f", minimum}' "$flat_output/reports/timing_hold.rpt")"
{
    echo "status=PASS_EXPLORATORY_FLAT_FAIR_AREA_DC"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "strict_fair_density_admitted=false_pending_formality_and_independent_review"
    echo "optimizer_contract=M35_DEFAULT_COMPILE_ULTRA_INCREMENTAL_HOLD"
    echo "hierarchy_resource_proof=NOT_ADMITTED_USE_FORMALITY_AND_SOURCE_CONTRACT"
    echo "clock_period_ns=$flat_period"
    echo "timing_status=MET"
    echo "check_design_warning_count=$flat_check_warning_count"
    echo "dc_log_warning_count=$flat_log_warning_count"
    echo "setup_wns_reported_ns=$flat_setup_wns"
    echo "hold_wns_reported_ns=$flat_hold_wns"
    echo "macro_db=NONE"
    echo "library_slow=$LIB_DB"
    echo "library_fast=$MIN_LIB_DB"
} > "$flat_output/admission.txt"

sha256sum \
    "$flat_hw_root/rtl_m31/qfit_signed_int8_mul96_pool.sv" \
    "$flat_hw_root/rtl_m33/qfit_threshold_late_scale_uq0p24_radix20x4.sv" \
    "$flat_filelist" "$flat_sdc" \
    "${flat_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24_flat.tcl" \
    "${flat_dc_root}/scripts/run_dc_m33_threshold_late_scale_uq0p24_flat.sh" \
    "$LIB_DB" "$MIN_LIB_DB" "$flat_output/dc.log" \
    "$flat_output/admission.txt" \
    "$flat_output/reports/check_design_precompile.rpt" \
    "$flat_output/reports/check_timing_precompile.rpt" \
    "$flat_output/reports/hierarchy_precompile.rpt" \
    "$flat_output/reports/resources_precompile.rpt" \
    "$flat_output/reports/references_precompile.rpt" \
    "$flat_output/reports/check_design_postcompile.rpt" \
    "$flat_output/reports/check_timing_postcompile.rpt" \
    "$flat_output/reports/m33_flat_audit.rpt" \
    "$flat_output/reports/hierarchy_postcompile.rpt" \
    "$flat_output/reports/resources_postcompile.rpt" \
    "$flat_output/reports/references_postcompile.rpt" \
    "$flat_output/reports/qor.rpt" "$flat_output/reports/area.rpt" \
    "$flat_output/reports/clocks.rpt" \
    "$flat_output/reports/timing_setup.rpt" \
    "$flat_output/reports/timing_hold.rpt" \
    "$flat_output/reports/constraint_violators.rpt" \
    "$flat_output/netlist/${flat_design}_mapped.v" \
    "$flat_output/netlist/${flat_design}_mapped.sdc" \
    "$flat_output/netlist/${flat_design}.ddc" \
    "$flat_output/netlist/${flat_design}.svf" \
    > "$flat_output/evidence.sha256"
echo "M33_FLAT_DC_COMPLETE timing=MET run=$flat_output"
