#!/usr/bin/env bash
set -euo pipefail

m66_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
m66_dc_root="$(cd "$m66_script_dir/.." && pwd)"
m66_hw_root="$(cd "$m66_dc_root/.." && pwd)"
m66_contract="$m66_hw_root/contracts/m54_m66_core_synopsys_ab_exact_sha_contract_r2_20260823.json"
m66_run="${M66_AB_DC_RUN_DIR:-$m66_dc_root/runs/m66_m54_same_resource_dc_ab_3p000ns_r2_20260823}"
m66_period="${CLOCK_PERIOD_NS:-3.000}"
m66_dc="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
m66_fm="${FM_SHELL:-/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell}"
m66_lib_slow="${LIB_DB:-/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db}"
m66_lib_fast="${MIN_LIB_DB:-/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db}"

if [[ -e "$m66_run" ]]; then
    echo "refusing to overwrite M54/M66 A/B DC evidence: $m66_run" >&2
    exit 5
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -x fm_shell >/dev/null; then
    echo "refusing concurrent DC/Formality invocation" >&2
    exit 4
fi
for m66_file in "$m66_dc" "$m66_fm" "$m66_lib_slow" "$m66_lib_fast" \
        "$m66_contract" \
        "$m66_hw_root/rtl_m54/qfit_k4_parent_delta_p8_l96_ctx16.sv" \
        "$m66_hw_root/rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv" \
        "$m66_dc_root/filelists/date_m66_ab_m54_baseline_dc.f" \
        "$m66_dc_root/filelists/date_m66_ab_lookahead_dc.f" \
        "$m66_dc_root/constraints/date_m54_m66_core_ab_3ns.sdc" \
        "$m66_script_dir/run_dc_m66_ab_exact_snapshot.tcl" \
        "$m66_script_dir/run_formality_m66_ab_exact_snapshot.tcl"; do
    [[ -f "$m66_file" ]] || { echo "missing M54/M66 A/B DC input: $m66_file" >&2; exit 3; }
done

snapshot="$m66_run/snapshot"
mkdir -p "$snapshot/hw_autoresearch_nts07/rtl_m54" \
    "$snapshot/hw_autoresearch_nts07/rtl_m66" \
    "$snapshot/hw_autoresearch_nts07/dc_handoff/filelists" \
    "$snapshot/hw_autoresearch_nts07/dc_handoff/constraints" \
    "$snapshot/hw_autoresearch_nts07/dc_handoff/scripts" \
    "$snapshot/hw_autoresearch_nts07/contracts" \
    "$snapshot/library" "$m66_run/baseline/work" "$m66_run/lookahead/work"
cp "$m66_hw_root/rtl_m54/qfit_k4_parent_delta_p8_l96_ctx16.sv" \
    "$snapshot/hw_autoresearch_nts07/rtl_m54/"
cp "$m66_hw_root/rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv" \
    "$snapshot/hw_autoresearch_nts07/rtl_m66/"
cp "$m66_dc_root/filelists/date_m66_ab_m54_baseline_dc.f" \
    "$m66_dc_root/filelists/date_m66_ab_lookahead_dc.f" \
    "$snapshot/hw_autoresearch_nts07/dc_handoff/filelists/"
cp "$m66_dc_root/constraints/date_m54_m66_core_ab_3ns.sdc" \
    "$snapshot/hw_autoresearch_nts07/dc_handoff/constraints/"
cp "$m66_script_dir/run_dc_m66_ab_exact_snapshot.tcl" \
    "$m66_script_dir/run_formality_m66_ab_exact_snapshot.tcl" \
    "$snapshot/hw_autoresearch_nts07/dc_handoff/scripts/"
cp "$m66_contract" "$snapshot/hw_autoresearch_nts07/contracts/"
cp "$m66_lib_slow" "$m66_lib_fast" "$snapshot/library/"
(
    cd "$snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum > "$m66_run/snapshot.sha256"
    sha256sum --strict -c "$m66_run/snapshot.sha256" > "$m66_run/snapshot_check.raw.log" 2>&1
)
find "$snapshot" -type f -exec chmod 0444 {} +

export SNAPSHOT_ROOT="$snapshot/hw_autoresearch_nts07"
export SDC_FILE="$SNAPSHOT_ROOT/dc_handoff/constraints/date_m54_m66_core_ab_3ns.sdc"
export CLOCK_PERIOD_NS="$m66_period"
export LIB_DB="$snapshot/library/$(basename "$m66_lib_slow")"
export MIN_LIB_DB="$snapshot/library/$(basename "$m66_lib_fast")"
export OPERATING_CONDITION=ssg0p9v125c
m66_tcl="$SNAPSHOT_ROOT/dc_handoff/scripts/run_dc_m66_ab_exact_snapshot.tcl"
m66_fm_tcl="$SNAPSHOT_ROOT/dc_handoff/scripts/run_formality_m66_ab_exact_snapshot.tcl"

{
    echo "status=RUNNING_NOT_CITABLE"
    echo "scope=standalone_m54_m66_same_resource_ab"
    echo "clock_period_ns=$m66_period"
    echo "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO"
    echo "system_speedup_admitted=false"
    echo "paper_ppa_ready=false"
    echo "power_or_energy_admitted=false"
} > "$m66_run/RUN_IN_PROGRESS.txt"
set +e
"$m66_dc" -version > "$m66_run/dc.version.raw.log" 2>&1
m66_dc_version_rc=$?
set -e
echo "$m66_dc_version_rc" > "$m66_run/dc.version.rc"
[[ "$m66_dc_version_rc" -eq 1 ]]
grep -q '^dc_shell version' "$m66_run/dc.version.raw.log"
sha256sum "$(readlink -f "$m66_dc")" > "$m66_run/dc.binary.sha256"
set +e
"$m66_fm" -version > "$m66_run/formality.version.raw.log" 2>&1
m66_fm_version_rc=$?
set -e
echo "$m66_fm_version_rc" > "$m66_run/formality.version.rc"
[[ "$m66_fm_version_rc" -eq 0 ]]
grep -q 'Formality' "$m66_run/formality.version.raw.log"
sha256sum "$(readlink -f "$m66_fm")" > "$m66_run/formality.binary.sha256"

run_variant() {
    local variant="$1"
    local design="$2"
    local filelist="$3"
    local output="$m66_run/$variant"
    export VARIANT_NAME="$variant"
    export DESIGN_NAME="$design"
    export RTL_FILELIST="$SNAPSHOT_ROOT/dc_handoff/filelists/$filelist"
    export OUTPUT_DIR="$output"
    printf '%s\n' "$m66_dc -f $m66_tcl" > "$output/dc.command.txt"
    set +e
    (cd "$output/work" && "$m66_dc" -f "$m66_tcl") > "$output/dc.raw.log" 2>&1
    local dc_rc=$?
    set -e
    printf '%s\n' "$dc_rc" > "$output/dc.rc"
    if [[ "$dc_rc" -ne 0 ]] || grep -Eq '^(Error|Fatal):' "$output/dc.raw.log"; then
        printf '%s\n' FAIL_OR_INCOMPLETE_DO_NOT_CITE > "$output/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
        return 20
    fi
    [[ "$(grep -xc 'M66_AB_DC_INTERNAL_COMPLETE=PASS' "$output/DC_INTERNAL_COMPLETE.txt")" -eq 1 ]]
    for required in reports/qor.rpt reports/area.rpt reports/timing_setup.rpt \
            reports/timing_hold.rpt reports/timing_reg2reg_setup.rpt \
            reports/timing_input2reg_setup.rpt reports/timing_reg2out_setup.rpt \
            reports/timing_input2out_setup.rpt reports/check_timing_postcompile.rpt \
            reports/check_design_postcompile.rpt netlist/${design}_mapped.v \
            netlist/${design}.ddc netlist/${design}.svf; do
        [[ -s "$output/$required" ]] || { echo "missing output: $variant/$required" >&2; return 21; }
    done
    grep -qx 'physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO' \
        "$output/reports/constraint_contract_postcompile.rpt"
    if grep -Eq '^Warning:.*(unconstrained|input delay|clock)' \
            "$output/reports/check_timing_postcompile.rpt"; then
        echo "timing contract warning in $variant" >&2
        return 22
    fi
}

run_variant baseline qfit_k4_parent_delta_p8_l96_ctx16 \
    date_m66_ab_m54_baseline_dc.f
run_variant lookahead qfit_k4_parent_delta_p8_l96_ctx16_lookahead \
    date_m66_ab_lookahead_dc.f

run_formality_variant() {
    local variant="$1"
    local design="$2"
    local filelist="$3"
    local output="$m66_run/$variant"
    mkdir -p "$output/work/formality"
    export VARIANT_NAME="$variant"
    export DESIGN_NAME="$design"
    export RTL_FILELIST="$SNAPSHOT_ROOT/dc_handoff/filelists/$filelist"
    export OUTPUT_DIR="$output"
    export MAPPED_NETLIST="$output/netlist/${design}_mapped.v"
    export SVF_FILE="$output/netlist/${design}.svf"
    printf '%s\n' "$m66_fm -f $m66_fm_tcl" > "$output/formality.command.txt"
    set +e
    (cd "$output/work/formality" && "$m66_fm" -f "$m66_fm_tcl") \
        > "$output/formality.raw.log" 2>&1
    local fm_rc=$?
    set -e
    printf '%s\n' "$fm_rc" > "$output/formality.rc"
    if [[ "$fm_rc" -ne 0 ]] || grep -Eq '^(Error|Fatal):' "$output/formality.raw.log"; then
        printf '%s\n' FAIL_OR_INCOMPLETE_DO_NOT_CITE > "$output/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
        return 30
    fi
    [[ "$(grep -xc 'M66_AB_FORMALITY_INTERNAL_COMPLETE=PASS' "$output/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
    grep -qx 'verify_return=1' "$output/reports/formality_status.rpt"
    grep -q 'Verification SUCCEEDED' "$output/reports/formality_status.rpt"
    grep -q 'No unmatched points' "$output/reports/formality_unmatched.rpt"
    ! grep -Eq 'Failing \(not equivalent\)[[:space:]]+[1-9]|Aborted[[:space:]]+[1-9]|Unverified[[:space:]]+[1-9]' \
        "$output/reports/formality_status.rpt"
}

run_formality_variant baseline qfit_k4_parent_delta_p8_l96_ctx16 \
    date_m66_ab_m54_baseline_dc.f
run_formality_variant lookahead qfit_k4_parent_delta_p8_l96_ctx16_lookahead \
    date_m66_ab_lookahead_dc.f

rm "$m66_run/RUN_IN_PROGRESS.txt"
{
    echo "status=PASS_EXACT_SNAPSHOT_M54_M66_SAME_RESOURCE_DC_STA_AB"
    echo "scope=standalone_m54_m66_same_resource_ab"
    echo "clock_period_ns=$m66_period"
    echo "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO"
    echo "system_speedup_admitted=false"
    echo "paper_ppa_ready=false"
    echo "power_or_energy_admitted=false"
} > "$m66_run/RUN_COMPLETE.txt"
(
    cd "$m66_run"
    find . -type f ! -name output.sha256 ! -name output_check.raw.log -print0 \
        | sort -z | xargs -0 sha256sum > output.sha256
    sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "M66_M54_SAME_RESOURCE_DC_AB=PASS run=$m66_run"
