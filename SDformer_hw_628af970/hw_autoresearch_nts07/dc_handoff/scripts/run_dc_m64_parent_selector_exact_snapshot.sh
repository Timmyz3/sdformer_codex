#!/usr/bin/env bash
set -euo pipefail

m64_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
m64_dc_root="$(cd "$m64_script_dir/.." && pwd)"
m64_hw_root="$(cd "$m64_dc_root/.." && pwd)"
m64_repo_root="$(cd "$m64_hw_root/.." && pwd)"
m64_run="${M64_DC_RUN_DIR:-$m64_dc_root/runs/m64_parent_selector_dc_3p000ns_r1_20260823}"
m64_period="${CLOCK_PERIOD_NS:-3.000}"
m64_dc="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
m64_lib_slow="${LIB_DB:-/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db}"
m64_lib_fast="${MIN_LIB_DB:-/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db}"

if [[ -e "$m64_run" ]]; then
    echo "refusing to overwrite M64 DC evidence: $m64_run" >&2
    exit 5
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || pgrep -x fm_shell >/dev/null; then
    echo "refusing concurrent DC/Formality invocation" >&2
    exit 4
fi
for m64_file in "$m64_dc" "$m64_lib_slow" "$m64_lib_fast" \
        "$m64_hw_root/rtl_m64/qfit_adaptive_parent_selector_p256.sv" \
        "$m64_dc_root/filelists/date_m64_parent_selector_dc.f" \
        "$m64_dc_root/constraints/date_m64_parent_selector_3ns.sdc" \
        "$m64_script_dir/run_dc_m64_parent_selector_exact_snapshot.tcl"; do
    [[ -f "$m64_file" ]] || { echo "missing M64 DC input: $m64_file" >&2; exit 3; }
done

mkdir -p "$m64_run/snapshot/hw_autoresearch_nts07/rtl_m64" \
    "$m64_run/snapshot/hw_autoresearch_nts07/dc_handoff/filelists" \
    "$m64_run/snapshot/hw_autoresearch_nts07/dc_handoff/constraints" \
    "$m64_run/snapshot/hw_autoresearch_nts07/dc_handoff/scripts" \
    "$m64_run/snapshot/library" "$m64_run/work"
cp "$m64_hw_root/rtl_m64/qfit_adaptive_parent_selector_p256.sv" \
    "$m64_run/snapshot/hw_autoresearch_nts07/rtl_m64/"
cp "$m64_dc_root/filelists/date_m64_parent_selector_dc.f" \
    "$m64_run/snapshot/hw_autoresearch_nts07/dc_handoff/filelists/"
cp "$m64_dc_root/constraints/date_m64_parent_selector_3ns.sdc" \
    "$m64_run/snapshot/hw_autoresearch_nts07/dc_handoff/constraints/"
cp "$m64_script_dir/run_dc_m64_parent_selector_exact_snapshot.tcl" \
    "$m64_run/snapshot/hw_autoresearch_nts07/dc_handoff/scripts/"
cp "$m64_lib_slow" "$m64_run/snapshot/library/"
cp "$m64_lib_fast" "$m64_run/snapshot/library/"
(
    cd "$m64_run/snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum > "$m64_run/snapshot.sha256"
    sha256sum --strict -c "$m64_run/snapshot.sha256" > "$m64_run/snapshot_check.raw.log" 2>&1
)
find "$m64_run/snapshot" -type f -exec chmod 0444 {} +

export SNAPSHOT_ROOT="$m64_run/snapshot/hw_autoresearch_nts07"
export RTL_FILELIST="$SNAPSHOT_ROOT/dc_handoff/filelists/date_m64_parent_selector_dc.f"
export SDC_FILE="$SNAPSHOT_ROOT/dc_handoff/constraints/date_m64_parent_selector_3ns.sdc"
export OUTPUT_DIR="$m64_run"
export CLOCK_PERIOD_NS="$m64_period"
export LIB_DB="$m64_run/snapshot/library/$(basename "$m64_lib_slow")"
export MIN_LIB_DB="$m64_run/snapshot/library/$(basename "$m64_lib_fast")"
export OPERATING_CONDITION=ssg0p9v125c
m64_tcl="$SNAPSHOT_ROOT/dc_handoff/scripts/run_dc_m64_parent_selector_exact_snapshot.tcl"

{
    echo "status=RUNNING_NOT_CITABLE"
    echo "scope=standalone_online_parent_selector_p256"
    echo "clock_period_ns=$m64_period"
    echo "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "power_or_energy_admitted=false"
} > "$m64_run/RUN_IN_PROGRESS.txt"
set +e
"$m64_dc" -version > "$m64_run/dc.version.raw.log" 2>&1
m64_version_rc=$?
set -e
echo "$m64_version_rc" > "$m64_run/dc.version.rc"
grep -q '^dc_shell version' "$m64_run/dc.version.raw.log"
sha256sum "$(readlink -f "$m64_dc")" > "$m64_run/dc.binary.sha256"
echo "$m64_dc -f $m64_tcl" > "$m64_run/dc.command.txt"
set +e
(cd "$m64_run/work" && "$m64_dc" -f "$m64_tcl") > "$m64_run/dc.raw.log" 2>&1
m64_rc=$?
set -e
echo "$m64_rc" > "$m64_run/dc.rc"
if [[ "$m64_rc" -ne 0 ]] || grep -Eq '^(Error|Fatal):' "$m64_run/dc.raw.log"; then
    echo "status=FAIL_OR_INCOMPLETE_DO_NOT_CITE" > "$m64_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    exit 20
fi
[[ "$(grep -xc 'M64_DC_INTERNAL_COMPLETE=PASS' "$m64_run/DC_INTERNAL_COMPLETE.txt")" -eq 1 ]]
for m64_required in reports/qor.rpt reports/area.rpt reports/timing_setup.rpt \
        reports/timing_hold.rpt reports/check_design_postcompile.rpt \
        reports/check_timing_postcompile.rpt \
        netlist/qfit_adaptive_parent_selector_p256_mapped.v \
        netlist/qfit_adaptive_parent_selector_p256.ddc \
        netlist/qfit_adaptive_parent_selector_p256.svf; do
    [[ -s "$m64_run/$m64_required" ]] || { echo "missing output: $m64_required" >&2; exit 21; }
done
grep -qx 'physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO' \
    "$m64_run/reports/constraint_contract_postcompile.rpt"
rm "$m64_run/RUN_IN_PROGRESS.txt"
{
    echo "status=PASS_EXACT_SNAPSHOT_M64_DC_STA"
    echo "scope=standalone_online_parent_selector_p256"
    echo "clock_period_ns=$m64_period"
    echo "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "power_or_energy_admitted=false"
} > "$m64_run/RUN_COMPLETE.txt"
(
    cd "$m64_run"
    find . -type f ! -name output.sha256 ! -name output_check.raw.log -print0 \
        | sort -z | xargs -0 sha256sum > output.sha256
    sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "M64_DC_STA=PASS run=$m64_run"
