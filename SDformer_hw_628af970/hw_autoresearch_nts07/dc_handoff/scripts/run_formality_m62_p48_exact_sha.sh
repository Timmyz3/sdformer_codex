#!/usr/bin/env bash
set -euo pipefail

m62_repo=/home/zhumd/work/sdformer_codex/SDformer
m62_hw="$m62_repo/hw_autoresearch_nts07"
m62_dc_run="$m62_hw/dc_handoff/runs/m62_p48_dc_3p000ns_r1b_20260823"
m62_run=${M62_FORMALITY_RUN:-$m62_hw/dc_handoff/runs/m62_p48_formality_r1_20260823}
m62_fm=/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell
m62_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m62_rtl="$m62_hw/rtl_m62/qfit_head_p48_signed_lane_fold.sv"
m62_filelist="$m62_hw/dc_handoff/filelists/date_m62_p48_dc.f"
m62_netlist="$m62_dc_run/netlist/qfit_head_p48_signed_lane_fold_mapped.v"
m62_svf="$m62_dc_run/netlist/qfit_head_p48_signed_lane_fold.svf"
m62_tcl="$m62_hw/dc_handoff/scripts/run_formality_m62_p48_exact_sha.tcl"

m62_sha() { sha256sum "$1" | awk '{print $1}'; }
m62_expect() {
    local m62_path=$1
    local m62_expected=$2
    [[ -f "$m62_path" && ! -L "$m62_path" ]] || {
        echo "missing or symlinked M62 input: $m62_path" >&2
        exit 3
    }
    [[ "$(m62_sha "$m62_path")" == "$m62_expected" ]] || {
        echo "M62 exact-SHA mismatch: $m62_path" >&2
        exit 3
    }
}

[[ ! -e "$m62_run" ]] || {
    echo "refusing to overwrite M62 Formality evidence: $m62_run" >&2
    exit 5
}
if pgrep -f '^/opt/synopsys/.*/(fm_shell|fm_shell_exec)( |$)' >/dev/null 2>&1; then
    echo "refusing concurrent Formality invocation" >&2
    exit 4
fi

m62_expect "$m62_fm" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m62_expect "$m62_lib" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m62_expect "$m62_rtl" 4ba42f70e664d7fc30716a04678acc955612008a2be5a0dad693778bbd776f0f
m62_expect "$m62_filelist" fe0e4b79734ba964145d997fc61f6b9abc66db689b47df04c9a9a941b1f9bc40
m62_expect "$m62_netlist" f5f6385841d115e41f0c7a8331ccb9664c35fd571d4f3fde9f0a17b6931e76b1
m62_expect "$m62_svf" 564916d58764ebdeaf981423d5cbe30488e9b2dac8b9ceb5aa318d120fd6655a
m62_expect "$m62_tcl" 98305b71c445a0b5f07ff2a5892982460c8489face9ed06184f39a9f29437b8d

mkdir -p "$m62_run/snapshot/rtl_m62" "$m62_run/snapshot/dc_handoff/filelists" \
    "$m62_run/snapshot/netlist" "$m62_run/snapshot/library" \
    "$m62_run/reports" "$m62_run/work"
cp "$m62_rtl" "$m62_run/snapshot/rtl_m62/"
cp "$m62_filelist" "$m62_run/snapshot/dc_handoff/filelists/"
cp "$m62_netlist" "$m62_run/snapshot/netlist/"
cp "$m62_svf" "$m62_run/snapshot/netlist/"
cp "$m62_lib" "$m62_run/snapshot/library/"

(
    cd "$m62_run/snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum > ../snapshot.sha256
    sha256sum --strict -c ../snapshot.sha256 > ../snapshot_check.raw.log 2>&1
)
find "$m62_run/snapshot" -type f -exec chmod 0444 {} +
find "$m62_run/snapshot" -type d -exec chmod 0555 {} +

{
    echo "status=RUNNING_NOT_CITABLE"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "power_or_energy_admitted=false"
    echo "scope=RTL_TO_EXISTING_3NS_PREMACRO_DC_NETLIST_FORMALITY"
} > "$m62_run/RUN_IN_PROGRESS.txt"
"$m62_fm" -version > "$m62_run/formality.version.raw.log" 2>&1

export DESIGN_NAME=qfit_head_p48_signed_lane_fold
export SNAPSHOT_ROOT="$m62_run/snapshot"
export RTL_FILELIST="$m62_run/snapshot/dc_handoff/filelists/date_m62_p48_dc.f"
export LIB_DB="$m62_run/snapshot/library/$(basename "$m62_lib")"
export MAPPED_NETLIST="$m62_run/snapshot/netlist/$(basename "$m62_netlist")"
export SVF_FILE="$m62_run/snapshot/netlist/$(basename "$m62_svf")"
export OUTPUT_DIR="$m62_run"

echo "$m62_fm -f $m62_tcl" > "$m62_run/formality.command.txt"
set +e
(cd "$m62_run/work" && "$m62_fm" -f "$m62_tcl") \
    > "$m62_run/formality.raw.log" 2>&1
m62_rc=$?
set -e
echo "$m62_rc" > "$m62_run/formality.rc"
[[ "$m62_rc" -eq 0 ]]
[[ "$(grep -xc 'M62_P48_FORMALITY_INTERNAL_COMPLETE=PASS' \
    "$m62_run/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
grep -q 'Verification SUCCEEDED' "$m62_run/reports/formality_status.rpt"
grep -q 'No failing compare points' "$m62_run/reports/formality_failing.rpt"
grep -q 'No aborted compare points' "$m62_run/reports/formality_aborted.rpt"
grep -q 'No unverified compare points' "$m62_run/reports/formality_unverified.rpt"
! grep -Eq '^(Error|Fatal):' "$m62_run/formality.raw.log"

mv "$m62_run/RUN_IN_PROGRESS.txt" "$m62_run/RUN_BOOTSTRAP_RECORD.txt"
{
    echo "status=PASS_EXACT_SHA_M62_P48_FORMALITY"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "power_or_energy_admitted=false"
    echo "scope=RTL_TO_EXISTING_3NS_PREMACRO_DC_NETLIST_FORMALITY"
} > "$m62_run/RUN_COMPLETE.txt"
(
    cd "$m62_run"
    find . -type f ! -path './work/*' ! -name output.sha256 \
        ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
        > output.sha256
    sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "M62_P48_FORMALITY=PASS run=$m62_run"
