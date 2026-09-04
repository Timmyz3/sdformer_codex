#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_design=phase_slack_guarded_wordpacked_pwp_stream
task_filelist="${task_dc_root}/filelists/date_m100_m99_phase_slack_logic_only_dc.f"
task_sdc="${task_dc_root}/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="${task_dc_root}/scripts/run_dc_m100_m99_phase_slack_logic_only.tcl"
task_contract="${task_hw_root}/contracts/m100_m99_phase_slack_logic_only_synopsys_contract_r1_20260824.json"
task_output="${task_dc_root}/runs/m100_m99_phase_slack_logic_only_dc_3p000ns_r1_20260824"
task_lock="${task_output}.launch_lock"

task_slow_db=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_fast_db=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M100 requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$task_slow_db" || ! -f "$task_fast_db" ]]; then
    echo "M100 slow/fast library is missing" >&2
    exit 3
fi
if [[ -n "${CLOCK_PERIOD_NS:-}" || -n "${OPERATING_CONDITION:-}" \
        || -n "${LIB_DB:-}" || -n "${MIN_LIB_DB:-}" \
        || -n "${OUTPUT_DIR:-}" ]]; then
    echo "M100 r1 forbids flow, library, period, corner, or output overrides" >&2
    exit 10
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -f '[c]ommon_shell_exec -shell dc_shell' >/dev/null; then
    echo "refusing M100 because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$task_output" ]]; then
    echo "refusing to overwrite M100 evidence: $task_output" >&2
    exit 5
fi
if ! mkdir "$task_lock"; then
    echo "refusing M100 because launch lock is held: $task_lock" >&2
    exit 8
fi
trap 'rmdir "$task_lock" 2>/dev/null || true' EXIT

declare -A task_expected=(
    ["$task_contract"]="9cd8865f4dce2b36e103eeac47362e5d0f15a4286d4577a41e5a9dc2a75becb1"
    ["$task_filelist"]="13c92bdef276680174c564ea5f45e360bbd45e7cd6a38513ca3b247a96b629c0"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="632d87dfaf8b978c7d364ff10f446ba76db07474b20da75913af35aba3cdde94"
    ["${task_hw_root}/rtl_m82/zero_bubble_elastic_pwp_stream.sv"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["${task_hw_root}/rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv"]="adb2dfd95ee3dd179cb373eb5ead937d9beb4db25648325634ebba755243b082"
    ["${task_hw_root}/contracts/m99_phase_slack_metadata_compiler_vcs_contract_r1_20260824.json"]="a89fde382fb19b639523a0b2d0b4500b498794a09ec960a529c25c390324c420"
    ["${task_dc_root}/runs/m99_phase_slack_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="159e6b6a6a88be19dd873def3ed6b6c4a81c6bd7718ffb5e9a73e501d3c8e513"
    ["${task_dc_root}/runs/m97_m85_logic_only_dc_3p000ns_r1_20260824/m97_m85_logic_only_dc_receipt_r1.json"]="6e8cae5d1591ff2ab6842d3495367c181c6b9b9f52dc296da9dfde604f3a16dc"
    ["${task_dc_root}/runs/m97_m85_logic_only_dc_3p000ns_r1_20260824/evidence_manifest.sha256"]="931642f6bcc1b79a890182dc4ab58045a968b08dcf1bb2f36a49bd8e39e91ae3"
    ["$task_slow_db"]="79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
    ["$task_fast_db"]="a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"
)
for task_path in "${!task_expected[@]}"; do
    task_actual="$(sha256sum "$task_path" | awk '{print $1}')"
    if [[ "$task_actual" != "${task_expected[$task_path]}" ]]; then
        echo "M100 frozen input mismatch: $task_path" >&2
        exit 7
    fi
done
python3 -m json.tool "$task_contract" >/dev/null
(
    cd "$task_hw_root/.."
    sha256sum -c \
        hw_autoresearch_nts07/dc_handoff/runs/m97_m85_logic_only_dc_3p000ns_r1_20260824/evidence_manifest.sha256 \
        >/dev/null
)

mkdir -p "$task_output"
export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_filelist"
export SDC_FILE="$task_sdc"
export OUTPUT_DIR="$task_output"
export CLOCK_PERIOD_NS=3.000
export LIB_DB="$task_slow_db"
export MIN_LIB_DB="$task_fast_db"
export OPERATING_CONDITION=ssg0p9v125c

{
    echo "status=LAUNCHED_EXACT_SHA_AWAITING_BACKEND_AND_POSTRUN_AUDIT"
    echo "paper_ppa_ready=false"
    echo "complete_pwp_lookup_timing=false"
    echo "system_speedup_admitted=false"
    echo "headline_admitted=false"
    echo "logic_only=true"
    echo "pre_macro=true"
    echo "ideal_clock=true"
    echo "wireload=ZeroWireload"
    echo "clock_period_ns=3.000"
    echo "operating_condition=ssg0p9v125c"
    echo "setup_library=$task_slow_db"
    echo "hold_library=$task_fast_db"
    echo "macros=0"
    echo "identity=M100_M99_PHASE_SLACK_LOGIC_ISLAND_ONLY"
    sha256sum "$0"
    for task_path in "${!task_expected[@]}"; do sha256sum "$task_path"; done
} > "$task_output/admission.txt"

set +e
dc_shell -f "$task_tcl" 2>&1 | tee "$task_output/dc.log"
task_dc_rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$task_dc_rc" > "$task_output/dc_backend.rc"
if [[ "$task_dc_rc" -ne 0 ]]; then
    echo "M100 DC backend failed rc=$task_dc_rc" >&2
    exit 9
fi
if grep -q '^Error:' "$task_output/dc.log"; then
    echo "M100 DC log contains a Tcl/DC error" >&2
    exit 11
fi
for task_report in \
    reports/qor.rpt reports/area.rpt reports/timing_setup.rpt \
    reports/timing_hold.rpt reports/constraint_violators.rpt \
    reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt \
    reports/references_postcompile.rpt reports/resources_precompile.rpt \
    reports/resources_postcompile.rpt \
    netlist/${task_design}_mapped.v netlist/${task_design}_mapped.sdc \
    netlist/${task_design}.ddc; do
    if [[ ! -s "$task_output/$task_report" ]]; then
        echo "M100 missing evidence: $task_report" >&2
        exit 6
    fi
done
{
    echo "status=BACKEND_COMPLETE_AWAITING_FAIL_CLOSED_POSTRUN_AUDIT"
    echo "dc_backend_rc=0"
    echo "run_complete=false"
    echo "paper_ppa_ready=false"
    echo "complete_pwp_lookup_timing=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_output/BACKEND_COMPLETE_AWAITING_AUDIT.txt"
echo "M100 DC backend completed; postrun admission still required at $task_output"
