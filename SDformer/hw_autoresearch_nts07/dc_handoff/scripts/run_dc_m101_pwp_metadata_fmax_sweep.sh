#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_sdc="${task_dc_root}/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="${task_dc_root}/scripts/run_dc_m101_pwp_metadata_fmax_sweep.tcl"
task_contract="${task_hw_root}/contracts/m101_pwp_metadata_fmax_sweep_synopsys_contract_r1_20260824.json"
task_output="${task_dc_root}/runs/m101_pwp_metadata_fmax_sweep_r1_20260824"
task_lock="${task_output}.launch_lock"

task_slow_db=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_fast_db=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M101 requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$task_slow_db" || ! -f "$task_fast_db" ]]; then
    echo "M101 slow/fast library is missing" >&2
    exit 3
fi
if [[ -n "${CLOCK_PERIOD_NS:-}" || -n "${DESIGN_NAME:-}" \
        || -n "${OPERATING_CONDITION:-}" || -n "${RTL_FILELIST:-}" \
        || -n "${LIB_DB:-}" || -n "${MIN_LIB_DB:-}" \
        || -n "${OUTPUT_DIR:-}" ]]; then
    echo "M101 r1 forbids flow, design, library, period, corner, or output overrides" >&2
    exit 10
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -f '[c]ommon_shell_exec -shell dc_shell' >/dev/null; then
    echo "refusing M101 because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$task_output" ]]; then
    echo "refusing to overwrite M101 evidence: $task_output" >&2
    exit 5
fi
if ! mkdir "$task_lock"; then
    echo "refusing M101 because launch lock is held: $task_lock" >&2
    exit 8
fi
trap 'rmdir "$task_lock" 2>/dev/null || true' EXIT

declare -A task_expected=(
    ["$task_contract"]="dad2b791d505b9532f7924b80e28cd899983e2b097f993f5b1df1c1a97a16c50"
    ["$task_tcl"]="a1527b772e2de1244b2c0ad45727e6ff07bf0b65af73c30948bbc466433e8094"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["${task_dc_root}/filelists/date_m97_m85_logic_only_dc.f"]="6e2c6c7f831eecadba604675447f8425c3427e6cf83a6c6310e7a20483789d00"
    ["${task_dc_root}/filelists/date_m100_m99_phase_slack_logic_only_dc.f"]="13c92bdef276680174c564ea5f45e360bbd45e7cd6a38513ca3b247a96b629c0"
    ["${task_hw_root}/rtl_m82/zero_bubble_elastic_pwp_stream.sv"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["${task_hw_root}/rtl_m85/guarded_wordpacked_pwp_stream.sv"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["${task_hw_root}/rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv"]="adb2dfd95ee3dd179cb373eb5ead937d9beb4db25648325634ebba755243b082"
    ["$task_slow_db"]="79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
    ["$task_fast_db"]="a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"
)
for task_path in "${!task_expected[@]}"; do
    task_actual="$(sha256sum "$task_path" | awk '{print $1}')"
    if [[ "$task_actual" != "${task_expected[$task_path]}" ]]; then
        echo "M101 frozen input mismatch: $task_path" >&2
        exit 7
    fi
done
python3 -m json.tool "$task_contract" >/dev/null

mkdir -p "$task_output"
{
    echo "status=LAUNCHED_EXACT_SHA_AWAITING_ALL_GRID_POINTS_AND_POSTRUN_AUDIT"
    echo "paper_ppa_ready=false"
    echo "complete_pwp_lookup_timing=false"
    echo "bit_sparse_physical_baseline=false"
    echo "m88_multiplication_admitted=false"
    echo "system_speedup_admitted=false"
    echo "headline_admitted=false"
    echo "logic_only=true"
    echo "pre_macro=true"
    echo "ideal_clock=true"
    echo "wireload=ZeroWireload"
    echo "period_grid_ns=2.750,3.000,3.250,3.500,3.750,4.000,4.250,4.500"
    echo "operating_condition=ssg0p9v125c"
    echo "setup_library=$task_slow_db"
    echo "hold_library=$task_fast_db"
    sha256sum "$0"
    for task_path in "${!task_expected[@]}"; do sha256sum "$task_path"; done
} > "$task_output/admission.txt"

task_design_keys=(m85 m99)
task_periods=(2.750 3.000 3.250 3.500 3.750 4.000 4.250 4.500)
task_backend_fail=0
for task_key in "${task_design_keys[@]}"; do
    if [[ "$task_key" == m85 ]]; then
        task_design=guarded_wordpacked_pwp_stream
        task_filelist="${task_dc_root}/filelists/date_m97_m85_logic_only_dc.f"
    else
        task_design=phase_slack_guarded_wordpacked_pwp_stream
        task_filelist="${task_dc_root}/filelists/date_m100_m99_phase_slack_logic_only_dc.f"
    fi
    for task_period in "${task_periods[@]}"; do
        task_point="${task_output}/${task_key}_${task_period//./p}ns"
        mkdir -p "$task_point"
        export HW_ROOT="$task_hw_root"
        export RTL_FILELIST="$task_filelist"
        export SDC_FILE="$task_sdc"
        export OUTPUT_DIR="$task_point"
        export CLOCK_PERIOD_NS="$task_period"
        export DESIGN_NAME="$task_design"
        export LIB_DB="$task_slow_db"
        export MIN_LIB_DB="$task_fast_db"
        export OPERATING_CONDITION=ssg0p9v125c
        {
            echo "design_key=$task_key"
            echo "design_name=$task_design"
            echo "clock_period_ns=$task_period"
            sha256sum "$task_filelist"
        } > "$task_point/point_identity.txt"

        set +e
        dc_shell -f "$task_tcl" 2>&1 | tee "$task_point/dc.log"
        task_dc_rc="${PIPESTATUS[0]}"
        set -e
        printf '%s\n' "$task_dc_rc" > "$task_point/dc_backend.rc"
        if [[ "$task_dc_rc" -ne 0 ]] || grep -q '^Error:' "$task_point/dc.log"; then
            task_backend_fail=1
            printf 'backend_complete=false\n' > "$task_point/BACKEND_FAILED.txt"
            continue
        fi
        task_missing=0
        for task_report in \
            reports/qor.rpt reports/area.rpt reports/timing_setup.rpt \
            reports/timing_hold.rpt reports/constraint_violators.rpt \
            reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt \
            reports/references_postcompile.rpt reports/resources_precompile.rpt \
            reports/resources_postcompile.rpt \
            netlist/${task_design}_mapped.v netlist/${task_design}_mapped.sdc \
            netlist/${task_design}.ddc; do
            if [[ ! -s "$task_point/$task_report" ]]; then
                echo "M101 missing evidence at $task_key $task_period: $task_report" >&2
                task_missing=1
            fi
        done
        if [[ "$task_missing" -ne 0 ]]; then
            task_backend_fail=1
            printf 'backend_complete=false\n' > "$task_point/BACKEND_FAILED.txt"
        else
            printf 'backend_complete=true\n' > "$task_point/BACKEND_COMPLETE.txt"
        fi
    done
done
unset HW_ROOT RTL_FILELIST SDC_FILE OUTPUT_DIR CLOCK_PERIOD_NS DESIGN_NAME \
    LIB_DB MIN_LIB_DB OPERATING_CONDITION

if [[ "$task_backend_fail" -ne 0 ]]; then
    echo "M101 grid has one or more backend failures; no sweep admission" >&2
    exit 9
fi
{
    echo "status=ALL_GRID_BACKENDS_COMPLETE_AWAITING_FAIL_CLOSED_POSTRUN_AUDIT"
    echo "grid_points=16"
    echo "run_complete=false"
    echo "paper_ppa_ready=false"
    echo "complete_pwp_lookup_timing=false"
    echo "bit_sparse_physical_baseline=false"
    echo "m88_multiplication_admitted=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_output/BACKEND_COMPLETE_AWAITING_AUDIT.txt"
echo "M101 all grid backends completed; postrun admission still required at $task_output"
