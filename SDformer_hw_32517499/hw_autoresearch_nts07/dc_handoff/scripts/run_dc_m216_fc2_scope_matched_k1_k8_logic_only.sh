#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m216_fc2_scope_matched_k1_k8_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_min_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M216 sealed matched DC run" >&2
    exit 2
}
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] \
    || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

task_files=dc_handoff/filelists/date_m216_fc2_source_cap_rtl.f
task_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
task_tcl=dc_handoff/scripts/run_dc_m216_flattened_source_cap_logic_only.tcl
declare -A task_expected=(
 ["rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"]="e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5"
 ["rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"]="8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0"
 ["rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"]="529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267"
 ["$task_files"]="3380352827a201a750a8bdecad1e09d269479d2fdb691d23c84b6a09b7110e48"
 ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
 ["$task_tcl"]="2565e750551f6f2a03abff96b462558416f3b9531a693f43b6194af8026d61d5"
 ["contracts/m216_fc2_scope_matched_k1_k8_logic_only_dc_contract_r1_20260825.json"]="8aaa8957a98f87731159a71dec4a56e97b959df239b39fcd44f9bb859af81818"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export DESIGN_NAME=m216_fc2_raw4_to_source_cap_frontend
export HW_ROOT="$task_hw_root" RTL_FILELIST="$task_hw_root/$task_files"
export LIB_DB="$task_lib" MIN_LIB_DB="$task_min_lib"
export SDC_FILE="$task_hw_root/$task_sdc" OPERATING_CONDITION=ssg0p9v125c

run_cap() {
    local task_cap="$1"
    local task_dir="$task_run/k${task_cap}"
    mkdir "$task_dir"
    export OUTPUT_DIR="$task_dir" ELAB_PARAMETERS="SOURCE_CAP=$task_cap"
    set +e
    "$task_dc_shell" -f "$task_hw_root/$task_tcl" \
        > "$task_dir/dc.log" 2>&1
    local task_rc=$?
    set -e
    echo "$task_rc" > "$task_dir/dc.rc"
    [[ "$task_rc" -eq 0 ]] || exit 20
    grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' \
        "$task_dir/dc.log" && exit 21 || true
    grep -Fq 'Thank you...' "$task_dir/dc.log" || exit 22
    for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt resources_postcompile.rpt; do
        [[ -s "$task_dir/reports/$task_report" ]] || exit 30
    done
    [[ -s "$task_dir/netlist/${DESIGN_NAME}_mapped.v" \
        && -s "$task_dir/netlist/${DESIGN_NAME}_mapped.sdc" \
        && -s "$task_dir/netlist/${DESIGN_NAME}.ddc" ]] || exit 31
    grep -Fq 'slack (VIOLATED)' "$task_dir/reports/timing_setup.rpt" \
        "$task_dir/reports/timing_hold.rpt" && exit 32 || true
    [[ "$(grep -Fc 'This design has no violated constraints.' \
        "$task_dir/reports/constraint_violators.rpt")" -eq 5 ]] || exit 34
    [[ "$(tr -d '[:space:]' \
        < "$task_dir/reports/check_design_postcompile.rpt")" == 1 ]] \
        || exit 35
    [[ "$(tail -n 1 "$task_dir/reports/check_timing_postcompile.rpt" \
        | tr -d '[:space:]')" == 1 ]] || exit 36
    grep -Fq 'Number of macros/black boxes:               0' \
        "$task_dir/reports/area.rpt" || exit 37
    grep -Eq 'DW_mult|mult_[0-9]' \
        "$task_dir/reports/resources_postcompile.rpt" && exit 38 || true
    local task_area task_cells task_seq task_levels task_path task_setup task_hold
    task_area="$(awk '/Total cell area:/ {print $4; exit}' \
        "$task_dir/reports/area.rpt")"
    task_cells="$(awk '/Number of cells:/ {print $4; exit}' \
        "$task_dir/reports/area.rpt")"
    task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' \
        "$task_dir/reports/area.rpt")"
    task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' \
        "$task_dir/reports/qor.rpt")"
    task_path="$(awk '/Critical Path Length:/ {print $4; exit}' \
        "$task_dir/reports/qor.rpt")"
    task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' \
        "$task_dir/reports/timing_setup.rpt")"
    task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' \
        "$task_dir/reports/timing_hold.rpt")"
    awk -v x="$task_area" 'BEGIN {exit !(x < 22000)}'
    awk -v x="$task_cells" 'BEGIN {exit !(x < 33000)}'
    awk -v x="$task_seq" 'BEGIN {exit !(x < 2900)}'
    awk -v x="$task_setup" 'BEGIN {exit !(x >= 0)}'
    awk -v x="$task_hold" 'BEGIN {exit !(x >= 0)}'
    awk -v x="$task_levels" 'BEGIN {exit !(x <= 100)}'
    {
        echo status=PASS_M216_K${task_cap}_LOGIC_ONLY_DC_3NS
        echo exact_sha=true
        echo tool=Synopsys_DC_V-2023.12-SP3
        echo elaboration_parameter=SOURCE_CAP=${task_cap}
        echo clock_period_ns=3.000
        echo clock_network=ideal
        echo wireload=ZeroWireload
        echo cell_area_um2="$task_area"
        echo cell_count="$task_cells"
        echo sequential_cells="$task_seq"
        echo logic_levels="$task_levels"
        echo critical_path_length_ns="$task_path"
        echo setup_worst_slack_ns="$task_setup"
        echo hold_worst_slack_ns="$task_hold"
        echo macro_count=0
        echo complete_fc2=false
        echo physical_speedup=false
        echo system_speedup=false
        echo paper_ppa_ready=false
        echo headline=false
    } > "$task_dir/RUN_COMPLETE.txt"
    sha256sum "$task_dir"/dc.log "$task_dir"/reports/*.rpt \
        "$task_dir"/netlist/* "$task_dir"/RUN_COMPLETE.txt \
        > "$task_dir/evidence_manifest.sha256"
}

run_cap 1
run_cap 8

task_k1_seq="$(awk -F= '/^sequential_cells=/{print $2}' \
    "$task_run/k1/RUN_COMPLETE.txt")"
task_k8_seq="$(awk -F= '/^sequential_cells=/{print $2}' \
    "$task_run/k8/RUN_COMPLETE.txt")"
[[ "$task_k1_seq" == "$task_k8_seq" ]] || exit 40
task_k1_area="$(awk -F= '/^cell_area_um2=/{print $2}' \
    "$task_run/k1/RUN_COMPLETE.txt")"
task_k8_area="$(awk -F= '/^cell_area_um2=/{print $2}' \
    "$task_run/k8/RUN_COMPLETE.txt")"
task_area_ratio="$(awk -v a="$task_k8_area" -v b="$task_k1_area" \
    'BEGIN {printf "%.12f", a/b}')"
{
    echo status=PASS_M216_SCOPE_MATCHED_K1_K8_LOGIC_ONLY_DC_3NS
    echo exact_sha=true
    echo same_rtl_filelist_sdc_tcl=true
    echo same_sequential_cells=true
    echo sequential_cells_each="$task_k1_seq"
    echo k1_cell_area_um2="$task_k1_area"
    echo k8_cell_area_um2="$task_k8_area"
    echo k8_over_k1_area_ratio="$task_area_ratio"
    echo macro_count_each=0
    echo complete_fc2=false
    echo physical_speedup=false
    echo system_speedup=false
    echo paper_ppa_ready=false
    echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$0" > "$task_run/runner_sha256.txt"
sha256sum "$task_run"/k1/evidence_manifest.sha256 \
    "$task_run"/k8/evidence_manifest.sha256 "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
task_complete=1
echo "PASS M216 matched K1/K8 logic-only DC sealed at $task_run"
