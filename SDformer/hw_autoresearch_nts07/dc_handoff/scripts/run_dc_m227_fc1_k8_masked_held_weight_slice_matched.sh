#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_dc_root/runs/m227_fc1_k8_masked_held_weight_slice_matched_dc_3p000ns_r1_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_min_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M227 matched DC run" >&2
    exit 2
}
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] \
    || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

task_files=dc_handoff/filelists/date_m227_fc1_k8_masked_held_weight_slice_rtl.f
task_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
task_tcl=dc_handoff/scripts/run_dc_m216_flattened_source_cap_logic_only.tcl
declare -A task_expected=(
 ["rtl_m227/m227_fc1_k8_masked_held_weight_slice.sv"]="939e3dc4dcdb20d0962fde84d0c8a8f576886b6f9a259f8a702130149e9bb1b0"
 ["$task_files"]="faacf0204c3b05bb7b2b92d85e7eb7eb14e8d5e46053e75a440ea15371480ff1"
 ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
 ["$task_tcl"]="2565e750551f6f2a03abff96b462558416f3b9531a693f43b6194af8026d61d5"
 ["contracts/m227_fc1_k8_masked_held_weight_slice_synopsys_contract_r1_20260825.json"]="2537092faf1bf46f9dc5632b2c67bb6b34aa0cfb3e825e784da8bb0ba71d06f5"
 ["results/m227_fc1_k8_masked_held_weight_slice_directed_vcs_r2_exact_20260825/RUN_COMPLETE.txt"]="7a2aa84c60cd7192a629ca5a251915d6ae4928adcba380fa86ddfb527bdaddf8"
 ["results/m227_fc1_k8_masked_held_weight_slice_directed_vcs_r2_exact_20260825/SHA256SUMS"]="71f6bcf436d2417e4b044baca6e10ae6225af6e9bf29deb3e773dfeb9bdaf257"
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

export DESIGN_NAME=m227_fc1_k8_masked_held_weight_slice
export HW_ROOT="$task_hw_root" RTL_FILELIST="$task_hw_root/$task_files"
export LIB_DB="$task_lib" MIN_LIB_DB="$task_min_lib"
export SDC_FILE="$task_hw_root/$task_sdc" OPERATING_CONDITION=ssg0p9v125c

run_fanout() {
    local task_fanout="$1"
    local task_dir="$task_run/f${task_fanout}"
    mkdir "$task_dir"
    export OUTPUT_DIR="$task_dir" ELAB_PARAMETERS="FANOUT=$task_fanout"
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
    [[ "$(tr -d '[:space:]' \
        < "$task_dir/reports/check_design_postcompile.rpt")" == 1 ]] \
        || exit 35
    [[ "$(tail -n 1 "$task_dir/reports/check_timing_postcompile.rpt" \
        | tr -d '[:space:]')" == 1 ]] || exit 36
    grep -Fq 'Number of macros/black boxes:               0' \
        "$task_dir/reports/area.rpt" || exit 37
    grep -Eq 'DW_mult|mult_[0-9]' \
        "$task_dir/reports/resources_postcompile.rpt" && exit 38 || true

    local task_area task_cells task_seq task_levels task_path task_setup
    local task_hold task_setup_met task_hold_met
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
    task_setup="$(awk '/slack \((MET|VIOLATED)\)/ {print $3; exit}' \
        "$task_dir/reports/timing_setup.rpt")"
    task_hold="$(awk '/slack \((MET|VIOLATED)\)/ {print $3; exit}' \
        "$task_dir/reports/timing_hold.rpt")"
    task_setup_met=false; task_hold_met=false
    awk -v x="$task_setup" 'BEGIN {exit !(x >= 0)}' && task_setup_met=true
    awk -v x="$task_hold" 'BEGIN {exit !(x >= 0)}' && task_hold_met=true
    awk -v x="$task_area" 'BEGIN {exit !(x > 0 && x < 500000)}'
    awk -v x="$task_cells" 'BEGIN {exit !(x > 0 && x < 500000)}'
    awk -v x="$task_seq" 'BEGIN {exit !(x >= 22000 && x < 30000)}'
    {
        echo status=COMPLETE_M227_F${task_fanout}_LOGIC_ONLY_DC_3NS_SCREEN
        echo exact_sha=true
        echo tool=Synopsys_DC_V-2023.12-SP3
        echo elaboration_parameter=FANOUT=${task_fanout}
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
        echo setup_met="$task_setup_met"
        echo hold_met="$task_hold_met"
        echo macro_count=0
        echo same_k8_mask_state_and_weight_port=true
        echo complete_fc1=false
        echo complete_ffn=false
        echo physical_speedup=false
        echo system_speedup=false
        echo paper_ppa_ready=false
        echo headline=false
    } > "$task_dir/RUN_COMPLETE.txt"
    sha256sum "$task_dir"/dc.log "$task_dir"/reports/*.rpt \
        "$task_dir"/netlist/* "$task_dir"/RUN_COMPLETE.txt \
        > "$task_dir/evidence_manifest.sha256"
}

run_fanout 1
run_fanout 2
run_fanout 4

task_f1_area="$(awk -F= '/^cell_area_um2=/{print $2}' \
    "$task_run/f1/RUN_COMPLETE.txt")"
task_f2_area="$(awk -F= '/^cell_area_um2=/{print $2}' \
    "$task_run/f2/RUN_COMPLETE.txt")"
task_f4_area="$(awk -F= '/^cell_area_um2=/{print $2}' \
    "$task_run/f4/RUN_COMPLETE.txt")"
task_f2_area_ratio="$(awk -v a="$task_f2_area" -v b="$task_f1_area" \
    'BEGIN {printf "%.12f", a/b}')"
task_f4_area_ratio="$(awk -v a="$task_f4_area" -v b="$task_f1_area" \
    'BEGIN {printf "%.12f", a/b}')"
task_f2_tpa="$(awk -v s=1.568695409173474 -v a="$task_f2_area_ratio" \
    'BEGIN {printf "%.12f", s/a}')"
task_f4_tpa="$(awk -v s=2.112901790755882 -v a="$task_f4_area_ratio" \
    'BEGIN {printf "%.12f", s/a}')"
{
    echo status=COMPLETE_M227_MATCHED_LOGIC_ONLY_DC_3NS_SCREEN
    echo exact_sha=true
    echo same_rtl_filelist_sdc_tcl=true
    echo same_k8_mask_state_and_weight_port=true
    echo f1_cell_area_um2="$task_f1_area"
    echo f2_cell_area_um2="$task_f2_area"
    echo f4_cell_area_um2="$task_f4_area"
    echo f2_over_f1_area_ratio="$task_f2_area_ratio"
    echo f4_over_f1_area_ratio="$task_f4_area_ratio"
    echo f2_m226_cycle_prior=1.568695409173474
    echo f4_m226_cycle_prior=2.112901790755882
    echo f2_prior_throughput_per_area_ratio="$task_f2_tpa"
    echo f4_prior_throughput_per_area_ratio="$task_f4_tpa"
    echo prior_ratios_are_not_rtl_cycle_measurements=true
    echo macro_count_each=0
    echo complete_fc1=false
    echo physical_speedup=false
    echo system_speedup=false
    echo paper_ppa_ready=false
    echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_runner" > "$task_run/runner_sha256.txt"
sha256sum "$task_run"/f1/evidence_manifest.sha256 \
    "$task_run"/f2/evidence_manifest.sha256 \
    "$task_run"/f4/evidence_manifest.sha256 "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
task_complete=1
echo "PASS M227 matched logic-only DC screen sealed at $task_run"
