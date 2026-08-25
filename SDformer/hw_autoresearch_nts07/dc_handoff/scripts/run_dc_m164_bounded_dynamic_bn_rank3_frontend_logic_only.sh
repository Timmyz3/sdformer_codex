#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m164_bounded_dynamic_bn_rank3_frontend_logic_only_dc_3p000ns_r1_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M164 sealed DC run: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m164_q8_bounded_dynamic_bn_rank3_frontend_rtl.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m164/m164_q8_bounded_dynamic_bn_rank3_frontend.sv"]="fd103faaebeee2a99535edf11402525e0b2bcdcb17e78d8451d9d9bf1dc2d5ae"
    ["$task_files"]="864a07493928a87c12fc91aabc3b0b4ed2e4bec8678849ccaa9b9e5e4cc70eea"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m164_bounded_dynamic_bn_rank3_frontend_logic_only_dc_contract_r1_20260824.json"]="ac61e250fce06a437d7fd1379933cda06de3a8e03219dda399239efac1271ce4"
    ["contracts/m164_bounded_dynamic_bn_rank3_frontend_vcs_contract_r1_20260824.json"]="12452ebdbb9fe85b2142ed661147d67b5f76cf6bbb6af065db5b538351de808a"
    ["dc_handoff/runs/m164_bounded_dynamic_bn_rank3_frontend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="2ed6dd31d4ab6d793b930ab51338bad075986230c04350f95658eca23561f690"
    ["results/m163r2_independent_hammer_review_r1_20260824/README.md"]="f945d2d8b4624b1dcae22d7ba6b897127e5b90eff3516e6378fc638b3f809c6c"
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

export DESIGN_NAME="m164_q8_bounded_dynamic_bn_rank3_frontend"
export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_hw_root/$task_files"
export LIB_DB="$task_lib"
export MIN_LIB_DB="$task_min_lib"
export SDC_FILE="$task_hw_root/$task_sdc"
export OUTPUT_DIR="$task_run"
export OPERATING_CONDITION="ssg0p9v125c"
set +e
"$task_dc_shell" -f "$task_hw_root/$task_tcl" > "$task_run/dc.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/dc.rc"
[[ "$task_rc" -eq 0 ]] || exit 20
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$task_run/dc.log"; then
    exit 21
fi
grep -Fq 'Thank you...' "$task_run/dc.log" || exit 22
for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt; do
    [[ -s "$task_run/reports/$task_report" ]] || exit 30
done
[[ -s "$task_run/netlist/${DESIGN_NAME}_mapped.v" \
   && -s "$task_run/netlist/${DESIGN_NAME}_mapped.sdc" \
   && -s "$task_run/netlist/${DESIGN_NAME}.ddc" ]] || exit 31
if grep -Fq 'slack (VIOLATED)' "$task_run/reports/timing_setup.rpt" \
        "$task_run/reports/timing_hold.rpt"; then
    exit 32
fi
grep -Fq 'slack (MET)' "$task_run/reports/timing_setup.rpt" || exit 33
grep -Fq 'slack (MET)' "$task_run/reports/timing_hold.rpt" || exit 33
[[ "$(grep -Fc 'This design has no violated constraints.' "$task_run/reports/constraint_violators.rpt")" -eq 5 ]] || exit 34
[[ "$(tr -d '[:space:]' < "$task_run/reports/check_design_postcompile.rpt")" == "1" ]] || exit 35
[[ "$(tail -n 1 "$task_run/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" == "1" ]] || exit 36
grep -Fq 'Number of macros/black boxes:               0' "$task_run/reports/area.rpt" || exit 37

task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_path="$(awk '/Critical Path Length:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_start="$(awk '/Startpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
task_end="$(awk '/Endpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
awk -v x="$task_area" 'BEGIN {exit !(x < 50000)}' || exit 38
awk -v x="$task_cells" 'BEGIN {exit !(x < 58000)}' || exit 39
awk -v x="$task_seq" 'BEGIN {exit !(x < 7000)}' || exit 40
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.1)}' || exit 41
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 42
awk -v x="$task_levels" 'BEGIN {exit !(x <= 90)}' || exit 43
[[ "$task_start" != "tile_channel_start" ]] || exit 44

task_area_saved="$(awk -v old=53662.139958 -v new="$task_area" 'BEGIN {printf "%.6f", old-new}')"
task_area_reduction="$(awk -v old=53662.139958 -v new="$task_area" 'BEGIN {printf "%.6f", 100*(old-new)/old}')"
task_seq_saved="$(awk -v old=9183 -v new="$task_seq" 'BEGIN {printf "%d", old-new}')"
task_seq_reduction="$(awk -v old=9183 -v new="$task_seq" 'BEGIN {printf "%.6f", 100*(old-new)/old}')"
task_setup_gain="$(awk -v old=0.1053 -v new="$task_setup" 'BEGIN {printf "%.6f", new-old}')"

{
    echo "status=PASS_M164_BOUNDED_DYNAMIC_BN_RANK3_FRONTEND_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "clock_network=ideal"
    echo "wireload=ZeroWireload"
    echo "hierarchy=flattened_before_mapping"
    echo "cell_area_um2=$task_area"
    echo "cell_count=$task_cells"
    echo "sequential_cells=$task_seq"
    echo "logic_levels=$task_levels"
    echo "critical_path_length_ns=$task_path"
    echo "critical_startpoint=$task_start"
    echo "critical_endpoint=$task_end"
    echo "setup_worst_slack_ns=$task_setup"
    echo "hold_worst_slack_ns=$task_hold"
    echo "macro_count=0"
    echo "matched_m163r2_area_um2=53662.139958"
    echo "matched_m163r2_sequential_cells=9183"
    echo "matched_m163r2_setup_worst_slack_ns=0.1053"
    echo "area_saved_um2=$task_area_saved"
    echo "area_reduction_percent=$task_area_reduction"
    echo "sequential_cells_saved=$task_seq_saved"
    echo "sequential_cells_reduction_percent=$task_seq_reduction"
    echo "setup_slack_gain_ns=$task_setup_gain"
    echo "tile_channel_start_is_critical_startpoint=false"
    echo "sum_bits=26"
    echo "sumsq_bits=32"
    echo "count_bits=18"
    echo "projection_bits=19"
    echo "maximum_samples_per_hidden_lane=192000"
    echo "rtl_signed_int8_product_slots=96"
    echo "rtl_square_issue_lanes=32"
    echo "independent_hidden_channel_moment_states=16"
    echo "shared_requant_lanes=16"
    echo "input_tile_ii_accepted_cycles=5"
    echo "clock_tree_extracted_parasitics=false"
    echo "placed_and_routed=false"
    echo "saif_power=false"
    echo "fc1_to_q8_early_requant_implemented=false"
    echo "checkpoint_factors_and_scales=false"
    echo "full_dynamic_bn_barrier=false"
    echo "dynamic_bn_coefficient_generation=false"
    echo "paft_valid825=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt \
    "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m164_bounded_dynamic_bn_rank3_frontend_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M164 bounded dynamic-BN rank3 frontend logic-only DC sealed at $task_run"
