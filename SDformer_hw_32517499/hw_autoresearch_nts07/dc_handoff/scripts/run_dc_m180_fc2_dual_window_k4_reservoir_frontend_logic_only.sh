#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_3p000ns_r1_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M180 sealed DC run: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m180_fc2_dual_window_k4_reservoir_frontend_rtl.f"
task_sdc="dc_handoff/constraints/date_m180_logic_only_3ns_maxfanout16.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m180/m180_fc2_dual_window_k4_reservoir_frontend.sv"]="83e72b7bd71f059a1e47dedaaf060d37d8d416979bf24271973910f31ac20a6c"
    ["$task_files"]="8afa6e5aa33e96339d90cf43c94b4f2eaf2adc0dd89d4ac0a40d82d4a79cee8c"
    ["$task_sdc"]="c4ad2bed6eb2851175d99a8a1a9fbd0f79cf637967de332876f4bd663d8a685a"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_contract_r1_20260824.json"]="c22a232ce0b2c0927092a2382d00e176446fd738b5fc05bc92979aeac088a191"
    ["contracts/m180_fc2_dual_window_k4_reservoir_frontend_vcs_contract_r1_20260824.json"]="8a6cd6ccf2e0f3653d7fbde36f8a480e68f4b9fec7d21c0cc70c063f95b8b02d"
    ["contracts/m179_r1_independent_review_baseline_and_selection_overlay_r1_20260824.json"]="6e8c0b7db0644b6a22545c9660828e311ef4e90b5d6f724b721de69373d40542"
    ["dc_handoff/runs/m180_fc2_dual_window_k4_reservoir_frontend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="0fe5ead64ecde8f7b82f1f1c7eee0f63b7baddf09aa6da7a78b10d8714aade63"
    ["results/m179_independent_hammer_review_r1_20260824/manifest.sha256"]="31f3e4baddcf1d5478d2cb011875154918d6fd9c998bf199c395f52561c3277b"
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

export DESIGN_NAME="m180_fc2_dual_window_k4_reservoir_frontend"
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
if grep -Eq 'DW_mult|mult_[0-9]' "$task_run/reports/resources_postcompile.rpt"; then
    exit 38
fi

task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_path="$(awk '/Critical Path Length:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_start="$(awk '/Startpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
task_end="$(awk '/Endpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
awk -v x="$task_area" 'BEGIN {exit !(x < 16000.0)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x < 25000)}' || exit 40
awk -v x="$task_seq" 'BEGIN {exit !(x <= 1900)}' || exit 41
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.0)}' || exit 42
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 43
awk -v x="$task_levels" 'BEGIN {exit !(x <= 180.0)}' || exit 44

task_over_m177_area="$(awk -v x="$task_area" 'BEGIN {printf "%.9f", x / 1314.684003}')"
task_area_overhead="$(awk -v x="$task_area" 'BEGIN {printf "%.6f", 100.0 * (x - 1314.684003) / 1314.684003}')"
task_conditional_td="$(awk -v x="$task_area" 'BEGIN {printf "%.9f", 1.12984127959043 * 1314.684003 / x}')"

{
    echo "status=PASS_M180_DUAL_WINDOW_K4_RESERVOIR_FRONTEND_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "clock_network=ideal"
    echo "wireload=ZeroWireload"
    echo "maximum_fanout=16"
    echo "m180_cell_area_um2=$task_area"
    echo "m180_cell_count=$task_cells"
    echo "m180_sequential_cells=$task_seq"
    echo "m180_logic_levels=$task_levels"
    echo "m180_critical_path_length_ns=$task_path"
    echo "m180_critical_startpoint=$task_start"
    echo "m180_critical_endpoint=$task_end"
    echo "m180_setup_worst_slack_ns=$task_setup"
    echo "m180_hold_worst_slack_ns=$task_hold"
    echo "m177_cell_area_um2=1314.684003"
    echo "m177_cell_count=1838"
    echo "m177_sequential_cells=235"
    echo "m180_over_m177_area_ratio=$task_over_m177_area"
    echo "m180_area_overhead_percent=$task_area_overhead"
    echo "m179_d1_over_selected_k4_analytic_cycle_ratio=1.129841280"
    echo "conditional_analytic_opportunity_per_logic_area_ratio=$task_conditional_td"
    echo "comparison_capacity_matched=false"
    echo "maximum_two_buffer_bitmap_payload_bits_without_metadata=1536"
    echo "macro_count=0"
    echo "multipliers_in_mapped_resource_report=0"
    echo "native_descriptor_producer=false"
    echo "token_directory_generation=false"
    echo "weight_sram_response=false"
    echo "arithmetic=false"
    echo "complete_fc2=false"
    echo "clock_tree_extracted_parasitics=false"
    echo "placed_and_routed=false"
    echo "saif_power=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt \
    "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m180_fc2_dual_window_k4_reservoir_frontend_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M180 dual-window K4 reservoir frontend logic-only DC sealed at $task_run"
