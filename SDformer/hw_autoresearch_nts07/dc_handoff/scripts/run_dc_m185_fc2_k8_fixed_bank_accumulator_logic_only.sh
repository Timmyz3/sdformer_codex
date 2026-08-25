#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m185_fc2_k8_fixed_bank_accumulator_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M185 sealed DC run" >&2; exit 2; fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m185_fc2_k8_fixed_bank_accumulator_rtl.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m185/m185_fc2_k8_fixed_bank_accumulator.sv"]="60c836e6d1cef03279dd3fa4b68e9d18926ae86e06ca43cbeb1a9eae0335e00e"
    ["$task_files"]="00a862e745ad0fc6e1c8ab44f9ddc9393712588584d84f8a97f9e4bd6ff9d44c"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m185_fc2_k8_fixed_bank_accumulator_logic_only_dc_contract_r1_20260825.json"]="0c6e842238ce72cf6e92059ee39b7902e0851abe565eefdcdc78eaf090ff1878"
    ["contracts/m185_fc2_k8_fixed_bank_accumulator_vcs_contract_r1_20260825.json"]="e1557381a80b3f700487e4031e8572e8bcbaab84f6b0de4e103e3be73aa7dbf7"
    ["dc_handoff/runs/m185_fc2_k8_fixed_bank_accumulator_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="1c5c22d3b42d5b705f99e66df0b22d0b3b3ca31e68d5d600d66a6c21e7a1e04d"
    ["dc_handoff/runs/m183_fc2_k8_unique_bank_accumulator_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="7c5b61e56619c40f70727786599953403ce868c7d0601583348a7f3cd1b31190"
    ["dc_handoff/runs/m169_fc2_k4_unique_bank_accumulator_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="9cc8ef38744687d80644df53fa838db03b0445a5df01cf559e98be3547cdd1cf"
    ["dc_handoff/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="ee772129cfa156349f98c737f53aa435e474dddfb70738833348927826641e8f"
    ["dc_handoff/runs/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="3bb31c56d23e960852e0b27e264bfa4e14506f234ed159bd7655f1cee4ba5b27"
    ["contracts/m180_m183_independent_review_admission_overlay_r1_20260825.json"]="6faaf375730eaa5e5bd680b6f90f399371d72587ec00861fbab3c0e553b1d315"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export DESIGN_NAME="m185_fc2_k8_fixed_bank_accumulator"
export HW_ROOT="$task_hw_root" RTL_FILELIST="$task_hw_root/$task_files"
export LIB_DB="$task_lib" MIN_LIB_DB="$task_min_lib"
export SDC_FILE="$task_hw_root/$task_sdc" OUTPUT_DIR="$task_run"
export OPERATING_CONDITION="ssg0p9v125c"
set +e
"$task_dc_shell" -f "$task_hw_root/$task_tcl" > "$task_run/dc.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/dc.rc"
[[ "$task_rc" -eq 0 ]] || exit 20
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$task_run/dc.log"; then exit 21; fi
grep -Fq 'Thank you...' "$task_run/dc.log" || exit 22
for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt constraint_violators.rpt check_design_postcompile.rpt check_timing_postcompile.rpt resources_postcompile.rpt; do
    [[ -s "$task_run/reports/$task_report" ]] || exit 30
done
[[ -s "$task_run/netlist/${DESIGN_NAME}_mapped.v" && -s "$task_run/netlist/${DESIGN_NAME}_mapped.sdc" && -s "$task_run/netlist/${DESIGN_NAME}.ddc" ]] || exit 31
if grep -Fq 'slack (VIOLATED)' "$task_run/reports/timing_setup.rpt" "$task_run/reports/timing_hold.rpt"; then exit 32; fi
grep -Fq 'slack (MET)' "$task_run/reports/timing_setup.rpt" || exit 33
grep -Fq 'slack (MET)' "$task_run/reports/timing_hold.rpt" || exit 33
[[ "$(grep -Fc 'This design has no violated constraints.' "$task_run/reports/constraint_violators.rpt")" -eq 5 ]] || exit 34
[[ "$(tr -d '[:space:]' < "$task_run/reports/check_design_postcompile.rpt")" == "1" ]] || exit 35
[[ "$(tail -n 1 "$task_run/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" == "1" ]] || exit 36
grep -Fq 'Number of macros/black boxes:               0' "$task_run/reports/area.rpt" || exit 37
if grep -Eq 'DW_mult|mult_[0-9]' "$task_run/reports/resources_postcompile.rpt"; then exit 38; fi

task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_path="$(awk '/Critical Path Length:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_start="$(awk '/Startpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
task_end="$(awk '/Endpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
awk -v x="$task_area" 'BEGIN {exit !(x < 30000 && x > 24210.885723)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x < 32000)}' || exit 40
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.2)}' || exit 41
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 42
awk -v x="$task_levels" 'BEGIN {exit !(x <= 50)}' || exit 43

task_over_m183="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/27031.031773}')"
task_over_m169="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/18522.881882}')"
task_schedule="$(awk 'BEGIN {printf "%.9f", 127581198/97607807}')"
task_td_acc="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (127581198/97607807)/(a/18522.881882)}')"
task_k4_sum="$(awk 'BEGIN {printf "%.6f", 14417.928053+18522.881882}')"
task_k8_sum="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", 10026.828029+a}')"
task_sum_area_ratio="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (10026.828029+a)/(14417.928053+18522.881882)}')"
task_sum_td="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (127581198/97607807)/((10026.828029+a)/(14417.928053+18522.881882))}')"

{
    echo "status=PASS_M185_FC2_K8_FIXED_BANK_ACCUMULATOR_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "clock_network=ideal"
    echo "wireload=ZeroWireload"
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
    echo "multipliers_in_mapped_resource_report=0"
    echo "m185_over_m183_area_ratio=$task_over_m183"
    echo "m185_over_m169_k4_area_ratio=$task_over_m169"
    echo "m169_k4_over_m185_k8_schedule_ratio=$task_schedule"
    echo "same_3ns_schedule_throughput_per_area_m185_over_m169=$task_td_acc"
    echo "standalone_k4_density_break_even_area_um2=24210.885723"
    echo "standalone_k4_density_break_even_met=false"
    echo "m180_plus_m169_sum_logic_area_um2=$task_k4_sum"
    echo "m184_plus_m185_sum_logic_area_um2=$task_k8_sum"
    echo "sum_standalone_k8_over_k4_area_ratio=$task_sum_area_ratio"
    echo "same_3ns_sum_standalone_schedule_throughput_per_area_k8_over_k4=$task_sum_td"
    echo "sum_of_standalone_logic_areas_only=true"
    echo "flat_composition=false"
    echo "bank_id_payload=false"
    echo "pairwise_bank_comparators=0"
    echo "prefix_packing=false"
    echo "weight_sram_response=false"
    echo "complete_fc2=false"
    echo "paft_valid825=false"
    echo "clock_tree_extracted_parasitics=false"
    echo "placed_and_routed=false"
    echo "saif_power=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m185_fc2_k8_fixed_bank_accumulator_logic_only.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M185 FC2 K8 fixed-bank accumulator logic-only DC sealed at $task_run"
