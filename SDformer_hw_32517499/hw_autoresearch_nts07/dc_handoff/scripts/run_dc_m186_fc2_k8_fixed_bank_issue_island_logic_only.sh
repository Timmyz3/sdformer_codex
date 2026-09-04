#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m186_fc2_k8_fixed_bank_issue_island_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M186 sealed DC run" >&2; exit 2; fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m186_fc2_k8_fixed_bank_issue_island_rtl.f"
task_sdc="dc_handoff/constraints/date_m180_logic_only_3ns_maxfanout16.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m184/m184_fc2_dual_window_k8_fixed_bank_frontend.sv"]="c6212049305faf42cda13f7f3408d5fa478c79a7c76c142501ec01d9f1e01cd6"
    ["rtl_m185/m185_fc2_k8_fixed_bank_accumulator.sv"]="60c836e6d1cef03279dd3fa4b68e9d18926ae86e06ca43cbeb1a9eae0335e00e"
    ["rtl_m186/m186_fc2_k8_fixed_bank_issue_island.sv"]="8925b78a93aaae7813363cd61d838f7cbf2ca74b2451be39df5facb6a4e5f3cf"
    ["$task_files"]="b888b4f41f40f6e25f763840e1df85104cdd8e6a2dedad7c1684e2eb7f6d609b"
    ["$task_sdc"]="c4ad2bed6eb2851175d99a8a1a9fbd0f79cf637967de332876f4bd663d8a685a"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m186_fc2_k8_fixed_bank_issue_island_logic_only_dc_contract_r1_20260825.json"]="d5cbd8cda8bb2e07e961f7da64472edc072ce428e3ceea74495251dac53a3fda"
    ["contracts/m186_fc2_k8_fixed_bank_issue_island_vcs_contract_r1_20260825.json"]="a8768d7f1ad2435e7785902085127bd5fc06efe0fd53bc2174a78ba7e90f0f11"
    ["dc_handoff/runs/m186_fc2_k8_fixed_bank_issue_island_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="4ff995f84ef3ce9a49770bfa7e975e73f25e0fc1677038b1991c34785ddc3f8c"
    ["dc_handoff/runs/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="3bb31c56d23e960852e0b27e264bfa4e14506f234ed159bd7655f1cee4ba5b27"
    ["dc_handoff/runs/m169_fc2_k4_unique_bank_accumulator_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="9cc8ef38744687d80644df53fa838db03b0445a5df01cf559e98be3547cdd1cf"
    ["dc_handoff/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="ee772129cfa156349f98c737f53aa435e474dddfb70738833348927826641e8f"
    ["dc_handoff/runs/m185_fc2_k8_fixed_bank_accumulator_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="16d1c042b3a7c21e27fdfde4dcb35fe7de62f9ac614770aaaee37788b8226c58"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export DESIGN_NAME="m186_fc2_k8_fixed_bank_issue_island"
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
awk -v x="$task_area" 'BEGIN {exit !(x < 39000)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x < 45000)}' || exit 40
awk -v x="$task_seq" 'BEGIN {exit !(x < 5000)}' || exit 41
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.0)}' || exit 42
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 43
awk -v x="$task_levels" 'BEGIN {exit !(x <= 150)}' || exit 44

task_sum_k4="32940.809935"
task_sum_k8="37156.643801"
task_schedule="$(awk 'BEGIN {printf "%.9f", 127581198/97607807}')"
task_over_k8_sum="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/37156.643801}')"
task_over_k4_sum="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/32940.809935}')"
task_flat_saving="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", 37156.643801-a}')"
task_conditional_td="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (127581198/97607807)/(a/32940.809935)}')"
task_density_break_even="$(awk 'BEGIN {printf "%.6f", 32940.809935*(127581198/97607807)}')"

{
    echo "status=PASS_M186_FC2_K8_FIXED_BANK_ISSUE_ISLAND_FLAT_LOGIC_ONLY_DC_3NS"
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
    echo "m180_plus_m169_standalone_area_um2=$task_sum_k4"
    echo "m184_plus_m185_standalone_area_um2=$task_sum_k8"
    echo "m186_flat_over_m184_plus_m185_area_ratio=$task_over_k8_sum"
    echo "m186_flat_area_saving_vs_m184_plus_m185_um2=$task_flat_saving"
    echo "m186_flat_over_m180_plus_m169_area_ratio=$task_over_k4_sum"
    echo "m179_k4_over_m182_k8_schedule_ratio=$task_schedule"
    echo "conditional_same_3ns_schedule_throughput_per_area_vs_k4_standalone_sum=$task_conditional_td"
    echo "conditional_density_break_even_area_um2=$task_density_break_even"
    echo "matched_flat_k4_baseline=false"
    echo "weight_sram_macro=false"
    echo "descriptor_producer=false"
    echo "accumulator_context_storage=false"
    echo "bn2=false"
    echo "residual=false"
    echo "complete_fc2=false"
    echo "clock_tree_extracted_parasitics=false"
    echo "placed_and_routed=false"
    echo "saif_power=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m186_fc2_k8_fixed_bank_issue_island_logic_only.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M186 FC2 K8 fixed-bank issue island flat logic-only DC sealed at $task_run"
