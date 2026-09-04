#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m190_fc2_k7_single_hole_elision_accumulator_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M190 sealed DC run" >&2; exit 2; fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m190_fc2_k7_single_hole_elision_accumulator_rtl.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m190/m190_fc2_k7_single_hole_elision_accumulator.sv"]="d607cb9f1a7c1bf7ed5917bcb42d87a2b13f9d414475113001c27b9e61e5bcd9"
    ["$task_files"]="9518cee611f4c65f3f1424c61e13ee39faa9d2d56ca62d848ed7c1da3e6f460f"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m190_fc2_k7_single_hole_elision_accumulator_logic_only_dc_contract_r1_20260825.json"]="b7810302788a3bebd610478344d72efaeb1bea416ae6e0d8b0cabd7234f0cef7"
    ["contracts/m190_fc2_k7_single_hole_elision_accumulator_vcs_contract_r1_20260825.json"]="1c153e449475e30649cf5e1d6eeca82708cec4056f357d87c92e152e313c5d7f"
    ["dc_handoff/runs/m190_fc2_k7_single_hole_elision_accumulator_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="340a9dfddc6421822e3240361cc5d86ab1fcbe2bf7b8a138cd56dd98ed4dae51"
    ["dc_handoff/runs/m189_fc2_k7_bank_compacting_accumulator_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="59e1d294bbe0791fe6884c8955abed08cd981e9ec0a1517909f10cec3ec1f5d3"
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

export DESIGN_NAME="m190_fc2_k7_single_hole_elision_accumulator"
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
awk -v x="$task_area" 'BEGIN {exit !(x < 26705.976561)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x < 32000)}' || exit 40
awk -v x="$task_seq" 'BEGIN {exit !(x < 2600)}' || exit 41
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.2)}' || exit 42
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 43
awk -v x="$task_levels" 'BEGIN {exit !(x <= 55)}' || exit 44

task_over_m185="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/27129.815772}')"
task_save="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", 27129.815772-a}')"
task_save_pct="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", (1-a/27129.815772)*100}')"
task_k7_sum="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", 10417.680032+a}')"
task_k8_sum="$(awk 'BEGIN {printf "%.6f", 10026.828029+27129.815772}')"
task_sum_ratio="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (10417.680032+a)/(10026.828029+27129.815772)}')"
task_td="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (97607807/97694539)/((10417.680032+a)/(10026.828029+27129.815772))}')"
task_margin="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", 37123.656593-(10417.680032+a)}')"

{
    echo "status=PASS_M190_FC2_K7_SINGLE_HOLE_ELISION_ACCUMULATOR_LOGIC_ONLY_DC_3NS_POSITIVE_SCREEN"
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
    echo "m190_over_m185_area_ratio=$task_over_m185"
    echo "m190_area_saving_vs_m185_um2=$task_save"
    echo "m190_area_saving_vs_m185_percent=$task_save_pct"
    echo "m190_winning_area_threshold_um2=26705.976561"
    echo "m190_winning_area_threshold_met=true"
    echo "m188_plus_m190_sum_logic_area_um2=$task_k7_sum"
    echo "m184_plus_m185_sum_logic_area_um2=$task_k8_sum"
    echo "k7_over_k8_sum_area_ratio=$task_sum_ratio"
    echo "k7_over_k8_schedule_throughput_factor=0.999112212"
    echo "conditional_k7_over_k8_sum_throughput_per_area=$task_td"
    echo "k7_winning_area_threshold_margin_um2=$task_margin"
    echo "standalone_sum_screen_pass=true"
    echo "sum_of_standalone_logic_areas_only=true"
    echo "flat_composition=false"
    echo "single_hole_elision=true"
    echo "stable_prefix_compaction=false"
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
sha256sum "dc_handoff/scripts/run_dc_m190_fc2_k7_single_hole_elision_accumulator_logic_only.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M190 FC2 K7 single-hole-elision accumulator logic-only DC positive screen sealed at $task_run"
