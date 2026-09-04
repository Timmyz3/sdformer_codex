#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M188 sealed DC run: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m188_fc2_dual_window_k7_weakest_bank_defer_frontend_rtl.f"
task_sdc="dc_handoff/constraints/date_m180_logic_only_3ns_maxfanout16.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m188/m188_fc2_dual_window_k7_weakest_bank_defer_frontend.sv"]="e5401fce15191b261ccf5c413a221d2e9042daf16c9e0773654b4f27b851f0ee"
    ["$task_files"]="ca33eed3b54e56aa960740210f7dc80bdc15f2a72aed11d95f682f32cd8a129c"
    ["$task_sdc"]="c4ad2bed6eb2851175d99a8a1a9fbd0f79cf637967de332876f4bd663d8a685a"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_logic_only_dc_contract_r1_20260825.json"]="2af9362b8c82480606a7bf8a9fb8508177731fbe06f132de3192dad35c6782f4"
    ["contracts/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_vcs_contract_r1_20260825.json"]="ec1473bb7930cd7b4d274099deb7a570b96b8d07967e082b502c9770cf055fe1"
    ["dc_handoff/runs/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="ee5d41c6e5b6701a553769d36d948ca9ada1033a1a7badee64fae62c1c2deb99"
    ["dc_handoff/runs/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="3bb31c56d23e960852e0b27e264bfa4e14506f234ed159bd7655f1cee4ba5b27"
    ["dc_handoff/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="ee772129cfa156349f98c737f53aa435e474dddfb70738833348927826641e8f"
    ["contracts/m187_h67_fc2_fixed_bank_kcap_exact_payload_dse_contract_r1_20260825.json"]="ad1316e92eedc2b6ca71cc21ca0795f401bbe6f9436652a0e01077a8fc5a1f9d"
    ["results/m187_h67_fc2_fixed_bank_kcap_exact_payload_dse_r1_20260825/m187_h67_fc2_fixed_bank_kcap_exact_payload_dse.json"]="411e61ff9c5e0a8b4ff27e86cf15d5ae87b8ef523fc25e43b0939417d65cd201"
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

export DESIGN_NAME="m188_fc2_dual_window_k7_weakest_bank_defer_frontend"
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
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$task_run/dc.log"; then exit 21; fi
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
        "$task_run/reports/timing_hold.rpt"; then exit 32; fi
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
awk -v x="$task_area" 'BEGIN {exit !(x < 12000)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x < 18000)}' || exit 40
awk -v x="$task_seq" 'BEGIN {exit !(x < 2200)}' || exit 41
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.0)}' || exit 42
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 43
awk -v x="$task_levels" 'BEGIN {exit !(x <= 160)}' || exit 44

task_over_m184="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/10026.828029}')"
task_over_m180="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/14417.928053}')"
task_k7_over_k8_throughput="$(awk 'BEGIN {printf "%.9f", 97607807/97694539}')"
task_k4_over_k7_schedule="$(awk 'BEGIN {printf "%.9f", 127581198/97694539}')"
task_td_vs_m184="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (97607807/97694539)/(a/10026.828029)}')"
task_td_vs_m180="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (127581198/97694539)/(a/14417.928053)}')"

{
    echo "status=PASS_M188_FC2_DUAL_WINDOW_K7_WEAKEST_BANK_DEFER_FRONTEND_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "max_fanout=16"
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
    echo "m184_k8_frontend_area_um2=10026.828029"
    echo "m180_k4_frontend_area_um2=14417.928053"
    echo "m188_over_m184_area_ratio=$task_over_m184"
    echo "m188_over_m180_area_ratio=$task_over_m180"
    echo "k7_over_k8_total_throughput_factor=$task_k7_over_k8_throughput"
    echo "m179_k4_over_m187_k7_schedule_ratio=$task_k4_over_k7_schedule"
    echo "conditional_frontend_throughput_per_area_m188_over_m184=$task_td_vs_m184"
    echo "conditional_frontend_throughput_per_area_m188_over_m180=$task_td_vs_m180"
    echo "k7_frontend_supersedes_m184=false"
    echo "maximum_sources_per_group=7"
    echo "weakest_bank_defer=true"
    echo "eight_to_seven_weight_compactor=false"
    echo "k7_accumulator=false"
    echo "weight_sram_response=false"
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
sha256sum "dc_handoff/scripts/run_dc_m188_fc2_dual_window_k7_weakest_bank_defer_frontend_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M188 FC2 dual-window K7 weakest-bank-defer frontend logic-only DC sealed at $task_run"

