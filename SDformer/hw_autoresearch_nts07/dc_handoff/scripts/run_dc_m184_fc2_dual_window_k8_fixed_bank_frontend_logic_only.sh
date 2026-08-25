#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M184 sealed DC run: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m184_fc2_dual_window_k8_fixed_bank_frontend_rtl.f"
task_sdc="dc_handoff/constraints/date_m180_logic_only_3ns_maxfanout16.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m184/m184_fc2_dual_window_k8_fixed_bank_frontend.sv"]="c6212049305faf42cda13f7f3408d5fa478c79a7c76c142501ec01d9f1e01cd6"
    ["$task_files"]="397e01858f59dbc9a840e10296ced0761d80a9246551e129e5261de835327516"
    ["$task_sdc"]="c4ad2bed6eb2851175d99a8a1a9fbd0f79cf637967de332876f4bd663d8a685a"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m184_fc2_dual_window_k8_fixed_bank_frontend_logic_only_dc_contract_r1_20260825.json"]="b5779d368c2f1ced9be31d9098fb76c6b25a8e86eabb5ff1edeb27b858267954"
    ["contracts/m184_fc2_dual_window_k8_fixed_bank_frontend_vcs_contract_r1_20260825.json"]="64883d54ebb69471198851078bcb17a336dc34f0df0f26b921938687804b8e1e"
    ["dc_handoff/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="178ff9b41b53779ebc66891defde36e19dae59677929ac57ceb2acad45bba139"
    ["contracts/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_contract_r1_20260824.json"]="c22a232ce0b2c0927092a2382d00e176446fd738b5fc05bc92979aeac088a191"
    ["dc_handoff/runs/m180_fc2_dual_window_k4_reservoir_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="3bb31c56d23e960852e0b27e264bfa4e14506f234ed159bd7655f1cee4ba5b27"
    ["contracts/m182_h67_fc2_k8_dual_window_depth_exact_payload_dse_contract_r1_20260824.json"]="4dea36a1ebcb544ea597a84c34fdf7759962adaaf6d6ca2f2ae3a7f511be642a"
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

export DESIGN_NAME="m184_fc2_dual_window_k8_fixed_bank_frontend"
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
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.0)}' || exit 41
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 42
awk -v x="$task_levels" 'BEGIN {exit !(x <= 150)}' || exit 43

task_area_ratio="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", a/14417.928053}')"
task_area_reduction="$(awk -v a="$task_area" 'BEGIN {printf "%.6f", (1-a/14417.928053)*100}')"
task_schedule_ratio="$(awk 'BEGIN {printf "%.9f", 127581198/97607807}')"
task_td_ratio="$(awk -v a="$task_area" 'BEGIN {printf "%.9f", (127581198/97607807)/(a/14417.928053)}')"

{
    echo "status=PASS_M184_FC2_DUAL_WINDOW_K8_FIXED_BANK_FRONTEND_LOGIC_ONLY_DC_3NS"
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
    echo "m180_k4_frontend_area_um2=14417.928053"
    echo "m184_over_m180_frontend_area_ratio=$task_area_ratio"
    echo "m184_frontend_area_reduction_percent=$task_area_reduction"
    echo "m180_k4_logic_levels=161"
    echo "m179_k4_wall_cycles=127581198"
    echo "m182_bounded_k8_wall_cycles=97607807"
    echo "m180_k4_over_m184_k8_schedule_ratio=$task_schedule_ratio"
    echo "same_3ns_schedule_throughput_per_area_m184_over_m180=$task_td_ratio"
    echo "global_topk_sort=false"
    echo "bank_id_payload=false"
    echo "bank_to_prefix_packing=false"
    echo "fixed_bank_valid_mask=true"
    echo "native_descriptor_producer=false"
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
sha256sum "dc_handoff/scripts/run_dc_m184_fc2_dual_window_k8_fixed_bank_frontend_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M184 FC2 dual-window K8 fixed-bank frontend logic-only DC sealed at $task_run"
