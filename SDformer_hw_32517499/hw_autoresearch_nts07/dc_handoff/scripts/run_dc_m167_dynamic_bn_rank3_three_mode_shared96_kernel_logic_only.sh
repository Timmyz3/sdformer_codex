#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m167_dynamic_bn_rank3_three_mode_shared96_kernel_logic_only_dc_3p000ns_r1_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M167 sealed DC run: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m167_three_mode_shared96_kernel_rtl.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m167/m167_dynamic_bn_rank3_three_mode_shared96_kernel.sv"]="9cb7bbeb4ef720c6d0ec09bb67df2a7ebd3438cde055fd7f6412fb55d1a9705c"
    ["$task_files"]="d383db02c1256322cdab0c238fe688a7c70943055cee57cac908e6aeb3ec4671"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m167_dynamic_bn_rank3_three_mode_shared96_kernel_logic_only_dc_contract_r1_20260824.json"]="45bface905a03b6d49756378f891853ef0f2668bff550770372a5385febe67a2"
    ["contracts/m167_dynamic_bn_rank3_three_mode_shared96_kernel_vcs_contract_r1_20260824.json"]="5492fb060df91c4f89475c9653598f03ff2bbe04b54f70ec6e06aad156fe2205"
    ["dc_handoff/runs/m167_dynamic_bn_rank3_three_mode_shared96_kernel_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="7efce8eb65d10db32d9d4dc3a0ce88d0d70f3aa333c0c605eab949717341337e"
    ["dc_handoff/runs/m165_owned_raw_bank_dynamic_bn_rank3_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="e08705a7f7d5cb2f80292471a6b5cd41821ee03cca5766f359639832ed3ed9fd"
    ["dc_handoff/runs/m166_prefolded_rank3_left_atlif_backend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="2c43b532d1862a0d780c9ad76a00a75f5ae9dab813650b6b2a2cd7f0a0b377ad"
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

export DESIGN_NAME="m167_dynamic_bn_rank3_three_mode_shared96_kernel"
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
awk -v x="$task_area" 'BEGIN {exit !(x < 60000)}' || exit 38
awk -v x="$task_cells" 'BEGIN {exit !(x < 55000)}' || exit 39
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.2)}' || exit 40
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 41
awk -v x="$task_levels" 'BEGIN {exit !(x <= 60)}' || exit 42

task_sum_area="83848.588969"
task_boundary_reduction="$(awk -v sum="$task_sum_area" -v current="$task_area" 'BEGIN {printf "%.6f", 100.0*(sum-current)/sum}')"
{
    echo "status=PASS_M167_DYNAMIC_BN_RANK3_THREE_MODE_SHARED96_KERNEL_LOGIC_ONLY_DC_3NS"
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
    echo "shared_signed_int8_main_product_slots_in_source=96"
    echo "front_signed_square_lanes_in_source=32"
    echo "three_mutually_exclusive_phase_modes=true"
    echo "m165_standalone_area_um2=39568.535846"
    echo "m166_standalone_area_um2=44280.053123"
    echo "m165_plus_m166_independent_area_um2=$task_sum_area"
    echo "m167_vs_independent_m165_m166_area_boundary_reduction_pct=$task_boundary_reduction"
    echo "m167_full_function_equivalence_to_m165_plus_m166=false"
    echo "full_controller=false"
    echo "configuration_sram=false"
    echo "rank_state_epoch_sram=false"
    echo "dynamic_bn_rsqrt=false"
    echo "fixed_point_checkpoint_equivalence=false"
    echo "paft_valid825=false"
    echo "full_ffn_cycles=false"
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
sha256sum "dc_handoff/scripts/run_dc_m167_dynamic_bn_rank3_three_mode_shared96_kernel_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M167 three-mode shared96 kernel logic-only DC sealed at $task_run"
