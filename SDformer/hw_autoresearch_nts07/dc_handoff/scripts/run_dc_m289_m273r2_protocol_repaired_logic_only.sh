#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
if [[ "${M289_TEST_WRONG_SHA:-0}" == "1" ]]; then
    task_run="$task_dc_root/runs/m289_m273r2_protocol_repaired_logic_only_dc_wrong_sha_preflight_r1_20260825"
else
    task_run="$task_dc_root/runs/m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825"
fi
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
task_files="dc_handoff/filelists/date_m273_integrated_rank3_atlif_rtl.f"
task_sdc="dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc"
task_tcl="dc_handoff/scripts/run_dc_m289_m273r2_flattened_logic_only.tcl"
task_contract="contracts/m289_m273r2_protocol_repaired_logic_only_dc_contract_r1_20260825.json"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M289 evidence: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    echo "refusing M289 DC because another dc_shell is active" >&2
    exit 4
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d"
    ["$task_files"]="c99fe329c43276ce40f7027d54baeaaf747553c9f0b8d4419dcf8e7574b1a02d"
    ["$task_sdc"]="73030f70b27909c1f8100bbc02af75c77fed246908027980912afd6499beb6e3"
    ["$task_tcl"]="9ef3912ff13b17afad739c4af1eaf74087a31919735772ef8c98d4bf569071dc"
    ["contracts/m285_m273r2_glitch_clean_zero_tile_vcs_contract_r1_20260825.json"]="0f0ebe41e70a2a599aa7202622e8fc472a912f8adb019c3e3ddf0357211445df"
    ["results/m285_m273r2_glitch_clean_zero_tile_vcs_r1_exact_20260825/RUN_COMPLETE.txt"]="d10f1a0fefb4b3fc62935cca64bd11facb357290ac5f73f06ec8fde8e8f2476d"
    ["results/m286_m285_m273r2_independent_review_r1_20260825/m286_m285_m273r2_independent_hammer_review_r1.json"]="da3fb7adcc86ada343449b10e6fdd551824e1f4ee6dc67f5420bb2b50fea1b28"
    ["results/m286_m285_m273r2_independent_review_r1_20260825/RUN_MANIFEST.seal.sha256"]="1ae630a94bd5639739d664c7cd05ed750a3d312dffd62446d24e81466d15d3fe"
    ["dc_handoff/runs/m277_integrated_rank3_atlif_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_FAILED_OR_INCOMPLETE.txt"]="5b05ac411bf5ff3ddf9eecf06475c5dc8bfd2f7bc79712d270a295dba8e87dd9"
    ["dc_handoff/runs/m277_integrated_rank3_atlif_logic_only_dc_3p000ns_r1_sealed_20260825/reports/constraint_violators.rpt"]="d8e3eb70e9f9b22a47b2819b3defc771e4e4d7011cf3f655936f168477b4a8d2"
    ["$task_contract"]="07efe93cbbd2e6998944f9f8c96422e8a489a3627c4d2f1f861c97a9d0675710"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
if [[ "${M289_TEST_WRONG_SHA:-0}" == "1" ]]; then
    task_expected["rtl_m273/m273_integrated_rank3_atlif.sv"]="0000000000000000000000000000000000000000000000000000000000000000"
fi
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" "$task_lib" "$task_min_lib" \
    > "$task_run/input_sha256.txt"

export DESIGN_NAME="m273_integrated_rank3_atlif"
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
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt; do
    [[ -s "$task_run/reports/$task_report" ]] || exit 30
done
[[ -s "$task_run/netlist/${DESIGN_NAME}_mapped.v" \
   && -s "$task_run/netlist/${DESIGN_NAME}_mapped.sdc" \
   && -s "$task_run/netlist/${DESIGN_NAME}.ddc" \
   && -s "$task_run/netlist/${DESIGN_NAME}.svf" ]] || exit 31
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
task_comb_area="$(awk '/Combinational area:/ {print $3; exit}' "$task_run/reports/area.rpt")"
task_seq_area="$(awk '/Noncombinational area:/ {print $3; exit}' "$task_run/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_path="$(awk '/Critical Path Length:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_start="$(awk '/Startpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
task_end="$(awk '/Endpoint:/ {print $2; exit}' "$task_run/reports/timing_setup.rpt")"
for task_value in "$task_area" "$task_cells" "$task_seq" "$task_comb_area" \
        "$task_seq_area" "$task_levels" "$task_path" "$task_setup" "$task_hold"; do
    [[ -n "$task_value" ]] || exit 38
done
awk -v x="$task_area" 'BEGIN {exit !(x > 0 && x < 500000)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x > 0 && x < 500000)}' || exit 40
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.0)}' || exit 41
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 42

{
    echo "status=PASS_M289_M273R2_PROTOCOL_REPAIRED_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "clock_network=ideal"
    echo "wireload=ZeroWireload"
    echo "hierarchy=flattened_before_mapping"
    echo "synthesis_max_fanout=24"
    echo "paper_requirement_max_fanout=32"
    echo "constraint_classes_without_violations=5"
    echo "cell_area_um2=$task_area"
    echo "combinational_area_um2=$task_comb_area"
    echo "sequential_area_um2=$task_seq_area"
    echo "cell_count=$task_cells"
    echo "sequential_cells=$task_seq"
    echo "logic_levels=$task_levels"
    echo "critical_path_length_ns=$task_path"
    echo "critical_startpoint=$task_start"
    echo "critical_endpoint=$task_end"
    echo "setup_worst_slack_ns=$task_setup"
    echo "hold_worst_slack_ns=$task_hold"
    echo "macro_count=0"
    echo "m273r2_protocol_repaired=true"
    echo "m285_vcs_bound=true"
    echo "m286_independent_review_bound=true"
    echo "m277_predecessor_admitted=false"
    echo "fixed_same_boundary_rtl=false"
    echo "area_matched_fixed_comparison=false"
    echo "throughput_per_area_admitted=false"
    echo "trained_rank3_accuracy=false"
    echo "sram_macro_inclusive=false"
    echo "clock_tree_extracted_parasitics=false"
    echo "saif_power=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt \
    "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m289_m273r2_protocol_repaired_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
sha256sum "$task_run/evidence_manifest.sha256" > "$task_run/evidence_manifest.seal.sha256"
task_complete=1
echo "PASS M289 M273r2 protocol-repaired logic-only DC sealed at $task_run"
