#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m177_fc2_indexed_nonzero96_vs_m175_raw96_matched_logic_only_dc_3p000ns_r2_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M177 sealed DC run: $task_run" >&2
    exit 2
fi
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
task_files="dc_handoff/filelists/date_m177_fc2_indexed_nonzero96_k4_replay_frontend_rtl.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m177/m177_fc2_indexed_nonzero96_k4_replay_frontend.sv"]="ef0e9f6075420f404dcb7617c74e7cc2a36af6db28cada0853c587432703a21f"
    ["$task_files"]="ecd63ed069d5ff61bd21942073081f79ed35acc775cf23e40f78ac7236c0007f"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m177_fc2_indexed_nonzero96_vs_m175_raw96_matched_logic_only_dc_contract_r1_20260824.json"]="78d334c2a707a2ff92cbb9ad64ccc0b07faea1ca5e425fca73486af001875e86"
    ["contracts/m177_fc2_indexed_nonzero96_k4_replay_frontend_vcs_contract_r1_20260824.json"]="7332a0975a38b16551bf43028e2bb559b1a5d0b7705c85d2c3c2f397327e732a"
    ["contracts/m177_r1_structural_timing_loop_correction_overlay_r2_20260824.json"]="941978606fc89171d7456f78630cb78fe64c9813e4f9e4da9c9b4b9645abfba9"
    ["contracts/m176_r1_beat_index_and_producer_admission_overlay_r1_20260824.json"]="c19ef872a5ca507bc29e2fe625bfe2700e07e671fd1d50ece8ff5342c1396dc9"
    ["dc_handoff/runs/m177_fc2_indexed_nonzero96_k4_replay_frontend_vcs_r2_sealed_20260824/RUN_COMPLETE.txt"]="54544ff43b6dfb1e0e0b0aa4973a200c9beba6701134a1dc6af8f642477dfd7d"
    ["dc_handoff/runs/m175_fc2_bitmap96_vs_m174_bitmap128_matched_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="cf1cd0a94ce9734bdb5d2b6ce7aec004ef0db5d00d1a29591d7bcf51ba55fab2"
    ["results/m176_independent_hammer_review_r1_20260824/manifest.sha256"]="bdc6ae8c0ba3b9ce5712f31107aef618385c09941358a8d460ae3fd898a6ee74"
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

export DESIGN_NAME="m177_fc2_indexed_nonzero96_k4_replay_frontend"
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
awk -v x="$task_area" 'BEGIN {exit !(x < 1800.0)}' || exit 39
awk -v x="$task_cells" 'BEGIN {exit !(x < 2500)}' || exit 40
awk -v x="$task_seq" 'BEGIN {exit !(x <= 300)}' || exit 41
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0.05)}' || exit 42
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0.0)}' || exit 43
awk -v x="$task_levels" 'BEGIN {exit !(x <= 80.0)}' || exit 44

task_m177_over_m175_area="$(awk -v x="$task_area" 'BEGIN {printf "%.9f", x / 1309.266002}')"
task_m177_area_overhead="$(awk -v x="$task_area" 'BEGIN {printf "%.6f", 100.0 * (x - 1309.266002) / 1309.266002}')"
task_conditional_td="$(awk -v x="$task_area" 'BEGIN {printf "%.9f", 1.0926702530364525 * 1309.266002 / x}')"

{
    echo "status=PASS_M177_INDEXED_NONZERO96_VS_M175_RAW96_MATCHED_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "clock_network=ideal"
    echo "wireload=ZeroWireload"
    echo "hierarchy=flattened_before_mapping"
    echo "m177_descriptor_bitmap_width_bits=96"
    echo "m177_beat_index_bits=5"
    echo "m177_cell_area_um2=$task_area"
    echo "m177_cell_count=$task_cells"
    echo "m177_sequential_cells=$task_seq"
    echo "m177_logic_levels=$task_levels"
    echo "m177_critical_path_length_ns=$task_path"
    echo "m177_critical_startpoint=$task_start"
    echo "m177_critical_endpoint=$task_end"
    echo "m177_setup_worst_slack_ns=$task_setup"
    echo "m177_hold_worst_slack_ns=$task_hold"
    echo "m175_cell_area_um2=1309.266002"
    echo "m175_cell_count=1783"
    echo "m175_sequential_cells=236"
    echo "m175_logic_levels=55.00"
    echo "m175_setup_worst_slack_ns=0.4731"
    echo "m175_hold_worst_slack_ns=0.0003"
    echo "m177_over_m175_area_ratio=$task_m177_over_m175_area"
    echo "m177_area_overhead_percent=$task_m177_area_overhead"
    echo "m176_native_preindexed_raw96_over_indexed96_K4_ratio=1.092670253"
    echo "conditional_native_preindexed_logic_only_throughput_density_ratio=$task_conditional_td"
    echo "macro_count=0"
    echo "multipliers_in_mapped_resource_report=0"
    echo "native_index_producer=false"
    echo "finite_producer_fifo=false"
    echo "posthoc_scanner_speedup=false"
    echo "descriptor_memory_delivery=false"
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
sha256sum "dc_handoff/scripts/run_dc_m177_fc2_indexed_nonzero96_vs_m175_raw96_matched_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M177 indexed-nonzero96 versus M175 raw96 matched logic-only DC sealed at $task_run"
