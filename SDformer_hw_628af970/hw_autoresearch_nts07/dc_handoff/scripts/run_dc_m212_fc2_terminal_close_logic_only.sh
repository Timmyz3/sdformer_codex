#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m212_fc2_terminal_close_logic_only_dc_3p000ns_r1_sealed_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_min_lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
[[ ! -e "$task_run" ]] || { echo "refusing to overwrite M212 sealed DC run" >&2; exit 2; }
[[ -x "$task_dc_shell" && -s "$task_lib" && -s "$task_min_lib" ]] || exit 3
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"
task_files=dc_handoff/filelists/date_m212_fc2_raw4_to_terminal_close_rtl.f
task_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
task_tcl=dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl
declare -A task_expected=(
 ["rtl_m212/m212_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"]="8d97ea77b72b0767e164a8d8902295440ceda84156732522963de8e038f6739f"
 ["rtl_m212/m212_fc2_descriptor4_terminal_close_frontend.sv"]="c229411f6b6b29020a2dd0250f12ac540a14cb06f064bddc3dfee55e3db34ec9"
 ["rtl_m212/m212_fc2_raw4_to_terminal_close_frontend.sv"]="15aa7db00545080e3318f0e98f8605639440d5e2ab47c2ad33455d3a33785159"
 ["$task_files"]="df4a509935ff5a4176eab8a2700982d5e5cb36d7d737052fa37b5d1d41394863"
 ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
 ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
 ["contracts/m212_fc2_terminal_close_logic_only_dc_contract_r1_20260825.json"]="5062aee8871acfa577e81e040f97cd6a2d6fc3be030b7215d37d53f0898a9506"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
 task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"
export DESIGN_NAME=m212_fc2_raw4_to_terminal_close_frontend HW_ROOT="$task_hw_root" RTL_FILELIST="$task_hw_root/$task_files"
export LIB_DB="$task_lib" MIN_LIB_DB="$task_min_lib" SDC_FILE="$task_hw_root/$task_sdc" OUTPUT_DIR="$task_run" OPERATING_CONDITION=ssg0p9v125c
set +e
"$task_dc_shell" -f "$task_hw_root/$task_tcl" > "$task_run/dc.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/dc.rc"
[[ "$task_rc" -eq 0 ]] || exit 20
grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$task_run/dc.log" && exit 21 || true
grep -Fq 'Thank you...' "$task_run/dc.log" || exit 22
for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt constraint_violators.rpt check_design_postcompile.rpt check_timing_postcompile.rpt resources_postcompile.rpt; do
 [[ -s "$task_run/reports/$task_report" ]] || exit 30
done
[[ -s "$task_run/netlist/${DESIGN_NAME}_mapped.v" && -s "$task_run/netlist/${DESIGN_NAME}_mapped.sdc" && -s "$task_run/netlist/${DESIGN_NAME}.ddc" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "$task_run/reports/timing_setup.rpt" "$task_run/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' "$task_run/reports/constraint_violators.rpt")" -eq 5 ]] || exit 34
[[ "$(tr -d '[:space:]' < "$task_run/reports/check_design_postcompile.rpt")" == 1 ]] || exit 35
[[ "$(tail -n 1 "$task_run/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" == 1 ]] || exit 36
grep -Fq 'Number of macros/black boxes:               0' "$task_run/reports/area.rpt" || exit 37
grep -Eq 'DW_mult|mult_[0-9]' "$task_run/reports/resources_postcompile.rpt" && exit 38 || true
task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_path="$(awk '/Critical Path Length:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
awk -v x="$task_area" 'BEGIN {exit !(x < 21200)}'
awk -v x="$task_cells" 'BEGIN {exit !(x < 31500)}'
awk -v x="$task_seq" 'BEGIN {exit !(x < 2800)}'
awk -v x="$task_setup" 'BEGIN {exit !(x >= 0)}'
awk -v x="$task_hold" 'BEGIN {exit !(x >= 0)}'
awk -v x="$task_levels" 'BEGIN {exit !(x <= 90)}'
{
 echo status=PASS_M212_FC2_TERMINAL_CLOSE_LOGIC_ONLY_DC_3NS
 echo exact_sha=true
 echo tool=Synopsys_DC_V-2023.12-SP3
 echo clock_period_ns=3.000
 echo clock_network=ideal
 echo wireload=ZeroWireload
 echo cell_area_um2="$task_area"
 echo cell_count="$task_cells"
 echo sequential_cells="$task_seq"
 echo logic_levels="$task_levels"
 echo critical_path_length_ns="$task_path"
 echo setup_worst_slack_ns="$task_setup"
 echo hold_worst_slack_ns="$task_hold"
 echo macro_count=0
 echo multipliers_in_mapped_resource_report=0
 echo terminal_hint_and_close_integrated=true
 echo complete_fc2=false
 echo physical_speedup=false
 echo system_speedup=false
 echo paper_ppa_ready=false
 echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt > "$task_run/evidence_manifest.sha256"
sha256sum dc_handoff/scripts/run_dc_m212_fc2_terminal_close_logic_only.sh > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M212 terminal-close frontend logic-only DC sealed at $task_run"
