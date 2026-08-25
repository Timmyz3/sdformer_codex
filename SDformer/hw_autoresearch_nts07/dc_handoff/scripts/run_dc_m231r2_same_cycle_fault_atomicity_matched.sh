#!/usr/bin/env bash
set -euo pipefail
dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "$dc_root/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
run="$dc_root/runs/m231r2_same_cycle_fault_atomicity_matched_dc_3p000ns_r1_20260825"
dc="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
minlib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
[[ ! -e "$run" && -x "$dc" && -s "$lib" && -s "$minlib" ]] || exit 2
mkdir -p "$(dirname "$run")"
mkdir "$run"
complete=0
trap 'rc=$?; if [[ $complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$rc" >"$run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$hw_root"
files=dc_handoff/filelists/date_m231_atlif32_to_fc2_raw4_pingpong_bridge_rtl.f
sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
tcl=dc_handoff/scripts/run_dc_m216_flattened_source_cap_logic_only.tcl
declare -A expected=(
 ["rtl_m231/m231_atlif32_to_fc2_raw4_pingpong_bridge.sv"]="7e129ab7724dcb66c10de51bbb8b444cb0d5e4f3c407b5f910afd1267ddfe89c"
 ["$files"]="36b3c4cd631f97a58762639044a475904b41297a249b3577e4a240a156196417"
 ["$sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
 ["$tcl"]="2565e750551f6f2a03abff96b462558416f3b9531a693f43b6194af8026d61d5"
 ["contracts/m231r2_same_cycle_fault_atomicity_correction_contract_r1_20260825.json"]="1c0ea65de1c71259fa96e4a3e14107eb15440096b58078e509351afff4afc1d5"
 ["results/m231r2_same_cycle_fault_atomicity_directed_vcs_r1_exact_20260825/RUN_COMPLETE.txt"]="9b4319fc17fc5ec1d442b8346f0df5a9340e76a36d7d04cf08c17dd44dc04c8e"
 ["results/m231r2_same_cycle_fault_atomicity_directed_vcs_r1_exact_20260825/SHA256SUMS"]="f49d2b447c07ff3eaa96a6523e5a49c94968c408088e2f9c8df576e99712bfcf"
 ["results/m231_independent_hammer_review_r1_20260825/SHA256SUMS"]="09dc0085cdf35fe1ca7175b2867c5fd444b98712c58d8e70474d0971474ed427"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"$run/preflight_sha_checks.txt"
for path in "${!expected[@]}"; do
 observed="$(sha256sum "$path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$path" "${expected[$path]}" "$observed" >>"$run/preflight_sha_checks.txt"
 [[ "$observed" == "${expected[$path]}" ]] || exit 10
done
sha256sum "${!expected[@]}" >"$run/input_sha256.txt"
run_one() {
 width="$1"
 directory="$run/w$width"
 mkdir "$directory"
 (
  export DESIGN_NAME=m231_atlif32_to_fc2_raw4_pingpong_bridge
  export HW_ROOT="$hw_root" RTL_FILELIST="$hw_root/$files"
  export LIB_DB="$lib" MIN_LIB_DB="$minlib" SDC_FILE="$hw_root/$sdc"
  export OUTPUT_DIR="$directory" OPERATING_CONDITION=ssg0p9v125c
  export ELAB_PARAMETERS="INPUT_WIDTH=$width"
  set +e
  "$dc" -f "$hw_root/$tcl" >"$directory/dc.log" 2>&1
  rc=$?
  set -e
  echo "$rc" >"$directory/dc.rc"
  [[ $rc -eq 0 ]] || exit 20
  grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$directory/dc.log" && exit 21 || true
  grep -Fq 'Thank you...' "$directory/dc.log" || exit 22
  for report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt constraint_violators.rpt check_design_postcompile.rpt check_timing_postcompile.rpt resources_postcompile.rpt; do
   [[ -s "$directory/reports/$report" ]] || exit 30
  done
  [[ -s "$directory/netlist/${DESIGN_NAME}_mapped.v" && -s "$directory/netlist/${DESIGN_NAME}_mapped.sdc" && -s "$directory/netlist/${DESIGN_NAME}.ddc" ]] || exit 31
  [[ "$(tr -d '[:space:]' <"$directory/reports/check_design_postcompile.rpt")" == 1 ]] || exit 35
  [[ "$(tail -n1 "$directory/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" == 1 ]] || exit 36
  grep -Fq 'Number of macros/black boxes:               0' "$directory/reports/area.rpt" || exit 37
  area="$(awk '/Total cell area:/{print $4;exit}' "$directory/reports/area.rpt")"
  cells="$(awk '/Number of cells:/{print $4;exit}' "$directory/reports/area.rpt")"
  sequential="$(awk '/Number of sequential cells:/{print $5;exit}' "$directory/reports/area.rpt")"
  levels="$(awk '/Levels of Logic:/{print $4;exit}' "$directory/reports/qor.rpt")"
  path_length="$(awk '/Critical Path Length:/{print $4;exit}' "$directory/reports/qor.rpt")"
  setup="$(awk '/slack \((MET|VIOLATED)\)/{print $3;exit}' "$directory/reports/timing_setup.rpt")"
  hold="$(awk '/slack \((MET|VIOLATED)\)/{print $3;exit}' "$directory/reports/timing_hold.rpt")"
  setup_met=false
  hold_met=false
  awk -v value="$setup" 'BEGIN{exit !(value>=0)}' && setup_met=true
  awk -v value="$hold" 'BEGIN{exit !(value>=0)}' && hold_met=true
  awk -v value="$area" 'BEGIN{exit !(value>0&&value<200000)}'
  awk -v value="$sequential" 'BEGIN{exit !(value>=1500&&value<15000)}'
  {
   echo status=COMPLETE_M231R2_W${width}_LOGIC_ONLY_DC_SCREEN
   echo exact_sha=true
   echo elaboration_parameter=INPUT_WIDTH=$width
   echo clock_period_ns=3.000
   echo cell_area_um2="$area"
   echo cell_count="$cells"
   echo sequential_cells="$sequential"
   echo logic_levels="$levels"
   echo critical_path_length_ns="$path_length"
   echo setup_worst_slack_ns="$setup"
   echo hold_worst_slack_ns="$hold"
   echo setup_met="$setup_met"
   echo hold_met="$hold_met"
   echo bridge_storage_bits=$((4*width))
   echo same_cycle_fault_atomicity=true
   echo macro_count=0
   echo complete_ffn=false
   echo system_speedup=false
   echo paper_ppa_ready=false
  } >"$directory/RUN_COMPLETE.txt"
  sha256sum "$directory"/dc.log "$directory"/reports/*.rpt "$directory"/netlist/* "$directory"/RUN_COMPLETE.txt >"$directory/evidence_manifest.sha256"
 )
}
run_one 384 & p384=$!
run_one 768 & p768=$!
run_one 1536 & p1536=$!
run_one 3072 & p3072=$!
rc=0
wait "$p384" || rc=1
wait "$p768" || rc=1
wait "$p1536" || rc=1
wait "$p3072" || rc=1
[[ $rc -eq 0 ]] || exit 40
{
 echo status=COMPLETE_M231R2_MATCHED_LOGIC_ONLY_DC_SCREEN
 echo exact_sha=true
 echo same_cycle_fault_atomicity=true
 echo parallel_matched_runs=true
 echo same_rtl_sdc_tcl=true
 for width in 384 768 1536 3072; do
  awk -v prefix="w${width}_" -F= '/^(cell_area_um2|cell_count|sequential_cells|setup_worst_slack_ns|hold_worst_slack_ns|bridge_storage_bits)=/{print prefix $1 "=" $2}' "$run/w$width/RUN_COMPLETE.txt"
 done
 echo macro_count_each=0
 echo complete_ffn=false
 echo system_speedup=false
 echo paper_ppa_ready=false
} >"$run/RUN_COMPLETE.txt"
sha256sum "$runner" >"$run/runner_sha256.txt"
sha256sum "$run"/w*/evidence_manifest.sha256 "$run/RUN_COMPLETE.txt" >"$run/evidence_manifest.sha256"
complete=1
echo "PASS M231r2 matched DC screen sealed at $run"
