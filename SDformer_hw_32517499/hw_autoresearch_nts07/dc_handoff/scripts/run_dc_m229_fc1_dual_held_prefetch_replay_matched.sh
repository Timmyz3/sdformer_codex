#!/usr/bin/env bash
set -euo pipefail
dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)";hw_root="$(cd "$dc_root/.."&&pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")";run="$dc_root/runs/m229_fc1_dual_held_prefetch_replay_matched_dc_3p000ns_r1_20260825"
dc="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
minlib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
[[ ! -e "$run"&&-x "$dc"&&-s "$lib"&&-s "$minlib" ]]||exit 2;mkdir -p "$(dirname "$run")";mkdir "$run"
complete=0;trap 'rc=$?;if [[ $complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$rc">"$run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$hw_root";files=dc_handoff/filelists/date_m229_fc1_dual_held_prefetch_replay_rtl.f;sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc;tcl=dc_handoff/scripts/run_dc_m216_flattened_source_cap_logic_only.tcl
declare -A expected=(
 ["rtl_m229/m229_fc1_dual_held_prefetch_replay_island.sv"]="c36fe753a16fbf76ec9c1654d7ee991ab999964e2b3491d2b86c0badc6cce1e9"
 ["$files"]="1feb42fe141d3c60dab9d9d3179fb364a0bc29bd853b255316b65a69b6f6bc58"
 ["$sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
 ["$tcl"]="2565e750551f6f2a03abff96b462558416f3b9531a693f43b6194af8026d61d5"
 ["contracts/m229_fc1_dual_held_prefetch_replay_synopsys_contract_r1_20260825.json"]="424aabfe0e36570d221b2aa255414af3a320a6291f82269117376f5448546ee2"
 ["results/m229_fc1_dual_held_prefetch_replay_directed_vcs_r1_exact_20260825/RUN_COMPLETE.txt"]="e3a6c9a9b01950e50a12d38b2b258ebdc7c047c83f5ca56aedc3bb3bb589db2f"
 ["results/m229_fc1_dual_held_prefetch_replay_directed_vcs_r1_exact_20260825/SHA256SUMS"]="7591869a0e519f32e309794a5f66d43bfd1b57d059f4cc2261d9be4ae5f9186e"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
:>"$run/preflight_sha_checks.txt";for p in "${!expected[@]}";do o="$(sha256sum "$p"|awk '{print $1}')";printf 'path=%s expected=%s observed=%s\n' "$p" "${expected[$p]}" "$o">>"$run/preflight_sha_checks.txt";[[ "$o" == "${expected[$p]}" ]]||exit 10;done
sha256sum "${!expected[@]}">"$run/input_sha256.txt"
run_one(){ f="$1";d="$run/f$f";mkdir "$d";(
 export DESIGN_NAME=m229_fc1_dual_held_prefetch_replay_island HW_ROOT="$hw_root" RTL_FILELIST="$hw_root/$files" LIB_DB="$lib" MIN_LIB_DB="$minlib" SDC_FILE="$hw_root/$sdc" OUTPUT_DIR="$d" OPERATING_CONDITION=ssg0p9v125c ELAB_PARAMETERS="FANOUT=$f"
 set +e;"$dc" -f "$hw_root/$tcl">"$d/dc.log" 2>&1;rc=$?;set -e;echo "$rc">"$d/dc.rc";[[ $rc -eq 0 ]]||exit 20
 grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$d/dc.log"&&exit 21||true;grep -Fq 'Thank you...' "$d/dc.log"||exit 22
 for r in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt constraint_violators.rpt check_design_postcompile.rpt check_timing_postcompile.rpt resources_postcompile.rpt;do [[ -s "$d/reports/$r" ]]||exit 30;done
 [[ -s "$d/netlist/${DESIGN_NAME}_mapped.v"&&-s "$d/netlist/${DESIGN_NAME}_mapped.sdc"&&-s "$d/netlist/${DESIGN_NAME}.ddc" ]]||exit 31
 [[ "$(tr -d '[:space:]'<"$d/reports/check_design_postcompile.rpt")" == 1 ]]||exit 35
 [[ "$(tail -n1 "$d/reports/check_timing_postcompile.rpt"|tr -d '[:space:]')" == 1 ]]||exit 36
 grep -Fq 'Number of macros/black boxes:               0' "$d/reports/area.rpt"||exit 37
 area="$(awk '/Total cell area:/{print $4;exit}' "$d/reports/area.rpt")";cells="$(awk '/Number of cells:/{print $4;exit}' "$d/reports/area.rpt")";seq="$(awk '/Number of sequential cells:/{print $5;exit}' "$d/reports/area.rpt")";levels="$(awk '/Levels of Logic:/{print $4;exit}' "$d/reports/qor.rpt")";path="$(awk '/Critical Path Length:/{print $4;exit}' "$d/reports/qor.rpt")";setup="$(awk '/slack \((MET|VIOLATED)\)/{print $3;exit}' "$d/reports/timing_setup.rpt")";hold="$(awk '/slack \((MET|VIOLATED)\)/{print $3;exit}' "$d/reports/timing_hold.rpt")"
 sm=false;hm=false;awk -v x="$setup" 'BEGIN{exit !(x>=0)}'&&sm=true;awk -v x="$hold" 'BEGIN{exit !(x>=0)}'&&hm=true
 awk -v x="$area" 'BEGIN{exit !(x>0&&x<100000)}';awk -v x="$seq" 'BEGIN{exit !(x>=3000&&x<5000)}'
 { echo status=COMPLETE_M229_F${f}_LOGIC_ONLY_DC_SCREEN;echo exact_sha=true;echo elaboration_parameter=FANOUT=$f;echo clock_period_ns=3.000;echo cell_area_um2="$area";echo cell_count="$cells";echo sequential_cells="$seq";echo logic_levels="$levels";echo critical_path_length_ns="$path";echo setup_worst_slack_ns="$setup";echo hold_worst_slack_ns="$hold";echo setup_met="$sm";echo hold_met="$hm";echo acc_capacity_port_cut_bits=14592;echo macro_count=0;echo complete_fc1=false;echo system_speedup=false;echo paper_ppa_ready=false;} >"$d/RUN_COMPLETE.txt"
 sha256sum "$d"/dc.log "$d"/reports/*.rpt "$d"/netlist/* "$d"/RUN_COMPLETE.txt>"$d/evidence_manifest.sha256"
 )
}
run_one 1&p1=$!;run_one 2&p2=$!;run_one 4&p4=$!;rc=0;wait "$p1"||rc=1;wait "$p2"||rc=1;wait "$p4"||rc=1;[[ $rc -eq 0 ]]||exit 40
a1="$(awk -F= '/^cell_area_um2=/{print $2}' "$run/f1/RUN_COMPLETE.txt")";a2="$(awk -F= '/^cell_area_um2=/{print $2}' "$run/f2/RUN_COMPLETE.txt")";a4="$(awk -F= '/^cell_area_um2=/{print $2}' "$run/f4/RUN_COMPLETE.txt")";r2="$(awk -v a="$a2" -v b="$a1" 'BEGIN{printf "%.12f",a/b}')";r4="$(awk -v a="$a4" -v b="$a1" 'BEGIN{printf "%.12f",a/b}')";t2="$(awk -v s=1.7 -v a="$r2" 'BEGIN{printf "%.12f",s/a}')";t4="$(awk -v s=2.586956521739 -v a="$r4" 'BEGIN{printf "%.12f",s/a}')"
{
 echo status=COMPLETE_M229_MATCHED_LOGIC_ONLY_DC_SCREEN;echo exact_sha=true;echo parallel_matched_runs=true;echo same_rtl_sdc_tcl=true
 echo f1_area_um2="$a1";echo f2_area_um2="$a2";echo f4_area_um2="$a4";echo f2_over_f1_area_ratio="$r2";echo f4_over_f1_area_ratio="$r4"
 echo clean_directed_service_f2_speedup=1.700000000000;echo clean_directed_service_f4_speedup=2.586956521739
 echo directed_f2_throughput_per_area="$t2";echo directed_f4_throughput_per_area="$t4";echo directed_ratios_are_not_trace_performance=true
 echo acc_capacity_port_cut_bits_each=14592;echo macro_count_each=0;echo complete_fc1=false;echo system_speedup=false;echo paper_ppa_ready=false
}>"$run/RUN_COMPLETE.txt"
sha256sum "$runner">"$run/runner_sha256.txt";sha256sum "$run"/f*/evidence_manifest.sha256 "$run/RUN_COMPLETE.txt">"$run/evidence_manifest.sha256";complete=1;echo "PASS M229 matched DC screen sealed at $run"
