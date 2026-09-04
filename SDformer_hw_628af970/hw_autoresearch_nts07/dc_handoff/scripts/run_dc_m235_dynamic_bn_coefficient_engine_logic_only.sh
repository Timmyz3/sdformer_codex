#!/usr/bin/env bash
set -euo pipefail
dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "$dc_root/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
run="$dc_root/runs/m235_dynamic_bn_coefficient_engine_logic_only_dc_3p000ns_r1_20260825"
dc="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
lib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
minlib=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
[[ ! -e "$run" && -x "$dc" && -s "$lib" && -s "$minlib" ]] || exit 2
mkdir -p "$(dirname "$run")"
mkdir "$run"
complete=0
trap 'rc=$?; if [[ $complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$rc" >"$run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$hw_root"
files=dc_handoff/filelists/date_m235_dynamic_bn_coefficient_engine_rtl.f
sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
tcl=dc_handoff/scripts/run_dc_m216_flattened_source_cap_logic_only.tcl
declare -A expected=(
 ["rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv"]="933b0cab1e63a06ae4ef3a13334806b3167b917c603e63b088c2d03ea9ac6fb0"
 ["$files"]="752b65fa62ac7f0b2a3443065677dbfc3ed64107cf331faf972f0c62d0d447d1"
 ["$sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
 ["$tcl"]="2565e750551f6f2a03abff96b462558416f3b9531a693f43b6194af8026d61d5"
 ["contracts/m235_dynamic_bn_segmented_lut_newton_coefficient_engine_contract_r1_20260825.json"]="fb661071ab72650f744c0501e712b31c6a0b29cb88c7bada817238e9ad103d35"
 ["results/m235_dynamic_bn_coefficient_engine_directed_vcs_r1_exact_20260825/RUN_COMPLETE.txt"]="f281d9ddb1a707b04625c593f619090fe06ac3efa5b23cb079d0cd184dc74ba6"
 ["results/m235_dynamic_bn_coefficient_engine_directed_vcs_r1_exact_20260825/SHA256SUMS"]="2815e91a1a30d0cbb94d52bdeb74560f46007dec475cb40b73b6847fa411e90a"
 ["results/m234_h67_dynamic_bn_lut_newton_coefficient_dse_r1_20260825/manifest.sha256"]="c7fc0ba9495a821e634644023cbbbd4477385f6b4126a61b668f0458b4bcec71"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"$run/preflight_sha_checks.txt"
for path in "${!expected[@]}"; do
 observed="$(sha256sum "$path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$path" "${expected[$path]}" "$observed" >>"$run/preflight_sha_checks.txt"
 [[ "$observed" == "${expected[$path]}" ]] || exit 10
done
sha256sum "${!expected[@]}" >"$run/input_sha256.txt"
export DESIGN_NAME=m235_dynamic_bn_segmented_lut_newton_coefficient_engine
export HW_ROOT="$hw_root" RTL_FILELIST="$hw_root/$files"
export LIB_DB="$lib" MIN_LIB_DB="$minlib" SDC_FILE="$hw_root/$sdc"
export OUTPUT_DIR="$run" OPERATING_CONDITION=ssg0p9v125c
set +e
"$dc" -f "$hw_root/$tcl" >"$run/dc.log" 2>&1
rc=$?
set -e
echo "$rc" >"$run/dc.rc"
[[ $rc -eq 0 ]] || exit 20
grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$run/dc.log" && exit 21 || true
grep -Fq 'Thank you...' "$run/dc.log" || exit 22
for report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt constraint_violators.rpt check_design_postcompile.rpt check_timing_postcompile.rpt resources_postcompile.rpt; do
 [[ -s "$run/reports/$report" ]] || exit 30
done
[[ -s "$run/netlist/${DESIGN_NAME}_mapped.v" && -s "$run/netlist/${DESIGN_NAME}_mapped.sdc" && -s "$run/netlist/${DESIGN_NAME}.ddc" ]] || exit 31
[[ "$(tr -d '[:space:]' <"$run/reports/check_design_postcompile.rpt")" == 1 ]] || exit 35
[[ "$(tail -n1 "$run/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" == 1 ]] || exit 36
grep -Fq 'Number of macros/black boxes:               0' "$run/reports/area.rpt" || exit 37
area="$(awk '/Total cell area:/{print $4;exit}' "$run/reports/area.rpt")"
cells="$(awk '/Number of cells:/{print $4;exit}' "$run/reports/area.rpt")"
sequential="$(awk '/Number of sequential cells:/{print $5;exit}' "$run/reports/area.rpt")"
levels="$(awk '/Levels of Logic:/{print $4;exit}' "$run/reports/qor.rpt")"
path_length="$(awk '/Critical Path Length:/{print $4;exit}' "$run/reports/qor.rpt")"
setup="$(awk '/slack \((MET|VIOLATED)\)/{print $3;exit}' "$run/reports/timing_setup.rpt")"
hold="$(awk '/slack \((MET|VIOLATED)\)/{print $3;exit}' "$run/reports/timing_hold.rpt")"
setup_met=false
hold_met=false
awk -v value="$setup" 'BEGIN{exit !(value>=0)}' && setup_met=true
awk -v value="$hold" 'BEGIN{exit !(value>=0)}' && hold_met=true
awk -v value="$area" 'BEGIN{exit !(value>0&&value<100000)}'
awk -v value="$sequential" 'BEGIN{exit !(value>0&&value<5000)}'
source_multiply_operators="$(rg -o 'multiplier_a \* multiplier_b' rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv | wc -l)"
[[ "$source_multiply_operators" -eq 1 ]] || exit 38
{
 echo status=COMPLETE_M235_DYNAMIC_BN_COEFFICIENT_ENGINE_LOGIC_ONLY_DC_SCREEN
 echo exact_sha=true
 echo tool=Synopsys_DC_V-2023.12-SP3
 echo technology=TSMC28_HPCplus
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
 echo source_shared_multiply_operators="$source_multiply_operators"
 echo mapped_resource_report_sha256="$(sha256sum "$run/reports/resources_postcompile.rpt" | awk '{print $1}')"
 echo macro_count=0
 echo ideal_clock=true
 echo wireload=ZeroWireload
 echo moment_finalizer=false
 echo event_equivalence=false
 echo system_speedup=false
 echo paper_ppa_ready=false
} >"$run/RUN_COMPLETE.txt"
sha256sum "$runner" >"$run/runner_sha256.txt"
sha256sum "$run"/dc.log "$run"/reports/*.rpt "$run"/netlist/* "$run"/RUN_COMPLETE.txt >"$run/evidence_manifest.sha256"
complete=1
echo "PASS M235 logic-only DC sealed at $run"
