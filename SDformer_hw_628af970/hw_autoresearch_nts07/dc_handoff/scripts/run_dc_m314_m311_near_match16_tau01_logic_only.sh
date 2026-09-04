#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_run="${task_dc_root}/runs/m314_m311_near_match16_tau01_logic_only_dc_3p000ns_r1_20260825"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
task_files="dc_handoff/filelists/date_m311_near_match16_tau01_rtl.f"
task_sdc="dc_handoff/constraints/date_m311_near_match16_tau01_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m289_m273r2_flattened_logic_only.tcl"
task_contract="contracts/m314_m311_near_match16_tau01_logic_only_dc_contract_r1_20260825.json"

[[ ! -e "${task_run}" ]] || exit 2
[[ -x "${task_dc_shell}" && -s "${task_lib}" && -s "${task_min_lib}" ]] || exit 3
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
mkdir "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m311/m311_near_match16_tau01.sv"]="c5e7a6dd7d522c1f7cd98b6e5fafda4deedefb70e36dd1ca1a72b41b2cd53170"
    ["${task_files}"]="e406ba2313f953563c6cf0d108345f1c0812149e4551090af98de8aa7d77d93d"
    ["${task_sdc}"]="4940447559fd3229baa2b33fb151a198c282ef0ac9a1864413ba675a966aad86"
    ["${task_tcl}"]="9ef3912ff13b17afad739c4af1eaf74087a31919735772ef8c98d4bf569071dc"
    ["${task_contract}"]="35bdcf1b0f7fcc4b811732630f462a53b4c567dc01a2210783c59eda735a0505"
    ["results/m311r4_near_match16_tau01_vcs_r1_20260825/m311_near_match16_tau01_vcs_receipt_r1.json"]="66b2935f412e32d9da7fe078ba2d48ed348bc213ca99f7759eaf27a82e4f714f"
    ["results/m311r4_near_match16_tau01_vcs_r1_20260825/RUN_MANIFEST.seal.sha256"]="8014f6f91749cb5a936cd1372f3f6c0b96b170d0a79cd56cb8cb71fd00a2660f"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "${task_path}" "${task_expected[${task_path}]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" "${task_lib}" "${task_min_lib}" \
    >"${task_run}/input_sha256.txt"
cp "${task_contract}" "${task_run}/contract.json"

export DESIGN_NAME="m311_near_match16_tau01"
export HW_ROOT="${task_hw_root}"
export RTL_FILELIST="${task_hw_root}/${task_files}"
export LIB_DB="${task_lib}"
export MIN_LIB_DB="${task_min_lib}"
export SDC_FILE="${task_hw_root}/${task_sdc}"
export OUTPUT_DIR="${task_run}"
export OPERATING_CONDITION="ssg0p9v125c"
set +e
"${task_dc_shell}" -f "${task_hw_root}/${task_tcl}" \
    >"${task_run}/dc.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/dc.rc"
[[ ${task_rc} -eq 0 ]] || exit 20
grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' \
    "${task_run}/dc.log" && exit 21 || true
grep -Fq 'Thank you...' "${task_run}/dc.log" || exit 22
for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt; do
    [[ -s "${task_run}/reports/${task_report}" ]] || exit 30
done
[[ -s "${task_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${task_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${task_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${task_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${task_run}/reports/timing_setup.rpt" \
    "${task_run}/reports/timing_hold.rpt" && exit 32 || true
grep -Fq 'slack (MET)' "${task_run}/reports/timing_setup.rpt" || exit 33
grep -Fq 'slack (MET)' "${task_run}/reports/timing_hold.rpt" || exit 33
[[ "$(grep -Fc 'This design has no violated constraints.' "${task_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 34
[[ "$(tr -d '[:space:]' <"${task_run}/reports/check_design_postcompile.rpt")" == "1" ]] || exit 35
[[ "$(tail -n 1 "${task_run}/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" == "1" ]] || exit 36
grep -Fq 'Number of macros/black boxes:               0' \
    "${task_run}/reports/area.rpt" || exit 37

task_area="$(awk '/Total cell area:/ {print $4; exit}' "${task_run}/reports/area.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "${task_run}/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "${task_run}/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "${task_run}/reports/qor.rpt")"
task_path="$(awk '/Critical Path Length:/ {print $4; exit}' "${task_run}/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "${task_run}/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "${task_run}/reports/timing_hold.rpt")"
for task_value in "${task_area}" "${task_cells}" "${task_seq}" \
        "${task_levels}" "${task_path}" "${task_setup}" "${task_hold}"; do
    [[ -n "${task_value}" ]] || exit 38
done
awk -v x="${task_area}" 'BEGIN {exit !(x > 0 && x < 100000)}' || exit 39
awk -v x="${task_setup}" 'BEGIN {exit !(x >= 0.0)}' || exit 40
awk -v x="${task_hold}" 'BEGIN {exit !(x >= 0.0)}' || exit 41

{
    echo "status=PASS_M314_M311_NEAR_MATCH16_TAU01_LOGIC_ONLY_DC_3NS"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "cell_area_um2=${task_area}"
    echo "cell_count=${task_cells}"
    echo "sequential_cells=${task_seq}"
    echo "logic_levels=${task_levels}"
    echo "critical_path_length_ns=${task_path}"
    echo "setup_worst_slack_ns=${task_setup}"
    echo "hold_worst_slack_ns=${task_hold}"
    echo "macro_count=0"
    echo "vcs_bound=true"
    echo "physical_metadata_sram=false"
    echo "saif_power=false"
    echo "complete_pwp_conv_rtl=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} >"${task_run}/RUN_COMPLETE.txt"
sha256sum "${task_run}"/dc.log "${task_run}"/reports/*.rpt \
    "${task_run}"/netlist/* "${task_run}"/RUN_COMPLETE.txt \
    >"${task_run}/evidence_manifest.sha256"
sha256sum "${task_run}/evidence_manifest.sha256" \
    >"${task_run}/evidence_manifest.seal.sha256"
task_complete=1
echo "PASS M314 M311 matcher logic-only DC sealed at ${task_run}"
