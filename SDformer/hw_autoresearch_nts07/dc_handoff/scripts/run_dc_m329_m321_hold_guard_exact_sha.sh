#!/usr/bin/env bash
set -euo pipefail

m329_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m329_hw="$(cd "${m329_dc_root}/.." && pwd)"
m329_run="${M329_DC_RUN:-${m329_dc_root}/runs/m329_m321_hold_guard_dc_3p000ns_r1b_20260825}"
m329_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m329_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m329_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m329_rtl="rtl_m321/m321_near_match16_tau01_tournament2.sv"
m329_filelist="dc_handoff/filelists/date_m321_near_match16_tau01_tournament2_rtl.f"
m329_sdc="dc_handoff/constraints/date_m311_near_match16_tau01_3ns.sdc"
m329_tcl="dc_handoff/scripts/run_dc_m329_m321_hold_guard_exact_sha.tcl"
m329_contract="contracts/m329_m321_hold_guard_dc_contract_r1_20260825.json"

m329_sha() { sha256sum "$1" | awk '{print $1}'; }
m329_expect() {
    local m329_path=$1
    local m329_expected=$2
    [[ -f "${m329_path}" ]] || exit 3
    [[ "$(m329_sha "${m329_path}")" == "${m329_expected}" ]] || exit 3
}

[[ ! -e "${m329_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m329_hw}"
m329_expect "${m329_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m329_expect "${m329_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m329_expect "${m329_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m329_expect "${m329_rtl}" e54fd31a0beca33c18938e8241e55a07939dc1ff924dce8b81ccbb3a57e242e2
m329_expect "${m329_filelist}" ea57b3fb731ccb1fa7452060a3a39647e97e4dcddcabd075c899b420a99d12e4
m329_expect "${m329_sdc}" 4940447559fd3229baa2b33fb151a198c282ef0ac9a1864413ba675a966aad86
m329_expect "${m329_tcl}" 764323f93a54a63d227b84696227016261c20096debb2a11fa6e323abad8a8bc
m329_expect "${m329_contract}" 320f077b101f9e103b94b765d04de07ce95b00e500532572a86650b64097ef90
m329_expect "dc_handoff/runs/m326_m321_prelayout_ptsta_r1_20260825/m326_prelayout_ptsta_receipt_r1.json" 5c64e2ba21591c5bdb01e3b1a3b54963491bb60d36bdfb1cbdb35fc3e1fec4cd
m329_expect "dc_handoff/runs/m326_m321_prelayout_ptsta_r1_20260825/output.sha256" 86460c93ef15655af0c5fe85ba0560277928d6a7bd5eabb8fdc656ebe52acc06
m329_expect "results/m323_m321_m322_independent_hammer_r1_20260825/m323_independent_hammer_review_r1.json" a831fdab1e16cc0c0e09b34706dd9c366921b2cdfc15ffc4ea8f9a0d4e598f81
m329_expect "results/m323_m321_m322_independent_hammer_r1_20260825/evidence_manifest.seal.sha256" 1cf45f3aab71834117218cf283f52bfa9a6a5be56dd54a3c2fd461a73a922915
m329_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m329_run}"
m329_complete=0
trap 'm329_rc=$?; if [[ ${m329_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m329_rc}" >"${m329_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
{
    sha256sum "${m329_rtl}" "${m329_filelist}" "${m329_sdc}" \
        "${m329_tcl}" "${m329_contract}" "${m329_slow}" "${m329_fast}"
} > "${m329_run}/input_sha256.txt"
cp "${m329_contract}" "${m329_run}/contract.json"
export DESIGN_NAME=m321_near_match16_tau01_tournament2
export HW_ROOT="${m329_hw}"
export RTL_FILELIST="${m329_hw}/${m329_filelist}"
export LIB_DB="${m329_slow}"
export MIN_LIB_DB="${m329_fast}"
export SDC_FILE="${m329_hw}/${m329_sdc}"
export OUTPUT_DIR="${m329_run}"
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m329_dc}" -f "${m329_hw}/${m329_tcl}" > "${m329_run}/dc.log" 2>&1
m329_rc=$?
set -e
echo "${m329_rc}" > "${m329_run}/dc.rc"
[[ "${m329_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m329_run}/dc.log"
grep -Fq 'Thank you...' "${m329_run}/dc.log"
for m329_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt hold_guard_contract.rpt; do
    [[ -s "${m329_run}/reports/${m329_report}" ]] || exit 30
done
[[ -s "${m329_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m329_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m329_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m329_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m329_run}/reports/timing_setup.rpt" \
    "${m329_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m329_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33
grep -qx 'additional_hold_guard_ns=0.025' \
    "${m329_run}/reports/hold_guard_contract.rpt"
grep -Eq 'set_clock_uncertainty( +-hold)? +0\.1(00)? ' \
    "${m329_run}/netlist/${DESIGN_NAME}_mapped.sdc"

m329_area=$(awk '/Total cell area:/ {print $4; exit}' "${m329_run}/reports/area.rpt")
m329_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m329_run}/reports/area.rpt")
m329_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m329_run}/reports/area.rpt")
m329_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m329_run}/reports/qor.rpt")
m329_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m329_run}/reports/timing_setup.rpt")
m329_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m329_run}/reports/timing_hold.rpt")
for m329_value in "${m329_area}" "${m329_cells}" "${m329_seq}" \
        "${m329_levels}" "${m329_setup}" "${m329_hold}"; do
    [[ -n "${m329_value}" ]] || exit 34
done
awk -v x="${m329_setup}" 'BEGIN {exit !(x >= 0.5000)}'
awk -v x="${m329_hold}" 'BEGIN {exit !(x >= 0.0150)}'
awk -v x="${m329_levels}" 'BEGIN {exit !(x <= 60)}'

{
    echo "status=PASS_M329_M321_25PS_HOLD_GUARD_DC"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "cell_area_um2=${m329_area}"
    echo "cell_count=${m329_cells}"
    echo "sequential_cells=${m329_seq}"
    echo "logic_levels=${m329_levels}"
    echo "setup_worst_slack_ns=${m329_setup}"
    echo "hold_worst_slack_ns=${m329_hold}"
    echo "synthesis_hold_guard_ns=0.025"
    echo "publication_hold_uncertainty_ns=0.100"
    echo "fresh_formality_required=true"
    echo "fresh_primetime_required=true"
    echo "macro_count=0"
    echo "physical_timing=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "${m329_run}/RUN_COMPLETE.txt"
sha256sum "${m329_run}"/dc.log "${m329_run}"/reports/*.rpt \
    "${m329_run}"/netlist/* "${m329_run}"/RUN_COMPLETE.txt \
    > "${m329_run}/evidence_manifest.sha256"
sha256sum "${m329_run}/evidence_manifest.sha256" \
    > "${m329_run}/evidence_manifest.seal.sha256"
m329_complete=1
echo "PASS_M329_M321_25PS_HOLD_GUARD_DC run=${m329_run}"
