#!/usr/bin/env bash
set -euo pipefail

m362_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m362_hw="$(cd "${m362_dc_root}/.." && pwd)"
m362_run="${M362_DC_RUN:-${m362_dc_root}/runs/m362_m356_failclosed_q128_matcher_dc_3p000ns_r1_20260825}"
m362_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m362_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m362_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m362_filelist="dc_handoff/filelists/date_m356_failclosed_q128_signed_residual_matcher_rtl.f"
m362_sdc="dc_handoff/constraints/date_m311_near_match16_tau01_3ns.sdc"
m362_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m362_contract="contracts/m362_m356_failclosed_q128_matcher_logic_only_dc_contract_r1_20260825.json"

m362_sha() { sha256sum "$1" | awk '{print $1}'; }
m362_expect() {
    local m362_path=$1
    local m362_expected=$2
    [[ -f "${m362_path}" ]] || exit 3
    [[ "$(m362_sha "${m362_path}")" == "${m362_expected}" ]] || exit 3
}

[[ ! -e "${m362_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m362_hw}"
m362_expect "${m362_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m362_expect "${m362_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m362_expect "${m362_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m362_expect "rtl_m348/m348_exact_q128_signed_residual_matcher.sv" 960b72268d526baad7e7d74cd159f2a9fbb286abac025678f239af3f5147eb1f
m362_expect "rtl_m356/m356_failclosed_q128_signed_residual_matcher.sv" 3a5dd5f7e3602f4f27f0744ebb13807877b4cc8a45e44dda11afe77dcae67fb8
m362_expect "${m362_filelist}" 7e6f09da3c6e642e77ec755067cc6a19b45db1b77e0142e25e282f4675184001
m362_expect "${m362_sdc}" 4940447559fd3229baa2b33fb151a198c282ef0ac9a1864413ba675a966aad86
m362_expect "${m362_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m362_expect "${m362_contract}" 7048bd924ff6a68eb6e041afe46ce25c49289fb6878a835e55c4c56f035ec0e0
m362_expect "results/m356_failclosed_q128_signed_residual_matcher_vcs_r1_20260825/m356_failclosed_q128_signed_residual_matcher_vcs_receipt_r1.json" 299da387473af4f70d5823baa7774864d3f4012da96a0452df921b19bfa6381b
m362_expect "results/m357_m356_failclosed_matcher_independent_hammer_r1_20260825/m357_m356_independent_hammer_review_r1.json" 20330f43387240806a4995a0a8832b375fed81cec31e1e4429aea7a88bab48ab
m362_expect "results/m357_m356_failclosed_matcher_independent_hammer_r1_20260825/SHA256SUMS.seal.sha256" 42984ae26412c139c0ccc279364303ef1c71108f6ef8af638af8ade20d46a4ac
m362_expect "dc_handoff/runs/m329_m321_hold_guard_dc_3p000ns_r1b_20260825/RUN_COMPLETE.txt" b9bf33a9d11b8c8aa8e5a03f916d1fa89937f48a04e7cd6f16dd8d2266b6d923
m362_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m362_run}"
m362_complete=0
trap 'm362_rc=$?; if [[ ${m362_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m362_rc}" >"${m362_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
{
    sha256sum \
        rtl_m348/m348_exact_q128_signed_residual_matcher.sv \
        rtl_m356/m356_failclosed_q128_signed_residual_matcher.sv \
        "${m362_filelist}" "${m362_sdc}" "${m362_tcl}" \
        "${m362_contract}" "${m362_slow}" "${m362_fast}"
} >"${m362_run}/input_sha256.txt"
cp "${m362_contract}" "${m362_run}/contract.json"

export DESIGN_NAME=m356_failclosed_q128_signed_residual_matcher
export HW_ROOT="${m362_hw}"
export RTL_FILELIST="${m362_hw}/${m362_filelist}"
export LIB_DB="${m362_slow}"
export MIN_LIB_DB="${m362_fast}"
export SDC_FILE="${m362_hw}/${m362_sdc}"
export OUTPUT_DIR="${m362_run}"
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m362_dc}" -f "${m362_hw}/${m362_tcl}" >"${m362_run}/dc.log" 2>&1
m362_rc=$?
set -e
echo "${m362_rc}" >"${m362_run}/dc.rc"
[[ "${m362_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m362_run}/dc.log"
grep -Fq 'Thank you...' "${m362_run}/dc.log"
for m362_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt hold_guard_contract.rpt; do
    [[ -s "${m362_run}/reports/${m362_report}" ]] || exit 30
done
[[ -s "${m362_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m362_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m362_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m362_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m362_run}/reports/timing_setup.rpt" \
    "${m362_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m362_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33
grep -qx 'additional_hold_guard_ns=0.025' \
    "${m362_run}/reports/hold_guard_contract.rpt"
grep -Eq 'set_clock_uncertainty( +-hold)? +0\.1(00)? ' \
    "${m362_run}/netlist/${DESIGN_NAME}_mapped.sdc"

m362_area=$(awk '/Total cell area:/ {print $4; exit}' "${m362_run}/reports/area.rpt")
m362_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m362_run}/reports/area.rpt")
m362_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m362_run}/reports/area.rpt")
m362_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m362_run}/reports/qor.rpt")
m362_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m362_run}/reports/timing_setup.rpt")
m362_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m362_run}/reports/timing_hold.rpt")
for m362_value in "${m362_area}" "${m362_cells}" "${m362_seq}" \
        "${m362_levels}" "${m362_setup}" "${m362_hold}"; do
    [[ -n "${m362_value}" ]] || exit 34
done
awk -v x="${m362_area}" 'BEGIN {exit !(x > 0)}'
awk -v x="${m362_setup}" 'BEGIN {exit !(x >= 0.0)}'
awk -v x="${m362_hold}" 'BEGIN {exit !(x >= 0.0)}'

python3 - "${m362_run}" "${m362_area}" "${m362_cells}" \
    "${m362_seq}" "${m362_levels}" "${m362_setup}" "${m362_hold}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
area = float(sys.argv[2])
cells = int(sys.argv[3])
seq = int(sys.argv[4])
levels = float(sys.argv[5])
setup = float(sys.argv[6])
hold = float(sys.argv[7])
q16_area = 1997.981971
receipt = {
    "schema": "m362_m356_failclosed_q128_matcher_logic_only_dc_receipt_v1",
    "status": "PASS_M362_M356_Q128_MATCHER_LOGIC_ONLY_DC_3NS",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "cell_area_um2": area,
    "cell_count": cells,
    "sequential_cells": seq,
    "logic_levels": levels,
    "setup_worst_slack_ns": setup,
    "hold_worst_slack_ns": hold,
    "macro_count": 0,
    "comparison": {
        "m329_q16_cell_area_um2": q16_area,
        "q128_over_q16_area_ratio": area / q16_area,
        "same_clock_match_issue_density_q128_over_q16": q16_area / area,
        "both_directed_initiation_interval_cycles": 1,
        "complete_executor_throughput_claimed": False,
    },
    "claim_boundary": {
        "isolated_matcher_logic_only_dc": True,
        "physical_timing": False,
        "complete_pwp_conv": False,
        "finite_queue_cycle_match": False,
        "formality": False,
        "primetime": False,
        "saif_power": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(run / "m362_m356_q128_matcher_logic_only_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
    echo "status=PASS_M362_M356_Q128_MATCHER_LOGIC_ONLY_DC_3NS"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "cell_area_um2=${m362_area}"
    echo "cell_count=${m362_cells}"
    echo "sequential_cells=${m362_seq}"
    echo "logic_levels=${m362_levels}"
    echo "setup_worst_slack_ns=${m362_setup}"
    echo "hold_worst_slack_ns=${m362_hold}"
    echo "synthesis_hold_guard_ns=0.025"
    echo "publication_hold_uncertainty_ns=0.100"
    echo "macro_count=0"
    echo "complete_pwp_conv=false"
    echo "physical_timing=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} >"${m362_run}/RUN_COMPLETE.txt"
sha256sum "${m362_run}"/dc.log "${m362_run}"/reports/*.rpt \
    "${m362_run}"/netlist/* \
    "${m362_run}"/m362_m356_q128_matcher_logic_only_dc_receipt_r1.json \
    "${m362_run}"/RUN_COMPLETE.txt >"${m362_run}/evidence_manifest.sha256"
sha256sum "${m362_run}/evidence_manifest.sha256" \
    >"${m362_run}/evidence_manifest.seal.sha256"
m362_complete=1
echo "PASS_M362_M356_Q128_MATCHER_LOGIC_ONLY_DC run=${m362_run}"
