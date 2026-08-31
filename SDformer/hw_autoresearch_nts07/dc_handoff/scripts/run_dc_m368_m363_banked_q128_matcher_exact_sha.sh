#!/usr/bin/env bash
set -euo pipefail

m368_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m368_hw="$(cd "${m368_dc_root}/.." && pwd)"
m368_run="${M368_DC_RUN:-${m368_dc_root}/runs/m368_m363_banked_q128_matcher_dc_3p000ns_r1_20260825}"
m368_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m368_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m368_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m368_filelist="dc_handoff/filelists/date_m363_banked_q128_exact_signed_residual_matcher_rtl.f"
m368_sdc="dc_handoff/constraints/date_m311_near_match16_tau01_3ns.sdc"
m368_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m368_contract="contracts/m368_m363_banked_q128_matcher_logic_only_dc_contract_r1_20260825.json"

m368_sha() { sha256sum "$1" | awk '{print $1}'; }
m368_expect() {
    local m368_path=$1
    local m368_expected=$2
    [[ -f "${m368_path}" ]] || exit 3
    [[ "$(m368_sha "${m368_path}")" == "${m368_expected}" ]] || exit 3
}

[[ ! -e "${m368_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m368_hw}"
m368_expect "${m368_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m368_expect "${m368_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m368_expect "${m368_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m368_expect "rtl_m363/m363_banked_q128_exact_signed_residual_matcher.sv" 257084916c312c9db4e2d6ad59a4fe20fb604fa6c9d0a0573039339e9614879d
m368_expect "${m368_filelist}" 1c3409aa73a632842be88a7a52dd29c2263c5eaaafc8744f17e44b5c16fa74ce
m368_expect "${m368_sdc}" 4940447559fd3229baa2b33fb151a198c282ef0ac9a1864413ba675a966aad86
m368_expect "${m368_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m368_expect "${m368_contract}" c5216b643dcf701a6af65b585d24b7d86c08a9ff964de0a24df6411489913d8c
m368_expect "results/m363_banked_q128_exact_signed_residual_matcher_vcs_r1_20260825/m363_banked_q128_exact_signed_residual_matcher_vcs_receipt_r1.json" 9ec43acadc73e02fa56f3a1efad846ee8866ded5204d6a8db5ae76903b13e3c4
m368_expect "results/m363_banked_q128_exact_signed_residual_matcher_vcs_r1_20260825/RUN_MANIFEST.seal.sha256" 560c6fb9dc4f2f02f6c543a52cd40c448c94aa8497f37b097bdfc91360dd304e
m368_expect "results/m364_m363_banked_q128_independent_hammer_vcs_r1_20260825/m364_m363_banked_q128_independent_hammer_review_r1.json" a1f3d2a389c6ff6d08a21c31dff979084f3c7838bfe7227adb5da8996b2cdd1c
m368_expect "results/m364_m363_banked_q128_independent_hammer_vcs_r1_20260825/SHA256SUMS.seal.sha256" e3cc9d3f8c40529425cd67eb9b270ffe53b9dce394dec40967517c4f724a5f7d
m368_expect "dc_handoff/runs/m329_m321_hold_guard_dc_3p000ns_r1b_20260825/RUN_COMPLETE.txt" b9bf33a9d11b8c8aa8e5a03f916d1fa89937f48a04e7cd6f16dd8d2266b6d923
m368_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m368_run}"
m368_complete=0
trap 'm368_rc=$?; if [[ ${m368_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m368_rc}" >"${m368_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
{
    sha256sum \
        rtl_m363/m363_banked_q128_exact_signed_residual_matcher.sv \
        "${m368_filelist}" "${m368_sdc}" "${m368_tcl}" \
        "${m368_contract}" "${m368_slow}" "${m368_fast}"
} >"${m368_run}/input_sha256.txt"
cp "${m368_contract}" "${m368_run}/contract.json"

export DESIGN_NAME=m363_banked_q128_exact_signed_residual_matcher
export HW_ROOT="${m368_hw}"
export RTL_FILELIST="${m368_hw}/${m368_filelist}"
export LIB_DB="${m368_slow}"
export MIN_LIB_DB="${m368_fast}"
export SDC_FILE="${m368_hw}/${m368_sdc}"
export OUTPUT_DIR="${m368_run}"
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m368_dc}" -f "${m368_hw}/${m368_tcl}" >"${m368_run}/dc.log" 2>&1
m368_rc=$?
set -e
echo "${m368_rc}" >"${m368_run}/dc.rc"
[[ "${m368_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m368_run}/dc.log"
grep -Fq 'Thank you...' "${m368_run}/dc.log"
for m368_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt hold_guard_contract.rpt; do
    [[ -s "${m368_run}/reports/${m368_report}" ]] || exit 30
done
[[ -s "${m368_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m368_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m368_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m368_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m368_run}/reports/timing_setup.rpt" \
    "${m368_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m368_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33
grep -qx 'additional_hold_guard_ns=0.025' \
    "${m368_run}/reports/hold_guard_contract.rpt"
grep -Eq 'set_clock_uncertainty( +-hold)? +0\.1(00)? ' \
    "${m368_run}/netlist/${DESIGN_NAME}_mapped.sdc"

m368_area=$(awk '/Total cell area:/ {print $4; exit}' "${m368_run}/reports/area.rpt")
m368_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m368_run}/reports/area.rpt")
m368_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m368_run}/reports/area.rpt")
m368_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m368_run}/reports/qor.rpt")
m368_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m368_run}/reports/timing_setup.rpt")
m368_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m368_run}/reports/timing_hold.rpt")
for m368_value in "${m368_area}" "${m368_cells}" "${m368_seq}" \
        "${m368_levels}" "${m368_setup}" "${m368_hold}"; do
    [[ -n "${m368_value}" ]] || exit 34
done
awk -v x="${m368_area}" 'BEGIN {exit !(x > 0)}'
awk -v x="${m368_setup}" 'BEGIN {exit !(x >= 0.0)}'
awk -v x="${m368_hold}" 'BEGIN {exit !(x >= 0.0)}'

python3 - "${m368_run}" "${m368_area}" "${m368_cells}" \
    "${m368_seq}" "${m368_levels}" "${m368_setup}" "${m368_hold}" <<'PY'
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
    "schema": "m368_m363_banked_q128_matcher_logic_only_dc_receipt_v1",
    "status": "PASS_M368_M363_BANKED_Q128_MATCHER_LOGIC_ONLY_DC_3NS",
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
        "m363_no_stall_latency_cycles": 4,
        "m356_no_stall_latency_cycles": 128,
        "latency_reduction_claimed_as_speedup": False,
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
(run / "m368_m363_q128_matcher_logic_only_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
    echo "status=PASS_M368_M363_BANKED_Q128_MATCHER_LOGIC_ONLY_DC_3NS"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "cell_area_um2=${m368_area}"
    echo "cell_count=${m368_cells}"
    echo "sequential_cells=${m368_seq}"
    echo "logic_levels=${m368_levels}"
    echo "setup_worst_slack_ns=${m368_setup}"
    echo "hold_worst_slack_ns=${m368_hold}"
    echo "synthesis_hold_guard_ns=0.025"
    echo "publication_hold_uncertainty_ns=0.100"
    echo "macro_count=0"
    echo "complete_pwp_conv=false"
    echo "physical_timing=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} >"${m368_run}/RUN_COMPLETE.txt"
sha256sum "${m368_run}"/dc.log "${m368_run}"/reports/*.rpt \
    "${m368_run}"/netlist/* \
    "${m368_run}"/m368_m363_q128_matcher_logic_only_dc_receipt_r1.json \
    "${m368_run}"/RUN_COMPLETE.txt >"${m368_run}/evidence_manifest.sha256"
sha256sum "${m368_run}/evidence_manifest.sha256" \
    >"${m368_run}/evidence_manifest.seal.sha256"
m368_complete=1
echo "PASS_M368_M363_BANKED_Q128_MATCHER_LOGIC_ONLY_DC run=${m368_run}"
