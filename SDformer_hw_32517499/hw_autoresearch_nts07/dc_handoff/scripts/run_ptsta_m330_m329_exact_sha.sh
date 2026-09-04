#!/usr/bin/env bash
set -euo pipefail

m330_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m330_hw="$(cd "${m330_dc_root}/.." && pwd)"
m330_run="${M330_PTSTA_RUN:-${m330_dc_root}/runs/m330_m329_m321_prelayout_ptsta_r1_20260825}"
m330_source="${m330_dc_root}/runs/m329_m321_hold_guard_dc_3p000ns_r1b_20260825"
m330_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m330_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m330_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m330_netlist="${m330_source}/netlist/m321_near_match16_tau01_tournament2_mapped.v"
m330_sdc="${m330_source}/netlist/m321_near_match16_tau01_tournament2_mapped.sdc"
m330_tcl="${m330_dc_root}/scripts/run_ptsta_m326_m321_exact_sha.tcl"
m330_contract="${m330_hw}/contracts/m330_m329_m321_prelayout_ptsta_contract_r1_20260825.json"

m330_sha() { sha256sum "$1" | awk '{print $1}'; }
m330_expect() {
    [[ -f "$1" ]] || exit 3
    [[ "$(m330_sha "$1")" == "$2" ]] || exit 3
}
[[ ! -e "${m330_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 4; fi
m330_expect "${m330_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m330_expect "${m330_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m330_expect "${m330_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m330_expect "${m330_netlist}" fe7e6db0c81107dff81425198d85935d79c9f8473c21671a8d697ade6b2c5b1e
m330_expect "${m330_sdc}" 883909684da4293ded56a524b29aa8ac57c07ae5764a93ec459628696102d534
m330_expect "${m330_tcl}" 457a5e8ae4c134533302e44e2f2e3909923d7f7efed4aac962cd0a5d9f9a401d
m330_expect "${m330_contract}" 570e991cda3afa1895f70eca7923d18e0f018d97edde54ee5d16da6c30393bb5
m330_expect "${m330_source}/RUN_COMPLETE.txt" b9bf33a9d11b8c8aa8e5a03f916d1fa89937f48a04e7cd6f16dd8d2266b6d923
m330_expect "${m330_source}/evidence_manifest.seal.sha256" b011cbf6926b8e1024efde579aa29156e64dae75818d7711efff3e9b7be8b685
m330_expect "${m330_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m330_run}/reports" "${m330_run}/work"
cp "${m330_contract}" "${m330_run}/contract.json"
sha256sum "${m330_netlist}" "${m330_sdc}" "${m330_slow}" "${m330_fast}" \
    "${m330_tcl}" "${m330_contract}" > "${m330_run}/input_sha256.txt"
export DESIGN_NAME=m321_near_match16_tau01_tournament2
export LIB_SLOW="${m330_slow}"
export LIB_FAST="${m330_fast}"
export MAPPED_NETLIST="${m330_netlist}"
export MAPPED_SDC="${m330_sdc}"
export OUTPUT_DIR="${m330_run}"
"${m330_pt}" -version > "${m330_run}/pt.version.raw.log" 2>&1
set +e
(cd "${m330_run}/work" && "${m330_pt}" -f "${m330_tcl}") \
    > "${m330_run}/pt.raw.log" 2>&1
m330_rc=$?
set -e
echo "${m330_rc}" > "${m330_run}/pt.rc"
[[ "${m330_rc}" -eq 0 ]]
! grep -Eq '^(Error|Fatal):' "${m330_run}/pt.raw.log"
[[ "$(grep -xc 'M326_M321_PTSTA_INTERNAL_COMPLETE=PASS' \
    "${m330_run}/PTSTA_INTERNAL_COMPLETE.txt")" -eq 1 ]]

python3 - "${m330_run}" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
def slack(name):
    text = (root / "reports" / name).read_text()
    match = re.search(r"slack \((MET|VIOLATED)\)\s+(-?\d+\.\d+)", text)
    if not match: raise RuntimeError("missing slack " + name)
    return match.group(1), float(match.group(2))
setup_state, setup = slack("ptsta_timing_setup.rpt")
hold_state, hold = slack("ptsta_timing_hold.rpt")
coverage = (root / "reports" / "ptsta_analysis_coverage.rpt").read_text()
match = re.search(r"^hold\s+156\s+(\d+) \([^\n]+?\)\s+(\d+) \(", coverage, re.M)
if not match: raise RuntimeError("missing hold coverage")
hold_met, hold_violated = map(int, match.groups())
constraint = (root / "reports" / "ptsta_constraint_violators.rpt").read_text()
violated_paths = constraint.count("slack (VIOLATED)")
check = (root / "reports" / "ptsta_check_timing.rpt").read_text()
reset_only = check.count("paths from such ports will be unconstrained") == 1 and "reset_n" in check
gate = (setup >= 0.5 and hold >= 0.0 and hold_met == 156 and
        hold_violated == 0 and violated_paths == 0 and reset_only)
receipt = {
  "schema": "m330_m329_m321_prelayout_ptsta_receipt_v1",
  "status": "PASS_M330_M329_HOLD_REPAIR_PTSTA" if gate else "NO_GO_M330_M329_HOLD_REPAIR_PTSTA",
  "setup_worst_slack_ns": setup, "setup_state": setup_state,
  "hold_worst_slack_ns": hold, "hold_state": hold_state,
  "hold_met_endpoints": hold_met, "hold_violated_endpoints": hold_violated,
  "constraint_report_violated_paths": violated_paths,
  "asynchronous_reset_only_no_input_delay_exception": reset_only,
  "timing_gate_pass": gate,
  "comparison": {"m326_hold_worst_slack_ns": -0.0071,
                 "m326_hold_violated_endpoints": 82},
  "claim_boundary": {"prelayout_no_spef": True, "ideal_clock": True,
                     "post_route_timing": False, "reset_timing_signoff": False,
                     "system_speedup": False, "paper_ppa_ready": False,
                     "headline": False},
}
(root / "m330_prelayout_ptsta_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
if not gate: raise SystemExit(42)
PY

{
    echo "status=PASS_M330_M329_HOLD_REPAIR_PTSTA"
    echo "scope=PRELAYOUT_NO_SPEF_IDEAL_CLOCK_ZERO_MACRO"
    echo "reset_timing_signoff=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
} > "${m330_run}/RUN_COMPLETE.txt"
(
  cd "${m330_run}"
  find . -type f ! -path './work/*' ! -name output.sha256 \
      ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
      > output.sha256
  sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "PASS_M330_M329_HOLD_REPAIR_PTSTA run=${m330_run}"
