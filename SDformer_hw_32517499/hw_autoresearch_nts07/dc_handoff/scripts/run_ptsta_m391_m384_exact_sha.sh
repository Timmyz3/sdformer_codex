#!/usr/bin/env bash
set -euo pipefail

m391_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m391_hw="$(cd "${m391_dc_root}/.." && pwd)"
m391_runner="$(realpath "${BASH_SOURCE[0]}")"
m391_source="${m391_dc_root}/runs/m387_m384_active_descriptor_controller_dc_3p000ns_r1b_20260826"
m391_formality="${m391_dc_root}/runs/m389_m384_to_m387r1b_formality_r1_20260826"
m391_run="${M391_PTSTA_RUN:-${m391_dc_root}/runs/m391_m387r1b_m384_prelayout_ptsta_r1b_20260826}"
m391_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m391_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m391_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m391_netlist="${m391_source}/netlist/m384_active_descriptor_streaming_controller_mapped.v"
m391_sdc="${m391_source}/netlist/m384_active_descriptor_streaming_controller_mapped.sdc"
m391_tcl="${m391_dc_root}/scripts/run_ptsta_m391_m384_exact_sha.tcl"
m391_contract="${m391_hw}/contracts/m391_m387r1b_m384_prelayout_ptsta_contract_r1_20260826.json"

m391_sha() { sha256sum "$1" | awk '{print $1}'; }
m391_expect() { [[ -f "$1" && "$(m391_sha "$1")" == "$2" ]] || exit 3; }
[[ ! -e "${m391_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 4; fi
m391_expect "${m391_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m391_expect "${m391_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m391_expect "${m391_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m391_expect "${m391_netlist}" fe478950d336eb28176125572983a02fb4f47fd97fcfeb28dedfb6063df03efd
m391_expect "${m391_sdc}" 6cd373d3ef195ac8744e3466b69c4fe6e831932d040636206c892038bb598f9b
m391_expect "${m391_tcl}" 8e3f402f86139bc7ae3334f622094c1385b1ff2ff72eb20bc94aacbcca1bad9a
m391_expect "${m391_contract}" 74d1577722ae8b42110c4e25ccd1f32d7f14b85269e6863b913da208fd623df5
m391_expect "${m391_source}/m387_m384_active_descriptor_controller_logic_only_dc_receipt_r1b.json" 896eba1d373fa8d8bb371a19e097047dabbfe391b39838887d8dd9b785b77b2b
m391_expect "${m391_source}/evidence_manifest.seal.sha256" c6e86050acb21576a5cd5073573a4941085b91295d57282aa126af1b46d0ce5f
m391_expect "${m391_formality}/m389_m384_to_m387r1b_formality_receipt_r1.json" 25cb25d9cd1266097354a7e7726a4cae6edcd957305215ae6831fd0649fe0980
m391_expect "${m391_formality}/output.seal.sha256" dffe307c98a1179cb01bc11bd51476f7d3d711a6c5a1f69f8621e5bd96cb2b18
m391_expect "${m391_hw}/results/m390_m389_m384_formality_independent_hammer_r1_20260826/m390_m389_m384_formality_independent_hammer_review_r1.json" c18215334b68dcab4b1e01c8719a37ef8bd10d2df65a09a2db23e5b529b9fcdf
m391_expect "${m391_hw}/results/m390_m389_m384_formality_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256" 0a947b79707b48b8a6c2e41dce833220ad09e2e9cca63f32a081ee5e60a61a59
m391_expect "${m391_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m391_run}/reports" "${m391_run}/work"
m391_complete=0
trap 'm391_rc=$?; if [[ ${m391_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m391_rc}" >"${m391_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cp "${m391_contract}" "${m391_run}/contract.json"
sha256sum "${m391_netlist}" "${m391_sdc}" "${m391_slow}" \
    "${m391_fast}" "${m391_tcl}" "${m391_contract}" \
    "${m391_source}/evidence_manifest.seal.sha256" \
    "${m391_formality}/output.seal.sha256" \
    "${m391_hw}/results/m390_m389_m384_formality_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256" \
    >"${m391_run}/input_sha256.txt"

export DESIGN_NAME=m384_active_descriptor_streaming_controller
export LIB_SLOW="${m391_slow}"
export LIB_FAST="${m391_fast}"
export MAPPED_NETLIST="${m391_netlist}"
export MAPPED_SDC="${m391_sdc}"
export OUTPUT_DIR="${m391_run}"
"${m391_pt}" -version >"${m391_run}/pt.version.raw.log" 2>&1
set +e
(cd "${m391_run}/work" && "${m391_pt}" -f "${m391_tcl}") \
    >"${m391_run}/pt.raw.log" 2>&1
m391_rc=$?
set -e
echo "${m391_rc}" >"${m391_run}/pt.rc"
[[ "${m391_rc}" -eq 0 ]]
! grep -Eq '^(Error|Fatal):' "${m391_run}/pt.raw.log"
[[ "$(grep -xc 'M391_M384_PTSTA_INTERNAL_COMPLETE=PASS' \
    "${m391_run}/PTSTA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
for m391_report in ptsta_scope.rpt ptsta_check_timing.rpt \
        ptsta_analysis_coverage.rpt ptsta_global_timing.rpt \
        ptsta_timing_setup.rpt ptsta_timing_hold.rpt \
        ptsta_constraint_violators.rpt ptsta_clock.rpt \
        ptsta_exceptions.rpt; do
    [[ -s "${m391_run}/reports/${m391_report}" ]] || exit 30
done

python3 - "${m391_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def slack(name):
    text = (root / "reports" / name).read_text(
        encoding="utf-8", errors="replace")
    match = re.search(r"slack \((MET|VIOLATED)\)\s+(-?\d+\.\d+)", text)
    if not match:
        raise RuntimeError("missing slack in " + name)
    return match.group(1), float(match.group(2))

def coverage_row(name):
    text = (root / "reports/ptsta_analysis_coverage.rpt").read_text(
        encoding="utf-8", errors="replace")
    match = re.search(
        rf"^{name}\s+(\d+)\s+(\d+) \([^\n]+?\)\s+(\d+) \([^\n]+?\)\s+(\d+) \(",
        text, re.M)
    if not match:
        raise RuntimeError("missing coverage row " + name)
    return tuple(map(int, match.groups()))

setup_state, setup = slack("ptsta_timing_setup.rpt")
hold_state, hold = slack("ptsta_timing_hold.rpt")
coverage = {name: coverage_row(name) for name in
            ("setup", "hold", "out_setup", "out_hold")}
constraint = (root / "reports/ptsta_constraint_violators.rpt").read_text(
    encoding="utf-8", errors="replace")
violated_paths = constraint.count("slack (VIOLATED)")
check = (root / "reports/ptsta_check_timing.rpt").read_text(
    encoding="utf-8", errors="replace")
reset_only = (check.count("paths from such ports will be unconstrained") == 1
              and "reset_n" in check)
coverage_text = (root / "reports/ptsta_analysis_coverage.rpt").read_text(
    encoding="utf-8", errors="replace")
reason_counts = {
    "constant_disabled": len(re.findall(r"untested\s+constant_disabled", coverage_text)),
    "no_paths": len(re.findall(r"untested\s+no_paths", coverage_text)),
    "no_startpoint_clock": len(re.findall(r"untested\s+no_startpoint_clock", coverage_text)),
    "no_clock": len(re.findall(r"untested\s+no_clock", coverage_text)),
}
expected_coverage = {
    "setup": (1482, 1170, 0, 312),
    "hold": (1482, 1170, 0, 312),
    "out_setup": (793, 731, 0, 62),
    "out_hold": (793, 731, 0, 62),
}
expected_reasons = {
    "constant_disabled": 1872,
    "no_paths": 124,
    "no_startpoint_clock": 1716,
    "no_clock": 3432,
}
coverage_gate = coverage == expected_coverage and reason_counts == expected_reasons
gate = (setup >= 0.0 and hold >= 0.0 and violated_paths == 0
        and reset_only and coverage_gate)
receipt = {
    "schema": "m391_m387r1b_m384_prelayout_ptsta_receipt_r1b",
    "status": ("PASS_M391_R1B_M384_PRELAYOUT_PTSTA" if gate
               else "NO_GO_M391_R1B_M384_PRELAYOUT_PTSTA"),
    "tool": "Synopsys PrimeTime W-2024.09-SP3",
    "setup_worst_slack_ns": setup,
    "setup_state": setup_state,
    "hold_worst_slack_ns": hold,
    "hold_state": hold_state,
    "coverage": {
        name: {"total": row[0], "met": row[1],
               "violated": row[2], "untested": row[3]}
        for name, row in coverage.items()
    },
    "untested_reason_counts": reason_counts,
    "untested_reason_gate": coverage_gate,
    "untested_interpretation": {
        "setup_hold": "312 alternate conditional arcs per check are constant-disabled; 1170 active DFF checks are met",
        "output": "62 synthesized constant outputs per check have no timing path; 731 active output checks are met",
        "recovery_removal": "reset_n is the sole unconstrained asynchronous reset and is explicitly false-pathed",
    },
    "constraint_report_violated_paths": violated_paths,
    "asynchronous_reset_only_no_input_delay_exception": reset_only,
    "timing_gate_pass": gate,
    "claim_boundary": {
        "prelayout_no_spef": True,
        "ideal_clock": True,
        "zero_macro": True,
        "physical_sram": False,
        "postroute_timing": False,
        "reset_recovery_removal_signoff": False,
        "activity_backed_ptpx": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(root / "m391_m384_prelayout_ptsta_receipt_r1b.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not gate:
    raise SystemExit(42)
PY

{
    echo "status=PASS_M391_R1B_M384_PRELAYOUT_PTSTA"
    echo "scope=PRELAYOUT_NO_SPEF_IDEAL_CLOCK_ZERO_MACRO"
    echo "reset_timing_signoff=false"
    echo "activity_backed_ptpx=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
} >"${m391_run}/RUN_COMPLETE.txt"
sha256sum "${m391_runner}" >"${m391_run}/runner_sha256.txt"
(
  cd "${m391_run}"
  find . -type f ! -path './work/*' ! -name output.sha256 \
      ! -name output.seal.sha256 ! -name output_check.raw.log \
      -print0 | sort -z | xargs -0 sha256sum >output.sha256
  sha256sum --strict -c output.sha256 >output_check.raw.log 2>&1
  sha256sum output.sha256 >output.seal.sha256
)
m391_complete=1
echo "PASS_M391_M384_PRELAYOUT_PTSTA run=${m391_run}"
