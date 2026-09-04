#!/usr/bin/env bash
set -euo pipefail

m422_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m422_hw="$(cd "${m422_dc_root}/.." && pwd)"
m422_runner="$(realpath "${BASH_SOURCE[0]}")"
m422_source="${m422_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826"
m422_formality="${m422_dc_root}/runs/m420_m414_dual_formality_r1_20260826"
m422_hammer="${m422_hw}/results/m421_m420_dual_formality_independent_hammer_r1_20260826"
m422_run="${M422_PTSTA_RUN:-${m422_dc_root}/runs/m422_m416_selected_slice_prelayout_ptsta_r1_20260826}"
m422_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m422_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m422_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m422_netlist="${m422_source}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m422_sdc="${m422_source}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m422_tcl="${m422_dc_root}/scripts/run_ptsta_m422_m416_selected_slice_exact_sha.tcl"
m422_contract="${m422_hw}/contracts/m422_m416_selected_slice_prelayout_ptsta_contract_r1_20260826.json"

m422_sha() { sha256sum "$1" | awk '{print $1}'; }
m422_expect() { [[ -f "$1" && "$(m422_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m422_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 4; fi

m422_expect "${m422_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m422_expect "${m422_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m422_expect "${m422_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m422_expect "${m422_netlist}" 4b07e83eba88508da3fc1aa27187b3fa8ca03a633b165ea68641bcd26b969fe2
m422_expect "${m422_sdc}" b4eb7d225256474c8629cb924db18260efbd691db8a76d3a622d4ae8d0479ed9
m422_expect "${m422_tcl}" e8a83147770495b466c8517681880ca4a215f325f619ebcafae8c42d18a128b6
m422_expect "${m422_contract}" c6b5e812039cec1f543461e7f08e19da7bb2513db6ed10a7bdedcda9a184616b
m422_expect "${m422_source}/m416_m414_balanced_selected_slice_dc_receipt_r1.json" bedb903268d3e94c858e8177a383a46f35427cd9a1bdad3ad9ad398b4bc85c02
m422_expect "${m422_source}/evidence_manifest.seal.sha256" 40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79
m422_expect "${m422_formality}/m420_m414_dual_formality_receipt_r1.json" 4df80e3964c0ced618dedaa67776e21e283c6026c241566ddc59427479c08949
m422_expect "${m422_formality}/output.seal.sha256" cf216915f3f0c8e1ee4e894734e81337d81baf70b36cbd9b51ee23b381e723d7
m422_expect "${m422_hammer}/m421_m420_dual_formality_independent_hammer_review_r1.json" 1a449050ebe5967431798ff13638fd27fccca1ae2ec37a636375b34f7c2070a0
m422_expect "${m422_hammer}/SHA256SUMS.seal.sha256" 53d71e23ae3f901e98196fb008131847bdd86b0079608d32c5165347a9450554
m422_expect "${m422_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(cd "${m422_source}" && sha256sum --strict -c evidence_manifest.seal.sha256 >/dev/null)
(cd "${m422_source}" && sha256sum --strict -c evidence_manifest.sha256 >/dev/null)
(cd "${m422_formality}" && sha256sum --strict -c output.seal.sha256 >/dev/null)
(cd "${m422_formality}" && sha256sum --strict -c output.sha256 >/dev/null)
(cd "${m422_hammer}" && sha256sum --strict -c SHA256SUMS.seal.sha256 >/dev/null)
(cd "${m422_hammer}" && sha256sum --strict -c SHA256SUMS >/dev/null)

mkdir -p "${m422_run}/reports" "${m422_run}/work"
m422_complete=0
trap 'm422_rc=$?; if [[ ${m422_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m422_rc}" >"${m422_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cp "${m422_contract}" "${m422_run}/contract.json"
sha256sum "${m422_netlist}" "${m422_sdc}" "${m422_slow}" \
    "${m422_fast}" "${m422_tcl}" "${m422_contract}" \
    "${m422_source}/m416_m414_balanced_selected_slice_dc_receipt_r1.json" \
    "${m422_source}/evidence_manifest.seal.sha256" \
    "${m422_formality}/m420_m414_dual_formality_receipt_r1.json" \
    "${m422_formality}/output.seal.sha256" \
    "${m422_hammer}/m421_m420_dual_formality_independent_hammer_review_r1.json" \
    "${m422_hammer}/SHA256SUMS.seal.sha256" \
    >"${m422_run}/input_sha256.txt"

export DESIGN_NAME=m405_q32_elastic_selected_slice
export LIB_SLOW="${m422_slow}"
export LIB_FAST="${m422_fast}"
export MAPPED_NETLIST="${m422_netlist}"
export MAPPED_SDC="${m422_sdc}"
export OUTPUT_DIR="${m422_run}"
"${m422_pt}" -version >"${m422_run}/pt.version.raw.log" 2>&1
set +e
(cd "${m422_run}/work" && "${m422_pt}" -f "${m422_tcl}") \
    >"${m422_run}/pt.raw.log" 2>&1
m422_rc=$?
set -e
echo "${m422_rc}" >"${m422_run}/pt.rc"
[[ "${m422_rc}" -eq 0 ]]
! grep -Eq '^(Error|Fatal):' "${m422_run}/pt.raw.log"
grep -Fqx "Design 'm405_q32_elastic_selected_slice' was successfully linked." \
    "${m422_run}/pt.raw.log"
[[ "$(grep -xc 'M422_M416_SELECTED_SLICE_PTSTA_INTERNAL_COMPLETE=PASS' \
    "${m422_run}/PTSTA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
for m422_report in ptsta_scope.rpt ptsta_check_timing.rpt \
        ptsta_analysis_coverage.rpt ptsta_global_timing.rpt \
        ptsta_timing_setup.rpt ptsta_timing_hold.rpt \
        ptsta_constraint_violators.rpt ptsta_clock.rpt \
        ptsta_exceptions.rpt; do
    [[ -s "${m422_run}/reports/${m422_report}" ]] || exit 30
done

python3 - "${m422_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])


def text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def slack(name):
    report = text(root / "reports" / name)
    match = re.search(r"slack \((MET|VIOLATED)\)\s+(-?\d+\.\d+)", report)
    if not match:
        raise RuntimeError("missing slack in " + name)
    return match.group(1), float(match.group(2))


def coverage_row(name):
    report = text(root / "reports" / "ptsta_analysis_coverage.rpt")
    match = re.search(
        rf"^{name}\s+(\d+)\s+(\d+) \([^\n]+?\)\s+(\d+) \([^\n]+?\)\s+(\d+) \(",
        report,
        re.M,
    )
    if not match:
        raise RuntimeError("missing coverage row " + name)
    return tuple(map(int, match.groups()))


setup_state, setup = slack("ptsta_timing_setup.rpt")
hold_state, hold = slack("ptsta_timing_hold.rpt")
coverage = {
    name: coverage_row(name) for name in ("setup", "hold", "out_setup", "out_hold")
}
coverage_gate = all(
    total > 0 and met > 0 and violated == 0 and total == met + violated + untested
    for total, met, violated, untested in coverage.values()
)
constraint_report = text(root / "reports" / "ptsta_constraint_violators.rpt")
violated_paths = constraint_report.count("slack (VIOLATED)")
check_report = text(root / "reports" / "ptsta_check_timing.rpt")
reset_only = (
    check_report.count("paths from such ports will be unconstrained") == 1
    and re.search(r"(?m)^reset_n\s*$", check_report) is not None
)
coverage_report = text(root / "reports" / "ptsta_analysis_coverage.rpt")
reason_counts = {
    reason: len(re.findall(rf"untested\s+{reason}\b", coverage_report))
    for reason in ("constant_disabled", "no_paths", "no_startpoint_clock", "no_clock")
}
raw = text(root / "pt.raw.log")
link_success_count = raw.count(
    "Design 'm405_q32_elastic_selected_slice' was successfully linked."
)
unlinked_reference_count = len(
    re.findall(r"(?im)^(?:Error|Warning):.*(?:unlinked|unresolved reference)", raw)
)
gate = (
    setup_state == "MET"
    and hold_state == "MET"
    and setup >= 0.0
    and hold >= 0.0
    and violated_paths == 0
    and coverage_gate
    and reset_only
    and link_success_count == 1
    and unlinked_reference_count == 0
)
receipt = {
    "schema": "m422_m416_selected_slice_prelayout_ptsta_receipt_r1",
    "status": (
        "PASS_M422_M416_SELECTED_SLICE_PRELAYOUT_PTSTA"
        if gate
        else "NO_GO_M422_M416_SELECTED_SLICE_PRELAYOUT_PTSTA"
    ),
    "tool": "Synopsys PrimeTime W-2024.09-SP3",
    "setup_corner": "ssg0p9v125c",
    "hold_corner": "ffg1p05vm40c via set_min_library",
    "setup_worst_slack_ns": setup,
    "setup_state": setup_state,
    "hold_worst_slack_ns": hold,
    "hold_state": hold_state,
    "coverage": {
        name: {
            "total": row[0],
            "met": row[1],
            "violated": row[2],
            "untested": row[3],
        }
        for name, row in coverage.items()
    },
    "coverage_gate_pass": coverage_gate,
    "untested_reason_counts": reason_counts,
    "untested_interpretation": {
        "status": "recorded_not_silently_promoted_to_met",
        "recovery_removal": "reset_n is the sole unconstrained asynchronous reset and is explicitly false-pathed; recovery/removal is not signed off",
    },
    "constraint_report_violated_paths": violated_paths,
    "reset_n_only_unconstrained_input_exception": reset_only,
    "link_success_count": link_success_count,
    "unlinked_design_references": unlinked_reference_count,
    "timing_gate_pass": gate,
    "required_wording": "PrimeTime prelayout timing diagnostic for the M416 logic-only selected slice at 3 ns. This is neither routed timing nor SRAM-inclusive paper PPA.",
    "claim_boundary": {
        "prelayout_no_spef": True,
        "ideal_clock": True,
        "zero_macro": True,
        "physical_sram": False,
        "postroute_timing": False,
        "reset_recovery_removal_signoff": False,
        "activity_backed_ptpx": False,
        "energy": False,
        "new_cycle_speedup": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(root / "m422_m416_selected_slice_prelayout_ptsta_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not gate:
    raise SystemExit(42)
PY

{
    echo "status=PASS_M422_M416_SELECTED_SLICE_PRELAYOUT_PTSTA"
    echo "scope=PRELAYOUT_NO_SPEF_IDEAL_CLOCK_ZERO_MACRO"
    echo "reset_timing_signoff=false"
    echo "activity_backed_ptpx=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
} >"${m422_run}/RUN_COMPLETE.txt"
sha256sum "${m422_runner}" >"${m422_run}/runner_sha256.txt"
(
    cd "${m422_run}"
    find . -type f ! -path './work/*' ! -name output.sha256 \
        ! -name output.seal.sha256 ! -name output_check.raw.log \
        -print0 | sort -z | xargs -0 sha256sum >output.sha256
    sha256sum --strict -c output.sha256 >output_check.raw.log 2>&1
    sha256sum output.sha256 >output.seal.sha256
)
m422_complete=1
echo "PASS_M422_M416_SELECTED_SLICE_PRELAYOUT_PTSTA run=${m422_run}"
