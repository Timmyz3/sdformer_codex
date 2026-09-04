#!/usr/bin/env bash
set -euo pipefail

m441_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m441_hw="$(cd "${m441_dc_root}/.." && pwd)"
m441_runner="$(realpath "${BASH_SOURCE[0]}")"
m441_run="${M441_RUN:-${m441_dc_root}/runs/m441_m433_to_m439_formality_ptsta_r1d_20260826}"
m441_m439="${m441_dc_root}/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826"
m441_m433="${m441_hw}/results/m433_exact_dualbank_coread_directed_vcs_r1_20260826"
m441_m434="${m441_hw}/reviews/m434_m433_independent_hammer_r1_20260826"
m441_fm="/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell"
m441_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m441_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m441_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m441_rtl_fl="${m441_dc_root}/filelists/date_m439_m433_dualcoread_adapter_rtl.f"
m441_rtl="${m441_hw}/rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv"
m441_netlist="${m441_m439}/dual_coread/netlist/m433_exact_dualbank_coread_pwp_adapter_mapped.v"
m441_svf="${m441_m439}/dual_coread/netlist/m433_exact_dualbank_coread_pwp_adapter.svf"
m441_sdc="${m441_m439}/dual_coread/netlist/m433_exact_dualbank_coread_pwp_adapter_mapped.sdc"
m441_fm_tcl="${m441_dc_root}/scripts/run_formality_m441_m433_to_m439_exact_sha.tcl"
m441_pt_tcl="${m441_dc_root}/scripts/run_ptsta_m441_m439_dualcoread_exact_sha.tcl"
m441_contract="${m441_hw}/contracts/m441_m433_to_m439_formality_ptsta_contract_r1_20260826.json"
m441_top="m433_exact_dualbank_coread_pwp_adapter"

m441_sha() { sha256sum "$1" | awk '{print $1}'; }
m441_expect() { [[ -f "$1" && "$(m441_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m441_run}" ]] || exit 5
if pgrep -x fm_shell >/dev/null 2>&1 || \
        pgrep -x fm_shell_exec >/dev/null 2>&1 || \
        pgrep -x pt_shell >/dev/null 2>&1; then
    exit 4
fi

cd "${m441_hw}"
m441_expect "${m441_fm}" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m441_expect "${m441_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m441_expect "${m441_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m441_expect "${m441_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m441_expect "${m441_rtl_fl}" 99f23e242a5eff6526e2118c6811b3b09493ce43bbc01d0b91e0ea9be7e19303
m441_expect "${m441_rtl}" 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
m441_expect "${m441_netlist}" f553da7b33d5966be445a3e02b11740c46f0776a3163f0654897777809fdd7d8
m441_expect "${m441_svf}" 74dfb47d834c221f6b61e89eb3b7b5272af59eb852a0062c0f5f949147b2e26d
m441_expect "${m441_sdc}" 5878b6bd2954d30a90673f116a2f3927c2f409a9a5b11963f959f64a73c1359a
m441_expect "${m441_fm_tcl}" 4d1c81f3f3b6aeff079ba60a5c750ffe8f620fd599adb770bdd8aa68be8ed2ec
m441_expect "${m441_pt_tcl}" 94840c0835f96eafd60ac2f950978c83bac630e5fae21f6c8a972f7f2c40bcbb
m441_expect "${m441_contract}" 9404533af56d6931d9d27a28834069d42492292c185c4f7c5ddde5bf5bef88bf
m441_expect "${m441_m433}/RUN_MANIFEST.seal.sha256" d57308dcabd40945f827fa0dfba0f18c7374f5d710722d2121e1084cd5b6d375
m441_expect "${m441_m434}/m434_m433_independent_review_r1.json" d2f445159c61df237820433a512e71ff47768220805f9e2c6c7e3bec11b9e56a
m441_expect "${m441_m434}/SHA256SUMS.seal.sha256" 5b271829327c5ae5632b42b44c22a37515f022426217663fac9a7e9001455aa1
m441_expect "${m441_m439}/m439_serial_vs_dualcoread_adapters_dc_receipt_r1.json" 9f8d14bec581114e80886c172397b04043965f8c9930acaca742167927072133
m441_expect "${m441_m439}/evidence_manifest.seal.sha256" 98696f3bd166172aa294d2d24fb5d16f6fa7211a8da939fb99c035506d3eaa1a
m441_expect "${m441_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(cd "${m441_m433}" && sha256sum --strict -c RUN_MANIFEST.seal.sha256 >/dev/null && \
    sha256sum --strict -c RUN_MANIFEST.sha256 >/dev/null)
(cd "${m441_m434}" && sha256sum --strict -c SHA256SUMS.seal.sha256 >/dev/null && \
    sha256sum --strict -c SHA256SUMS >/dev/null)
(cd "${m441_m439}" && sha256sum --strict -c evidence_manifest.seal.sha256 >/dev/null && \
    sha256sum --strict -c evidence_manifest.sha256 >/dev/null)

mkdir -p "${m441_run}/formality/reports" "${m441_run}/formality/work" \
    "${m441_run}/ptsta/reports" "${m441_run}/ptsta/work"
m441_complete=0
trap 'm441_rc=$?; if [[ ${m441_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m441_rc}" >"${m441_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cp "${m441_contract}" "${m441_run}/contract.json"
sha256sum "${m441_rtl_fl}" "${m441_rtl}" "${m441_netlist}" \
    "${m441_svf}" "${m441_sdc}" "${m441_slow}" "${m441_fast}" \
    "${m441_fm_tcl}" "${m441_pt_tcl}" "${m441_contract}" \
    "${m441_m433}/RUN_MANIFEST.seal.sha256" \
    "${m441_m434}/m434_m433_independent_review_r1.json" \
    "${m441_m434}/SHA256SUMS.seal.sha256" \
    "${m441_m439}/m439_serial_vs_dualcoread_adapters_dc_receipt_r1.json" \
    "${m441_m439}/evidence_manifest.seal.sha256" \
    "${m441_hw}/docs/359_DATE终局冻结_20260813.md" \
    >"${m441_run}/input_sha256.txt"

"${m441_fm}" -version >"${m441_run}/formality/formality.version.raw.log" 2>&1
export M441_DESIGN_NAME="${m441_top}"
export M441_SNAPSHOT_ROOT="${m441_hw}"
export M441_RTL_FILELIST="${m441_rtl_fl}"
export M441_LIB_DB="${m441_slow}"
export M441_MAPPED_NETLIST="${m441_netlist}"
export M441_SVF_FILE="${m441_svf}"
export M441_FM_OUTPUT_DIR="${m441_run}/formality"
set +e
(cd "${m441_run}/formality/work" && "${m441_fm}" -f "${m441_fm_tcl}") \
    >"${m441_run}/formality/formality.raw.log" 2>&1
m441_fm_rc=$?
set -e
printf '%s\n' "${m441_fm_rc}" >"${m441_run}/formality/formality.rc"
[[ "${m441_fm_rc}" -eq 0 ]]
if grep -Eq '^(Error|Fatal):' "${m441_run}/formality/formality.raw.log"; then exit 31; fi
grep -Fqx 'M441_M433_RTL_TO_M439_NETLIST_FORMALITY_INTERNAL_COMPLETE=PASS' \
    "${m441_run}/formality/FORMALITY_INTERNAL_COMPLETE.txt"
grep -Fq 'Verification SUCCEEDED' "${m441_run}/formality/reports/formality_status.rpt"
grep -Eq '[1-9][0-9]* Passing compare points' "${m441_run}/formality/reports/formality_status.rpt"
grep -Fq 'No unmatched points' "${m441_run}/formality/reports/formality_unmatched.rpt"
grep -Fq 'No failing compare points' "${m441_run}/formality/reports/formality_failing.rpt"
grep -Fq 'No aborted compare points' "${m441_run}/formality/reports/formality_aborted.rpt"
grep -Fq 'No unverified compare points' "${m441_run}/formality/reports/formality_unverified.rpt"

"${m441_pt}" -version >"${m441_run}/ptsta/pt.version.raw.log" 2>&1
export M441_LIB_SLOW="${m441_slow}"
export M441_LIB_FAST="${m441_fast}"
export M441_MAPPED_SDC="${m441_sdc}"
export M441_PT_OUTPUT_DIR="${m441_run}/ptsta"
set +e
(cd "${m441_run}/ptsta/work" && "${m441_pt}" -f "${m441_pt_tcl}") \
    >"${m441_run}/ptsta/pt.raw.log" 2>&1
m441_pt_rc=$?
set -e
printf '%s\n' "${m441_pt_rc}" >"${m441_run}/ptsta/pt.rc"
[[ "${m441_pt_rc}" -eq 0 ]]
if grep -Eq '^(Error|Fatal):' "${m441_run}/ptsta/pt.raw.log"; then exit 41; fi
grep -Fqx "Design '${m441_top}' was successfully linked." "${m441_run}/ptsta/pt.raw.log"
grep -Fqx 'M441_M439_DUALCOREAD_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS' \
    "${m441_run}/ptsta/PTSTA_INTERNAL_COMPLETE.txt"
for m441_report in check_timing.rpt analysis_coverage.rpt global_timing.rpt \
        timing_setup_slow.rpt timing_hold_fast.rpt constraint_violators.rpt \
        clock.rpt exceptions.rpt design.rpt wire_load.rpt libraries.rpt \
        runtime_scope.rpt; do
    [[ -s "${m441_run}/ptsta/reports/${m441_report}" ]] || exit 42
done

python3 - "${m441_run}" "${m441_sdc}" "${m441_pt_tcl}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
sdc_path = Path(sys.argv[2])
pt_tcl_path = Path(sys.argv[3])

def read(path):
    return path.read_text(encoding="utf-8", errors="replace")

status = read(run / "formality/reports/formality_status.rpt")
passing_match = re.search(r"(\d+) Passing compare points", status)
passing_row = re.search(
    r"Passing \(equivalent\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
    status,
)
if not passing_match or not passing_row:
    raise SystemExit("missing Formality passing counts")
passing = int(passing_match.group(1))
passing_fields = tuple(map(int, passing_row.groups()))
if passing != passing_fields[-1]:
    raise SystemExit("Formality passing total mismatch")
all_compared_points_passing = (
    passing_fields[0] == 0
    and passing_fields[1] == 0
    and passing_fields[2] == 0
    and passing_fields[3] == 0
    and passing_fields[4] + passing_fields[5] + passing_fields[6] == passing
)

def unread_count(path, noun):
    report = read(path)
    if "No unmatched points" in report or "No not-compared points" in report:
        return 0
    match = re.search(rf"(\d+) {noun} points", report, re.I)
    if not match:
        raise SystemExit(f"cannot parse unread report {path}")
    return int(match.group(1))

unmatched_unread_dffs = unread_count(
    run / "formality/reports/formality_unmatched_unread_dff.rpt", "Unmatched")
not_compared_unread_dffs = unread_count(
    run / "formality/reports/formality_not_compared_unread_dff.rpt", "not-compared")
fm_gate = (
    "Verification SUCCEEDED" in status
    and "verify_return=1" in status
    and passing > 0
    and all_compared_points_passing
    and "No unmatched points" in read(run / "formality/reports/formality_unmatched.rpt")
    and "No failing compare points" in read(run / "formality/reports/formality_failing.rpt")
    and "No aborted compare points" in read(run / "formality/reports/formality_aborted.rpt")
    and "No unverified compare points" in read(run / "formality/reports/formality_unverified.rpt")
)

def slack(path):
    match = re.search(r"slack \((MET|VIOLATED)\)\s+(-?\d+\.\d+)", read(path))
    if not match:
        raise SystemExit(f"missing slack in {path}")
    return match.group(1), float(match.group(2))

setup_state, setup_slack = slack(run / "ptsta/reports/timing_setup_slow.rpt")
hold_state, hold_slack = slack(run / "ptsta/reports/timing_hold_fast.rpt")
constraint_report = read(run / "ptsta/reports/constraint_violators.rpt")
constraint_violations = constraint_report.count("slack (VIOLATED)")
design_report = read(run / "ptsta/reports/design.rpt")
libraries_report = read(run / "ptsta/reports/libraries.rpt")
scope_report = read(run / "ptsta/reports/runtime_scope.rpt")
clock_report = read(run / "ptsta/reports/clock.rpt")
coverage_report = read(run / "ptsta/reports/analysis_coverage.rpt")
check_report = read(run / "ptsta/reports/check_timing.rpt")
sdc = read(sdc_path)
pt_tcl = read(pt_tcl_path)

def coverage_row(name):
    match = re.search(
        rf"^{name}\s+(\d+)\s+(\d+) \([^\n]+?\)\s+(\d+) \([^\n]+?\)\s+(\d+) \(",
        coverage_report,
        re.M,
    )
    if not match:
        raise SystemExit(f"missing analysis coverage row {name}")
    return tuple(map(int, match.groups()))

coverage = {
    name: coverage_row(name)
    for name in ("setup", "hold", "recovery", "removal", "min_pulse_width", "out_setup", "out_hold")
}
functional_coverage_gate = all(
    total > 0 and met == total and violated == 0 and untested == 0
    for total, met, violated, untested in (
        coverage["setup"], coverage["hold"], coverage["out_setup"], coverage["out_hold"])
)
reset_only_unconstrained = (
    check_report.count("paths from such ports will be unconstrained") == 1
    and re.search(r"(?m)^reset_n\s*$", check_report) is not None
)

sdc_clock = re.search(
    r"create_clock \[get_ports clk_core\]\s+-name core_clk\s+-period 3(?:\.0+)?\b",
    sdc,
) is not None
sdc_slow_opcond = (
    "set_operating_conditions ssg0p9v125c" in sdc
    and "tcbn28hpcplusbwp35p140ssg0p9v125c" in sdc
)
sdc_zero_wireload = "set_wire_load_model -name ZeroWireload" in sdc
sdc_reset_false_path = "set_false_path   -from [get_ports reset_n]" in sdc
corner_gate = (
    re.search(r"operating_condition_max_name\s+ssg0p9v125c", design_report) is not None
    and re.search(r"operating_condition_min_name\s+ffg1p05vm40c", design_report) is not None
    and "tcbn28hpcplusbwp35p140ssg0p9v125c" in libraries_report
    and "tcbn28hpcplusbwp35p140ffg1p05vm40c" in libraries_report
)
scope_gate = all(token in scope_report for token in (
    "parasitics=none_no_read_parasitics_command",
    "clock=ideal_from_frozen_m439_mapped_sdc",
    "wireload=ZeroWireload_from_frozen_m439_mapped_sdc",
    "macro_count=0",
    "physical_sram=false",
    "physical_interconnect=false",
))
clock_gate = "core_clk" in clock_report and re.search(r"\b3\.00\b", clock_report) is not None
no_parasitic_command = re.search(r"(?m)^\s*read_parasitics\b", pt_tcl) is None
min_library_command = (
    "set_min_library $lib_slow -min_version $lib_fast" in pt_tcl
    and "-min $fast_opcond -min_library $fast_lib_name" in pt_tcl
)
pt_gate = (
    setup_state == "MET" and setup_slack >= 0.0
    and hold_state == "MET" and hold_slack >= 0.0
    and constraint_violations == 0
    and corner_gate and scope_gate and clock_gate
    and functional_coverage_gate and reset_only_unconstrained
    and no_parasitic_command and min_library_command
    and sdc_clock and sdc_slow_opcond and sdc_zero_wireload and sdc_reset_false_path
)

if not fm_gate or not pt_gate:
    raise SystemExit("M441 formal or timing gate failed")

all_internal_state_wording = unmatched_unread_dffs == 0 and not_compared_unread_dffs == 0
if all_internal_state_wording:
    required_wording = (
        f"Unconstrained-primary-input sequential equivalence passes for all {passing} "
        "Formality compare points of the M433 standalone dual-coread adapter."
    )
else:
    required_wording = (
        f"Unconstrained-primary-input sequential observational equivalence passes for all {passing} "
        f"compared points; {unmatched_unread_dffs} unmatched-unread and "
        f"{not_compared_unread_dffs} not-compared-unread DFF rows are separately disclosed, "
        "so this is not an all-internal-state claim."
    )

receipt = {
    "schema": "m441_m433_to_m439_formality_ptsta_receipt_v1",
    "status": "PASS_M441_M433_TO_M439_FORMALITY_AND_PRELAYOUT_PTSTA",
    "formality": {
        "tool": "Synopsys Formality V-2023.12-SP3",
        "reference": "M433 standalone dual-coread RTL",
        "implementation": "M439 mapped dual-coread netlist",
        "primary_inputs": "unconstrained",
        "verify_return": 1,
        "passing_compare_points": passing,
        "all_compared_points_passing": all_compared_points_passing,
        "passing_blackbox_pins": passing_fields[0],
        "passing_ports": passing_fields[4],
        "passing_dffs": passing_fields[5],
        "failing_compare_points": 0,
        "aborted_compare_points": 0,
        "unverified_compare_points": 0,
        "general_unmatched_points": 0,
        "unmatched_unread_reference_dffs": unmatched_unread_dffs,
        "not_compared_unread_dff_rows": not_compared_unread_dffs,
        "all_internal_state_wording_allowed": all_internal_state_wording,
        "blackboxes": 0,
        "required_wording": required_wording,
    },
    "primetime": {
        "tool": "Synopsys PrimeTime W-2024.09-SP3",
        "clock_period_ns": 3.0,
        "setup_corner": "ssg0p9v125c",
        "setup_worst_slack_ns": setup_slack,
        "setup_state": setup_state,
        "hold_corner": "ffg1p05vm40c via slow-to-fast min-library binding",
        "hold_worst_slack_ns": hold_slack,
        "hold_state": hold_state,
        "constraint_violated_paths": constraint_violations,
        "analysis_coverage": {
            name: {
                "total": row[0],
                "met": row[1],
                "violated": row[2],
                "untested": row[3],
            }
            for name, row in coverage.items()
        },
        "functional_setup_hold_coverage_gate_pass": functional_coverage_gate,
        "reset_n_only_unconstrained_input": reset_only_unconstrained,
        "reset_recovery_removal_interpretation": "reset_n is the sole unconstrained asynchronous reset and is false-pathed by the frozen mapped SDC; recovery/removal is not signed off",
        "mapped_sdc_clock_verified": sdc_clock,
        "mapped_sdc_slow_operating_condition_verified": sdc_slow_opcond,
        "slow_max_fast_min_operating_conditions_verified": corner_gate,
        "mapped_sdc_zero_wireload_verified": sdc_zero_wireload,
        "mapped_sdc_reset_false_path_verified": sdc_reset_false_path,
        "prelayout_no_spef": True,
        "no_read_parasitics_command_verified": no_parasitic_command,
        "slow_to_fast_min_library_binding_verified": min_library_command,
        "ideal_clock": True,
        "macro_count": 0,
    },
    "claim_boundary": {
        "standalone_dual_adapter_only": True,
        "serial_vs_dual_functional_equivalence": False,
        "rtl_to_own_mapped_netlist_equivalence": True,
        "prelayout_logic_only_timing": True,
        "physical_sram": False,
        "physical_interconnect": False,
        "postroute_timing": False,
        "reset_recovery_removal_signoff": False,
        "power": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(run / "m441_m433_to_m439_formality_ptsta_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
    echo 'status=PASS_M441_M433_TO_M439_FORMALITY_AND_PRELAYOUT_PTSTA'
    echo 'scope=STANDALONE_DUAL_ADAPTER_ONLY'
    echo 'serial_vs_dual_functional_equivalence=false'
    echo 'prelayout_no_spef_ideal_clock_zero_macro=true'
    echo 'physical_sram_or_interconnect=false'
    echo 'power=false'
    echo 'system_speedup=false'
    echo 'paper_ppa_ready=false'
    echo 'headline=false'
} >"${m441_run}/RUN_COMPLETE.txt"
sha256sum "${m441_runner}" >"${m441_run}/runner_sha256.txt"
(
    cd "${m441_run}"
    find . -type f ! -path './formality/work/*' ! -path './ptsta/work/*' \
        ! -name evidence_manifest.sha256 ! -name evidence_manifest.seal.sha256 \
        ! -name evidence_manifest_check.raw.log -print0 | sort -z | \
        xargs -0 sha256sum >evidence_manifest.sha256
    sha256sum --strict -c evidence_manifest.sha256 >evidence_manifest_check.raw.log 2>&1
    sha256sum evidence_manifest.sha256 >evidence_manifest.seal.sha256
)
m441_complete=1
echo "PASS_M441_M433_TO_M439_FORMALITY_AND_PRELAYOUT_PTSTA run=${m441_run}"
