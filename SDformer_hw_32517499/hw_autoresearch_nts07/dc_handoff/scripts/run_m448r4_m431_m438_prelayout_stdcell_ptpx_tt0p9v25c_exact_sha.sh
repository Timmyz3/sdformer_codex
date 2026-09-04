#!/usr/bin/env bash
set -euo pipefail

m448r4_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m448r4_hw="$(cd "${m448r4_dc_root}/.." && pwd)"
m448r4_runner="$(realpath "${BASH_SOURCE[0]}")"
m448r4_run="${M448R4_RUN_DIR:-${m448r4_dc_root}/runs/m448r4_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r4_20260826}"
m448r4_contract="${m448r4_hw}/contracts/m448r4_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_contract_r4_20260826.json"
m448r4_inner="${m448r4_dc_root}/scripts/run_m448r3_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_exact_sha.sh"
m448r4_tcl="${m448r4_dc_root}/scripts/run_ptpx_m448r3_m431_m438_prelayout_stdcell_tt0p9v25c.tcl"
m448r4_r3_contract="${m448r4_hw}/contracts/m448r3_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_contract_r3_20260826.json"
m448r4_docs359="${m448r4_hw}/docs/359_DATE终局冻结_20260813.md"
m448r4_r3_failed="${m448r4_dc_root}/runs/m448r3_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r3_20260826"
m448r4_tmp="$(mktemp)"
m448r4_complete=0

m448r4_sha() { sha256sum "$1" | awk '{print $1}'; }

m448r4_cleanup() {
    m448r4_rc=$?
    rm -f "${m448r4_tmp}"
    if [[ ${m448r4_complete} -ne 1 && -d "${m448r4_run}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${m448r4_rc}" >"${m448r4_run}/RUN_FAILED_OR_INCOMPLETE_R4.txt"
    fi
}
trap m448r4_cleanup EXIT

[[ ! -e "${m448r4_run}" ]] || exit 2

declare -A m448r4_expected=(
    ["${m448r4_contract}"]="653e6548289095e37bc871c9cd08112718e178c053061370493665a96bd2235b"
    ["${m448r4_r3_contract}"]="1f8a5aeec6bd26548686d5ab5db5ac8828d64e323813f7c976374fa85fc1ef6a"
    ["${m448r4_inner}"]="a243e8a495ddf9f0d9495a4fc9905f4a82725ee3462d56712629b23e5da32596"
    ["${m448r4_tcl}"]="9a9628fe92722cf9f98e8fa8db8839dbf0bffd2189bab8c661a2eb69d7554ed6"
    ["${m448r4_docs359}"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    ["${m448r4_r3_failed}/RUN_MANIFEST.sha256"]="abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a"
    ["${m448r4_r3_failed}/RUN_MANIFEST.seal.sha256"]="657898a0d7f7d1b421281f509718e371357863be28c93023a7e6ccba32d11f35"
)

for m448r4_path in "${!m448r4_expected[@]}"; do
    [[ "$(m448r4_sha "${m448r4_path}")" == "${m448r4_expected[${m448r4_path}]}" ]] || exit 10
done

python3 - "${m448r4_contract}" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text())
if contract["milestone"] != "M448R4":
    raise SystemExit("M448R4 milestone drift")
if contract["status_before_execution"] != "FROZEN_BEFORE_FULL_R4_RERUN_AND_NONVACUOUS_SEAL":
    raise SystemExit("M448R4 contract is not frozen")
if contract["r4_manifest_contract"]["minimum_entries"] != 30:
    raise SystemExit("M448R4 manifest minimum-entry contract drift")
if not contract["claim_boundary"]["requires_independent_hammer_before_admission"]:
    raise SystemExit("M448R4 independent-hammer gate was removed")
PY

set +e
M448R3_RUN_DIR="${m448r4_run}" "${m448r4_inner}" >"${m448r4_tmp}" 2>&1
m448r4_inner_rc=$?
set -e
if [[ -d "${m448r4_run}" ]]; then
    cp "${m448r4_tmp}" "${m448r4_run}/r4_inner_runner.log"
fi
[[ ${m448r4_inner_rc} -eq 0 ]] || exit 20

cp "${m448r4_contract}" "${m448r4_run}/contract_r4.json"
sha256sum "${m448r4_runner}" >"${m448r4_run}/outer_runner_sha256.txt"
: >"${m448r4_run}/r4_outer_preflight_sha_checks.txt"
for m448r4_path in "${!m448r4_expected[@]}"; do
    printf 'path=%s expected=%s observed=%s\n' "${m448r4_path}" \
        "${m448r4_expected[${m448r4_path}]}" "$(m448r4_sha "${m448r4_path}")" \
        >>"${m448r4_run}/r4_outer_preflight_sha_checks.txt"
done

python3 - "${m448r4_run}" "${m448r4_contract}" "${m448r4_inner}" "${m448r4_tcl}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
contract_path = Path(sys.argv[2])
inner_runner = Path(sys.argv[3])
tcl = Path(sys.argv[4])
inner_receipt_path = run / "m448r3_m431_m438_prelayout_stdcell_ptpx_receipt_r3.json"

required_nonempty = [
    run / "ptpx.log",
    run / "ptpx.rc",
    run / "power_call_ledger.txt",
    run / "reports/ptpx_power_primary_100ps.rpt",
    run / "reports/ptpx_power_sensitivity_050ps.rpt",
    run / "reports/ptpx_power_sensitivity_200ps.rpt",
    run / "reports/saif_annotation_summary.rpt",
    run / "reports/switching_coverage.rpt",
    inner_receipt_path,
]
for path in required_nonempty:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"M448R4 missing/empty required output: {path}")
if (run / "ptpx.rc").read_text().strip() != "0":
    raise SystemExit("M448R4 pt_shell exit code is not zero")

expected_ledger = [
    "primary_100ps check_power_pass",
    "primary_100ps update_power_complete",
    "primary_100ps report_power_complete",
    "sensitivity_050ps check_power_pass",
    "sensitivity_050ps update_power_complete",
    "sensitivity_050ps report_power_complete",
    "sensitivity_200ps check_power_pass",
    "sensitivity_200ps update_power_complete",
    "sensitivity_200ps report_power_complete",
]
ledger = (run / "power_call_ledger.txt").read_text().splitlines()
if ledger != expected_ledger:
    raise SystemExit(f"M448R4 runtime ledger mismatch: {ledger}")

for label in ("primary_100ps", "sensitivity_050ps", "sensitivity_200ps"):
    check = (run / f"reports/ptpx_check_power_{label}_pre_update.rpt").read_text(errors="replace")
    if "check_power succeeded." not in check:
        raise SystemExit(f"M448R4 {label} check_power did not succeed")
    if re.search(r"Warning:|out_of_range|out of ramp range|missing table|missing function", check, re.I):
        raise SystemExit(f"M448R4 {label} check_power warning/ramp/table/function finding")
    power_path = run / f"reports/ptpx_power_{label}.rpt"
    power_text = power_path.read_text(errors="replace")
    for field in ("Cell Internal Power", "Net Switching Power", "Cell Leakage Power", "Total Power"):
        matches = re.findall(rf"{re.escape(field)}\s*=\s*([0-9.eE+-]+)", power_text)
        if len(matches) != 1:
            raise SystemExit(f"M448R4 {label} nonunique {field}: {len(matches)}")

annotation = (run / "reports/saif_annotation_summary.rpt").read_text(errors="replace")
coverage = (run / "reports/switching_coverage.rpt").read_text(errors="replace")
ann = re.search(r"Total number of nets = (\d+).*?Number of annotated nets = (\d+) \(([0-9.]+)%\).*?Total number of leaf cells = (\d+).*?Number of fully annotated leaf cells = (\d+) \(([0-9.]+)%\)", annotation, re.S)
cov = re.search(r"^m405_q32_elastic_selected_slice\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$", coverage, re.M)
if not ann or not cov:
    raise SystemExit("M448R4 cannot parse activity coverage")
activity_tuple = (
    int(ann.group(1)), int(ann.group(2)), float(ann.group(3)),
    int(ann.group(4)), int(ann.group(5)), float(ann.group(6)),
    int(cov.group(2)), int(cov.group(3)),
)
if activity_tuple != (22800, 22800, 100.0, 20803, 20803, 100.0, 21827, 22800):
    raise SystemExit(f"M448R4 activity gate failed: {activity_tuple}")
if float(cov.group(1)) < 95.0:
    raise SystemExit("M448R4 nonzero toggle coverage below 95%")

ptlog = (run / "ptpx.log").read_text(errors="replace")
if re.search(r"^Error:|^Fatal:", ptlog, re.M):
    raise SystemExit("M448R4 Synopsys log contains Error/Fatal")

inner = json.loads(inner_receipt_path.read_text())
if inner["status"] != "PASS_M448R3_PRELAYOUT_STDCELL_SELECTED_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER":
    raise SystemExit("M448R4 fresh inner analysis receipt did not pass")
if inner["activity"]["nonzero_tx_entries"] != 0:
    raise SystemExit("M448R4 fresh inner receipt has nonzero TX")
if inner["runtime_power_call_ledger"] != expected_ledger:
    raise SystemExit("M448R4 fresh inner receipt ledger drift")

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(inner)
receipt["schema"] = "date.m448r4_m431_m438_prelayout_stdcell_ptpx_receipt.v4"
receipt["milestone"] = "M448R4"
receipt["status"] = "PASS_M448R4_PRELAYOUT_STDCELL_SELECTED_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER"
receipt["fresh_r4_execution"] = {
    "full_synopsys_rerun_from_beginning": True,
    "fresh_run_directory": str(run),
    "exact_inner_runner_path": str(inner_runner),
    "exact_inner_runner_sha256": sha(inner_runner),
    "exact_ptpx_tcl_path": str(tcl),
    "exact_ptpx_tcl_sha256": sha(tcl),
    "fresh_inner_receipt_path": str(inner_receipt_path),
    "fresh_inner_receipt_sha256": sha(inner_receipt_path),
    "note": "The R3-named receipt was generated by the exact reused engine inside this fresh R4 run; no prior R1/R2/R3 numeric report was copied.",
}
receipt["seal_correction"] = {
    "old_inner_RUN_MANIFEST_files_ignored": True,
    "reason": "The exact inner runner retains the known absolute-path manifest bug; R4 replaces it with R4_RUN_MANIFEST.sha256 built from relative paths.",
    "r4_manifest_paths_relative": True,
    "r4_manifest_minimum_entries": 30,
    "r4_manifest_forbids_stdin_dash_target": True,
    "r4_manifest_and_seal_pending_until_after_receipt_generation": True,
}
receipt["supersession"] = {
    "M448_R1": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
    "M448R2": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
    "M448R3": "FAILED_INVALID_VACUOUS_SEAL_DO_NOT_CITE",
}
receipt["claim_boundary"]["pending_independent_hammer"] = True
receipt["claim_boundary"]["paper_ppa_ready"] = False
receipt["claim_boundary"]["headline"] = False
receipt_path = run / "m448r4_m431_m438_prelayout_stdcell_ptpx_receipt_r4.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

contract = json.loads(contract_path.read_text())
summary = "# M448R4 M431/M438 prelayout standard-cell PTPX\n\n"
summary += "Status: **PASS, pending independent hammer.** R4 is a fresh full PrimeTime PX rerun with a corrected nonvacuous outer seal.\n\n"
summary += "The exact numeric power and energy fields are in the JSON receipt and raw reports. Scope is only the M416 balanced selected slice at TT 0.9 V / 25 C, 3.0 ns ideal clock, ZeroWireload, no SPEF, zero macro. SRAM, CTS, extracted interconnect, full Conv, full network and system energy/speedup are excluded. reset_n input slew is not reset signoff.\n\n"
summary += "R1/R2/R3 remain DO_NOT_CITE. Admission requires an independent hammer of the R4 relative-path manifest, raw reports, receipt derivations and claim boundary.\n"
(run / "m448r4_m431_m438_prelayout_stdcell_ptpx_receipt_r4.md").write_text(summary)
PY

[[ "$(m448r4_sha "${m448r4_docs359}")" == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4" ]] || exit 30
printf '%s\n' 'PASS_M448R4_PRELAYOUT_STDCELL_SELECTED_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER' >"${m448r4_run}/RUN_COMPLETE_R4.txt"
printf '%s\n' \
    'R4_RUN_MANIFEST.sha256 is generated from the R4 run directory with relative ./ paths.' \
    'The ./work subtree is pruned; R4 manifest and seal files are excluded from manifest self-hashing.' \
    'Minimum entries: 30. The stdin target - is forbidden. Verification is fail closed.' \
    >"${m448r4_run}/r4_manifest_contract.txt"

(
    cd "${m448r4_run}"
    find . -path './work' -prune -o -type f \
        ! -name 'R4_RUN_MANIFEST.sha256' \
        ! -name 'R4_RUN_MANIFEST.seal.sha256' -print0 \
        | sort -z \
        | xargs -0 -r sha256sum >R4_RUN_MANIFEST.sha256
)

python3 - "${m448r4_run}" <<'PY'
import sys
from pathlib import Path

run = Path(sys.argv[1])
manifest = run / "R4_RUN_MANIFEST.sha256"
lines = manifest.read_text().splitlines()
if len(lines) < 30:
    raise SystemExit(f"M448R4 manifest is too small: {len(lines)}")
targets = []
for line in lines:
    fields = line.split(maxsplit=1)
    if len(fields) != 2:
        raise SystemExit(f"M448R4 malformed manifest line: {line}")
    target = fields[1].lstrip("*")
    if target == "-":
        raise SystemExit("M448R4 manifest contains forbidden stdin target -")
    if not target.startswith("./"):
        raise SystemExit(f"M448R4 manifest target is not relative ./ path: {target}")
    targets.append(target)
required = {
    "./m448r3_m431_m438_prelayout_stdcell_ptpx_receipt_r3.json",
    "./m448r4_m431_m438_prelayout_stdcell_ptpx_receipt_r4.json",
    "./power_call_ledger.txt",
    "./reports/ptpx_power_primary_100ps.rpt",
    "./reports/ptpx_power_sensitivity_050ps.rpt",
    "./reports/ptpx_power_sensitivity_200ps.rpt",
    "./reports/saif_annotation_summary.rpt",
    "./reports/switching_coverage.rpt",
    "./contract_r4.json",
    "./RUN_COMPLETE_R4.txt",
}
missing = sorted(required - set(targets))
if missing:
    raise SystemExit(f"M448R4 manifest is missing required files: {missing}")
PY

(
    cd "${m448r4_run}"
    sha256sum -c R4_RUN_MANIFEST.sha256
    sha256sum R4_RUN_MANIFEST.sha256 >R4_RUN_MANIFEST.seal.sha256
    sha256sum -c R4_RUN_MANIFEST.seal.sha256
)

[[ "$(m448r4_sha "${m448r4_docs359}")" == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4" ]] || exit 31
m448r4_complete=1
echo "M448R4 prelayout standard-cell PTPX complete at ${m448r4_run}"
