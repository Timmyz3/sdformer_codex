#!/usr/bin/env bash
set -euo pipefail

m389_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m389_hw="$(cd "${m389_dc_root}/.." && pwd)"
m389_runner="$(realpath "${BASH_SOURCE[0]}")"
m389_source="${m389_dc_root}/runs/m387_m384_active_descriptor_controller_dc_3p000ns_r1b_20260826"
m389_run="${M389_FORMALITY_RUN:-${m389_dc_root}/runs/m389_m384_to_m387r1b_formality_r1_20260826}"
m389_fm="/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell"
m389_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m389_rtl="${m389_hw}/rtl_m384/m384_active_descriptor_streaming_controller.sv"
m389_filelist="${m389_dc_root}/filelists/date_m384_active_descriptor_streaming_controller_rtl.f"
m389_netlist="${m389_source}/netlist/m384_active_descriptor_streaming_controller_mapped.v"
m389_svf="${m389_source}/netlist/m384_active_descriptor_streaming_controller.svf"
m389_tcl="${m389_dc_root}/scripts/run_formality_m389_m384_exact_sha.tcl"
m389_contract="${m389_hw}/contracts/m389_m384_rtl_to_m387r1b_netlist_formality_contract_r1_20260826.json"

m389_sha() { sha256sum "$1" | awk '{print $1}'; }
m389_expect() { [[ -f "$1" && "$(m389_sha "$1")" == "$2" ]] || exit 3; }
[[ ! -e "${m389_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/(fm_shell|fm_shell_exec|common_shell_exec.*fm_shell)( |$)' >/dev/null 2>&1; then
    exit 4
fi

m389_expect "${m389_fm}" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m389_expect "${m389_lib}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m389_expect "${m389_rtl}" 15f0e1d8aebfcb66ed58cefed988bde855a8b2a351e32c86beb2381a8c4e6b38
m389_expect "${m389_filelist}" c3db231e355357c138247c0c76a0352d80d5574a863988fb9af2746be9c37467
m389_expect "${m389_netlist}" fe478950d336eb28176125572983a02fb4f47fd97fcfeb28dedfb6063df03efd
m389_expect "${m389_svf}" 1a88a6c84552a63cb73d92219f0c2387b771b2129bf786502117305eeaaab67c
m389_expect "${m389_tcl}" 14c12f3ec36a1e9605dee8dacde87474180f26cbc9265aa3f0c5ae7c5bf16349
m389_expect "${m389_contract}" 24c8b7efe22ef459a2f1a90afdd8ea6a24fdb20eb45e36d67f142bc0d5ae707b
m389_expect "${m389_source}/m387_m384_active_descriptor_controller_logic_only_dc_receipt_r1b.json" 896eba1d373fa8d8bb371a19e097047dabbfe391b39838887d8dd9b785b77b2b
m389_expect "${m389_source}/evidence_manifest.seal.sha256" c6e86050acb21576a5cd5073573a4941085b91295d57282aa126af1b46d0ce5f
m389_expect "${m389_hw}/results/m388_m387_m384_controller_dc_independent_hammer_r1_20260826/m388_m387_m384_controller_dc_independent_hammer_review_r1.json" b34137151cd608f34771ac2b7998e028f16f3191580bde2a67e915fecbd264e8
m389_expect "${m389_hw}/results/m388_m387_m384_controller_dc_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256" 9092be23f300c3475ed3547ce8643a6953bbeecdce698c6e91fe439ba3796ef3
m389_expect "${m389_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m389_run}/reports" "${m389_run}/work"
m389_complete=0
trap 'm389_rc=$?; if [[ ${m389_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m389_rc}" >"${m389_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cp "${m389_contract}" "${m389_run}/contract.json"
sha256sum "${m389_rtl}" "${m389_filelist}" "${m389_netlist}" \
    "${m389_svf}" "${m389_lib}" "${m389_tcl}" "${m389_contract}" \
    "${m389_source}/evidence_manifest.seal.sha256" \
    "${m389_hw}/results/m388_m387_m384_controller_dc_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256" \
    >"${m389_run}/input_sha256.txt"

export DESIGN_NAME=m384_active_descriptor_streaming_controller
export SNAPSHOT_ROOT="${m389_hw}"
export RTL_FILELIST="${m389_filelist}"
export LIB_DB="${m389_lib}"
export MAPPED_NETLIST="${m389_netlist}"
export SVF_FILE="${m389_svf}"
export OUTPUT_DIR="${m389_run}"
"${m389_fm}" -version >"${m389_run}/formality.version.raw.log" 2>&1
set +e
(cd "${m389_run}/work" && "${m389_fm}" -f "${m389_tcl}") \
    >"${m389_run}/formality.raw.log" 2>&1
m389_rc=$?
set -e
echo "${m389_rc}" >"${m389_run}/formality.rc"
[[ "${m389_rc}" -eq 0 ]]
[[ "$(grep -xc 'M389_M384_FORMALITY_INTERNAL_COMPLETE=PASS' \
    "${m389_run}/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
grep -q 'Verification SUCCEEDED' "${m389_run}/reports/formality_status.rpt"
grep -Eq '[1-9][0-9]* Passing compare points' "${m389_run}/reports/formality_status.rpt"
grep -q 'No unmatched points' "${m389_run}/reports/formality_unmatched.rpt"
grep -q 'No failing compare points' "${m389_run}/reports/formality_failing.rpt"
grep -q 'No aborted compare points' "${m389_run}/reports/formality_aborted.rpt"
grep -q 'No unverified compare points' "${m389_run}/reports/formality_unverified.rpt"
! grep -Eq '^(Error|Fatal):' "${m389_run}/formality.raw.log"

python3 - "${m389_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
status = (run / "reports/formality_status.rpt").read_text(
    encoding="utf-8", errors="replace")
match = re.search(r"(\d+) Passing compare points", status)
if not match:
    raise SystemExit("missing passing compare point count")
receipt = {
    "schema": "m389_m384_to_m387r1b_formality_receipt_v1",
    "status": "PASS_M389_M384_TO_M387R1B_FORMALITY",
    "tool": "Synopsys Formality V-2023.12-SP3",
    "passing_compare_points": int(match.group(1)),
    "failing_compare_points": 0,
    "aborted_compare_points": 0,
    "unverified_compare_points": 0,
    "unmatched_points": 0,
    "claim_boundary": {
        "rtl_to_mapped_netlist_equivalence": True,
        "physical_sram": False,
        "physical_timing": False,
        "primetime": False,
        "activity_backed_ptpx": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(run / "m389_m384_to_m387r1b_formality_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
    echo "status=PASS_EXACT_SHA_M389_M384_TO_M387R1B_FORMALITY"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "activity_backed_ptpx=false"
    echo "scope=M384_RTL_TO_M387R1B_HOLD_GUARDED_PRELAYOUT_NETLIST"
} >"${m389_run}/RUN_COMPLETE.txt"
sha256sum "${m389_runner}" >"${m389_run}/runner_sha256.txt"
(
  cd "${m389_run}"
  find . -type f ! -path './work/*' ! -name output.sha256 \
      ! -name output.seal.sha256 ! -name output_check.raw.log \
      -print0 | sort -z | xargs -0 sha256sum >output.sha256
  sha256sum --strict -c output.sha256 >output_check.raw.log 2>&1
  sha256sum output.sha256 >output.seal.sha256
)
m389_complete=1
echo "PASS_EXACT_SHA_M389_M384_TO_M387R1B_FORMALITY run=${m389_run}"
