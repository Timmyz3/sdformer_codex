#!/usr/bin/env bash
set -euo pipefail

m420_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m420_hw="$(cd "${m420_dc_root}/.." && pwd)"
m420_runner="$(realpath "${BASH_SOURCE[0]}")"
m420_run="${M420_FORMALITY_RUN:-${m420_dc_root}/runs/m420_m414_dual_formality_r1_20260826}"
m420_fm="/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell"
m420_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m420_serial_fl="${m420_dc_root}/filelists/date_m412_m405_selected_slice_rtl.f"
m420_balanced_fl="${m420_dc_root}/filelists/date_m414_balanced_selected_slice_rtl.f"
m420_netlist="${m420_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/netlist/m405_q32_elastic_selected_slice_mapped.v"
m420_svf="${m420_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/netlist/m405_q32_elastic_selected_slice.svf"
m420_r2r_tcl="${m420_dc_root}/scripts/run_formality_m420_serial_to_balanced_rtl_exact_sha.tcl"
m420_r2n_tcl="${m420_dc_root}/scripts/run_formality_m420_balanced_rtl_to_m416_netlist_exact_sha.tcl"
m420_contract="${m420_hw}/contracts/m420_m414_dual_formality_contract_r1_20260826.json"
m420_top="m405_q32_elastic_selected_slice"

m420_sha() { sha256sum "$1" | awk '{print $1}'; }
m420_expect() {
    local m420_path=$1
    local m420_expected=$2
    [[ -f "${m420_path}" ]] || exit 3
    [[ "$(m420_sha "${m420_path}")" == "${m420_expected}" ]] || exit 3
}
m420_check_proof() {
    local m420_dir=$1
    local m420_marker=$2
    [[ "$(grep -xc "${m420_marker}" "${m420_dir}/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
    grep -q 'Verification SUCCEEDED' "${m420_dir}/reports/formality_status.rpt"
    grep -Eq '[1-9][0-9]* Passing compare points' "${m420_dir}/reports/formality_status.rpt"
    grep -q 'No unmatched points' "${m420_dir}/reports/formality_unmatched.rpt"
    grep -q 'No failing compare points' "${m420_dir}/reports/formality_failing.rpt"
    grep -q 'No aborted compare points' "${m420_dir}/reports/formality_aborted.rpt"
    grep -q 'No unverified compare points' "${m420_dir}/reports/formality_unverified.rpt"
    ! grep -Eq '^(Error|Fatal):' "${m420_dir}/formality.raw.log"
}

[[ ! -e "${m420_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/(fm_shell|fm_shell_exec|common_shell_exec.*fm_shell)( |$)' >/dev/null 2>&1; then
    exit 4
fi
cd "${m420_hw}"
m420_expect "${m420_fm}" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m420_expect "${m420_lib}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m420_expect "${m420_serial_fl}" e0ef128a2ae9e351ecd98c45c19e9706e983cb9cef82913600febf48dae0f58e
m420_expect "${m420_balanced_fl}" 5e93db53c751df3ca3c4cefec8376434f31cbe6561b15e0848ec2d872adc1f92
m420_expect rtl_m405/m405_q32_serial16_zero_stop_controller.sv f412ab817eb29ab303da9ec011379a853efc567fa5a6a29a73fced52cf503b42
m420_expect rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
m420_expect rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv 819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744
m420_expect rtl_m405/m405_q32_elastic_selected_slice.sv 91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd
m420_expect "${m420_netlist}" 4b07e83eba88508da3fc1aa27187b3fa8ca03a633b165ea68641bcd26b969fe2
m420_expect "${m420_svf}" 8d332db9efc87f70d266b01612a5f8a29b63c4168043d2e0dd1c46b936e7edaf
m420_expect "${m420_r2r_tcl}" 99a6f8abe4cf9b383d20531e03c96a1acab85bfb99d6697230414fec336b5e02
m420_expect "${m420_r2n_tcl}" 4e48713bb9804a55e5bd15ea65ef5d28a132fa0aeebbb268bdc318285a6f1119
m420_expect "${m420_contract}" ce1dd1806bc81c74b04a47be3dc7306b0fc911780ecdb72a40bcd2f907b7ae7f
m420_expect dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/m416_m414_balanced_selected_slice_dc_receipt_r1.json bedb903268d3e94c858e8177a383a46f35427cd9a1bdad3ad9ad398b4bc85c02
m420_expect dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79
m420_expect results/m417_m416_balanced_dc_independent_hammer_r1_20260826/m417_m416_balanced_dc_independent_hammer_review_r1.json e8dcf24620bf6fa74e84adf8a60fd8f4245084675b653b4633a5b942c6de368a
m420_expect results/m417_m416_balanced_dc_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 7b309ae3b9f66c793c1a862e67c346115c3dce28ffd80b0781e8c3afea38d7fc
m420_expect results/m414_q32_balanced16_vcs_r1_20260826/m414_q32_balanced16_zero_stop_vcs_receipt_r1.json 032bedb7ee15080083cacceee49b7ecb0a2fe92ccb456aceeebc8e86928a183d
m420_expect results/m415_m414_balanced_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 98db30a32a11957795bc4a1e937864718c486d11eb0b77b25aecfdd852278167
m420_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m420_run}/serial_to_balanced/reports" \
    "${m420_run}/serial_to_balanced/work" \
    "${m420_run}/balanced_to_netlist/reports" \
    "${m420_run}/balanced_to_netlist/work"
m420_complete=0
trap 'm420_rc=$?; if [[ ${m420_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m420_rc}" >"${m420_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
(
    cd dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826
    sha256sum -c evidence_manifest.sha256
    sha256sum -c evidence_manifest.seal.sha256
) >"${m420_run}/m416_seal_check.log" 2>&1
(
    cd results/m417_m416_balanced_dc_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m420_run}/m417_seal_check.log" 2>&1
cp "${m420_contract}" "${m420_run}/contract.json"
sha256sum "${m420_serial_fl}" "${m420_balanced_fl}" \
    rtl_m405/m405_q32_serial16_zero_stop_controller.sv \
    rtl_m414/m414_q32_balanced16_zero_stop_controller.sv \
    rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv \
    rtl_m405/m405_q32_elastic_selected_slice.sv \
    "${m420_netlist}" "${m420_svf}" "${m420_lib}" \
    "${m420_r2r_tcl}" "${m420_r2n_tcl}" "${m420_contract}" \
    dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 \
    results/m417_m416_balanced_dc_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m420_run}/input_sha256.txt"
"${m420_fm}" -version >"${m420_run}/formality.version.raw.log" 2>&1

export DESIGN_NAME="${m420_top}"
export SNAPSHOT_ROOT="${m420_hw}"
export REFERENCE_RTL_FILELIST="${m420_serial_fl}"
export IMPLEMENTATION_RTL_FILELIST="${m420_balanced_fl}"
export OUTPUT_DIR="${m420_run}/serial_to_balanced"
set +e
(cd "${m420_run}/serial_to_balanced/work" && "${m420_fm}" -f "${m420_r2r_tcl}") \
    >"${m420_run}/serial_to_balanced/formality.raw.log" 2>&1
m420_r2r_rc=$?
set -e
echo "${m420_r2r_rc}" >"${m420_run}/serial_to_balanced/formality.rc"
[[ "${m420_r2r_rc}" -eq 0 ]]
m420_check_proof "${m420_run}/serial_to_balanced" \
    'M420_SERIAL_TO_BALANCED_RTL_FORMALITY_INTERNAL_COMPLETE=PASS'

export RTL_FILELIST="${m420_balanced_fl}"
export LIB_DB="${m420_lib}"
export MAPPED_NETLIST="${m420_netlist}"
export SVF_FILE="${m420_svf}"
export OUTPUT_DIR="${m420_run}/balanced_to_netlist"
set +e
(cd "${m420_run}/balanced_to_netlist/work" && "${m420_fm}" -f "${m420_r2n_tcl}") \
    >"${m420_run}/balanced_to_netlist/formality.raw.log" 2>&1
m420_r2n_rc=$?
set -e
echo "${m420_r2n_rc}" >"${m420_run}/balanced_to_netlist/formality.rc"
[[ "${m420_r2n_rc}" -eq 0 ]]
m420_check_proof "${m420_run}/balanced_to_netlist" \
    'M420_BALANCED_RTL_TO_M416_NETLIST_FORMALITY_INTERNAL_COMPLETE=PASS'

python3 - "${m420_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
def points(name):
    text = (run / name / "reports/formality_status.rpt").read_text(
        encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) Passing compare points", text)
    if not match:
        raise SystemExit("missing passing compare-point count for " + name)
    return int(match.group(1))

r2r = points("serial_to_balanced")
r2n = points("balanced_to_netlist")
receipt = {
    "schema": "m420_m414_dual_formality_receipt_v1",
    "status": "PASS_M420_SERIAL_TO_BALANCED_AND_BALANCED_TO_M416_NETLIST_FORMALITY",
    "tool": "Synopsys Formality V-2023.12-SP3",
    "proofs": {
        "serial_rtl_to_balanced_rtl": {
            "passing_compare_points": r2r, "failing_compare_points": 0,
            "aborted_compare_points": 0, "unverified_compare_points": 0,
            "unmatched_points": 0, "all_state_equivalence": True},
        "balanced_rtl_to_m416_mapped_netlist": {
            "passing_compare_points": r2n, "failing_compare_points": 0,
            "aborted_compare_points": 0, "unverified_compare_points": 0,
            "unmatched_points": 0, "rtl_to_netlist_equivalence": True},
    },
    "functional_context": {
        "full_runtime_rows": 51840000,
        "m401_matcher_cycles": 67912100,
        "task_ledger_change": 0,
        "accuracy_change": False,
    },
    "claim_boundary": {
        "serial_to_balanced_all_state_equivalence": True,
        "balanced_rtl_to_m416_netlist_equivalence": True,
        "primetime": False, "physical_sram": False,
        "physical_timing": False, "saif_or_ptpx": False,
        "energy": False, "new_cycle_speedup": False,
        "system_speedup": False, "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(run / "m420_m414_dual_formality_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
    echo 'status=PASS_M420_DUAL_FORMALITY'
    echo 'serial_to_balanced_all_state_equivalence=true'
    echo 'balanced_rtl_to_m416_netlist_equivalence=true'
    echo 'task_ledger_change=0'
    echo 'new_cycle_speedup=false'
    echo 'primetime=false'
    echo 'power=false'
    echo 'system_speedup=false'
    echo 'paper_ppa_ready=false'
    echo 'headline=false'
} >"${m420_run}/RUN_COMPLETE.txt"
sha256sum "${m420_runner}" >"${m420_run}/runner_sha256.txt"
(
    cd "${m420_run}"
    find . -type f ! -path './serial_to_balanced/work/*' \
        ! -path './balanced_to_netlist/work/*' \
        ! -name output.sha256 ! -name output.seal.sha256 \
        ! -name output_check.raw.log -print0 | sort -z | \
        xargs -0 sha256sum >output.sha256
    sha256sum --strict -c output.sha256 >output_check.raw.log 2>&1
    sha256sum output.sha256 >output.seal.sha256
)
m420_complete=1
echo "PASS_M420_DUAL_FORMALITY run=${m420_run}"
