#!/usr/bin/env bash
set -euo pipefail

m421_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m421_hw="$(cd "${m421_dc_root}/.." && pwd)"
m421_runner="$(realpath "${BASH_SOURCE[0]}")"
m421_run="${M421_DIAGNOSTIC_RUN:-${m421_hw}/results/m421_m420_dual_formality_independent_hammer_r1_20260826/independent_formality}"
m421_fm="/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell"
m421_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m421_tcl="${m421_dc_root}/scripts/run_formality_m421_unread_diagnostic.tcl"
m421_contract="${m421_hw}/contracts/m421_m420_dual_formality_independent_hammer_contract_r1_20260826.json"
m421_serial_fl="${m421_dc_root}/filelists/date_m412_m405_selected_slice_rtl.f"
m421_balanced_fl="${m421_dc_root}/filelists/date_m414_balanced_selected_slice_rtl.f"
m421_netlist="${m421_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/netlist/m405_q32_elastic_selected_slice_mapped.v"
m421_svf="${m421_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/netlist/m405_q32_elastic_selected_slice.svf"
m421_m420="${m421_dc_root}/runs/m420_m414_dual_formality_r1_20260826"
m421_m416="${m421_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826"
m421_m417="${m421_hw}/results/m417_m416_balanced_dc_independent_hammer_r1_20260826"

m421_sha() { sha256sum "$1" | awk '{print $1}'; }
m421_expect() {
    local m421_path=$1
    local m421_expected=$2
    [[ -f "${m421_path}" ]] || exit 3
    [[ "$(m421_sha "${m421_path}")" == "${m421_expected}" ]] || exit 3
}
m421_check_proof() {
    local m421_dir=$1
    [[ "$(cat "${m421_dir}/formality.rc")" == "0" ]]
    grep -q 'Verification SUCCEEDED' "${m421_dir}/reports/formality_status.rpt"
    grep -q '5368 Passing compare points' "${m421_dir}/reports/formality_status.rpt"
    grep -q 'No failing compare points' "${m421_dir}/reports/formality_failing.rpt"
    grep -q 'No aborted compare points' "${m421_dir}/reports/formality_aborted.rpt"
    grep -q 'No unverified compare points' "${m421_dir}/reports/formality_unverified.rpt"
    ! grep -Eq '^(Error|Fatal):' "${m421_dir}/formality.raw.log"
}

[[ ! -e "${m421_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/(fm_shell|fm_shell_exec|common_shell_exec.*fm_shell)( |$)' >/dev/null 2>&1; then
    exit 4
fi
cd "${m421_hw}"
m421_expect "${m421_fm}" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m421_expect "${m421_lib}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m421_expect "${m421_tcl}" 7d34bb22534a2f743c325e282baaa79f00e76eb389562e93d33ec0ce1a6eaa8e
m421_expect "${m421_contract}" a52b7942cdce1d799c1c03c6f34081aab441598316b560ed8da6641268894fe4
m421_expect "${m421_serial_fl}" e0ef128a2ae9e351ecd98c45c19e9706e983cb9cef82913600febf48dae0f58e
m421_expect "${m421_balanced_fl}" 5e93db53c751df3ca3c4cefec8376434f31cbe6561b15e0848ec2d872adc1f92
m421_expect "${m421_netlist}" 4b07e83eba88508da3fc1aa27187b3fa8ca03a633b165ea68641bcd26b969fe2
m421_expect "${m421_svf}" 8d332db9efc87f70d266b01612a5f8a29b63c4168043d2e0dd1c46b936e7edaf
m421_expect "${m421_m420}/output.seal.sha256" cf216915f3f0c8e1ee4e894734e81337d81baf70b36cbd9b51ee23b381e723d7
m421_expect "${m421_m420}/m420_m414_dual_formality_receipt_r1.json" 4df80e3964c0ced618dedaa67776e21e283c6026c241566ddc59427479c08949
m421_expect "${m421_m416}/evidence_manifest.seal.sha256" 40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79
m421_expect "${m421_m417}/SHA256SUMS.seal.sha256" 7b309ae3b9f66c793c1a862e67c346115c3dce28ffd80b0781e8c3afea38d7fc
m421_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m421_run}/serial_to_balanced/reports" \
    "${m421_run}/serial_to_balanced/work" \
    "${m421_run}/balanced_to_netlist/reports" \
    "${m421_run}/balanced_to_netlist/work"
(
    cd "${m421_m420}"
    sha256sum --strict -c output.seal.sha256
    sha256sum --strict -c output.sha256
) >"${m421_run}/m420_seal_check.log" 2>&1
(
    cd "${m421_m416}"
    sha256sum --strict -c evidence_manifest.sha256
    sha256sum --strict -c evidence_manifest.seal.sha256
) >"${m421_run}/m416_seal_check.log" 2>&1
(
    cd "${m421_m417}"
    sha256sum --strict -c SHA256SUMS
    sha256sum --strict -c SHA256SUMS.seal.sha256
) >"${m421_run}/m417_seal_check.log" 2>&1
cp "${m421_contract}" "${m421_run}/contract.json"
sha256sum "${m421_tcl}" "${m421_contract}" "${m421_serial_fl}" \
    "${m421_balanced_fl}" "${m421_netlist}" "${m421_svf}" "${m421_lib}" \
    "${m421_m420}/output.seal.sha256" "${m421_m416}/evidence_manifest.seal.sha256" \
    "${m421_m417}/SHA256SUMS.seal.sha256" docs/359_DATE终局冻结_20260813.md \
    >"${m421_run}/input_sha256.txt"
"${m421_fm}" -version >"${m421_run}/formality.version.raw.log" 2>&1

export DESIGN_NAME=m405_q32_elastic_selected_slice
export SNAPSHOT_ROOT="${m421_hw}"
export REFERENCE_RTL_FILELIST="${m421_serial_fl}"
export IMPLEMENTATION_KIND=rtl
export IMPLEMENTATION_RTL_FILELIST="${m421_balanced_fl}"
export OUTPUT_DIR="${m421_run}/serial_to_balanced"
set +e
(cd "${m421_run}/serial_to_balanced/work" && "${m421_fm}" -f "${m421_tcl}") \
    >"${m421_run}/serial_to_balanced/formality.raw.log" 2>&1
m421_r2r_rc=$?
set -e
echo "${m421_r2r_rc}" >"${m421_run}/serial_to_balanced/formality.rc"
[[ "${m421_r2r_rc}" -eq 0 ]]
m421_check_proof "${m421_run}/serial_to_balanced"

export REFERENCE_RTL_FILELIST="${m421_balanced_fl}"
export IMPLEMENTATION_KIND=netlist
export LIB_DB="${m421_lib}"
export MAPPED_NETLIST="${m421_netlist}"
export SVF_FILE="${m421_svf}"
export OUTPUT_DIR="${m421_run}/balanced_to_netlist"
set +e
(cd "${m421_run}/balanced_to_netlist/work" && "${m421_fm}" -f "${m421_tcl}") \
    >"${m421_run}/balanced_to_netlist/formality.raw.log" 2>&1
m421_r2n_rc=$?
set -e
echo "${m421_r2n_rc}" >"${m421_run}/balanced_to_netlist/formality.rc"
[[ "${m421_r2n_rc}" -eq 0 ]]
m421_check_proof "${m421_run}/balanced_to_netlist"

sha256sum "${m421_runner}" >"${m421_run}/runner_sha256.txt"
echo 'status=PASS_M421_INDEPENDENT_DUAL_FORMALITY_DIAGNOSTIC' >"${m421_run}/RUN_COMPLETE.txt"
echo "PASS_M421_INDEPENDENT_DUAL_FORMALITY_DIAGNOSTIC run=${m421_run}"
