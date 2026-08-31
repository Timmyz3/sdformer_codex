#!/usr/bin/env bash
set -euo pipefail

m325_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m325_hw="$(cd "${m325_dc_root}/.." && pwd)"
m325_dc_run="${m325_dc_root}/runs/m322_m321_tournament2_logic_only_dc_3p000ns_r1_20260825"
m325_run="${M325_FORMALITY_RUN:-${m325_dc_root}/runs/m325_m321_formality_r1_20260825}"
m325_fm="/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell"
m325_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m325_rtl="${m325_hw}/rtl_m321/m321_near_match16_tau01_tournament2.sv"
m325_filelist="${m325_dc_root}/filelists/date_m321_near_match16_tau01_tournament2_rtl.f"
m325_netlist="${m325_dc_run}/netlist/m321_near_match16_tau01_tournament2_mapped.v"
m325_svf="${m325_dc_run}/netlist/m321_near_match16_tau01_tournament2.svf"
m325_tcl="${m325_dc_root}/scripts/run_formality_m325_m321_exact_sha.tcl"
m325_contract="${m325_hw}/contracts/m325_m321_rtl_to_m322_netlist_formality_contract_r1_20260825.json"
m325_docs="${m325_hw}/docs/359_DATE终局冻结_20260813.md"

m325_sha() { sha256sum "$1" | awk '{print $1}'; }
m325_expect() {
    local m325_path=$1
    local m325_expected=$2
    [[ -f "${m325_path}" && ! -L "${m325_path}" ]] || exit 3
    [[ "$(m325_sha "${m325_path}")" == "${m325_expected}" ]] || exit 3
}

[[ ! -e "${m325_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/(fm_shell|fm_shell_exec)( |$)' >/dev/null 2>&1; then
    exit 4
fi

m325_expect "${m325_fm}" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m325_expect "${m325_lib}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m325_expect "${m325_rtl}" e54fd31a0beca33c18938e8241e55a07939dc1ff924dce8b81ccbb3a57e242e2
m325_expect "${m325_filelist}" ea57b3fb731ccb1fa7452060a3a39647e97e4dcddcabd075c899b420a99d12e4
m325_expect "${m325_netlist}" f8bb8c85a6459b0f5d7bde9b45ea206b829014421b682fe73574c025b09dbe13
m325_expect "${m325_svf}" 230824b94e39b999cacfe84e21b17000611c5253708a431d227a77dfc04db541
m325_expect "${m325_tcl}" 50afcd77db48773f4db693090b1297d9a74c6dc03bd12c54d86593a8ae87ca69
m325_expect "${m325_contract}" 4964fb979cd13a9fca3f569dfaad3c859a6e247d13da20bcb303bf193334861a
m325_expect "${m325_docs}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m325_run}/snapshot/rtl_m321" \
    "${m325_run}/snapshot/dc_handoff/filelists" \
    "${m325_run}/snapshot/netlist" "${m325_run}/snapshot/library" \
    "${m325_run}/reports" "${m325_run}/work"
cp "${m325_rtl}" "${m325_run}/snapshot/rtl_m321/"
cp "${m325_filelist}" "${m325_run}/snapshot/dc_handoff/filelists/"
cp "${m325_netlist}" "${m325_run}/snapshot/netlist/"
cp "${m325_svf}" "${m325_run}/snapshot/netlist/"
cp "${m325_lib}" "${m325_run}/snapshot/library/"
cp "${m325_contract}" "${m325_run}/contract.json"

(
    cd "${m325_run}/snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum \
        > ../snapshot.sha256
    sha256sum --strict -c ../snapshot.sha256 \
        > ../snapshot_check.raw.log 2>&1
)
find "${m325_run}/snapshot" -type f -exec chmod 0444 {} +
find "${m325_run}/snapshot" -type d -exec chmod 0555 {} +

{
    echo "status=RUNNING_NOT_CITABLE"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "power_or_energy_admitted=false"
    echo "scope=M321_RTL_TO_M322_3NS_PREMACRO_DC_NETLIST_FORMALITY"
} > "${m325_run}/RUN_IN_PROGRESS.txt"
"${m325_fm}" -version > "${m325_run}/formality.version.raw.log" 2>&1

export DESIGN_NAME=m321_near_match16_tau01_tournament2
export SNAPSHOT_ROOT="${m325_run}/snapshot"
export RTL_FILELIST="${m325_run}/snapshot/dc_handoff/filelists/date_m321_near_match16_tau01_tournament2_rtl.f"
export LIB_DB="${m325_run}/snapshot/library/$(basename "${m325_lib}")"
export MAPPED_NETLIST="${m325_run}/snapshot/netlist/$(basename "${m325_netlist}")"
export SVF_FILE="${m325_run}/snapshot/netlist/$(basename "${m325_svf}")"
export OUTPUT_DIR="${m325_run}"

echo "${m325_fm} -f ${m325_tcl}" > "${m325_run}/formality.command.txt"
set +e
(cd "${m325_run}/work" && "${m325_fm}" -f "${m325_tcl}") \
    > "${m325_run}/formality.raw.log" 2>&1
m325_rc=$?
set -e
echo "${m325_rc}" > "${m325_run}/formality.rc"
[[ "${m325_rc}" -eq 0 ]]
[[ "$(grep -xc 'M325_M321_FORMALITY_INTERNAL_COMPLETE=PASS' \
    "${m325_run}/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
grep -q 'Verification SUCCEEDED' "${m325_run}/reports/formality_status.rpt"
grep -q 'No failing compare points' "${m325_run}/reports/formality_failing.rpt"
grep -q 'No aborted compare points' "${m325_run}/reports/formality_aborted.rpt"
grep -q 'No unverified compare points' "${m325_run}/reports/formality_unverified.rpt"
! grep -Eq '^(Error|Fatal):' "${m325_run}/formality.raw.log"

mv "${m325_run}/RUN_IN_PROGRESS.txt" "${m325_run}/RUN_BOOTSTRAP_RECORD.txt"
{
    echo "status=PASS_EXACT_SHA_M325_M321_FORMALITY"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "power_or_energy_admitted=false"
    echo "scope=M321_RTL_TO_M322_3NS_PREMACRO_DC_NETLIST_FORMALITY"
} > "${m325_run}/RUN_COMPLETE.txt"
(
    cd "${m325_run}"
    find . -type f ! -path './work/*' ! -name output.sha256 \
        ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
        > output.sha256
    sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "M325_M321_FORMALITY=PASS run=${m325_run}"
