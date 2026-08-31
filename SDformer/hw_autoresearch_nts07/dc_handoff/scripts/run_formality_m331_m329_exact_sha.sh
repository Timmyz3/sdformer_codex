#!/usr/bin/env bash
set -euo pipefail

m331_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m331_hw="$(cd "${m331_dc_root}/.." && pwd)"
m331_source="${m331_dc_root}/runs/m329_m321_hold_guard_dc_3p000ns_r1b_20260825"
m331_run="${M331_FORMALITY_RUN:-${m331_dc_root}/runs/m331_m321_to_m329_formality_r1_20260825}"
m331_fm="/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell"
m331_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m331_rtl="${m331_hw}/rtl_m321/m321_near_match16_tau01_tournament2.sv"
m331_filelist="${m331_dc_root}/filelists/date_m321_near_match16_tau01_tournament2_rtl.f"
m331_netlist="${m331_source}/netlist/m321_near_match16_tau01_tournament2_mapped.v"
m331_svf="${m331_source}/netlist/m321_near_match16_tau01_tournament2.svf"
m331_tcl="${m331_dc_root}/scripts/run_formality_m331_m329_exact_sha.tcl"
m331_contract="${m331_hw}/contracts/m331_m321_rtl_to_m329_netlist_formality_contract_r1_20260825.json"

m331_sha() { sha256sum "$1" | awk '{print $1}'; }
m331_expect() { [[ -f "$1" && "$(m331_sha "$1")" == "$2" ]] || exit 3; }
[[ ! -e "${m331_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/(fm_shell|fm_shell_exec)( |$)' >/dev/null 2>&1; then exit 4; fi
m331_expect "${m331_fm}" aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m331_expect "${m331_lib}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m331_expect "${m331_rtl}" e54fd31a0beca33c18938e8241e55a07939dc1ff924dce8b81ccbb3a57e242e2
m331_expect "${m331_filelist}" ea57b3fb731ccb1fa7452060a3a39647e97e4dcddcabd075c899b420a99d12e4
m331_expect "${m331_netlist}" fe7e6db0c81107dff81425198d85935d79c9f8473c21671a8d697ade6b2c5b1e
m331_expect "${m331_svf}" 1760d052b6890912231c422f78d25754a5abd99c9083f6c82a1bd6d9083ae546
m331_expect "${m331_tcl}" 7d3ad9dac57cf63c664f1be7fb3c8d6b44711830772adfcc5b934c1cd9194b8e
m331_expect "${m331_contract}" d6304a9db0699b561a50c3602cef185de7495c2217727f3efaddd26baf5541a1
m331_expect "${m331_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m331_run}/reports" "${m331_run}/work"
cp "${m331_contract}" "${m331_run}/contract.json"
sha256sum "${m331_rtl}" "${m331_filelist}" "${m331_netlist}" "${m331_svf}" \
    "${m331_lib}" "${m331_tcl}" "${m331_contract}" > "${m331_run}/input_sha256.txt"
export DESIGN_NAME=m321_near_match16_tau01_tournament2
export SNAPSHOT_ROOT="${m331_hw}"
export RTL_FILELIST="${m331_filelist}"
export LIB_DB="${m331_lib}"
export MAPPED_NETLIST="${m331_netlist}"
export SVF_FILE="${m331_svf}"
export OUTPUT_DIR="${m331_run}"
"${m331_fm}" -version > "${m331_run}/formality.version.raw.log" 2>&1
set +e
(cd "${m331_run}/work" && "${m331_fm}" -f "${m331_tcl}") \
    > "${m331_run}/formality.raw.log" 2>&1
m331_rc=$?
set -e
echo "${m331_rc}" > "${m331_run}/formality.rc"
[[ "${m331_rc}" -eq 0 ]]
[[ "$(grep -xc 'M331_M329_FORMALITY_INTERNAL_COMPLETE=PASS' \
    "${m331_run}/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
grep -q 'Verification SUCCEEDED' "${m331_run}/reports/formality_status.rpt"
grep -q 'No failing compare points' "${m331_run}/reports/formality_failing.rpt"
grep -q 'No aborted compare points' "${m331_run}/reports/formality_aborted.rpt"
grep -q 'No unverified compare points' "${m331_run}/reports/formality_unverified.rpt"
! grep -Eq '^(Error|Fatal):' "${m331_run}/formality.raw.log"
for m331_pair in \
    "${m331_rtl}:e54fd31a0beca33c18938e8241e55a07939dc1ff924dce8b81ccbb3a57e242e2" \
    "${m331_netlist}:fe7e6db0c81107dff81425198d85935d79c9f8473c21671a8d697ade6b2c5b1e" \
    "${m331_svf}:1760d052b6890912231c422f78d25754a5abd99c9083f6c82a1bd6d9083ae546" \
    "${m331_contract}:d6304a9db0699b561a50c3602cef185de7495c2217727f3efaddd26baf5541a1"; do
    m331_expect "${m331_pair%%:*}" "${m331_pair##*:}"
done
{
    echo "status=PASS_EXACT_SHA_M331_M321_TO_M329_FORMALITY"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
    echo "scope=M321_RTL_TO_M329_HOLD_GUARDED_PRELAYOUT_NETLIST"
} > "${m331_run}/RUN_COMPLETE.txt"
(
  cd "${m331_run}"
  find . -type f ! -path './work/*' ! -name output.sha256 \
      ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
      > output.sha256
  sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "PASS_EXACT_SHA_M331_M321_TO_M329_FORMALITY run=${m331_run}"
