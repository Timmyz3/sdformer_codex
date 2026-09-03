#!/usr/bin/env bash
# Additive two-axis C2 hold-repair campaign.  No EDA is authorized until a
# corrected M1893R2 source identity and a separate M1897 runner review exist.
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m1892_m1811_c2_fastmin_hold_repair_candidate.tcl"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M1811="${HW_ROOT}/dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830="${HW_ROOT}/reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1893R2="${HW_ROOT}/reviews/m1893r2_m1893_m1892_c2_fastmin_hold_source_identity_correction_r1_20260902"
M1897="${HW_ROOT}/reviews/m1897_m1896_c2_fastmin_hold_two_axis_runner_hammer_r1_20260902"

DESIGN=m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24
K8_DESIGN=${DESIGN}_ARCH_MODE0
K1X8_DESIGN=${DESIGN}_ARCH_MODE1
K8_DDC="${M1811}/k8/netlist/${DESIGN}.ddc"
K8_SDC="${M1811}/k8/netlist/${DESIGN}_mapped.sdc"
K1X8_DDC="${M1811}/k1x8/netlist/${DESIGN}.ddc"
K1X8_SDC="${M1811}/k1x8/netlist/${DESIGN}_mapped.sdc"

DC_SHELL=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
SLOW_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
FAST_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db

RESULT="${HW_ROOT}/dc_handoff/runs/m1896_m1892_m1811_c2_fastmin_hold_repair_two_axis_r1_20260902"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1896_m1892_m1811_c2_fastmin_hold_repair_two_axis_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1896_m1892_m1811_c2_fastmin_hold_repair_two_axis_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1896_m1892_m1811_c2_fastmin_hold_repair_two_axis_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 3; }
  [[ "$(sha_file "${path}")" == "${expected}" ]] || { echo "ERROR: SHA ${path}" >&2; exit 3; }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
}
seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
on_exit() {
  local rc=$?
  set +e
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

sha_exact b01b22661dbd3789984aa78eb86f6b996f41a398e749a8e874e917b070e9885f "${TCL}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact c2f2b7b538cccb39efb76dc3f524efd1777327a6732a7bd498d58cd208e43ad7 "${K8_DDC}"
sha_exact af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018 "${K8_SDC}"
sha_exact 7c73ef9ed0a2c224a006023fc46b136c7c15783b5df6bd085805130d57c2dfda "${K1X8_DDC}"
sha_exact 1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a "${K1X8_SDC}"
sha_exact 695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066 "${M1811}/SHA256SUMS"
sha_exact 04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b "${M1811}/SHA256SUMS.seal.sha256"
verify_dir_seal "${M1811}"
sha_exact 79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b "${M1830}/review.json"
sha_exact d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06 "${M1830}/SHA256SUMS"
sha_exact 0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d "${M1830}/SHA256SUMS.seal.sha256"
verify_dir_seal "${M1830}"

verify_dir_seal "${M1893R2}"
[[ -n "${M1896_EXPECTED_M1893R2_REVIEW_SHA256:-}" ]] || { echo "ERROR: M1893R2 pin absent" >&2; exit 3; }
sha_exact "${M1896_EXPECTED_M1893R2_REVIEW_SHA256}" "${M1893R2}/review.json"
grep -Fq 'PASS_M1893R2_M1893_M1892_C2_FASTMIN_HOLD_IDENTITY_CORRECTION__AUTHORIZE_RUNNER_AUTHORING_ONLY' "${M1893R2}/review.json"

verify_dir_seal "${M1897}"
[[ -n "${M1896_EXPECTED_RUNNER_REVIEW_SHA256:-}" ]] || { echo "ERROR: M1897 pin absent" >&2; exit 3; }
sha_exact "${M1896_EXPECTED_RUNNER_REVIEW_SHA256}" "${M1897}/review.json"
grep -Fq 'PASS_M1897_M1896_C2_FASTMIN_HOLD_TWO_AXIS_RUNNER_HAMMER__AUTHORIZE_ONE_ATTEMPT' "${M1897}/review.json"
[[ -n "${M1896_EXPECTED_RUNNER_SHA256:-}" ]] || { echo "ERROR: runner pin absent" >&2; exit 3; }
sha_exact "${M1896_EXPECTED_RUNNER_SHA256}" "${RUNNER}"
grep -Fq "\"runner_sha256\": \"${M1896_EXPECTED_RUNNER_SHA256}\"" "${M1897}/review.json"

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || {
  echo "ERROR: M1896 namespace is not fresh" >&2; exit 4;
}

# Reject any live same-UID synthesis/formal/timing process.
blocked=' dc_shell dc_shell-t pt_shell fm_shell icc2_shell common_shell_exec common_shell_exe '
for proc in /proc/[0-9]*; do
  [[ -r "${proc}/status" && -r "${proc}/comm" ]] || continue
  real_uid=''
  while IFS=$'\t' read -r key value rest; do
    [[ ${key} == 'Uid:' ]] && { real_uid=${value}; break; }
  done <"${proc}/status"
  [[ ${real_uid} == "${EUID}" ]] || continue
  comm=''; IFS= read -r comm <"${proc}/comm" || continue
  [[ " ${blocked} " != *" ${comm} "* ]] || { echo "ERROR: same-UID EDA collision ${comm}" >&2; exit 4; }
done

mem_available=0; commit_limit=0; committed_as=0
while IFS=' :' read -r key value unit; do
  case "${key}" in
    MemAvailable) mem_available=${value} ;;
    CommitLimit) commit_limit=${value} ;;
    Committed_AS) committed_as=${value} ;;
  esac
done </proc/meminfo
[[ ${mem_available} -ge 67108864 && $((commit_limit-committed_as)) -ge 33554432 ]] || {
  echo "ERROR: resource gate" >&2; exit 4;
}

"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f Design-Compiler >/dev/null
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M1896_ATTEMPT_CONSUMED\ndc_shell_runs=2\naxes=k8,k1x8\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1

axis_names=(k8 k1x8)
axis_designs=("${K8_DESIGN}" "${K1X8_DESIGN}")
axis_ddcs=("${K8_DDC}" "${K1X8_DDC}")
axis_sdcs=("${K8_SDC}" "${K1X8_SDC}")
axis_areas=(130822.775176 585534.971643)
axis_ceilings=(137363.9139348 614811.72022515)

for index in 0 1; do
  axis="${axis_names[$index]}"
  axis_dir="${WORK}/${axis}"
  mkdir -- "${axis_dir}"
  env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
    SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
    M1892_INPUT_DDC="${axis_ddcs[$index]}" \
    M1892_INPUT_SDC="${axis_sdcs[$index]}" \
    M1892_STD_SLOW_DB="${SLOW_DB}" M1892_STD_FAST_DB="${FAST_DB}" \
    M1892_OUTPUT_DIR="${axis_dir}" M1892_EXPECTED_DESIGN="${axis_designs[$index]}" \
    M1892_AXIS="${axis}" M1892_AREA_BASELINE_UM2="${axis_areas[$index]}" \
    M1892_AREA_CEILING_UM2="${axis_ceilings[$index]}" \
    "${DC_SHELL}" -f "${TCL}" >"${axis_dir}/dc.log" 2>&1
  printf '0\n' >"${axis_dir}/dc.rc"

  for artifact in TCL_INTERNAL_COMPLETE.txt reports/flow_contract.rpt \
      reports/setup_posthold_summary_machine.txt reports/hold_posthold_summary_machine.txt \
      reports/area_posthold.rpt reports/constraint_setup_posthold_all.rpt \
      reports/constraint_hold_posthold_all.rpt \
      "netlist/${axis_designs[$index]}_m1892_fastmin_hold_repaired_mapped.v" \
      "netlist/${axis_designs[$index]}_m1892_fastmin_hold_repaired_mapped.sdc" \
      "netlist/${axis_designs[$index]}_m1892_fastmin_hold_repaired.ddc" \
      "netlist/${axis_designs[$index]}_m1892_fastmin_hold_repaired.svf"; do
    [[ -s "${axis_dir}/${artifact}" && ! -L "${axis_dir}/${artifact}" ]] || {
      echo "ERROR: missing ${axis}/${artifact}" >&2; exit 6;
    }
  done
  grep -Fxq 'status=MET' "${axis_dir}/reports/setup_posthold_summary_machine.txt"
  grep -Fxq 'violating_paths=0' "${axis_dir}/reports/setup_posthold_summary_machine.txt"
  grep -Fxq 'status=MET' "${axis_dir}/reports/hold_posthold_summary_machine.txt"
  grep -Fxq 'violating_paths=0' "${axis_dir}/reports/hold_posthold_summary_machine.txt"
  grep -Fq 'set_fix_hold_count=1' "${axis_dir}/reports/flow_contract.rpt"
  grep -Fq 'hold_only_incremental_mapping_count=1' "${axis_dir}/reports/flow_contract.rpt"
  grep -Fq 'optimization_hold_uncertainty_ns=0.070' "${axis_dir}/reports/flow_contract.rpt"
  grep -Fq 'reported_hold_uncertainty_ns=0.050' "${axis_dir}/reports/flow_contract.rpt"
done

printf '%s\n' \
  'schema=m1896_m1892_m1811_c2_fastmin_hold_repair_two_axis_receipt_r1_v1' \
  'status=RAW_PASS_DC_HOLD_REPAIRED_AWAIT_RESULT_HAMMER_AND_TRANSITIVE_FORMALITY_PT' \
  'axes=k8,k1x8' 'dc_shell_runs=2' 'retry=false' \
  'clock_period_ns=3.000' 'setup_uncertainty_ns=0.200' \
  'reported_hold_uncertainty_ns=0.050' 'optimization_hold_uncertainty_ns=0.070' \
  'functional_rtl_modified=false' 'logic_only=true' 'ideal_clock=true' \
  'formality=false' 'prime_time=false' 'power=false' \
  'paper_ppa_ready=false' 'system_speedup=false' >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1896_C2_TWO_AXIS_HOLD_REPAIR__AWAIT_RESULT_HAMMER_FORMALITY_PT\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -n -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM HUP
rmdir -- "${LOCK}"
echo 'M1896 raw C2 two-axis hold repair published; result hammer and transitive Formality/PT required'
