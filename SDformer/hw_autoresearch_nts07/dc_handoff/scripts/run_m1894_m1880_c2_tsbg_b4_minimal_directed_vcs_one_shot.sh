#!/usr/bin/env bash
# M1894 minimal one-shot TSBG VCS campaign.  This additive successor does not
# reuse M1882/M1887 attempts or outputs.  It is inert until an independent
# M1895 source review is present and the caller pins both runner and review.
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
FILELIST="${HW_ROOT}/dc_handoff/filelists/iscas_m1880_c2_tsbg_b4_real_channel_signed_frontend_directed_vcs.f"
RTL="${HW_ROOT}/rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
ADAPTER="${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA="${HW_ROOT}/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB="${HW_ROOT}/tb_m1880/tb_m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
REVIEW_DIR="${HW_ROOT}/reviews/m1895_m1894_c2_tsbg_b4_minimal_vcs_source_hammer_r1_20260902"
REVIEW="${REVIEW_DIR}/review.json"

VCS=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
TOP=tb_m1880_c2_tsbg_b4_real_channel_signed_frontend

ATTEMPT="${HW_ROOT}/results/.m1894_m1880_c2_tsbg_b4_minimal_directed_vcs_attempt_consumed"
RESULT="${HW_ROOT}/results/m1894_m1880_c2_tsbg_b4_minimal_directed_vcs_r1_20260902"
FAILURE="${HW_ROOT}/results/m1894_m1880_c2_tsbg_b4_minimal_directed_vcs_r1_20260902.failed_or_incomplete.quarantine"
WORK="${HW_ROOT}/results/.m1894_m1880_c2_tsbg_b4_minimal_directed_vcs_work.$$"
LOCK="${HW_ROOT}/results/.m1894_m1880_c2_tsbg_b4_minimal_directed_vcs_launch_lock"
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
    mv -T -n -- "${WORK}" "${FAILURE}" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

# Static identities.  These local checks are not license or EDA operations.
sha_exact 300702cdfec07ba83d1b85c5464002e411ea838846d623d3a09b1045391e71d2 "${FILELIST}"
sha_exact 8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05 "${RTL}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${ADAPTER}"
sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"
sha_exact 07f638b3a6a2ae99c3d24fcf96088ed84bfa61ab3c34bd626f65965fa1fed2d5 "${TB}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
[[ -n "${M1894_EXPECTED_RUNNER_SHA256:-}" ]] || { echo "ERROR: runner pin absent" >&2; exit 3; }
sha_exact "${M1894_EXPECTED_RUNNER_SHA256}" "${RUNNER}"
verify_dir_seal "${REVIEW_DIR}"
[[ -n "${M1894_EXPECTED_REVIEW_SHA256:-}" ]] || { echo "ERROR: review pin absent" >&2; exit 3; }
sha_exact "${M1894_EXPECTED_REVIEW_SHA256}" "${REVIEW}"
grep -Fxq '  "status": "PASS_M1895_M1894_C2_TSBG_B4_MINIMAL_VCS_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT",' "${REVIEW}"
grep -Fxq '  "p0_count": 0,' "${REVIEW}"
grep -Fxq '  "p1_count": 0,' "${REVIEW}"
grep -Fxq '  "p2_count": 0,' "${REVIEW}"

[[ ! -e "${ATTEMPT}" && ! -e "${RESULT}" && ! -e "${FAILURE}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || {
  echo "ERROR: M1894 namespace is not fresh" >&2; exit 4;
}

# Same-UID EDA exclusion uses only procfs and Bash builtins.
blocked=' vcs vcs1 vlogan simv dc_shell dc_shell-t pt_shell fm_shell icc2_shell common_shell_exec common_shell_exe '
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
[[ ${mem_available} -ge 16777216 && $((commit_limit-committed_as)) -ge 16777216 ]] || {
  echo "ERROR: resource gate" >&2; exit 4;
}

# The durable latch is consumed before the sole license query and both EDA
# commands.  Any later failure publishes one sealed, non-retry quarantine.
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M1894_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1

env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}" >"${WORK}/license_preflight.log" 2>&1

env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${VCS}" -full64 -sverilog -assert svaext -top "${TOP}" -f "${FILELIST}" \
  -o "${WORK}/simv" -Mdir="${WORK}/csrc" >"${WORK}/vcs_compile.log" 2>&1

env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${WORK}/simv" >"${WORK}/simv.log" 2>&1

[[ "$(grep -Fc 'PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED' "${WORK}/simv.log")" -eq 1 ]]
! grep -Eq 'Assertion failed|Error-|\$fatal|Fatal:' "${WORK}/simv.log"

rm -rf -- "${WORK}/simv" "${WORK}/csrc" "${WORK}/simv.daidir" 2>/dev/null || true
printf '%s\n' \
  'schema=m1894_m1880_c2_tsbg_b4_minimal_directed_vcs_receipt_r1_v1' \
  'status=RAW_PASS_AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER' \
  'license_queries=1' 'vcs_compiles=1' 'simv_runs=1' \
  'sva_enabled=true' 'behavioral_rtl_directed_only=true' \
  'same_area=false' 'system_speedup=false' 'paper_admitted=false' \
  'result_hammer_required=true' >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1894_M1880_C2_TSBG_B4_DIRECTED_VCS__AWAIT_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -n -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM HUP
rmdir -- "${LOCK}"
echo 'M1894 raw TSBG VCS result published; independent result hammer required'
