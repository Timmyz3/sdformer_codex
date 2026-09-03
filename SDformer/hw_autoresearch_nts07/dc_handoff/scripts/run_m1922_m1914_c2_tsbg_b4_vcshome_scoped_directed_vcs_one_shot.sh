#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash
# M1922 additive TSBG VCS successor.  The clean shebang removes inherited
# Bash functions/BASH_ENV/PATH.  Two inert identity arguments are required:
# the exact runner SHA and the exact independently sealed M1923 review SHA.
set -euo pipefail
umask 002

[[ $# -eq 2 ]] || { echo "ERROR: expected runner_sha review_sha" >&2; exit 2; }
EXPECTED_RUNNER_SHA=$1
EXPECTED_REVIEW_SHA=$2
[[ ${EXPECTED_RUNNER_SHA} =~ ^[0-9a-f]{64}$ && ${EXPECTED_REVIEW_SHA} =~ ^[0-9a-f]{64}$ ]] || exit 2

HW_ROOT=/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
REPO_ROOT=/home/zhumd/work/sdformer_codex/SDformer
RUNNER=${HW_ROOT}/dc_handoff/scripts/run_m1922_m1914_c2_tsbg_b4_vcshome_scoped_directed_vcs_one_shot.sh
FILELIST=${HW_ROOT}/dc_handoff/filelists/iscas_m1880_c2_tsbg_b4_real_channel_signed_frontend_directed_vcs.f
RTL=${HW_ROOT}/rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv
ADAPTER=${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv
SVA=${HW_ROOT}/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv
TB=${HW_ROOT}/tb_m1880/tb_m1880_c2_tsbg_b4_real_channel_signed_frontend.sv
DOCS359=${HW_ROOT}/docs/359_DATE终局冻结_20260813.md
M1895=${HW_ROOT}/reviews/m1895_m1894_c2_tsbg_b4_minimal_vcs_source_hammer_r1_20260902
M1907FAIL=${HW_ROOT}/reviews/m1907_m1906_c2_tsbg_b4_cleanenv_vcs_source_hammer_r1_20260902
M1915FAIL=${HW_ROOT}/reviews/m1915_m1914_c2_tsbg_b4_vcshome_scoped_vcs_source_hammer_r1_20260902
M1923=${HW_ROOT}/reviews/m1923_m1922_c2_tsbg_b4_vcshome_scoped_vcs_source_hammer_r1_20260902
M1898_FAILURE=${HW_ROOT}/results/m1898_m1880_c2_tsbg_b4_cleanenv_directed_vcs_r1_20260902.failed_or_incomplete.quarantine
M1898_ATTEMPT=${HW_ROOT}/results/.m1898_m1880_c2_tsbg_b4_cleanenv_directed_vcs_attempt_consumed

VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
VCS=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
TOP=tb_m1880_c2_tsbg_b4_real_channel_signed_frontend

SHA=/usr/bin/sha256sum
AWK=/usr/bin/awk
FIND=/usr/bin/find
SORT=/usr/bin/sort
XARGS=/usr/bin/xargs
GREP=/usr/bin/grep
MKDIR=/usr/bin/mkdir
MV=/usr/bin/mv
RM=/usr/bin/rm
RMDIR=/usr/bin/rmdir
ENV=/usr/bin/env

ATTEMPT=${HW_ROOT}/results/.m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_attempt_consumed
RESULT=${HW_ROOT}/results/m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_r1_20260902
FAILURE=${HW_ROOT}/results/m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_r1_20260902.failed_or_incomplete.quarantine
WORK=${HW_ROOT}/results/.m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_work.$$
LOCK=${HW_ROOT}/results/.m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_launch_lock
WORK_ACTIVE=0
LOCK_HELD=0

sha_file() { "${SHA}" -- "$1" | "${AWK}" '{print $1}'; }
sha_exact() {
  local expected=$1 path=$2
  [[ -f ${path} && ! -L ${path} && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: identity ${path}" >&2; exit 3;
  }
}
verify_dir_seal() {
  local dir=$1
  [[ -d ${dir} && ! -L ${dir} ]] || return 1
  (cd -- "${dir}" && "${SHA}" -c SHA256SUMS >/dev/null &&
    "${SHA}" -c SHA256SUMS.seal.sha256 >/dev/null)
}
seal_dir() {
  local dir=$1
  (cd -- "${dir}" &&
    "${FIND}" -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C "${SORT}" -z | "${XARGS}" -0 -r "${SHA}" -- >SHA256SUMS &&
    "${SHA}" -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    "${SHA}" -c SHA256SUMS >/dev/null &&
    "${SHA}" -c SHA256SUMS.seal.sha256 >/dev/null)
}
publish_no_replace_checked() {
  local source=$1 destination=$2
  [[ -d ${source} && ! -e ${destination} ]] || return 1
  "${MV}" -T -n -- "${source}" "${destination}"
  [[ ! -e ${source} && -d ${destination} && ! -L ${destination} ]] || return 1
  verify_dir_seal "${destination}"
}
on_exit() {
  local original_rc=$1
  trap - EXIT INT TERM HUP
  set +e
  if [[ ${original_rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d ${WORK} && ! -L ${WORK} ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${original_rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    set -e
    seal_dir "${WORK}"
    publish_no_replace_checked "${WORK}" "${FAILURE}"
    set +e
  fi
  if [[ ${LOCK_HELD} -eq 1 ]]; then
    "${RMDIR}" -- "${LOCK}" 2>/dev/null
  fi
  exit "${original_rc}"
}
trap 'on_exit $?' EXIT
trap 'on_exit 130' INT
trap 'on_exit 143' TERM
trap 'on_exit 129' HUP

sha_exact "${EXPECTED_RUNNER_SHA}" "${RUNNER}"
sha_exact 300702cdfec07ba83d1b85c5464002e411ea838846d623d3a09b1045391e71d2 "${FILELIST}"
sha_exact 8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05 "${RTL}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${ADAPTER}"
sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"
sha_exact 07f638b3a6a2ae99c3d24fcf96088ed84bfa61ab3c34bd626f65965fa1fed2d5 "${TB}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact 08bf19bddf0f05b949843cc36c8ed5817c80081c6efe707763eb4168e2266be0 "${M1895}/review.json"
verify_dir_seal "${M1895}"
sha_exact 1b22b563a5606963853835dcb587e037d8b285311db1690a058474ec0641e6b1 "${M1907FAIL}/review.json"
verify_dir_seal "${M1907FAIL}"
"${GREP}" -Fq '"status": "FAIL_M1907_M1906_C2_TSBG_B4_CLEANENV_VCS_SOURCE_HAMMER__DO_NOT_AUTHORIZE_ATTEMPT"' "${M1907FAIL}/review.json"
sha_exact 359b6549efa120e84cb6b4db978c60e618e30194f4e8ebd496eb539e2dd3cc72 "${M1915FAIL}/review.json"
verify_dir_seal "${M1915FAIL}"
"${GREP}" -Fq '"status": "FAIL_M1915_M1914_C2_TSBG_B4_CLEANENV_VCS_SOURCE_HAMMER_DO_NOT_RUN"' "${M1915FAIL}/review.json"
sha_exact 80de2b9ddf826309880adcdd2c6487c4b66e8cf4ede3c33a97b7d71b085047f2 "${M1898_FAILURE}/SHA256SUMS"
sha_exact 4d6a7d299f788d45c9e0b8a0f47224b4e9e184b4255ef4af8950b6ae6fd829b9 "${M1898_FAILURE}/SHA256SUMS.seal.sha256"
sha_exact e6c82b04bb23fb157dcd54c1d877aa5dec686c7816e8fece8ae0b220035887a7 "${M1898_FAILURE}/vcs_compile.log"
verify_dir_seal "${M1898_FAILURE}"
sha_exact 44ce3243845d9e9901b73ed83e1948205d5e9b7aec21fb381faefe287f5e1464 "${M1898_ATTEMPT}/SHA256SUMS"
sha_exact ffaa869f8804e73d2afa8af6ea4f54db30bf7b94aa9e1190038bf0ae2acb5a40 "${M1898_ATTEMPT}/SHA256SUMS.seal.sha256"
verify_dir_seal "${M1898_ATTEMPT}"
sha_exact "${EXPECTED_REVIEW_SHA}" "${M1923}/review.json"
verify_dir_seal "${M1923}"
"${GREP}" -Fq 'PASS_M1923_M1922_C2_TSBG_B4_VCSHOME_SCOPED_VCS_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT' "${M1923}/review.json"
"${GREP}" -Fq "\"runner_sha256\": \"${EXPECTED_RUNNER_SHA}\"" "${M1923}/review.json"
"${GREP}" -Fq '"p0_count": 0' "${M1923}/review.json"
"${GREP}" -Fq '"p1_count": 0' "${M1923}/review.json"

[[ ! -e ${ATTEMPT} && ! -e ${RESULT} && ! -e ${FAILURE} && ! -e ${WORK} && ! -e ${LOCK} ]] || {
  echo 'ERROR: M1922 namespace is not fresh' >&2; exit 4;
}

blocked=' vcs vcs1 vlogan simv dc_shell dc_shell-t pt_shell fm_shell icc2_shell common_shell_ex common_shell_exec common_shell_exe '
for proc in /proc/[0-9]*; do
  [[ -r ${proc}/status && -r ${proc}/comm ]] || continue
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
[[ ${mem_available} -ge 16777216 && $((commit_limit-committed_as)) -ge 16777216 ]] || exit 4

cd -- "${REPO_ROOT}"
"${MKDIR}" -- "${LOCK}"
LOCK_HELD=1
WORK_ACTIVE=1
"${MKDIR}" -- "${WORK}"
"${MKDIR}" -- "${ATTEMPT}"
printf 'status=M1922_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"

"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}" >"${WORK}/license_preflight.log" 2>&1

"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${VCS}" -full64 -sverilog -assert svaext -top "${TOP}" -f "${FILELIST}" \
  -o "${WORK}/simv" -Mdir="${WORK}/csrc" >"${WORK}/vcs_compile.log" 2>&1

"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${WORK}/simv" >"${WORK}/simv.log" 2>&1

[[ "$("${GREP}" -Fc 'PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED' "${WORK}/simv.log")" -eq 1 ]]
! "${GREP}" -Eq 'Assertion failed|Error-|\$fatal|Fatal:' "${WORK}/simv.log"

"${RM}" -rf -- "${WORK}/simv" "${WORK}/csrc" "${WORK}/simv.daidir"
printf '%s\n' \
  'schema=m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_receipt_r1_v1' \
  'status=RAW_PASS_AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER' \
  'license_queries=1' 'vcs_compiles=1' 'simv_runs=1' \
  'sva_enabled=true' 'behavioral_rtl_directed_only=true' \
  'same_area=false' 'system_speedup=false' 'paper_admitted=false' \
  'result_hammer_required=true' >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1922_M1880_C2_TSBG_B4_DIRECTED_VCS__AWAIT_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
publish_no_replace_checked "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM HUP
"${RMDIR}" -- "${LOCK}"
LOCK_HELD=0
echo 'M1922 raw TSBG VCS result published; independent result hammer required'

