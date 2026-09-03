#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash
# M1926 additive TSBG VCS successor.  The clean shebang removes inherited
# Bash functions/BASH_ENV/PATH.  Four inert identity arguments are required:
# the exact runner, independently sealed M1927 review, M1928 release, and
# M1929 audit SHA values.
set -euo pipefail
umask 002

[[ $# -eq 4 ]] || { echo "ERROR: expected runner_sha review_sha release_sha audit_sha" >&2; exit 2; }
EXPECTED_RUNNER_SHA=$1
EXPECTED_REVIEW_SHA=$2
EXPECTED_RELEASE_SHA=$3
EXPECTED_AUDIT_SHA=$4
for digest in "$@"; do [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || exit 2; done

HW_ROOT=/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
REPO_ROOT=/home/zhumd/work/sdformer_codex/SDformer
RUNNER=${HW_ROOT}/dc_handoff/scripts/run_m1926_m1924_c2_tsbg_b4_vcs_legal_stats_one_shot.sh
FILELIST=${HW_ROOT}/dc_handoff/filelists/iscas_m1924_c2_tsbg_b4_vcs_legal_stats_directed_vcs.f
RTL=${HW_ROOT}/rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv
ADAPTER=${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv
SVA=${HW_ROOT}/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv
TB=${HW_ROOT}/tb_m1924/tb_m1924_c2_tsbg_b4_vcs_legal_stats.sv
DOCS359=${HW_ROOT}/docs/359_DATE终局冻结_20260813.md
M1895=${HW_ROOT}/reviews/m1895_m1894_c2_tsbg_b4_minimal_vcs_source_hammer_r1_20260902
M1907FAIL=${HW_ROOT}/reviews/m1907_m1906_c2_tsbg_b4_cleanenv_vcs_source_hammer_r1_20260902
M1915FAIL=${HW_ROOT}/reviews/m1915_m1914_c2_tsbg_b4_vcshome_scoped_vcs_source_hammer_r1_20260902
M1925=${HW_ROOT}/reviews/m1925_m1924_c2_tsbg_b4_vcs_legal_stats_source_hammer_r1_20260902
M1927=${HW_ROOT}/reviews/m1927_m1926_c2_tsbg_b4_vcs_runner_hammer_r1_20260902
M1928=${HW_ROOT}/contracts/m1928_m1927_m1926_c2_tsbg_b4_vcs_launch_release_r1_20260902.json
M1929=${HW_ROOT}/reviews/m1929_m1928_c2_tsbg_b4_vcs_launch_release_audit_r1_20260902
M1922_FAILURE=${HW_ROOT}/results/m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_r1_20260902.failed_or_incomplete.quarantine
M1922_ATTEMPT=${HW_ROOT}/results/.m1922_m1880_c2_tsbg_b4_cleanenv_directed_vcs_attempt_consumed
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
PY=/usr/bin/python3

ATTEMPT=${HW_ROOT}/results/.m1926_m1880_c2_tsbg_b4_vcs_legal_stats_directed_vcs_attempt_consumed
RESULT=${HW_ROOT}/results/m1926_m1880_c2_tsbg_b4_vcs_legal_stats_directed_vcs_r1_20260902
FAILURE=${HW_ROOT}/results/m1926_m1880_c2_tsbg_b4_vcs_legal_stats_directed_vcs_r1_20260902.failed_or_incomplete.quarantine
WORK=${HW_ROOT}/results/.m1926_m1880_c2_tsbg_b4_vcs_legal_stats_directed_vcs_work.$$
LOCK=${HW_ROOT}/results/.m1926_m1880_c2_tsbg_b4_vcs_legal_stats_directed_vcs_launch_lock
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
sha_exact 29bdb976655174cf9f3dace8dfaa87f57a6a2bd7ff02ba483ee9b2885e90ae21 "${FILELIST}"
sha_exact 8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05 "${RTL}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${ADAPTER}"
sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"
sha_exact df99e881e62ef2172f8658d36384d49640dcd86c8785e44cd7fbcfea97f264f1 "${TB}"
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
sha_exact 512de0a94a8fbeb508e24b172e416e5684ab148b3f46cd9b7bb3804dccaf61f1 "${M1925}/review.json"
verify_dir_seal "${M1925}"
"${PY}" -I - "${M1925}/review.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['schema']=='m1925_m1924_c2_tsbg_b4_vcs_legal_stats_source_hammer_review_r1_v1'
assert r['milestone']=='M1925'
assert r['reviewer_identity']=='/root/m1925_tsbg_tb_review'
assert r['status']=='PASS_M1925_M1924_C2_TSBG_B4_VCS_LEGAL_STATS_SOURCE_HAMMER__P0_P1_P2_0_0_0__M1926_FRESH_RUNNER_SOURCE_ONLY_NEXT__NO_EDA'
assert [r['p0_count'],r['p1_count'],r['p2_count']]==[0,0,0]
PY
sha_exact 19ffda599587e22c5093f05fb32d58bc0b1d1f2f68eda37b595b17ff7574f838 "${M1922_FAILURE}/SHA256SUMS"
sha_exact 284a2cf074305fbff0f882f5eefc57829732b7197864c7ebb074cd72aa7bed7a "${M1922_FAILURE}/SHA256SUMS.seal.sha256"
sha_exact fcf994799fd8e240105d5a5d587daea49b0a8f8bceeb27b927d84f8d6ac12adf "${M1922_FAILURE}/vcs_compile.log"
verify_dir_seal "${M1922_FAILURE}"
sha_exact 9965d0de44d6e0f5faceeff93bf9afbabea483ee95667c90ccc39a6c67dbb32c "${M1922_ATTEMPT}/SHA256SUMS"
sha_exact 41573654b0c4943667ed4c25a9f4667b0f8340b16d46a9285171df937f70aa12 "${M1922_ATTEMPT}/SHA256SUMS.seal.sha256"
verify_dir_seal "${M1922_ATTEMPT}"
sha_exact "${EXPECTED_REVIEW_SHA}" "${M1927}/review.json"
verify_dir_seal "${M1927}"
sha_exact "${EXPECTED_RELEASE_SHA}" "${M1928}"
sha_exact "${EXPECTED_AUDIT_SHA}" "${M1929}/review.json"
verify_dir_seal "${M1929}"
"${PY}" -I - "${M1927}/review.json" "${M1928}" "${M1929}/review.json" "${EXPECTED_RUNNER_SHA}" "${EXPECTED_REVIEW_SHA}" "${EXPECTED_RELEASE_SHA}" <<'PY'
import json,sys
from pathlib import Path
r,l,a=(json.loads(Path(x).read_text()) for x in sys.argv[1:4])
rs,vs,ls=sys.argv[4:7]
assert r['schema']=='m1927_m1926_c2_tsbg_b4_vcs_runner_hammer_review_r1_v1'
assert r['milestone']=='M1927'
assert r['reviewer_identity']=='/root/m1927_tsbg_runner_review'
assert r['status']=='PASS_M1927_M1926_C2_TSBG_B4_VCS_RUNNER_HAMMER__AUTHORIZE_RELEASE_ONLY'
assert [r['p0_count'],r['p1_count'],r['p2_count']]==[0,0,0]
assert r['identity']['runner_sha256']==rs
assert l['schema']=='m1928_m1927_m1926_c2_tsbg_b4_vcs_launch_release_r1_v1'
assert l['status']=='AUTHORIZE_ONE_M1926_C2_TSBG_B4_VCS_ATTEMPT'
assert l['identity']=={'runner_sha256':rs,'runner_review_sha256':vs,'m1925_source_review_sha256':'512de0a94a8fbeb508e24b172e416e5684ab148b3f46cd9b7bb3804dccaf61f1','m1924_filelist_sha256':'29bdb976655174cf9f3dace8dfaa87f57a6a2bd7ff02ba483ee9b2885e90ae21','m1924_tb_sha256':'df99e881e62ef2172f8658d36384d49640dcd86c8785e44cd7fbcfea97f264f1','m1922_failure_manifest_sha256':'19ffda599587e22c5093f05fb32d58bc0b1d1f2f68eda37b595b17ff7574f838','m1922_attempt_manifest_sha256':'9965d0de44d6e0f5faceeff93bf9afbabea483ee95667c90ccc39a6c67dbb32c'}
assert l['budget']=={'license_queries':1,'vcs_compiles':1,'simv_runs':1,'automatic_retry':False}
assert l['gates']=={'sva_enabled':True,'unique_pass_token':True,'result_hammer_required':True}
assert a['schema']=='m1929_m1928_c2_tsbg_b4_vcs_launch_release_audit_review_r1_v1'
assert a['milestone']=='M1929'
assert a['reviewer_identity']=='/root/m1929_tsbg_release_audit'
assert a['status']=='PASS_M1929_M1928_C2_TSBG_B4_VCS_LAUNCH_RELEASE_AUDIT__AUTHORIZE_ONE_ATTEMPT'
assert [a['p0_count'],a['p1_count'],a['p2_count']]==[0,0,0]
assert a['identity']=={'runner_sha256':rs,'runner_review_sha256':vs,'release_sha256':ls}
PY

[[ ! -e ${ATTEMPT} && ! -e ${RESULT} && ! -e ${FAILURE} && ! -e ${WORK} && ! -e ${LOCK} ]] || {
  echo 'ERROR: M1926 namespace is not fresh' >&2; exit 4;
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
printf 'status=M1926_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
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
  'schema=m1926_m1880_c2_tsbg_b4_vcs_legal_stats_directed_vcs_receipt_r1_v1' \
  'status=RAW_PASS_AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER' \
  'license_queries=1' 'vcs_compiles=1' 'simv_runs=1' \
  'sva_enabled=true' 'behavioral_rtl_directed_only=true' \
  'same_area=false' 'system_speedup=false' 'paper_admitted=false' \
  'result_hammer_required=true' >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1926_M1880_C2_TSBG_B4_VCS_LEGAL_STATS_DIRECTED__AWAIT_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
publish_no_replace_checked "${WORK}" "${RESULT}"
WORK_ACTIVE=0
"${RMDIR}" -- "${LOCK}"
LOCK_HELD=0
trap - EXIT INT TERM HUP
echo 'M1926 raw TSBG VCS result published; independent result hammer required'

