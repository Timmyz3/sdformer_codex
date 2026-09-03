#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash
# M1978 one-shot TSBG VCS runner.  M1970 repairs the split load handshake;
# this runner additionally bounds simv in wall-clock time and rejects native
# VCS assertion-failure diagnostics before publishing any raw result.
set -euo pipefail
umask 002

[[ $# -eq 4 ]] || { echo "ERROR: expected runner_sha review_sha release_sha audit_sha" >&2; exit 2; }
EXPECTED_RUNNER_SHA=$1
EXPECTED_REVIEW_SHA=$2
EXPECTED_RELEASE_SHA=$3
EXPECTED_AUDIT_SHA=$4
for digest in "$@"; do [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || exit 2; done

REPO_ROOT=/home/zhumd/work/sdformer_codex/SDformer
HW_ROOT=${REPO_ROOT}/hw_autoresearch_nts07
RUNNER=${HW_ROOT}/dc_handoff/scripts/run_m1978_m1970_c2_tsbg_b4_bounded_failclosed_one_shot.sh
FILELIST=${HW_ROOT}/dc_handoff/filelists/iscas_m1970_c2_tsbg_b4_bounded_independent_load_handshake_directed_vcs.f
RTL=${HW_ROOT}/rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv
ADAPTER=${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv
SVA=${HW_ROOT}/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv
TB=${HW_ROOT}/tb_m1970/tb_m1970_c2_tsbg_b4_bounded_independent_load_handshake.sv
DOCS359=${HW_ROOT}/docs/359_DATE终局冻结_20260813.md
M1965=${HW_ROOT}/reviews/m1965_m1956_m1964_c2_tsbg_b4_hang_failure_readonly_review_r1_20260902
M1967=${HW_ROOT}/reviews/m1967_m1966_m1965_c2_tsbg_b4_independent_load_handshake_source_hammer_r1_20260902
M1971=${HW_ROOT}/reviews/m1971_m1970_m1967_m1965_c2_tsbg_b4_bounded_independent_load_source_hammer_r1_20260902
M1972=${HW_ROOT}/reviews/m1972_m1970_m1971_c2_tsbg_b4_runner_ready_source_hammer_r1_20260902
M1975FAIL=${HW_ROOT}/reviews/m1975_m1974_c2_tsbg_b4_bounded_runner_hammer_r1_20260902
M1979=${HW_ROOT}/reviews/m1979_m1978_c2_tsbg_b4_bounded_runner_hammer_r1_20260902
M1980=${HW_ROOT}/contracts/m1980_m1979_m1978_c2_tsbg_b4_bounded_launch_release_r1_20260902.json
M1981=${HW_ROOT}/reviews/m1981_m1980_c2_tsbg_b4_bounded_launch_release_audit_r1_20260902
M1956_FAILURE=${HW_ROOT}/results/m1956_m1880_c2_tsbg_b4_sva_failclosed_directed_vcs_r1_20260902.failed_or_incomplete.quarantine
M1956_ATTEMPT=${HW_ROOT}/results/.m1956_m1880_c2_tsbg_b4_sva_failclosed_directed_vcs_attempt_consumed

VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
VCS=${VCS_HOME}/bin/vcs
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
TIMEOUT=/usr/bin/timeout

ATTEMPT=${HW_ROOT}/results/.m1978_m1880_c2_tsbg_b4_bounded_directed_vcs_attempt_consumed
RESULT=${HW_ROOT}/results/m1978_m1880_c2_tsbg_b4_bounded_directed_vcs_r1_20260902
FAILURE=${HW_ROOT}/results/m1978_m1880_c2_tsbg_b4_bounded_directed_vcs_r1_20260902.failed_or_incomplete.quarantine
WORK=${HW_ROOT}/results/.m1978_m1880_c2_tsbg_b4_bounded_directed_vcs_work.$$
LOCK=${HW_ROOT}/results/.m1978_m1880_c2_tsbg_b4_bounded_directed_vcs_launch_lock
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
  if [[ ${LOCK_HELD} -eq 1 ]]; then "${RMDIR}" -- "${LOCK}" 2>/dev/null; fi
  exit "${original_rc}"
}
trap 'on_exit $?' EXIT
trap 'on_exit 130' INT
trap 'on_exit 143' TERM
trap 'on_exit 129' HUP

sha_exact "${EXPECTED_RUNNER_SHA}" "${RUNNER}"
sha_exact d29a10c3f6b66854b44db72286cff8f0bac16cc00d2608399026f51139a975c5 "${FILELIST}"
sha_exact 8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05 "${RTL}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${ADAPTER}"
sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"
sha_exact 545cc5f0908f78e787efc25e937cb5a8051d29c2152b6158c3c0755fbed69555 "${TB}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact c0f90256dff7a39c9f2d64e9887b212aad3045fbbcf41f38ccf35d72b13d060d "${M1965}/review.json"
sha_exact 8f39a78a5861092a4f3c939589b1311e2f697d1f7ac36f5589956e6461864f64 "${M1967}/review.json"
sha_exact e2a577976bed8e69fa81d78994a8c34348b0842dc30878285d009c15eaa10be4 "${M1971}/review.json"
sha_exact 33517b5e6d661ed9a2955176bd952623a203055630ccaf0024f5756740d9b32f "${M1972}/review.json"
sha_exact a0851becbb63fcc32e50328110952ce9a9d84ef4ceb36aa4145170afa9a4645e "${M1975FAIL}/review.json"
verify_dir_seal "${M1965}"
verify_dir_seal "${M1967}"
verify_dir_seal "${M1971}"
verify_dir_seal "${M1972}"
verify_dir_seal "${M1975FAIL}"
sha_exact 2b756fe0c989afad0b1e612a9b0017400d483c83fbf90f23766a9b2900b4c41d "${M1956_FAILURE}/SHA256SUMS"
sha_exact 9c576642a4474af372683a5e436fc3ab356730bb40520da1de1c2591fc5026c0 "${M1956_FAILURE}/SHA256SUMS.seal.sha256"
sha_exact e4478d551ba0645aeb8a88acc446b0914209437becdc131358114829fe2b7470 "${M1956_ATTEMPT}/SHA256SUMS"
sha_exact 2c58e2aadb4f9fd312a16e9f1f029cccea4ab90977b47006c82f7ac9eef3933f "${M1956_ATTEMPT}/SHA256SUMS.seal.sha256"
verify_dir_seal "${M1956_FAILURE}"
verify_dir_seal "${M1956_ATTEMPT}"
sha_exact "${EXPECTED_REVIEW_SHA}" "${M1979}/review.json"
verify_dir_seal "${M1979}"
sha_exact "${EXPECTED_RELEASE_SHA}" "${M1980}"
sha_exact "${EXPECTED_AUDIT_SHA}" "${M1981}/review.json"
verify_dir_seal "${M1981}"

"${PY}" -I - "${M1979}/review.json" "${M1980}" "${M1981}/review.json" \
  "${EXPECTED_RUNNER_SHA}" "${EXPECTED_REVIEW_SHA}" "${EXPECTED_RELEASE_SHA}" <<'PY'
import json,sys
from pathlib import Path
r,l,a=(json.loads(Path(x).read_text()) for x in sys.argv[1:4])
runner_sha,review_sha,release_sha=sys.argv[4:7]
assert r['schema']=='m1979_m1978_c2_tsbg_b4_bounded_runner_hammer_r1_v1'
assert r['status']=='PASS_M1979_M1978_C2_TSBG_B4_BOUNDED_RUNNER_HAMMER__AUTHORIZE_RELEASE_ONLY'
assert r['severity_counts']=={'p0':0,'p1':0,'p2':0}
assert r['identity']['runner_sha256']==runner_sha
assert l['schema']=='m1980_m1979_m1978_c2_tsbg_b4_bounded_launch_release_r1_v1'
assert l['status']=='AUTHORIZE_ONE_M1978_C2_TSBG_B4_BOUNDED_VCS_ATTEMPT'
assert l['identity']=={'runner_sha256':runner_sha,'runner_review_sha256':review_sha,
    'm1972_source_review_sha256':'33517b5e6d661ed9a2955176bd952623a203055630ccaf0024f5756740d9b32f',
    'm1975_failure_review_sha256':'a0851becbb63fcc32e50328110952ce9a9d84ef4ceb36aa4145170afa9a4645e',
    'm1970_tb_sha256':'545cc5f0908f78e787efc25e937cb5a8051d29c2152b6158c3c0755fbed69555',
    'm1970_filelist_sha256':'d29a10c3f6b66854b44db72286cff8f0bac16cc00d2608399026f51139a975c5'}
assert l['budget']=={'license_queries':1,'vcs_compiles':1,'simv_runs':1,'automatic_retry':False}
assert l['gates']=={'sva_compile_enabled':True,'sva_runtime_maxfail':1,
    'simv_wall_timeout_s':180,'unique_pass_token':True,'result_hammer_required':True}
assert a['schema']=='m1981_m1980_c2_tsbg_b4_bounded_launch_release_audit_r1_v1'
assert a['status']=='PASS_M1981_M1980_C2_TSBG_B4_BOUNDED_RELEASE_AUDIT__AUTHORIZE_ONE_ATTEMPT'
assert a['severity_counts']=={'p0':0,'p1':0,'p2':0}
assert a['identity']=={'runner_sha256':runner_sha,'runner_review_sha256':review_sha,
    'release_sha256':release_sha}
PY

[[ ! -e ${ATTEMPT} && ! -e ${RESULT} && ! -e ${FAILURE} && ! -e ${WORK} && ! -e ${LOCK} ]] || {
  echo 'ERROR: M1978 namespace is not fresh' >&2; exit 4;
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
printf 'status=M1978_ATTEMPT_CONSUMED\nlicense_queries_authorized=1\nvcs_compiles_authorized=1\nsimv_runs_authorized=1\nlicense_queries_observed=0\nvcs_compiles_observed=0\nsimv_runs_observed=0\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"

"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}" >"${WORK}/license_preflight.log" 2>&1

"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${VCS}" -full64 -sverilog -assert svaext -top "${TOP}" -f "${FILELIST}" \
  -o "${WORK}/simv" -Mdir="${WORK}/csrc" >"${WORK}/vcs_compile.log" 2>&1
! "${GREP}" -Eiq 'Warning-\[SVAA-RNF\]|Ignoring.*global_finish_maxfail|global_finish_maxfail.*(ignored|unknown)|Unknown.*global_finish_maxfail|Error-' "${WORK}/vcs_compile.log"

"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${TIMEOUT}" --signal=TERM --kill-after=10s 180s "${WORK}/simv" \
  -assert global_finish_maxfail=1 >"${WORK}/simv.log" 2>&1

[[ "$("${GREP}" -Fc 'PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED' "${WORK}/simv.log")" -eq 1 ]]
for phase in reset full_load full_execute retired_replay replay_reset_recovery stale_attack stale_reset_recovery recovery_load recovery_execute final_checks; do
  [[ "$("${GREP}" -Fc "M1970_PHASE ${phase}_begin" "${WORK}/simv.log")" -eq 1 ]]
  [[ "$("${GREP}" -Fc "M1970_PHASE ${phase}_complete" "${WORK}/simv.log")" -eq 1 ]]
done
[[ "$("${GREP}" -Fc 'M1970_LOAD_BEGIN' "${WORK}/simv.log")" -eq 52 ]]
[[ "$("${GREP}" -Fc 'M1970_LOAD_COMPLETE' "${WORK}/simv.log")" -eq 52 ]]
[[ "$("${GREP}" -Fc 'M1970_LOAD_TIMEOUT' "${WORK}/simv.log")" -eq 0 ]]
! "${GREP}" -Eiq 'Warning-\[SVAA-RNF\]|Ignoring.*global_finish_maxfail|global_finish_maxfail.*(ignored|unknown)|Unknown.*global_finish_maxfail|: started at .* failed at|Assertion[^[:cntrl:]]*failed|Error-\[SVA|\$(error|fatal)|Fatal:|whole-test watchdog expired|directed timeout|post-reset legal-service timeout' "${WORK}/simv.log"

"${RM}" -rf -- "${WORK}/simv" "${WORK}/csrc" "${WORK}/simv.daidir"
printf '%s\n' \
  'schema=m1978_m1880_c2_tsbg_b4_bounded_directed_vcs_receipt_r1_v1' \
  'status=RAW_PASS_AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER' \
  'license_queries=1' 'vcs_compiles=1' 'simv_runs=1' \
  'sva_compile_enabled=true' 'sva_runtime_maxfail=1' 'simv_wall_timeout_s=180' \
  'behavioral_rtl_directed_only=true' 'same_area=false' 'system_speedup=false' \
  'paper_admitted=false' 'result_hammer_required=true' >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1978_M1880_C2_TSBG_B4_BOUNDED_DIRECTED__AWAIT_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
publish_no_replace_checked "${WORK}" "${RESULT}"
WORK_ACTIVE=0
"${RMDIR}" -- "${LOCK}"
LOCK_HELD=0
trap - EXIT INT TERM HUP
echo 'M1978 raw TSBG VCS result published; independent result hammer required'
