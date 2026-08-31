#!/usr/bin/env bash
set -Eeuo pipefail

# Source-only one-shot runner candidate.  It is not launch-authorized until a
# separate fresh static hammer binds this exact SHA.  The four GPU-idle checks
# happen before the attempt identity is consumed.

if [[ $# -ne 0 ]]; then
  echo "ERROR: no arguments or overrides are accepted" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
CONTRACT="${HW_ROOT}/contracts/m714_h67_ep35_pctda_pattern_s10_contract_r2_20260828.json"
CAPTURE="${HW_ROOT}/system_simulator/scripts/trace_m714_h67_ep35_pctda_pattern_s10.py"
PYTHON_BIN="/opt/conda/envs/sdformerflow/bin/python"
STATIC_REVIEW_DIR="${HW_ROOT}/reviews/m731_m714_r2_terminal_identity_revalidation_fresh_static_hammer_r1_20260828"
STATIC_REVIEW="${STATIC_REVIEW_DIR}/review.json"
RESULT_PARENT="${HW_ROOT}/results"
RESULT="${RESULT_PARENT}/m714_h67_ep35_pctda_pattern_s10_r2_20260828"
ATTEMPT="${RESULT_PARENT}/.m714_h67_ep35_pctda_pattern_s10_r2_20260828.attempt_consumed"
STAGING=""
IDLE_LOG=""
SUCCESS=0
EXPECTED_CONTRACT_SHA256="8e58fe96c1c05b1c6713231e36e799f7e68b55f073c4044433a01eb0b308ebd5"
EXPECTED_CAPTURE_SHA256="28457d9d2cb94bfe10c8655affdeb4bb51199d72cbb94b6d4398eb893a44c63c"
START_RUNNER_SHA256=""
START_CONTRACT_SHA256=""
START_CAPTURE_SHA256=""

: "${M714_R2_EXPECTED_RUNNER_SHA256:?missing independently pinned runner SHA}"
: "${M714_R2_EXPECTED_STATIC_REVIEW_OUTER_SHA256:?missing independently pinned static-review outer seal SHA}"

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

terminal_revalidate_identity() {
  local runner_now contract_now capture_now
  runner_now="$(sha256_file "${RUNNER_PATH}")"
  contract_now="$(sha256_file "${CONTRACT}")"
  capture_now="$(sha256_file "${CAPTURE}")"
  [[ "${runner_now}" == "${M714_R2_EXPECTED_RUNNER_SHA256}" &&
     "${runner_now}" == "${START_RUNNER_SHA256}" ]] || {
    echo "ERROR: terminal runner identity drift" >&2; return 71; }
  [[ "${contract_now}" == "${EXPECTED_CONTRACT_SHA256}" &&
     "${contract_now}" == "${START_CONTRACT_SHA256}" ]] || {
    echo "ERROR: terminal contract identity drift" >&2; return 72; }
  [[ "${capture_now}" == "${EXPECTED_CAPTURE_SHA256}" &&
     "${capture_now}" == "${START_CAPTURE_SHA256}" ]] || {
    echo "ERROR: terminal capture identity drift" >&2; return 73; }
  [[ -f "${ATTEMPT}/IDENTITY" && ! -L "${ATTEMPT}/IDENTITY" ]] || {
    echo "ERROR: terminal attempt identity missing" >&2; return 74; }
  grep -Fxq -- "runner_sha256=${runner_now}" "${ATTEMPT}/IDENTITY" &&
    grep -Fxq -- "contract_sha256=${contract_now}" "${ATTEMPT}/IDENTITY" &&
    grep -Fxq -- "capture_sha256=${capture_now}" "${ATTEMPT}/IDENTITY" || {
      echo "ERROR: terminal attempt identity mismatch" >&2; return 75; }
}

seal_tree() {
  local root=$1
  (
    cd -- "${root}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- > SHA256SUMS
    sha256sum -- SHA256SUMS > SHA256SUMS.seal.sha256
    sha256sum -c -- SHA256SUMS >/dev/null
    sha256sum -c -- SHA256SUMS.seal.sha256 >/dev/null
  )
}

quarantine_failure() {
  local rc=$1 quarantine
  if [[ -n "${IDLE_LOG}" && -f "${IDLE_LOG}" ]]; then
    quarantine="${RESULT}.preflight_failed.$(date +%s).$$.log"
    mv -- "${IDLE_LOG}" "${quarantine}" || true
    IDLE_LOG=""
  fi
  if [[ -n "${STAGING}" && -d "${STAGING}" ]]; then
    printf 'FAILED_DO_NOT_CITE rc=%s timestamp=%s\n' \
      "${rc}" "$(date --iso-8601=seconds)" > "${STAGING}/FAILED_DO_NOT_CITE"
    seal_tree "${STAGING}" || true
    quarantine="${RESULT}.failed_or_incomplete.$(date +%s).$$"
    mv -- "${STAGING}" "${quarantine}" || true
    STAGING=""
  fi
}

on_exit() {
  local rc=$?
  if [[ "${SUCCESS}" -ne 1 ]]; then
    quarantine_failure "${rc}"
  fi
  exit "${rc}"
}
trap on_exit EXIT

[[ -d "${RESULT_PARENT}" && ! -L "${RESULT_PARENT}" ]] || {
  echo "ERROR: invalid result parent" >&2; exit 3; }
[[ -f "${CONTRACT}" && ! -L "${CONTRACT}" ]] || {
  echo "ERROR: missing contract" >&2; exit 4; }
[[ -f "${CAPTURE}" && ! -L "${CAPTURE}" ]] || {
  echo "ERROR: missing capture" >&2; exit 5; }
[[ "$(sha256_file "${CONTRACT}")" == "${EXPECTED_CONTRACT_SHA256}" ]] || {
  echo "ERROR: contract SHA drift" >&2; exit 5; }
[[ "$(sha256_file "${CAPTURE}")" == "${EXPECTED_CAPTURE_SHA256}" ]] || {
  echo "ERROR: capture SHA drift" >&2; exit 5; }
[[ -f "${RUNNER_PATH}" && ! -L "${RUNNER_PATH}" ]] || {
  echo "ERROR: invalid canonical runner path" >&2; exit 5; }
START_RUNNER_SHA256="$(sha256_file "${RUNNER_PATH}")"
START_CONTRACT_SHA256="$(sha256_file "${CONTRACT}")"
START_CAPTURE_SHA256="$(sha256_file "${CAPTURE}")"
[[ "${START_RUNNER_SHA256}" == \
   "${M714_R2_EXPECTED_RUNNER_SHA256}" ]] || {
  echo "ERROR: runner SHA differs from independent admission" >&2; exit 5; }
[[ -x "${PYTHON_BIN}" && -f "$(readlink -f "${PYTHON_BIN}")" ]] || {
  echo "ERROR: frozen remote Python missing/non-executable" >&2; exit 6; }
[[ -f "${STATIC_REVIEW}" && ! -L "${STATIC_REVIEW}" &&
   -f "${STATIC_REVIEW_DIR}/SHA256SUMS" &&
   -f "${STATIC_REVIEW_DIR}/SHA256SUMS.seal.sha256" ]] || {
  echo "ERROR: independently sealed static review missing" >&2; exit 6; }
(
  cd -- "${STATIC_REVIEW_DIR}"
  sha256sum -c -- SHA256SUMS >/dev/null
  sha256sum -c -- SHA256SUMS.seal.sha256 >/dev/null
)
[[ "$(sha256_file "${STATIC_REVIEW_DIR}/SHA256SUMS.seal.sha256")" == \
   "${M714_R2_EXPECTED_STATIC_REVIEW_OUTER_SHA256}" ]] || {
  echo "ERROR: static-review outer seal differs from admission" >&2; exit 6; }
"${PYTHON_BIN}" -I - "${STATIC_REVIEW}" \
  "${M714_R2_EXPECTED_RUNNER_SHA256}" "${EXPECTED_CONTRACT_SHA256}" \
  "${EXPECTED_CAPTURE_SHA256}" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as h: d=json.load(h)
if d.get('schema')!='m731_m714_r2_terminal_identity_revalidation_fresh_static_hammer_v1': raise SystemExit(61)
if d.get('status')!='PASS_M731_M714_R2_TERMINAL_IDENTITY_REVALIDATION_STATIC_HAMMER': raise SystemExit(62)
if d.get('verdict')!='PASS' or d.get('score_100')!=100: raise SystemExit(63)
if [d.get(k) for k in ('p0_count','p1_count','p2_count')]!=[0,0,0]: raise SystemExit(64)
i=d.get('identity',{})
if i.get('runner_sha256')!=sys.argv[2]: raise SystemExit(65)
if i.get('contract_sha256')!=sys.argv[3]: raise SystemExit(66)
if i.get('capture_sha256')!=sys.argv[4]: raise SystemExit(67)
decision=d.get('decision',{})
if decision.get('exactly_one_remote_gpu_capture_authorized') is not True: raise SystemExit(68)
if decision.get('four_fresh_idle_checks_required_at_launch') is not True: raise SystemExit(69)
PY
[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" ]] || {
  echo "ERROR: result or attempt identity already exists" >&2; exit 7; }

IDLE_LOG="$(mktemp "${RESULT_PARENT}/.m714_r2_gpu_idle.XXXXXXXX")"
for sample in 1 2 3 4; do
  GPU_SAMPLE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits)"
  GPU_APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits)"
  PROCESS_HITS="$(${PYTHON_BIN} - <<'PY'
import os, re
from pathlib import Path
hits=[]
me=os.getpid(); parent=os.getppid()
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    pid=int(entry.name)
    if pid in (me,parent):
        continue
    try:
        argv=[x.decode('utf-8','replace') for x in
              (entry/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        continue
    joined=' '.join(argv)
    # Match the project's real pre-CUDA naming families: profile100,
    # valid825, validate, trainer/trainonly, plus the plain aliases.
    if re.search(
        r'(^|[/_.-])(?:train(?:ing|er|only)?|eval(?:uation)?|'
        r'valid(?:ate|ation)?[0-9]*|profile(?:ing)?[0-9]*)(?=$|[/_. -])',
        joined, re.IGNORECASE):
        hits.append('{}:{}'.format(pid,joined))
print('\n'.join(hits))
PY
)"
  printf 'sample=%s timestamp=%s gpu=%q apps=%q process_hits=%q\n' \
    "${sample}" "$(date --iso-8601=seconds)" "${GPU_SAMPLE}" \
    "${GPU_APPS}" "${PROCESS_HITS}" >> "${IDLE_LOG}"
  UTIL="$(awk -F, 'NR==1 {gsub(/ /,"",$1); print $1}' <<<"${GPU_SAMPLE}")"
  USED="$(awk -F, 'NR==1 {gsub(/ /,"",$2); print $2}' <<<"${GPU_SAMPLE}")"
  [[ "${UTIL}" =~ ^[0-9]+$ && "${USED}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: malformed GPU sample" >&2; exit 8; }
  [[ -z "${GPU_APPS}" && -z "${PROCESS_HITS}" && "${UTIL}" -le 5 && \
     "${USED}" -le 1024 ]] || {
    echo "ERROR: GPU/process idle gate failed at sample ${sample}" >&2
    exit 9
  }
  [[ "${sample}" -eq 4 ]] || sleep 5
done

mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\ncontract_sha256=%s\ncapture_sha256=%s\n' \
  "$(sha256_file "${RUNNER_PATH}")" "$(sha256_file "${CONTRACT}")" \
  "$(sha256_file "${CAPTURE}")" > "${ATTEMPT}/IDENTITY"
STAGING="$(mktemp -d "${RESULT_PARENT}/.m714_r2_staging.XXXXXXXX")"
mv -- "${IDLE_LOG}" "${STAGING}/gpu_idle_prelaunch.log"
IDLE_LOG=""

"${PYTHON_BIN}" "${CAPTURE}" --contract "${CONTRACT}" \
  --output-dir "${STAGING}/payload" 2>&1 | tee "${STAGING}/capture.log"

terminal_revalidate_identity

"${PYTHON_BIN}" -I - "${STAGING}/payload" "${CONTRACT}" \
  "${CAPTURE}" "${RUNNER_PATH}" "${EXPECTED_CONTRACT_SHA256}" \
  "${EXPECTED_CAPTURE_SHA256}" "${M714_R2_EXPECTED_RUNNER_SHA256}" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
payload_dir,contract_path,capture_path,runner_path=map(Path,sys.argv[1:5])
expected_contract,expected_capture,expected_runner=sys.argv[5:]
def strict(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise RuntimeError('duplicate key '+k)
            out[k]=v
        return out
    def reject(token): raise RuntimeError('bad JSON token '+token)
    with path.open(encoding='utf-8') as h:
        return json.load(h,object_pairs_hook=pairs,parse_constant=reject)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
m366=payload_dir/'m366_h67_ep35_atlif_remaining_budget_s10_capture.json'
m714=payload_dir/'m714_h67_ep35_pctda_pattern_s10_capture.json'
pointer=payload_dir/'M714_PAYLOAD_PATH.txt'
if not all(p.is_file() and not p.is_symlink() for p in (m366,m714,pointer)):
    raise RuntimeError('payload inventory missing')
d=strict(m714); c=strict(contract_path)
if sha(contract_path)!=expected_contract: raise RuntimeError('terminal expected contract SHA')
if sha(capture_path)!=expected_capture: raise RuntimeError('terminal expected capture SHA')
if sha(runner_path)!=expected_runner: raise RuntimeError('terminal expected runner SHA')
if c.get('identity',{}).get('m714_script',{}).get('sha256')!=expected_capture:
    raise RuntimeError('contract-pinned capture SHA')
if d.get('schema')!='m714_h67_ep35_pctda_pattern_s10_capture_v1':
    raise RuntimeError('M714 schema')
if d.get('status')!='PASS_M714_R2_PCTDA_PATTERN_CAPTURE__IDEAL_RESOURCE_LOWER_BOUND_ONLY':
    raise RuntimeError('M714 status')
a=d.get('admission',{})
required_true=('pctda_s10_pattern_capture','pctda_ideal_resource_issue_lower_bound')
required_false=('pctda_executable_cycle','pctda_real_output_miter','pctda_rtl',
                'pctda_ppa','pctda_system_speedup','pctda_headline')
if not all(a.get(k) is True for k in required_true): raise RuntimeError('true admission')
if not all(a.get(k) is False for k in required_false): raise RuntimeError('false admission')
i=d['m714_pctda']['identity']
if i.get('m714_contract_sha256')!=sha(contract_path): raise RuntimeError('contract SHA')
if i.get('m714_script_sha256')!=sha(capture_path): raise RuntimeError('capture SHA')
if Path(pointer.read_text().strip()).name!=m714.name: raise RuntimeError('payload pointer')
print('PASS_M714_R2_TERMINAL_VALIDATION')
PY

terminal_revalidate_identity
"${PYTHON_BIN}" -I - "${STAGING}" "${CONTRACT}" "${CAPTURE}" \
  "${RUNNER_PATH}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,contract,capture,runner=map(Path,sys.argv[1:])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
receipt={
  'schema':'m714_h67_ep35_pctda_pattern_s10_terminal_receipt_v2',
  'status':'PASS_M714_R2_PATTERN_CAPTURE__IDEAL_RESOURCE_LOWER_BOUND_ONLY',
  'four_consecutive_gpu_idle_checks':True,
  'identity':{
    'contract_sha256':sha(contract),'capture_sha256':sha(capture),
    'runner_sha256':sha(runner)},
  'admission':{
    'pattern_capture':True,'ideal_resource_lower_bound':True,
    'real_output_miter':False,'executable_cycle':False,'rtl':False,
    'vcs':False,'synopsys_ppa':False,'energy':False,'accuracy':False,
    'system_speedup':False,'headline':False},
  'claim_boundary':'Pattern statistics and ideal-resource lower bounds only.'}
(root/'RUN_COMPLETE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

terminal_revalidate_identity
seal_tree "${STAGING}"
mv -- "${STAGING}" "${RESULT}"
STAGING=""
(
  cd -- "${RESULT}"
  sha256sum -c -- SHA256SUMS >/dev/null
  sha256sum -c -- SHA256SUMS.seal.sha256 >/dev/null
)
SUCCESS=1
trap - EXIT
printf 'PASS_M714_R2_RESULT=%s\n' "${RESULT}"
