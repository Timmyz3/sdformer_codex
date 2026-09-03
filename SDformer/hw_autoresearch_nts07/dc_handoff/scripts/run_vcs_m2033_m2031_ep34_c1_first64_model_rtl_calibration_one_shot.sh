#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash
set -euo pipefail
unset BASH_ENV ENV CDPATH GLOBIGNORE
export PATH=/usr/bin:/bin LANG=C LC_ALL=C

# M2037 is one fresh successor VCS attempt after M2033 failed only at canonical
# sealing.  A PASS for the single real ep34 C1 tile proves
# numerical equivalence and event-counter alignment only.  It never upgrades
# the frozen 1.694510x CPU cycle model to an RTL or system speedup.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"

TOP="${HW_ROOT}/rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
MACRO="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
TB="${HW_ROOT}/tb_m528_dw1rw/tb_m2031_ep34_c1_first64_model_rtl_calibration.sv"
FIXTURE="${HW_ROOT}/tb_m528_dw1rw/fixtures/m2031_ep34_c1_first64_support16.memh"
AUDIT="${HW_ROOT}/system_simulator/scripts/check_m2031_ep34_c1_first64_model_rtl_calibration_source.py"
M2032_DIR="${HW_ROOT}/reviews/m2032_m2031_ep34_c1_first64_model_rtl_calibration_source_hammer_r1_20260902"
M2032="${M2032_DIR}/review.json"
M2034_DIR="${HW_ROOT}/reviews/m2034_m2033_ep34_c1_first64_model_rtl_calibration_runner_source_hammer_r1_20260902"
M2034="${M2034_DIR}/review.json"
M2035_DIR="${HW_ROOT}/reviews/m2035_m2033_ep34_c1_first64_vcs_seal_failure_hammer_r1_20260902"
M2035="${M2035_DIR}/review.json"
M2036_DIR="${HW_ROOT}/reviews/m2036_m2037_ep34_c1_first64_model_rtl_calibration_successor_runner_source_hammer_r1_20260902"
M2036="${M2036_DIR}/review.json"
LAUNCH_RELEASE="${M2036_DIR}/launch_release.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

ASSET_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
FOUNDRY_V="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON="/opt/anaconda3/bin/python3.12"

RESULT="${HW_ROOT}/results/m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_r1_20260902"
ATTEMPT="${HW_ROOT}/results/.m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_attempt_consumed"
STAGE="${HW_ROOT}/results/.m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_stage.$$"
FAILED="${HW_ROOT}/results/m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_r1_20260902.failed_or_incomplete.$$.quarantine"
RUN_UID="$(/usr/bin/id -u)"
LOCK="/tmp/hw_autoresearch_m2037_vcs_uid_${RUN_UID}.lock"

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }

require_sha() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "ERROR missing/non-regular input: ${path}" >&2; exit 2; }
  [[ "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR SHA drift: ${path}" >&2; exit 2; }
}

verify_double_seal() {
  local directory="$1"
  (cd -- "${directory}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

seal_dir() {
  local directory="$1"
  [[ -z "$(find "${directory}" -type l -print -quit)" ]] || {
    echo "ERROR symlink in result tree: ${directory}" >&2; return 1; }
  (cd -- "${directory}" &&
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
}

reject_same_uid_vcs() {
  "${PYTHON}" -I - <<'PY'
import os
from pathlib import Path
blocked = {'vcs', 'vcs1', 'vlogan', 'simv'}
hits = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        if path.stat().st_uid != os.getuid():
            continue
        comm = (path.parent / 'comm').read_text(errors='replace').strip()
        parts = [p.decode(errors='replace') for p in path.read_bytes().split(b'\0') if p]
        exe = Path(os.readlink(path.parent / 'exe')).name
    except (OSError, PermissionError):
        continue
    argv0 = Path(parts[0]).name if parts else ''
    joined = ' '.join(parts)
    wrapper = comm.startswith('common_shell_ex') or exe.startswith('common_shell_ex')
    is_vcs = bool({comm, exe, argv0} & blocked)
    if wrapper and ('/vcs/' in joined or any('/' + x in joined for x in blocked)):
        is_vcs = True
    if is_vcs:
        hits.append((path.parent.name, comm, exe, joined[:240]))
if hits:
    raise SystemExit('same-UID VCS-family collision: %r' % hits)
PY
}

stage_active=0
cleanup() {
  local rc=$?
  if [[ "${stage_active}" -eq 1 && -d "${STAGE}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" \
      >"${STAGE}/FAILED_DO_NOT_CITE"
    seal_dir "${STAGE}" || true
    mv -T -- "${STAGE}" "${FAILED}" || true
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM HUP
umask 077

[[ $# -eq 0 ]] || { echo "ERROR runner accepts no arguments" >&2; exit 2; }
[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${STAGE}" ]] || {
  echo "ERROR production identity already consumed" >&2; exit 2; }

require_sha 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1 "${TOP}"
require_sha 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO}"
require_sha 8cac9b384ce6812336d6961bc9ae50ca5a46e636ee8e74d2d49de40c0b4d74f1 "${TB}"
require_sha 4601182ca0dbba23d444de7d65cd2d7969159aa8564fd54a516a1934bf8112b3 "${FIXTURE}"
require_sha c3937a5d069f56cee3bd641eda0b78777acda8c15aae54e8650360e1105c485a "${AUDIT}"
require_sha f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a "${M2032}"
require_sha 3eb091f8385e73745deea40e82cb4a04711b22f3b91e619692c5d0156b027544 "${M2034}"
require_sha e3b8bffe5b9c0d33d326b5431ba79c9bcacec67527c4f996786cb5dd5f634654 "${M2035}"
require_sha 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
require_sha 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS}"
require_sha 873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161 "${PYTHON}"
require_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
verify_double_seal "${M2032_DIR}"
verify_double_seal "${M2034_DIR}"
verify_double_seal "${M2035_DIR}"
verify_double_seal "${M2036_DIR}"

# M2036 is authored after this successor runner.  Its double-sealed launch release pins
# the reviewed runner SHA and is the only live EDA authority consumed here.
"${PYTHON}" -I - "${RUNNER}" "${M2036}" "${LAUNCH_RELEASE}" "${RESULT}" "${ATTEMPT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
runner, review_path, release_path, result, attempt = map(Path, sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review=json.loads(review_path.read_text())
release=json.loads(release_path.read_text())
runner_sha=sha(runner)
if review.get('status') != 'PASS_M2036_M2037_SUCCESSOR_RUNNER_SOURCE_HAMMER':
    raise SystemExit('M2036 review status drift')
if review.get('score', 0) < 90 or review.get('severity_counts', {}).get('P0') != 0:
    raise SystemExit('M2036 review does not authorize execution')
if review.get('runner_sha256') != runner_sha:
    raise SystemExit('runner SHA differs from M2036 review')
if release.get('status') != 'AUTHORIZED_EXACTLY_ONE_M2037_SUCCESSOR_VCS_COMPILE_AND_SIM':
    raise SystemExit('launch-release status drift')
if release.get('runner_sha256') != runner_sha or release.get('review_sha256') != sha(review_path):
    raise SystemExit('launch-release identity drift')
if Path(release.get('result_path', '')).resolve() != result.resolve():
    raise SystemExit('launch-release result path drift')
if Path(release.get('attempt_path', '')).resolve() != attempt.resolve():
    raise SystemExit('launch-release attempt path drift')
if release.get('execution_budget') != {'vcs_compile_runs': 1, 'simv_runs': 1, 'automatic_retry': False}:
    raise SystemExit('launch-release execution budget drift')
PY

audit_output="$(${PYTHON} "${AUDIT}")"
grep -Fq 'PASS_M2031_EP34_C1_FIRST64_SOURCE_AUDIT__NO_EDA' <<<"${audit_output}"
grep -Fq '"fixture_is_exact_ledger_prefix": true' <<<"${audit_output}"

# Serialize cooperating jobs and scan the whole same-UID VCS process family.
# A second scan immediately precedes VCS to close the ordinary check/launch gap.
exec 9>"${LOCK}"
/usr/bin/flock -n 9 || { echo "ERROR same-UID VCS lock is held" >&2; exit 2; }
reject_same_uid_vcs

mkdir -- "${ATTEMPT}"
printf 'status=M2037_SUCCESSOR_ATTEMPT_CONSUMED\nvcs_compile_runs=1\nsimv_runs=1\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -- "${STAGE}"
stage_active=1

cd -- "${STAGE}"
reject_same_uid_vcs
set +e
/usr/bin/timeout --signal=TERM --kill-after=60s 900s \
  /usr/bin/env -i PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin \
  LANG=C LC_ALL=C TMPDIR=/tmp PWD="${STAGE}" \
  VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 \
  VCS_ARCH_OVERRIDE=linux SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
  LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat "${VCS}" \
  -full64 -sverilog -timescale=1ns/1ps -debug_access+pp \
  +define+UNIT_DELAY +vcs+lic+wait "${FOUNDRY_V}" "${MACRO}" "${TOP}" "${TB}" \
  -top tb_m2031_ep34_c1_first64_model_rtl_calibration -o simv \
  >compile.log 2>&1
compile_rc=$?
set -e
printf '%s\n' "${compile_rc}" >compile.rc
[[ "${compile_rc}" -eq 0 ]] || exit 3

set +e
/usr/bin/timeout --signal=TERM --kill-after=30s 180s \
  /usr/bin/env -i PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin \
  LANG=C LC_ALL=C TMPDIR=/tmp PWD="${STAGE}" \
  VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
  LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat ./simv -no_save >sim.log 2>&1
sim_rc=$?
set -e
printf '%s\n' "${sim_rc}" >sim.rc
[[ "${sim_rc}" -eq 0 ]] || exit 4

expected_pass='PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION rows=64 active=64 input_nnz=565 residual_nnz=192 exact_parent_rows=4 issue=196 parent_edges=58 dead_elisions=31 macro_reads=54 macro_writes=33 forwards=4 deadline_holds=6 stalls=14 psum_commits=64 row_completions=64 numeric_commits=64 rtl_cycle_speedup=false full_network=false system_speedup=false'
[[ "$(grep -Fxc "${expected_pass}" sim.log)" -eq 1 ]]
! grep -Eq '(^|[^A-Za-z])(Error|Fatal|Assertion.*failed)|\$fatal|global watchdog expired|counter mismatch|numeric mismatch|protocol_error' compile.log sim.log

"${PYTHON}" -I - "${STAGE}" <<'PY'
import hashlib, json, os, re, sys
from pathlib import Path
stage = Path(sys.argv[1]).resolve()
links = [p for p in stage.rglob('*') if p.is_symlink()]
if len(links) != 1:
    raise SystemExit('expected exactly one VCS archive symlink, found %d' % len(links))
link = links[0]
relative = link.relative_to(stage).as_posix()
if link.parent != stage / 'csrc' or not re.fullmatch(r'_\d+_archive_1\.so', link.name):
    raise SystemExit('unexpected VCS symlink path: ' + relative)
raw_target = os.readlink(str(link))
target = link.resolve(strict=True)
expected_target = (stage / 'simv.daidir' / link.name).resolve(strict=True)
if target != expected_target or not target.is_file() or target.is_symlink():
    raise SystemExit('unexpected VCS symlink target')
digest = hashlib.sha256(target.read_bytes()).hexdigest()
record = {
  'schema':'m2037_expected_vcs_archive_symlink_removal_r1_v1',
  'status':'RECORDED_AND_UNLINKED_EXPECTED_VCS_ARCHIVE_SYMLINK',
  'link_path':relative,
  'raw_target':raw_target,
  'resolved_target_path':target.relative_to(stage).as_posix(),
  'target_size_bytes':target.stat().st_size,
  'target_sha256':digest,
  'remaining_symlinks_after_unlink':0}
link.unlink()
remaining = [p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_symlink()]
if remaining:
    raise SystemExit('symlinks remain after exact removal: ' + repr(remaining))
(stage/'generated_symlink_removal.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY

"${PYTHON}" -I - "${STAGE}" "${RUNNER}" "${TOP}" "${MACRO}" "${FOUNDRY_V}" "${TB}" "${FIXTURE}" "${AUDIT}" "${M2032}" "${M2034}" "${M2035}" "${M2036}" "${LAUNCH_RELEASE}" "${VCS}" "${PYTHON}" "${DOCS359}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
stage, runner, top, macro, foundry, tb, fixture, audit, source_review, old_runner_review, failure_review, successor_review, release, vcs, python, docs359 = map(Path, sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sim=(stage/'sim.log').read_text(errors='replace')
line=re.findall(r'^PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION .+$', sim, re.M)
if len(line) != 1:
    raise SystemExit('terminal token cardinality drift')
receipt={
  'schema':'m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_receipt_r1_v1',
  'status':'PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW',
  'cohort':{'checkpoint':'Motion C12 ep34 live93','rows':64,'source':'exact prefix of sealed M1590 ledger'},
  'model_to_rtl_counts':{'issue_accepts':196,'parent_edges':58,'dead_write_elisions':31,'macro_reads':54,'macro_writes':33,'forwards':4,'deadline_holds':6,'issue_stalls':14,'psum_commits':64,'row_completions':64,'numeric_commits':64},
  'identity':{'runner_sha256':sha(runner),'top_rtl_sha256':sha(top),'macro_wrapper_sha256':sha(macro),'foundry_model_sha256':sha(foundry),'tb_sha256':sha(tb),'fixture_sha256':sha(fixture),'source_audit_sha256':sha(audit),'source_review_sha256':sha(source_review),'old_runner_review_sha256':sha(old_runner_review),'failure_review_sha256':sha(failure_review),'successor_runner_review_sha256':sha(successor_review),'launch_release_sha256':sha(release),'vcs_sha256':sha(vcs),'python_sha256':sha(python),'docs359_sha256':sha(docs359),'compile_log_sha256':sha(stage/'compile.log'),'sim_log_sha256':sha(stage/'sim.log'),'generated_symlink_removal_sha256':sha(stage/'generated_symlink_removal.json')},
  'execution':{'vcs_compile_runs':1,'simv_runs':1,'automatic_retry':False,'macro_model':'foundry UNIT_DELAY functional'},
  'payload_boundary':{'masks':'real ep34 sealed-ledger prefix','signed12_values':'synthetic deterministic function of source index and lane','psum_prior':'all zero','real_weight_or_real_psum_numeric_calibration':False},
  'claim_boundary':{'single_real_tile_event_and_synthetic_numeric_calibration':True,'functional_vcs':True,'cpu_model_1p694510x_upgraded_to_rtl':False,'rtl_cycle_speedup':False,'same_area':False,'timing':False,'power':False,'energy':False,'full_network':False,'system_speedup':False,'headline':False}}
(stage/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY

printf 'PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW\n' \
  >RUN_COMPLETE.txt
rm -f -- simv.daidir/.vcs.timestamp 2>/dev/null || true
seal_dir "${STAGE}"
verify_double_seal "${STAGE}"
mv -T -n -- "${STAGE}" "${RESULT}"
[[ ! -e "${STAGE}" && -d "${RESULT}" && ! -L "${RESULT}" ]]
verify_double_seal "${RESULT}"
stage_active=0
trap - EXIT INT TERM HUP
printf 'PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW\n'
