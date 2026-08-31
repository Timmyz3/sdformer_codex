#!/usr/bin/env bash
set -euo pipefail
umask 077

# Inert until a fresh disjoint M1251 hammer supplies all four release digests.
# Exactly one bounded UNIT_DELAY compile and one bounded simulation are admitted.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
for name in M1250_EXPECTED_RELEASE_SHA256 \
            M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256 \
            M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256 \
            M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256; do
  value="${!name:-}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: ${name} absent/invalid" >&2; exit 2; }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TB="${HW_ROOT}/verif_m1232r11_c1_common_charge_protocol/tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv"
R11_CHECKER="${HW_ROOT}/verif_m1232r11_c1_common_charge_protocol/check_m1232r11_source.py"
R11_TESTS="${HW_ROOT}/verif_m1232r11_c1_common_charge_protocol/test_m1232r11_source.py"
FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
PARENT="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER="${HW_ROOT}/rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
M1246_CONTRACT="${HW_ROOT}/contracts/m1246_m1242_m1239_c1_r11_checker_source_contract_r1_20260830.json"
M1246_AUTHOR="${HW_ROOT}/reviews/m1246_m1242_c1_r11_checker_hardening_source_author_r1_20260830"
M1247_HAMMER="${HW_ROOT}/reviews/m1247_m1246_c1_r11_checker_tests_independent_hammer_r1_20260830"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m1250_m1247_m1246_c1_r11_vcs_release_source_contract_r1_20260830.json"
RELEASE="${HW_ROOT}/contracts/m1250_m1247_m1246_c1_r11_vcs_launch_release_r1_20260830.json"
AUTHOR_DIR="${HW_ROOT}/reviews/m1250_m1247_m1246_c1_r11_vcs_release_author_r1_20260830"
RELEASE_CHECKER="${HW_ROOT}/verif_m1250_c1_r11_vcs_release/static_check_m1250_c1_r11_vcs_release_source.py"
RELEASE_HAMMER="${HW_ROOT}/reviews/m1251_m1250_c1_r11_vcs_release_independent_hammer_r1_20260830"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON_BIN="/opt/anaconda3/envs/pytorch310/bin/python3.10"
ATTEMPT="${HW_ROOT}/results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_attempt_consumed"
RESULT="${HW_ROOT}/results/m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830"
WORK="${HW_ROOT}/results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_work.$$"
QUARANTINE="${RESULT}.failed_or_incomplete.$$.quarantine"
TOP="tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11"
PASS_TOKEN="PASS_M1232R11_M1162_COMMON_CHARGE_PROTOCOL_SOURCE_CANDIDATE"
COMPILE_TIMEOUT_SECONDS=1200
SIM_TIMEOUT_SECONDS=1800

sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "ERROR: missing/nonregular ${path}" >&2; exit 3; }
  got="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${got}" == "${expected}" ]] || {
    echo "ERROR: SHA mismatch ${path}: ${got}" >&2; exit 3; }
}

seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

verify_recursive_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || {
    echo "ERROR: recursive seal absent ${dir}" >&2; exit 3; }
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  "${PYTHON_BIN}" -I - "${dir}" <<'PY'
import os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set(); actual=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    if line.strip(): listed.add(line.split(None,1)[1].lstrip('*'))
for root,dirs,files in os.walk(d,followlinks=False):
    base=Path(root); dirs[:]=[n for n in dirs if not (base/n).is_symlink()]
    for name in files:
        p=base/name; rel=str(p.relative_to(d))
        if rel in {'SHA256SUMS','SHA256SUMS.seal.sha256'} or p.is_symlink(): continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(rel)
assert listed==actual,(listed-actual,actual-listed)
PY
}

failure_dump() {
  local destination="$1"
  {
    echo "M1250_FAILURE_PHASE_OR_TOOL_TIMEOUT_DUMP"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "compile_timeout_seconds=${COMPILE_TIMEOUT_SECONDS}"
    echo "sim_timeout_seconds=${SIM_TIMEOUT_SECONDS}"
    echo "compile_exit_codes=$(cat compile.exit_codes 2>/dev/null || echo unavailable)"
    echo "sim_exit_codes=$(cat sim.exit_codes 2>/dev/null || echo unavailable)"
    if [[ -f compile.log ]]; then
      echo "--- final 100 compile lines ---"
      tail -n 100 compile.log || true
    fi
    if [[ -f sim.log ]]; then
      echo "--- phase tokens ---"
      rg '^PHASE_M1219R9_' sim.log || true
      echo "--- internal watchdog dumps ---"
      rg '^TIMEOUT_M1219R9 ' sim.log || true
      echo "--- assertion/error/fatal lines ---"
      rg -ni '(^|[^[:alnum:]_])(Error|Fatal|Assertion|\$error|\$fatal)([^[:alnum:]_]|$)' sim.log || true
      echo "--- final 200 simulation lines ---"
      tail -n 200 sim.log || true
    else
      echo "sim.log absent: failure occurred before simulation"
    fi
  } >"${destination}"
}

WORK_ACTIVE=0
on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    cd -- "${WORK}" || return
    failure_dump "failure_phase_and_timeout_dump.txt" || true
    printf 'status=FAILED_OR_INCOMPLETE\nexit_code=%s\nfunctional_vcs_verified=false\nautomatic_retry=false\n' \
      "${rc}" >RUN_FAILED_OR_INCOMPLETE.txt
    seal_dir "${WORK}" || true
    [[ ! -e "${QUARANTINE}" ]] && mv -- "${WORK}" "${QUARANTINE}" || true
  fi
}
trap on_exit EXIT

# Frozen technical corpus and tool identities.
sha_exact 850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776 "${TB}"
sha_exact 154860a16dfa3e2175653e81c14db645da3718af2c8d659c35299d80248e68fd "${R11_CHECKER}"
sha_exact de89c87210e8782d38b84b8202d229a418ebb153583a02043f4080e25aac4605 "${R11_TESTS}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${PARENT}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${M935}"
sha_exact 639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595 "${WRAPPER}"
sha_exact c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472 "${SVA}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115 "${PYTHON_BIN}"
sha_exact 7f956c2343a596da25dc8658a79e6f50da462370fa3ddd4c7b4a650ab8c6c88d "${M1246_CONTRACT}"
sha_exact d6cbcb89e1712c9fd1690a9f9ff5ba038275e75c882f7b2620c95837d4d6c8eb "${M1246_AUTHOR}/mechanical_checks.json"
sha_exact a67fdce5307e7a35e84e162aa748404b1c56cccfc902458e1679b09b1acc1c52 "${M1246_AUTHOR}/SHA256SUMS"
sha_exact ff6d811aa64078feea2ad01fb30b19f10edf550d8da550ddb850939e5473f144 "${M1246_AUTHOR}/SHA256SUMS.seal.sha256"
sha_exact 32bdfcdafe3039eb9e44f318c2133e997cb182227fc0c18367d3ba9393bc807b "${M1247_HAMMER}/review.json"
sha_exact 8440f0f6111f6df9df1cfe0f85847fc2743ce2b9cd6f857c33d2581fa6ec0132 "${M1247_HAMMER}/SHA256SUMS"
sha_exact b9eb60767d829ecb0bde4e95bacc73c533c4da764702541803c69a4fa062c57d "${M1247_HAMMER}/SHA256SUMS.seal.sha256"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

for artifact in "${M1246_CONTRACT}" "${SOURCE_CONTRACT}" "${RELEASE}"; do
  [[ -f "${artifact}.sha256" && -f "${artifact}.sha256.seal.sha256" ]] || exit 3
  (cd -- "$(dirname -- "${artifact}")" &&
    sha256sum -c "$(basename -- "${artifact}.sha256")" >/dev/null &&
    sha256sum -c "$(basename -- "${artifact}.sha256.seal.sha256")" >/dev/null)
done
for sealed in "${M1246_AUTHOR}" "${M1247_HAMMER}" "${AUTHOR_DIR}" "${RELEASE_HAMMER}"; do
  verify_recursive_seal "${sealed}"
done
sha_exact "${M1250_EXPECTED_RELEASE_SHA256}" "${RELEASE}"
sha_exact "${M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" "${RELEASE_HAMMER}/review.json"
sha_exact "${M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" "${RELEASE_HAMMER}/SHA256SUMS"
sha_exact "${M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}" "${RELEASE_HAMMER}/SHA256SUMS.seal.sha256"

"${PYTHON_BIN}" -I - "${RUNNER}" "${RELEASE_CHECKER}" "${SOURCE_CONTRACT}" \
  "${RELEASE}" "${AUTHOR_DIR}/review.json" "${M1247_HAMMER}/review.json" \
  "${RELEASE_HAMMER}/review.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
runner,checker,contract,release,author,m1247,hammer=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); r=json.loads(release.read_text())
a=json.loads(author.read_text()); p=json.loads(m1247.read_text()); h=json.loads(hammer.read_text())
assert c['status']=='M1250_C1_R11_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1251_REQUIRED__NO_VCS_NO_EDA'
assert r['status']=='AUTHORIZE_ONE_M1250_R11_UNIT_DELAY_VCS_ATTEMPT_AFTER_FRESH_M1251_HAMMER'
assert c['identity']['runner_sha256']==sha(runner)==r['identity']['runner_sha256']
assert c['identity']['release_checker_sha256']==sha(checker)
assert r['identity']['source_contract_sha256']==sha(contract)
assert a['schema']=='m1250_m1247_m1246_c1_r11_vcs_release_author_review_r1_v1'
assert a['status']=='PASS_M1250_R11_ONE_SHOT_RELEASE_SOURCE__FRESH_M1251_HAMMER_REQUIRED'
assert a['bindings']['runner_sha256']==sha(runner)
assert a['bindings']['source_contract_sha256']==sha(contract)
assert a['bindings']['release_sha256']==sha(release)
assert p['status']=='PASS_M1247_RELEASE_AUTHORING_GO' and p['score']==100
assert p['p0_count']==0 and p['p1_count']==0 and p['p2_count']==0
assert p['authorization']['fresh_disjoint_release_authoring'] is True
assert h['schema']=='m1251_m1250_c1_r11_vcs_release_independent_hammer_r1_v1'
assert h['status']=='PASS_M1251_AUTHORIZE_ONE_M1250_R11_UNIT_DELAY_VCS_LAUNCH'
assert h['verdict']=='GO' and h['score']>=95 and h['p0_count']==0 and h['p1_count']==0
assert h['bindings']['runner_sha256']==sha(runner)
assert h['bindings']['source_contract_sha256']==sha(contract)
assert h['bindings']['release_sha256']==sha(release)
assert h['authorization']=={'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0,'automatic_retry':False}
for forbidden in ('hammer_manifest_sha256','hammer_outer_file_sha256','manifest_sha256','outer_file_sha256'):
    assert forbidden not in h.get('bindings',{}), 'acyclic hammer review violation '+forbidden
for env_name in ('M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256','M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256'):
    assert os.environ[env_name] not in hammer.read_text(), 'self digest embedded in review'
for d in (c,r,a,h):
    for key in ('functional_vcs_verified','timing_verified','cycles_measured','speedup','ppa','power','energy','system_speedup','paper_citable'):
        assert d['claim_boundary'][key] is False
PY

"${PYTHON_BIN}" -I "${R11_CHECKER}" >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -I "${R11_TESTS}" >/dev/null
"${PYTHON_BIN}" -I "${RELEASE_CHECKER}" >/dev/null

[[ ! -e "${ATTEMPT}" && ! -e "${RESULT}" && ! -e "${WORK}" && ! -e "${QUARANTINE}" ]] || {
  echo "ERROR: M1250 namespace not fresh" >&2; exit 4; }
compgen -G "${HW_ROOT}/results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_work.*" >/dev/null && {
  echo "ERROR: stale M1250 work" >&2; exit 4; } || true
compgen -G "${RESULT}.failed_or_incomplete.*" >/dev/null && {
  echo "ERROR: prior M1250 quarantine" >&2; exit 4; } || true

"${PYTHON_BIN}" -I - <<'PY'
import os
from pathlib import Path
blocked={'vcs','vcs1','simv','dc_shell','pt_shell','fm_shell','icc2_shell','common_shell_exec','common_shell_exe'}
ancestry=set(); pid=os.getpid()
while pid>1 and pid not in ancestry:
    ancestry.add(pid)
    try: pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
    except Exception: break
hits=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) in ancestry: continue
    try:
        if p.stat().st_uid!=os.getuid(): continue
        comm=(p/'comm').read_text().strip(); argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or (blocked & {Path(x).name for x in argv}): hits.append((p.name,comm,argv[:4]))
if hits: raise SystemExit('EDA collision: %r' % hits)
PY
mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]] || {
  echo "ERROR: MemAvailable below 64 GiB" >&2; exit 5; }

/bin/mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\nrelease_sha256=%s\nhammer_review_sha256=%s\nhammer_manifest_sha256=%s\nhammer_outer_file_sha256=%s\ncreated_utc=%s\nautomatic_retry=false\n' \
  "$(sha256sum -- "${RUNNER}" | awk '{print $1}')" "${M1250_EXPECTED_RELEASE_SHA256}" \
  "${M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" "${M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
  "${M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"${ATTEMPT}/identity.txt"
/bin/mkdir -- "${WORK}"
WORK_ACTIVE=1
cd -- "${WORK}"

export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"
export SNPSLMD_LICENSE_FILE="27030@ic.ismd-nemo"
export LM_LICENSE_FILE="/opt/synopsys/Synopsys.dat"
set +e
/usr/bin/timeout --signal=TERM --kill-after=30s "${COMPILE_TIMEOUT_SECONDS}s" \
  "${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  +define+UNIT_DELAY +vcs+lic+wait -f "${FILELIST}" -top "${TOP}" -o simv 2>&1 | tee compile.log
compile_rc=("${PIPESTATUS[@]}"); printf '%s %s\n' "${compile_rc[0]}" "${compile_rc[1]}" >compile.exit_codes
set -e
[[ "${compile_rc[0]}" -eq 0 && "${compile_rc[1]}" -eq 0 ]] || exit 20
set +e
/usr/bin/timeout --signal=TERM --kill-after=30s "${SIM_TIMEOUT_SECONDS}s" ./simv -no_save 2>&1 | tee sim.log
sim_rc=("${PIPESTATUS[@]}"); printf '%s %s\n' "${sim_rc[0]}" "${sim_rc[1]}" >sim.exit_codes
set -e
[[ "${sim_rc[0]}" -eq 0 && "${sim_rc[1]}" -eq 0 ]] || exit 21

for phase in DIRECTED RESET_PENDING STICKY_ATTACKS SERVICE_ATTACKS RANDOM NORMAL_M935 CLEAN_RESET_PREP; do
  [[ "$(rg -c "^PHASE_M1219R9_${phase}_ENTER( |$)" sim.log || true)" -eq 1 ]] || exit 30
  [[ "$(rg -c "^PHASE_M1219R9_${phase}_COMPLETE( |$)" sim.log || true)" -eq 1 ]] || exit 31
done
for index in $(seq 0 23); do
  [[ "$(rg -c "^PHASE_M1219R9_RANDOM_TRANSACTION_ENTER index=${index}$" sim.log || true)" -eq 1 ]] || exit 32
  [[ "$(rg -c "^PHASE_M1219R9_RANDOM_TRANSACTION_COMPLETE index=${index}$" sim.log || true)" -eq 1 ]] || exit 33
done
if rg -q '^TIMEOUT_M1219R9 ' sim.log; then exit 34; fi
if rg -qi '(^|[^[:alnum:]_])(Error|Fatal|Assertion|\$error|\$fatal)([^[:alnum:]_]|$)' compile.log sim.log; then exit 35; fi
[[ "$(rg -c "^${PASS_TOKEN} " sim.log || true)" -eq 1 ]] || exit 36
rg -q '^COVERAGE_M1219R9_PROTOCOL weight_first=1 psum_first=1 weight_rsp_first=1 psum_rsp_first=1 long_request=5 long_response=5 nonfirst=1 ii2=1 no_duplicate_request=25 random=24 legal_masks_clear=29 random_request_quiesce=24 bounded_waits=4$' sim.log || exit 37
rg -q '^COVERAGE_M1219R9_RESETS_ATTACKS reset_partial=1 reset_complete=1 reset_skew=1 unsolicited_weight=1 unsolicited_psum=1 same_cycle=1 duplicate_response=1 cancel=1 tuple_mutation=1 nonfirst_psum=1 request_attack_windows=2$' sim.log || exit 38
rg -q '^COVERAGE_M1219R9_SERVICE_ASSUMPTIONS weight_payload_mutation=1 psum_valid_drop=1 weight_windows=1 psum_windows=1 independent_checker=1 race_free_negedge_sample=1 skew_isolated=1 reachable_core_ready_force=0 boundary_fault=0 core_fault=0 dut_fault_claim=0$' sim.log || exit 39
rg -q '^COVERAGE_M1219R9_FROZEN_M935 normal_issues=2 normal_rows=1 normal_tasks=1 epoch=36865 clean_reset_prep_bounded=1$' sim.log || exit 40
rg -q '^PASS_M1232R11_M1162_COMMON_CHARGE_PROTOCOL_SOURCE_CANDIDATE directed_random=24 protocol_attacks=7 service_assumption_attacks=2 request_attack_windows=2 legal_masks_clear=29 reset_states=3 ii=2 normal_m935_rows=1 normal_m935_tasks=1 random_request_quiesce=24 exactly_one_random_request_handshake=1 bounded_waits=4 clean_reset_prep_bounded=1 phase_observability=1 service_skew_isolated=1 reachable_core_ready_force=0 random_request_single_fire=1 random_response_exact_accept=1 random_tuple_retire=1 random_post_retire_edge=1 normal_request_single_fire=1 normal_response_exact_accept=1 normal_beat_boundary_gate=1 zero_sva_failures_required=true boundary_fault=0 core_fault=0 functional_vcs_only=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false$' sim.log || exit 41

"${PYTHON_BIN}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${RELEASE}" \
  "${RELEASE_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,release,hammer=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1250_m1232r11_m1162_c1_unit_delay_vcs_receipt_r11_v1','status':'PASS_FUNCTIONAL_VCS_ONLY',
   'created_utc':datetime.now(timezone.utc).isoformat(),
   'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),
               'release_sha256':sha(release),'m1251_release_hammer_review_sha256':sha(hammer)},
   'macro_model':'foundry_UNIT_DELAY_functional',
   'one_shot':{'attempt_consumed':True,'vcs_compiles':1,'simv_runs':1,'automatic_retry':False,
               'compile_timeout_seconds':1200,'sim_timeout_seconds':1800},
   'observability':{'phase_pairs':7,'random_transaction_pairs':24,'bounded_waits':4,
                    'clean_reset_prep_bounded':True,'failure_quarantine_recursive_seal':True},
   'coverage':{'directed_random_transactions':24,'protocol_attacks':7,'service_assumption_attacks':2,
               'request_attack_windows':2,'legal_masks_clear':29,'minimum_completed_issue_ii':2,
               'normal_m935_rows':1,'normal_m935_tasks':1,'zero_sva_error_fatal_lines':True,
               'boundary_fault':False,'core_fault':False},
   'claim_boundary':{'functional_vcs_verified':True,'timing_verified':False,'cycles_measured':False,
                     'speedup':False,'ppa':False,'power':False,'energy':False,
                     'system_speedup':False,'paper_citable':False}}
Path('m1250_m1232r11_m1162_c1_unit_delay_vcs_receipt_r11.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M1250/R11 functional VCS result=%s\n' "${RESULT}"
