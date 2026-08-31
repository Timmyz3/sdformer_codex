#!/usr/bin/env bash
set -euo pipefail
umask 077

# M1354 is inert until a fresh different-author M1355 hammer and an exact
# M1356 release provide all four external digests.  It consumes at most one
# foundry UNIT_DELAY functional VCS attempt and never retries automatically.
[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
for name in M1354_EXPECTED_RELEASE_SHA256 \
            M1354_EXPECTED_HAMMER_REVIEW_SHA256 \
            M1354_EXPECTED_HAMMER_MANIFEST_SHA256 \
            M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256; do
  value="${!name:-}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: ${name} absent/invalid" >&2; exit 2; }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
FILELIST="${HW_ROOT}/verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_unit_delay_filelist.f"
WITNESS="${HW_ROOT}/verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_m935_runtime_witness.sv"
TB="${HW_ROOT}/verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
PARENT="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER="${HW_ROOT}/rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
R16_CHECKER="${HW_ROOT}/verif_m1345r16_c1_real_m935_runtime_witness/check_m1345r16_source.py"
R16_TESTS="${HW_ROOT}/verif_m1345r16_c1_real_m935_runtime_witness/test_m1345r16_source.py"
R16_CONTRACT="${HW_ROOT}/contracts/m1345_c1_r16_real_m935_runtime_witness_source_contract_r1_20260831.json"
R16_AUTHOR="${HW_ROOT}/reviews/m1345_c1_r16_real_m935_runtime_witness_source_author_r1_20260831"
R16_HAMMER="${HW_ROOT}/reviews/m1352_m1345_c1_r16_runtime_witness_source_blind_hammer_r1_20260831"
SOURCE_CHECKER="${HW_ROOT}/verif_m1354_c1_r16_vcs_release/check_m1354_c1_r16_vcs_release_source.py"
SOURCE_TESTS="${HW_ROOT}/verif_m1354_c1_r16_vcs_release/test_m1354_c1_r16_vcs_release_source.py"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_contract_r1_20260831.json"
AUTHOR_DIR="${HW_ROOT}/reviews/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_author_r1_20260831"
RELEASE_HAMMER="${HW_ROOT}/reviews/m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_release_blind_hammer_r1_20260831"
RELEASE="${HW_ROOT}/contracts/m1356_m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_launch_release_r1_20260831.json"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON_BIN="/opt/anaconda3/envs/pytorch310/bin/python3.10"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
ATTEMPT="${HW_ROOT}/results/.m1354_c1_r16_real_m935_runtime_witness_vcs_attempt_consumed"
RESULT="${HW_ROOT}/results/m1354_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
WORK="${HW_ROOT}/results/.m1354_c1_r16_real_m935_runtime_witness_vcs_work.$$"
QUARANTINE="${RESULT}.failed_or_incomplete.$$.quarantine"
TOP="tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13"
R13_PASS="PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE real_m935=true parent_issue_override=0 child_issue_override=0 first_beats=1 nonfirst_beats=1 weight_requests=2 psum_requests=1 response_join_hold_cycles=2 ii_ge_2=true row_completions=1 task_completions=1 boundary_fault=0 core_fault=0 m935_fault=0 every_oracle_operands=true zero_sva_failures_required=true functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false"
R15_PASS="PASS_M1337R15_REAL_M935_RUNTIME_WITNESS wrapper_functional_candidate=true strict_registered_stages=true unknown_fail_closed=true structural_bind=true ledger_bytes=214912 functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false headline=false"
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
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  "${PYTHON_BIN}" -I - "${dir}" <<'PY'
import os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); listed={}; actual=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    if not line.strip(): continue
    digest,name=line.split(None,1); name=name.lstrip('*')
    assert name not in listed and not Path(name).is_absolute() and '..' not in Path(name).parts
    listed[name]=digest
for root,dirs,files in os.walk(d,followlinks=False):
    base=Path(root); dirs[:]=[n for n in dirs if not (base/n).is_symlink()]
    for name in files:
        p=base/name; rel=p.relative_to(d).as_posix()
        if rel in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        assert not p.is_symlink()
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(rel)
assert set(listed)==actual,(set(listed)-actual,actual-set(listed))
for name,digest in listed.items():
    import hashlib
    assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
PY
}

WORK_ACTIVE=0
on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    cd -- "${WORK}" || return
    printf 'status=FAILED_OR_INCOMPLETE\nexit_code=%s\nfunctional_vcs_verified=false\nautomatic_retry=false\n' \
      "${rc}" >RUN_FAILED_OR_INCOMPLETE.txt
    [[ -f compile.log ]] && tail -n 200 compile.log >compile_tail.txt || true
    [[ -f sim.log ]] && tail -n 300 sim.log >sim_tail.txt || true
    seal_dir "${WORK}" || true
    [[ ! -e "${QUARANTINE}" ]] && mv -- "${WORK}" "${QUARANTINE}" || true
  fi
}
trap on_exit EXIT

# Exact-byte technical corpus and tool identities.
sha_exact 87a8b5e7500808a8afbd4339668aae3a44db2de7924a948020e2c7bffce4289e "${FILELIST}"
sha_exact 0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af "${WITNESS}"
sha_exact b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263 "${TB}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${PARENT}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${M935}"
sha_exact 639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595 "${WRAPPER}"
sha_exact c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472 "${SVA}"
sha_exact b570eeb7a49bb042de2abca2f6739df09ab1895f208103dbe4dfdac2e340cea4 "${R16_CHECKER}"
sha_exact 5427063ef93e89cd7059b6e48422626a71fd0913427f9614da65faf9fca29929 "${R16_TESTS}"
sha_exact c9749b4a7f9e3e6f8b38cbaf4735b036d7753f79a407e208d28f09aecd375f33 "${R16_CONTRACT}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115 "${PYTHON_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"

verify_recursive_seal "${R16_AUTHOR}"
verify_recursive_seal "${R16_HAMMER}"
sha_exact a5b136fce2bc3c5b5a5920b1e88cff092b1228b49a7ff6fd9959ff95e06772e5 "${R16_AUTHOR}/review.json"
sha_exact bd875634a0be33cb5dc2f0600734fa90e014ade961658c3d1f480ce40425a616 "${R16_AUTHOR}/SHA256SUMS"
sha_exact c9700d4411dd087b12494e4aaf2f5fde0de52f7e30b7397573b205371837e99f "${R16_AUTHOR}/SHA256SUMS.seal.sha256"
sha_exact 74969404ea26e5a522c205328c05a3527fca6daeefb74f6fb103cacb990e94ea "${R16_HAMMER}/review.json"
sha_exact d703fb23ff2a7726049f58d09e7d304d0e4e8adcaa781f34856115dcb4de40e6 "${R16_HAMMER}/SHA256SUMS"
sha_exact 29c6bf6de6a7ed91dc523dfc3360d7731c324a24cd3548a0fe3a346018e37ec7 "${R16_HAMMER}/SHA256SUMS.seal.sha256"
"${PYTHON_BIN}" -I "${R16_CHECKER}" >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -I "${R16_TESTS}" >/dev/null

# Bind M1354 source, future different-author hammer, and future release before
# any attempt directory or VCS license request can exist.
for artifact in "${SOURCE_CONTRACT}" "${RELEASE}"; do
  [[ -f "${artifact}.sha256" && -f "${artifact}.sha256.seal.sha256" ]] || exit 3
  (cd -- "$(dirname -- "${artifact}")" &&
    sha256sum -c "$(basename -- "${artifact}.sha256")" >/dev/null &&
    sha256sum -c "$(basename -- "${artifact}.sha256.seal.sha256")" >/dev/null)
done
for sealed in "${AUTHOR_DIR}" "${RELEASE_HAMMER}"; do verify_recursive_seal "${sealed}"; done
sha_exact "${M1354_EXPECTED_RELEASE_SHA256}" "${RELEASE}"
sha_exact "${M1354_EXPECTED_HAMMER_REVIEW_SHA256}" "${RELEASE_HAMMER}/review.json"
sha_exact "${M1354_EXPECTED_HAMMER_MANIFEST_SHA256}" "${RELEASE_HAMMER}/SHA256SUMS"
sha_exact "${M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256}" "${RELEASE_HAMMER}/SHA256SUMS.seal.sha256"

"${PYTHON_BIN}" -I - "${RUNNER}" "${FILELIST}" "${SOURCE_CHECKER}" \
  "${SOURCE_TESTS}" "${SOURCE_CONTRACT}" "${RELEASE}" \
  "${AUTHOR_DIR}/review.json" "${RELEASE_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
runner,filelist,checker,tests,contract,release,author,hammer=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); r=json.loads(release.read_text())
a=json.loads(author.read_text()); h=json.loads(hammer.read_text())
assert c['status']=='M1354_C1_R16_VCS_RELEASE_SOURCE_READY__FRESH_M1355_REQUIRED__NO_LAUNCH'
for key,path in [('runner_sha256',runner),('filelist_sha256',filelist),
                 ('source_checker_sha256',checker),('source_tests_sha256',tests)]:
    assert c['identity'][key]==sha(path)
assert r['status']=='AUTHORIZE_ONE_M1354_C1_R16_RUNTIME_WITNESS_UNIT_DELAY_VCS_ATTEMPT'
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert a['status']=='PASS_M1354_C1_R16_VCS_RELEASE_SOURCE__FRESH_M1355_REQUIRED'
assert a['bindings']['runner_sha256']==sha(runner)
assert a['bindings']['source_contract_sha256']==sha(contract)
assert h['status']=='PASS_M1355_AUTHORIZE_ONE_M1354_C1_R16_RUNTIME_WITNESS_VCS_LAUNCH'
assert h['score']>=95 and h['p0_count']==0 and h['p1_count']==0
assert h['bindings']['runner_sha256']==sha(runner)
assert h['bindings']['source_contract_sha256']==sha(contract)
assert h['authorization']=={'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0,'automatic_retry':False}
for d in (c,r,a,h):
    for key in ('functional_vcs','timing_verified','cycles_measured','speedup','ppa','power','energy','system_speedup','headline'):
        assert d['claim_boundary'][key] is False
PY

"${PYTHON_BIN}" -I "${SOURCE_CHECKER}" >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -I "${SOURCE_TESTS}" >/dev/null

[[ ! -e "${ATTEMPT}" && ! -e "${RESULT}" && ! -e "${WORK}" && ! -e "${QUARANTINE}" ]] || exit 4
compgen -G "${HW_ROOT}/results/.m1354_c1_r16_real_m935_runtime_witness_vcs_work.*" >/dev/null && exit 4 || true
compgen -G "${RESULT}.failed_or_incomplete.*" >/dev/null && exit 4 || true

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
        comm=(p/'comm').read_text().strip()
        argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or (blocked & {Path(x).name for x in argv}): hits.append((p.name,comm,argv[:4]))
if hits: raise SystemExit('EDA collision: %r' % hits)
PY
mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]] || exit 5

/bin/mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\nrelease_sha256=%s\nhammer_review_sha256=%s\nhammer_manifest_sha256=%s\nhammer_outer_file_sha256=%s\ncreated_utc=%s\nautomatic_retry=false\n' \
  "$(sha256sum -- "${RUNNER}" | awk '{print $1}')" "${M1354_EXPECTED_RELEASE_SHA256}" \
  "${M1354_EXPECTED_HAMMER_REVIEW_SHA256}" "${M1354_EXPECTED_HAMMER_MANIFEST_SHA256}" \
  "${M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
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

[[ "$(rg -Fxc "${R13_PASS}" sim.log || true)" -eq 1 ]] || exit 30
[[ "$(rg -Fxc "${R15_PASS}" sim.log || true)" -eq 1 ]] || exit 31
[[ "$(rg -c '^PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER$' sim.log || true)" -eq 1 ]] || exit 32
[[ "$(rg -c '^PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE$' sim.log || true)" -eq 1 ]] || exit 33
[[ "$(rg -c '^M1337R15_WITNESS_OPERANDS pass=1 ' sim.log || true)" -eq 1 ]] || exit 34
[[ "$(rg -c '^COVERAGE_M1270R13_REAL_M935 first_beats=1 nonfirst_beats=1 join_hold_cycles=2 issue_accepts=2 psum_reads=1 row_completions=1 task_completions=1 response_cycle_gap=[2-9][0-9]* oracle_records=[8-9][0-9]* parent_issue_override=0 child_issue_override=0$' sim.log || true)" -eq 1 ]] || exit 35
if rg -qi '(^|[^[:alnum:]_])(Error|Fatal|Assertion|\$error|\$fatal)([^[:alnum:]_]|$)' sim.log; then exit 36; fi

"${PYTHON_BIN}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${RELEASE}" "${RELEASE_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,release,hammer=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1354_c1_r16_real_m935_runtime_witness_unit_delay_vcs_receipt_r1_v1',
'status':'PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS','created_utc':datetime.now(timezone.utc).isoformat(),
'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),'release_sha256':sha(release),'m1355_release_hammer_review_sha256':sha(hammer)},
'macro_model':'foundry_UNIT_DELAY_functional','one_shot':{'attempt_consumed':True,'vcs_compiles':1,'simv_runs':1,'automatic_retry':False,'compile_timeout_seconds':1200,'sim_timeout_seconds':1800},
'coverage':{'real_m935':True,'parent_issue_override':False,'child_issue_override':False,'first_beats':1,'nonfirst_beats':1,'issue_accepts':2,'psum_reads':1,'row_completions':1,'task_completions':1,'runtime_witness_pass':True,'zero_sva_error_fatal_lines':True},
'claim_boundary':{'functional_vcs':True,'timing_verified':False,'cycles_measured':False,'speedup':False,'ppa':False,'power':False,'energy':False,'system_speedup':False,'headline':False}}
Path('m1354_c1_r16_real_m935_runtime_witness_unit_delay_vcs_receipt_r1.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M1354 C1/R16 functional VCS result=%s\n' "${RESULT}"
