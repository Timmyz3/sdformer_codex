#!/usr/bin/env bash
set -euo pipefail
umask 077

# Inert one-shot C1/R16 source.  No tool is reachable until a fresh M1364
# source hammer, M1365 launch release and M1366 final hammer are supplied by
# exact external SHA.  Source authoring never executes this file.
[[ $# -eq 0 ]] || { echo "M1363: no arguments accepted" >&2; exit 2; }
for name in M1363_EXPECTED_RUNNER_SHA256 \
            M1363_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256 \
            M1363_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256 \
            M1363_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256 \
            M1363_EXPECTED_LAUNCH_RELEASE_SHA256 \
            M1363_EXPECTED_FINAL_HAMMER_REVIEW_SHA256 \
            M1363_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256 \
            M1363_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256; do
  value="${!name:-}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "M1363: ${name} absent/invalid" >&2; exit 2; }
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
M1354_AUTHOR="${HW_ROOT}/reviews/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_author_r1_20260831"
M1355_FAIL="${HW_ROOT}/reviews/m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_release_blind_hammer_r1_20260831"
SOURCE_CHECKER="${HW_ROOT}/verif_m1363_c1_r16_vcs_release_exact/check_m1363_c1_r16_vcs_release_exact_source.py"
SOURCE_TESTS="${HW_ROOT}/verif_m1363_c1_r16_vcs_release_exact/test_m1363_c1_r16_vcs_release_exact_source.py"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_contract_r1_20260831.json"
AUTHOR_DIR="${HW_ROOT}/reviews/m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_author_r1_20260831"
SOURCE_HAMMER="${HW_ROOT}/reviews/m1364_m1363_c1_r16_real_m935_runtime_witness_vcs_release_source_blind_hammer_r1_20260831"
LAUNCH_RELEASE="${HW_ROOT}/contracts/m1365_m1364_m1363_c1_r16_real_m935_runtime_witness_vcs_launch_release_r1_20260831.json"
FINAL_HAMMER="${HW_ROOT}/reviews/m1366_m1365_m1363_c1_r16_real_m935_runtime_witness_vcs_final_launch_hammer_r1_20260831"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON_BIN="/opt/anaconda3/envs/pytorch310/bin/python3.10"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

ATTEMPT="${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_attempt_consumed"
RESULT="${HW_ROOT}/results/m1363_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
QUARANTINE="${RESULT}.failed_or_incomplete.quarantine"
WORK="${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_work.$$"
ATTEMPT_STAGE="${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_attempt_stage.$$"
FAILURE_STAGE="${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_failure_stage.$$"
TOP="tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13"
R13_PASS="PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE real_m935=true parent_issue_override=0 child_issue_override=0 first_beats=1 nonfirst_beats=1 weight_requests=2 psum_requests=1 response_join_hold_cycles=2 ii_ge_2=true row_completions=1 task_completions=1 boundary_fault=0 core_fault=0 m935_fault=0 every_oracle_operands=true zero_sva_failures_required=true functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false"
R15_PASS="PASS_M1337R15_REAL_M935_RUNTIME_WITNESS wrapper_functional_candidate=true strict_registered_stages=true unknown_fail_closed=true structural_bind=true ledger_bytes=214912 functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false headline=false"
COMPILE_TIMEOUT_SECONDS=1200
SIM_TIMEOUT_SECONDS=1800
MIN_HEADROOM_KIB=16777216
phase="SOURCE_CHAIN"
failure_armed=0
complete=0
compile_count=0
sim_count=0

sha() { sha256sum -- "$1" | awk '{print $1}'; }
fail() { echo "M1363 gate failure: $*" >&2; exit 3; }
exact() {
  local path="$1" expected="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha "${path}")" == "${expected}" ]] \
    || fail "identity drift: ${path}"
}
verify_file_sidecar() {
  local path="$1" sum="${1}.sha256" outer="${1}.sha256.seal.sha256"
  [[ -f "${path}" && ! -L "${path}" && -f "${sum}" && ! -L "${sum}" \
      && -f "${outer}" && ! -L "${outer}" ]] || fail "sidecar absent: ${path}"
  (cd -- "$(dirname -- "${path}")" &&
    sha256sum -c "$(basename -- "${sum}")" >/dev/null &&
    sha256sum -c "$(basename -- "${outer}")" >/dev/null) || fail "sidecar mismatch: ${path}"
}
verify_recursive_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
      && ! -L "${dir}/SHA256SUMS" && -f "${dir}/SHA256SUMS.seal.sha256" \
      && ! -L "${dir}/SHA256SUMS.seal.sha256" ]] || fail "seal absent: ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "seal mismatch: ${dir}"
  "${PYTHON_BIN}" -I - "${dir}" <<'PY'
import hashlib,os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set(); actual=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(None,1); name=name.lstrip('*'); p=Path(name)
    assert len(digest)==64 and name not in listed and not p.is_absolute() and '..' not in p.parts
    q=d/p; assert q.is_file() and not q.is_symlink(); assert hashlib.sha256(q.read_bytes()).hexdigest()==digest
    listed.add(name)
for root,dirs,files in os.walk(d,followlinks=False):
    base=Path(root); assert all(not (base/name).is_symlink() for name in dirs+files)
    for name in files:
        p=base/name; rel=p.relative_to(d).as_posix()
        if rel not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}:
            assert stat.S_ISREG(os.lstat(p).st_mode); actual.add(rel)
assert listed==actual,(listed-actual,actual-listed)
PY
}
seal_dir() {
  local dir="$1"
  "${PYTHON_BIN}" -I - "${dir}" <<'PY'
import hashlib,os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); rows=[]
for root,dirs,files in os.walk(d,followlinks=False):
    base=Path(root); assert all(not (base/name).is_symlink() for name in dirs+files)
    for name in files:
        p=base/name; rel=p.relative_to(d).as_posix()
        if rel in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        assert stat.S_ISREG(os.lstat(p).st_mode); rows.append((rel,hashlib.sha256(p.read_bytes()).hexdigest()))
rows.sort(); manifest=d/'SHA256SUMS'; manifest.write_text(''.join('{}  {}\n'.format(h,n) for n,h in rows))
(d/'SHA256SUMS.seal.sha256').write_text('{}  SHA256SUMS\n'.format(hashlib.sha256(manifest.read_bytes()).hexdigest()))
PY
  verify_recursive_seal "${dir}"
}
publish_no_replace() {
  local source="$1" destination="$2"
  "${PYTHON_BIN}" -I - "${source}" "${destination}" <<'PY'
import ctypes,os,sys
src=os.fsencode(sys.argv[1]); dst=os.fsencode(sys.argv[2]); libc=ctypes.CDLL(None,use_errno=True)
fn=getattr(libc,'renameat2'); fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
if fn(-100,src,-100,dst,1)!=0:
    err=ctypes.get_errno(); raise OSError(err,os.strerror(err),sys.argv[2])
PY
}
collision_gate() {
  "${PYTHON_BIN}" -I - <<'PY'
import os
from pathlib import Path
blocked={'vcs','vcs1','vlogan','simv','dc_shell','dc_shell-t','pt_shell','fm_shell','icc2_shell','common_shell_exec','common_shell_exe'}
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
        argv={Path(x.decode(errors='replace')).name for x in (p/'cmdline').read_bytes().split(b'\0') if x}
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or blocked.intersection(argv): hits.append((p.name,comm,sorted(argv)[:4]))
if hits: raise SystemExit('same-UID EDA collision: %r' % hits)
PY
}
resource_gate() {
  local available limit committed
  available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
  committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
  [[ "${available}" =~ ^[0-9]+$ && "${limit}" =~ ^[0-9]+$ && "${committed}" =~ ^[0-9]+$ \
      && "${available}" -ge "${MIN_HEADROOM_KIB}" \
      && $((limit-committed)) -ge "${MIN_HEADROOM_KIB}" ]] || fail "resource preflight below 16 GiB"
}
namespace_gate() {
  for path in "${ATTEMPT}" "${RESULT}" "${QUARANTINE}" "${WORK}" \
              "${ATTEMPT_STAGE}" "${FAILURE_STAGE}"; do
    [[ ! -e "${path}" && ! -L "${path}" ]] || fail "namespace residue: ${path}"
  done
  compgen -G "${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_work.*" >/dev/null && fail "stale work" || true
  compgen -G "${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_attempt_stage.*" >/dev/null && fail "stale attempt stage" || true
  compgen -G "${HW_ROOT}/results/.m1363_c1_r16_real_m935_runtime_witness_vcs_failure_stage.*" >/dev/null && fail "stale failure stage" || true
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ "${complete}" -ne 1 && "${failure_armed}" -eq 1 ]]; then
    [[ "${rc}" -ne 0 ]] || rc=97
    mkdir -- "${FAILURE_STAGE}" 2>/dev/null || true
    if [[ -d "${WORK}" && ! -L "${WORK}" ]]; then mv -- "${WORK}" "${FAILURE_STAGE}/private_build" || true; fi
    printf 'status=FAILED_OR_INCOMPLETE\nphase=%s\nreturn_code=%s\ncompile_count=%s\nsim_count=%s\nautomatic_retry=false\nfunctional_vcs=false\ntiming_verified=false\ncycles_measured=false\nspeedup=false\nppa=false\npower=false\nenergy=false\nsystem_speedup=false\nheadline=false\n' \
      "${phase}" "${rc}" "${compile_count}" "${sim_count}" >"${FAILURE_STAGE}/RUN_FAILED_OR_INCOMPLETE.txt" || true
    seal_dir "${FAILURE_STAGE}" || true
    publish_no_replace "${FAILURE_STAGE}" "${QUARANTINE}" || true
  fi
  exit "${rc}"
}

# Exact frozen technical corpus and admitted/failed predecessor authorities.
exact "${RUNNER}" "${M1363_EXPECTED_RUNNER_SHA256}"
exact "${FILELIST}" 87a8b5e7500808a8afbd4339668aae3a44db2de7924a948020e2c7bffce4289e
exact "${WITNESS}" 0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af
exact "${TB}" b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263
exact "${PARENT}" 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783
exact "${M935}" e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8
exact "${WRAPPER}" 639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595
exact "${SVA}" c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472
exact "${R16_CHECKER}" b570eeb7a49bb042de2abca2f6739df09ab1895f208103dbe4dfdac2e340cea4
exact "${R16_TESTS}" 5427063ef93e89cd7059b6e48422626a71fd0913427f9614da65faf9fca29929
exact "${R16_CONTRACT}" c9749b4a7f9e3e6f8b38cbaf4735b036d7753f79a407e208d28f09aecd375f33
exact "${FOUNDRY_V}" 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d
exact "${VCS_BIN}" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
exact "${PYTHON_BIN}" 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
exact "${DOCS359}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
verify_recursive_seal "${R16_AUTHOR}"
verify_recursive_seal "${R16_HAMMER}"
verify_recursive_seal "${M1354_AUTHOR}"
verify_recursive_seal "${M1355_FAIL}"
exact "${M1354_AUTHOR}/review.json" 378ce7f6e8b0ae20f98c94d197c2fad1dcd7e1082fa269320041480319daddae
exact "${M1354_AUTHOR}/SHA256SUMS" 799616b204bb88333193baad0188aac846cdca9a0493c19476f31ca1f7f866f2
exact "${M1354_AUTHOR}/SHA256SUMS.seal.sha256" 862b93fa2e781f48e4c1a59cc63262fe6541787e32171f28109ce6fd3eb0cbb6
exact "${M1355_FAIL}/review.json" 7c06c50e2087e2794957508cf042d6931d73cb22ce3a3cada5628a2d55ae4c8d
exact "${M1355_FAIL}/SHA256SUMS" 9709d1c21ce13df3b84efa19d4dfa47d2116fa661327f18d0666b17d924ec5f8
exact "${M1355_FAIL}/SHA256SUMS.seal.sha256" 8b7aea4d1bc0764c1e9137196e2fc0ea3b86cee27baf6ab459c2c717bd201105

# New source/release chain, all before attempt consumption or any tool call.
verify_file_sidecar "${SOURCE_CONTRACT}"
verify_recursive_seal "${AUTHOR_DIR}"
verify_recursive_seal "${SOURCE_HAMMER}"
exact "${SOURCE_HAMMER}/review.json" "${M1363_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256}"
exact "${SOURCE_HAMMER}/SHA256SUMS" "${M1363_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256}"
exact "${SOURCE_HAMMER}/SHA256SUMS.seal.sha256" "${M1363_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256}"
verify_file_sidecar "${LAUNCH_RELEASE}"
exact "${LAUNCH_RELEASE}" "${M1363_EXPECTED_LAUNCH_RELEASE_SHA256}"
verify_recursive_seal "${FINAL_HAMMER}"
exact "${FINAL_HAMMER}/review.json" "${M1363_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}"
exact "${FINAL_HAMMER}/SHA256SUMS" "${M1363_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256}"
exact "${FINAL_HAMMER}/SHA256SUMS.seal.sha256" "${M1363_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256}"

"${PYTHON_BIN}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${AUTHOR_DIR}/review.json" \
  "${SOURCE_HAMMER}/review.json" "${LAUNCH_RELEASE}" "${FINAL_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
def strict(path):
    def pairs(items):
        d={}
        for k,v in items: assert k not in d; d[k]=v
        return d
    return json.loads(path.read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda x:(_ for _ in ()).throw(AssertionError(x)))
runner,contract,author,source_hammer,release,final_hammer=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); c,a,s,r,f=map(strict,(contract,author,source_hammer,release,final_hammer))
claims={'source_only':True,'functional_vcs':False,'timing_verified':False,'cycles_measured':False,'speedup':False,'ppa':False,'power':False,'energy':False,'system_speedup':False,'headline':False}
assert c['status']=='M1363_C1_R16_VCS_RELEASE_EXACT_SOURCE_READY__FRESH_M1364_REQUIRED__NO_LAUNCH'
assert c['identity']['runner_sha256']==sha(runner)
assert a['status']=='PASS_M1363_C1_R16_VCS_RELEASE_EXACT_SOURCE__FRESH_M1364_REQUIRED'
assert a['bindings']['runner_sha256']==sha(runner) and a['bindings']['source_contract_sha256']==sha(contract)
assert s['status']=='PASS_M1364_M1363_C1_R16_VCS_RELEASE_SOURCE__LAUNCH_RELEASE_MAY_BE_AUTHORED'
assert s['bindings']['runner_sha256']==sha(runner) and s['bindings']['source_contract_sha256']==sha(contract)
assert r['status']=='AUTHORIZE_ONE_M1363_C1_R16_RUNTIME_WITNESS_UNIT_DELAY_VCS_ATTEMPT'
assert r['identity']['runner_sha256']==sha(runner) and r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['source_hammer_review_sha256']==sha(source_hammer)
assert f['status']=='PASS_M1366_AUTHORIZE_ONE_M1363_C1_R16_RUNTIME_WITNESS_VCS_LAUNCH'
assert f['bindings']['runner_sha256']==sha(runner) and f['bindings']['source_contract_sha256']==sha(contract)
assert f['bindings']['launch_release_sha256']==sha(release)
authorization={'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0,'automatic_retry':False}
assert r['authorization']==authorization and f['authorization']==authorization
for d in (c,a,s,r,f): assert d['claim_boundary']==claims
PY
"${PYTHON_BIN}" -I "${SOURCE_CHECKER}" --mode runtime_present >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -I "${SOURCE_TESTS}" >/dev/null
namespace_gate

phase="RESOURCE_PREFLIGHT"
collision_gate
resource_gate
collision_gate
phase="ATTEMPT_CONSUME"
failure_armed=1
trap on_exit EXIT
trap 'exit 130' INT TERM HUP
mkdir -- "${ATTEMPT_STAGE}"
printf 'status=M1363_ATTEMPT_CONSUMED\nrunner_sha256=%s\nsource_contract_sha256=%s\nsource_hammer_review_sha256=%s\nlaunch_release_sha256=%s\nfinal_hammer_review_sha256=%s\nautomatic_retry=false\nmaximum_vcs_compiles=1\nmaximum_simv_runs=1\n' \
  "$(sha "${RUNNER}")" "$(sha "${SOURCE_CONTRACT}")" "${M1363_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256}" \
  "${M1363_EXPECTED_LAUNCH_RELEASE_SHA256}" "${M1363_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
  >"${ATTEMPT_STAGE}/attempt.txt"
seal_dir "${ATTEMPT_STAGE}"
publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"
mkdir -- "${WORK}"
cd -- "${WORK}"

export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"
export SNPSLMD_LICENSE_FILE="27030@ic.ismd-nemo"
export LM_LICENSE_FILE="/opt/synopsys/Synopsys.dat"
phase="COMPILE"
compile_count=1
set +e
/usr/bin/timeout --signal=TERM --kill-after=30s "${COMPILE_TIMEOUT_SECONDS}s" \
  "${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  +define+UNIT_DELAY +vcs+lic+wait -f "${FILELIST}" -top "${TOP}" -o simv 2>&1 | tee compile.log
compile_rc=("${PIPESTATUS[@]}"); printf '%s %s\n' "${compile_rc[0]}" "${compile_rc[1]}" >compile.exit_codes
set -e
[[ "${compile_rc[0]}" -eq 0 && "${compile_rc[1]}" -eq 0 && -x simv ]] || exit 20
phase="SIMULATE"
sim_count=1
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

"${PYTHON_BIN}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${LAUNCH_RELEASE}" "${FINAL_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,release,hammer=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1363_c1_r16_real_m935_runtime_witness_unit_delay_vcs_receipt_r1_v1',
'status':'PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS','created_utc':datetime.now(timezone.utc).isoformat(),
'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),'release_sha256':sha(release),'final_hammer_review_sha256':sha(hammer)},
'macro_model':'foundry_UNIT_DELAY_functional','one_shot':{'attempt_consumed':True,'vcs_compiles':1,'simv_runs':1,'automatic_retry':False,'compile_timeout_seconds':1200,'sim_timeout_seconds':1800},
'coverage':{'real_m935':True,'parent_issue_override':False,'child_issue_override':False,'first_beats':1,'nonfirst_beats':1,'issue_accepts':2,'psum_reads':1,'row_completions':1,'task_completions':1,'runtime_witness_pass':True,'zero_sva_error_fatal_lines':True},
'claim_boundary':{'source_only':False,'functional_vcs':True,'timing_verified':False,'cycles_measured':False,'speedup':False,'ppa':False,'power':False,'energy':False,'system_speedup':False,'headline':False}}
Path('m1363_c1_r16_real_m935_runtime_witness_unit_delay_vcs_receipt_r1.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS\n' >RUN_COMPLETE.txt
phase="SUCCESS_PUBLISH"
seal_dir "${WORK}"
publish_no_replace "${WORK}" "${RESULT}"
complete=1
trap - EXIT INT TERM HUP
echo "PASS M1363 C1/R16 functional VCS result=${RESULT}"
