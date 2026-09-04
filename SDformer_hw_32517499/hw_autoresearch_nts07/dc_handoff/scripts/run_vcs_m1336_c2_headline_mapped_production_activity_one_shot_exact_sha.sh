#!/usr/bin/env bash
set -euo pipefail
umask 077

# Inert one-shot release source.  It cannot reach a license query or VCS until
# a different-author source hammer, a launch release, and a final launch hammer
# are all present and supplied by exact external SHA.  Source authoring never
# executes this file.
[[ $# -eq 0 ]] || { echo "M1336: no arguments accepted" >&2; exit 2; }
for name in M1336_EXPECTED_RUNNER_SHA256 \
            M1336_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256 \
            M1336_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256 \
            M1336_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256 \
            M1336_EXPECTED_LAUNCH_RELEASE_SHA256 \
            M1336_EXPECTED_FINAL_HAMMER_REVIEW_SHA256 \
            M1336_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256 \
            M1336_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256; do
  value="${!name:-}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "M1336: ${name} absent/invalid" >&2; exit 2; }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m1336_c2_headline_mapped_production_activity_vcs_release_source_contract_r1_20260831.json"
SOURCE_AUTHOR="${HW_ROOT}/reviews/m1336_c2_headline_mapped_production_activity_vcs_release_source_author_r1_20260831"
SOURCE_HAMMER="${HW_ROOT}/reviews/m1337_m1336_c2_headline_mapped_production_activity_vcs_release_source_blind_hammer_r1_20260831"
LAUNCH_RELEASE="${HW_ROOT}/contracts/m1338_m1336_c2_headline_mapped_production_activity_vcs_launch_release_r1_20260831.json"
FINAL_HAMMER="${HW_ROOT}/reviews/m1339_m1338_m1336_c2_headline_mapped_production_activity_vcs_final_launch_hammer_r1_20260831"
SOURCE_CHECKER="${HW_ROOT}/system_simulator/scripts/check_m1334_c2_headline_mapped_production_activity_source.py"
RELEASE_CHECKER="${HW_ROOT}/verif_m1336_c2_activity_release/static_check_m1336_c2_activity_vcs_release_source.py"
RELEASE_TESTS="${HW_ROOT}/verif_m1336_c2_activity_release/test_m1336_c2_activity_vcs_release_source.py"
M1334_AUTHOR="${HW_ROOT}/reviews/m1334_c2_headline_mapped_production_activity_source_author_r1_20260831"
M1335_BLIND="${HW_ROOT}/reviews/m1335_m1334_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831"
M872="${HW_ROOT}/dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
M903="${HW_ROOT}/reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
FILELIST_K8="${HW_ROOT}/dc_handoff/filelists/date_m1334_c2_k8_mapped_production_activity.f"
FILELIST_K1X8="${HW_ROOT}/dc_handoff/filelists/date_m1334_c2_k1x8_mapped_production_activity.f"
UCLI="${HW_ROOT}/dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
LMUTIL="/opt/synopsys/scl/2025.03/linux64/bin/lmutil"
PYTHON="/opt/anaconda3/envs/pytorch310/bin/python3.10"
TIMEOUT="/usr/bin/timeout"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
LICENSE_SERVER="27030@ic.ismd-nemo"
LICENSE_FILE="/opt/synopsys/Synopsys.dat"
TOP="tb_m1334_c2_headline_mapped_production_activity"

ATTEMPT="${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_attempt_consumed"
RESULT="${HW_ROOT}/results/m1336_c2_headline_mapped_production_activity_vcs_r1_20260831"
FAILURE="${RESULT}.failed_or_incomplete.quarantine"
PRIVATE="${RESULT}.private_build.unsealed_do_not_cite"
FAILURE_PRIVATE="${RESULT}.failed_private_build.unsealed_do_not_cite"
WORK="${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_work.$$"
RESULT_STAGE="${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_result_stage.$$"
ATTEMPT_STAGE="${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_attempt_stage.$$"
FAILURE_STAGE="${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_failure_stage.$$"
COMPILE_TIMEOUT_SECONDS=1800
SIM_TIMEOUT_SECONDS=600
MIN_HEADROOM_KIB=16777216
phase="SOURCE_CHAIN"
failure_armed=0
complete=0
compile_count=0
sim_count=0

sha() { sha256sum -- "$1" | awk '{print $1}'; }
fail() { echo "M1336 gate failure: $*" >&2; exit 3; }
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
    sha256sum -c "$(basename -- "${outer}")" >/dev/null) \
    || fail "sidecar mismatch: ${path}"
}
verify_recursive_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
      && ! -L "${dir}/SHA256SUMS" && -f "${dir}/SHA256SUMS.seal.sha256" \
      && ! -L "${dir}/SHA256SUMS.seal.sha256" ]] || fail "seal absent: ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "seal mismatch: ${dir}"
  "${PYTHON}" -I - "${dir}" <<'PY'
import os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set(); actual=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    fields=line.split(None,1); assert len(fields)==2
    name=fields[1].lstrip('*'); p=Path(name)
    assert name not in listed and not p.is_absolute() and '..' not in p.parts
    listed.add(name)
for base_text,dirs,files in os.walk(str(d),followlinks=False):
    base=Path(base_text)
    assert all(not (base/name).is_symlink() for name in dirs+files)
    for name in files:
        p=base/name; rel=p.relative_to(d).as_posix()
        if rel not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}:
            assert stat.S_ISREG(os.lstat(str(p)).st_mode); actual.add(rel)
assert listed==actual,(listed-actual,actual-listed)
PY
}
seal_dir() {
  local dir="$1"
  "${PYTHON}" -I - "${dir}" <<'PY'
import hashlib,os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); rows=[]
for base_text,dirs,files in os.walk(str(d),followlinks=False):
    base=Path(base_text)
    assert all(not (base/name).is_symlink() for name in dirs+files)
    for name in files:
        p=base/name; rel=p.relative_to(d).as_posix()
        if rel in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        assert stat.S_ISREG(os.lstat(str(p)).st_mode)
        h=hashlib.sha256(p.read_bytes()).hexdigest(); rows.append((rel,h))
rows.sort(); manifest=d/'SHA256SUMS'
manifest.write_text(''.join('{}  {}\n'.format(h,n) for n,h in rows))
outer=d/'SHA256SUMS.seal.sha256'
outer.write_text('{}  SHA256SUMS\n'.format(hashlib.sha256(manifest.read_bytes()).hexdigest()))
PY
  verify_recursive_seal "${dir}"
}
publish_no_replace() {
  local source="$1" destination="$2"
  "${PYTHON}" -I - "${source}" "${destination}" <<'PY'
import ctypes,os,sys
source=os.fsencode(sys.argv[1]); destination=os.fsencode(sys.argv[2])
libc=ctypes.CDLL(None,use_errno=True)
renameat2=getattr(libc,'renameat2'); renameat2.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
if renameat2(-100,source,-100,destination,1)!=0:
    err=ctypes.get_errno(); raise OSError(err,os.strerror(err),sys.argv[2])
PY
}
collision_gate() {
  "${PYTHON}" -I - <<'PY'
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
        argv=[Path(x.decode(errors='replace')).name for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or blocked.intersection(argv): hits.append((p.name,comm,argv[:4]))
if hits: raise SystemExit('same-UID EDA collision: %r' % hits)
PY
}
namespace_gate() {
  for path in "${ATTEMPT}" "${RESULT}" "${FAILURE}" "${PRIVATE}" \
              "${FAILURE_PRIVATE}" "${WORK}" "${RESULT_STAGE}" \
              "${ATTEMPT_STAGE}" "${FAILURE_STAGE}"; do
    [[ ! -e "${path}" && ! -L "${path}" ]] || fail "namespace residue: ${path}"
  done
  compgen -G "${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_work.*" >/dev/null \
    && fail "stale work namespace" || true
  compgen -G "${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_result_stage.*" >/dev/null \
    && fail "stale result-stage namespace" || true
  compgen -G "${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_attempt_stage.*" >/dev/null \
    && fail "stale attempt-stage namespace" || true
  compgen -G "${HW_ROOT}/results/.m1336_c2_headline_mapped_production_activity_vcs_failure_stage.*" >/dev/null \
    && fail "stale failure-stage namespace" || true
}
resource_gate() {
  local available limit committed
  available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
  committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
  [[ "${available}" =~ ^[0-9]+$ && "${limit}" =~ ^[0-9]+$ && "${committed}" =~ ^[0-9]+$ \
      && "${available}" -ge "${MIN_HEADROOM_KIB}" \
      && $((limit-committed)) -ge "${MIN_HEADROOM_KIB}" ]] \
    || fail "resource preflight below 16 GiB memory/commit headroom"
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ "${complete}" -ne 1 && "${failure_armed}" -eq 1 ]]; then
    [[ "${rc}" -ne 0 ]] || rc=97
    if [[ ! -e "${FAILURE_STAGE}" && ! -L "${FAILURE_STAGE}" ]]; then
      mkdir -- "${FAILURE_STAGE}" || true
    fi
    if [[ -d "${FAILURE_STAGE}" && ! -L "${FAILURE_STAGE}" ]]; then
      printf 'status=FAILED_OR_INCOMPLETE\nphase=%s\nreturn_code=%s\ncompile_count=%s\nsim_count=%s\nautomatic_retry=false\ncanonical_result=false\nraw_private_build_citable=false\n' \
        "${phase}" "${rc}" "${compile_count}" "${sim_count}" \
        >"${FAILURE_STAGE}/RUN_FAILED_OR_INCOMPLETE.txt" || true
      seal_dir "${FAILURE_STAGE}" || true
      if [[ ! -e "${FAILURE}" && ! -L "${FAILURE}" ]]; then
        publish_no_replace "${FAILURE_STAGE}" "${FAILURE}" || true
      fi
    fi
    if [[ -d "${WORK}" && ! -L "${WORK}" && ! -e "${FAILURE_PRIVATE}" \
        && ! -L "${FAILURE_PRIVATE}" ]]; then
      publish_no_replace "${WORK}" "${FAILURE_PRIVATE}" || true
    fi
  fi
  exit "${rc}"
}

# Exact source/admission chain.  Compilation and simulation run only inside
# private axis directories and do not consume workspace-side simulator state.
exact "${RUNNER}" "${M1336_EXPECTED_RUNNER_SHA256}"
verify_file_sidecar "${SOURCE_CONTRACT}"
verify_recursive_seal "${SOURCE_AUTHOR}"
verify_recursive_seal "${M1334_AUTHOR}"
verify_recursive_seal "${M1335_BLIND}"
exact "${M1335_BLIND}/review.json" 2905fdec0e8799bd3790cadf3ca8c29b901deb564c2a290b83b63990465528c0
exact "${M1335_BLIND}/SHA256SUMS" 29efd256bdf7d328e7f965afa06d5b3b6a266447b5c6c8737e61e807d5958d55
exact "${M1335_BLIND}/SHA256SUMS.seal.sha256" 7ca5e39a9abeb85049de24b25b3f019df51f74d8519c105aa092c3e5cfd004b4
verify_recursive_seal "${SOURCE_HAMMER}"
exact "${SOURCE_HAMMER}/review.json" "${M1336_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256}"
exact "${SOURCE_HAMMER}/SHA256SUMS" "${M1336_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256}"
exact "${SOURCE_HAMMER}/SHA256SUMS.seal.sha256" "${M1336_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256}"
verify_file_sidecar "${LAUNCH_RELEASE}"
exact "${LAUNCH_RELEASE}" "${M1336_EXPECTED_LAUNCH_RELEASE_SHA256}"
verify_recursive_seal "${FINAL_HAMMER}"
exact "${FINAL_HAMMER}/review.json" "${M1336_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}"
exact "${FINAL_HAMMER}/SHA256SUMS" "${M1336_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256}"
exact "${FINAL_HAMMER}/SHA256SUMS.seal.sha256" "${M1336_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256}"

exact "${VCS_BIN}" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
exact "${LMUTIL}" e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07
exact "${PYTHON}" 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
exact "${DOCS359}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
exact "${FILELIST_K8}" 9030ca8f6e42a21546332f25009e08033e6a6740f5d95fd8c5a36f190ac00e6d
exact "${FILELIST_K1X8}" cca8a9b0bfe0c32d85f554994ab61c2b78dba425e6dee194fe9f1557b54998e9
exact "${UCLI}" c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1
exact "${M872}/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v" 6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5
exact "${M872}/k1x8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v" 65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc
exact "${M903}/review.json" 89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a

"${PYTHON}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${SOURCE_AUTHOR}/review.json" \
  "${SOURCE_HAMMER}/review.json" "${LAUNCH_RELEASE}" "${FINAL_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
runner,contract,author,source_hammer,release,final_hammer=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); a=json.loads(author.read_text())
s=json.loads(source_hammer.read_text()); r=json.loads(release.read_text()); f=json.loads(final_hammer.read_text())
assert c['status']=='M1336_C2_ACTIVITY_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1337_REQUIRED'
assert c['identity']['runner_sha256']==sha(runner)
assert a['status']=='PASS_M1336_C2_ACTIVITY_RELEASE_SOURCE__FRESH_M1337_HAMMER_REQUIRED'
assert a['bindings']['runner_sha256']==sha(runner) and a['bindings']['source_contract_sha256']==sha(contract)
assert s['status']=='PASS_M1337_M1336_C2_ACTIVITY_RELEASE_SOURCE__LAUNCH_RELEASE_MAY_BE_AUTHORED'
assert s['bindings']['runner_sha256']==sha(runner) and s['bindings']['source_contract_sha256']==sha(contract)
assert r['status']=='AUTHORIZE_ONE_M1336_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_ATTEMPT'
assert r['launch_now'] is True and r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['authorization']=={'vcs_compiles':2,'simv_runs':10,'all_other_eda_runs':0,'automatic_retry':False}
assert f['status']=='PASS_M1339_AUTHORIZE_ONE_M1336_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_LAUNCH'
for key,path in [('runner_sha256',runner),('source_contract_sha256',contract),('launch_release_sha256',release)]:
    assert f['bindings'][key]==sha(path)
assert f['authorization']==r['authorization']
for d in (c,a,s,r,f):
    for key in ('functional_vcs_verified','production_saif','ptpx','power','energy','performance','system_speedup','paper_ppa_ready','headline'):
        assert d['claim_boundary'][key] is False
PY
"${PYTHON}" -I "${SOURCE_CHECKER}" >/dev/null
"${PYTHON}" -I "${RELEASE_CHECKER}" >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -I "${RELEASE_TESTS}" >/dev/null
namespace_gate

failure_armed=1
trap on_exit EXIT
trap 'exit 130' INT TERM HUP
phase="RESOURCE_PREFLIGHT"
collision_gate
resource_gate
phase="LICENSE_PREFLIGHT"
[[ -f "${LICENSE_FILE}" && ! -L "${LICENSE_FILE}" ]] || fail "license file absent/nonregular"
"${TIMEOUT}" --signal=TERM --kill-after=10s 60s "${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}" >/dev/null 2>&1 \
  || fail "license preflight failed"
collision_gate

phase="ATTEMPT_CONSUME"
mkdir -- "${ATTEMPT_STAGE}"
printf 'status=M1336_ATTEMPT_CONSUMED\nrunner_sha256=%s\nsource_contract_sha256=%s\nlaunch_release_sha256=%s\nautomatic_retry=false\nmaximum_vcs_compiles=2\nmaximum_simv_runs=10\n' \
  "$(sha "${RUNNER}")" "$(sha "${SOURCE_CONTRACT}")" "$(sha "${LAUNCH_RELEASE}")" \
  >"${ATTEMPT_STAGE}/attempt.txt"
seal_dir "${ATTEMPT_STAGE}"
publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"
mkdir -- "${WORK}"
mkdir -- "${WORK}/build" "${WORK}/candidate"

for axis in k8 k1x8; do
  phase="COMPILE_${axis}"
  axis_dir="${WORK}/build/${axis}"
  mkdir -- "${axis_dir}"
  if [[ "${axis}" == k8 ]]; then filelist="${FILELIST_K8}"; axis_display=K8; cycles=(51 131 486 1231 14); else filelist="${FILELIST_K1X8}"; axis_display=K1x8; cycles=(53 133 499 1246 14); fi
  compile_count=$((compile_count+1))
  (cd -- "${axis_dir}" &&
    env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux \
      SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
      "${TIMEOUT}" --signal=TERM --kill-after=30s "${COMPILE_TIMEOUT_SECONDS}s" \
      "${VCS_BIN}" -full64 -sverilog +v2k -timescale=1ns/1ps -assert svaext \
      +vcs+lic+wait -Mdir=csrc -f "${filelist}" -top "${TOP}" -o simv \
      >compile.log 2>&1)
  [[ -x "${axis_dir}/simv" ]] || fail "compile did not create simv: ${axis}"
  if /usr/bin/rg -qi '(^|[^[:alnum:]_])(Error|Fatal)([^[:alnum:]_]|$)' "${axis_dir}/compile.log"; then fail "compile diagnostics: ${axis}"; fi
  for case_id in 0 1 2 3 4; do
    phase="RUN_${axis}_CASE${case_id}"
    sim_count=$((sim_count+1))
    saif="${WORK}/candidate/${axis}_case${case_id}.saif"
    log="${WORK}/candidate/${axis}_case${case_id}.log"
    report="${WORK}/candidate/${axis}_case${case_id}.assert.report"
    (cd -- "${axis_dir}" &&
      env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux \
        SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
        M1334_SAIF_FILE="${saif}" \
        "${TIMEOUT}" --signal=TERM --kill-after=30s "${SIM_TIMEOUT_SECONDS}s" \
        ./simv +M979_UCLI_SAIF +M979_CASE="${case_id}" -no_save \
        -assert report="${report}" -ucli -i "${UCLI}" >"${log}" 2>&1)
    expected="${cycles[case_id]}"; events=(20 41 90 110 0)
    /usr/bin/rg -q "^PASS M979 mapped replay axis=${axis_display} case=${case_id} events=${events[case_id]} cycles=${expected} saif_duration_ns=$((expected*3)) numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 protocol_errors=0$" "${log}" \
      || fail "frozen workload/cycle PASS mismatch: ${axis}/case${case_id}"
    "${PYTHON}" -I "${SOURCE_CHECKER}" --saif "${saif}" --axis "${axis}" \
      --case "${case_id}" --cycles "${expected}" \
      >"${WORK}/candidate/${axis}_case${case_id}.saif_check.json"
  done
done
[[ "${compile_count}" -eq 2 && "${sim_count}" -eq 10 ]] || fail "one-shot execution cardinality drift"

phase="INVENTORY"
"${PYTHON}" -I - "${WORK}/candidate" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); cycles={'k8':[51,131,486,1231,14],'k1x8':[53,133,499,1246,14]}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); entries=[]
for axis in ('k8','k1x8'):
  for case,value in enumerate(cycles[axis]):
    sp=root/('{}_case{}.saif'.format(axis,case)); lp=root/('{}_case{}.log'.format(axis,case))
    entries.append({'axis':axis,'case':case,'cycles':value,'saif':sp.name,'saif_sha256':sha(sp),'runtime_log':lp.name,'runtime_log_sha256':sha(lp)})
d={'schema':'m1334_c2_production_activity_inventory_r1','status':'CANDIDATE_UNSEALED_DO_NOT_CITE','entries':entries}
(root/'inventory.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
"${PYTHON}" -I "${SOURCE_CHECKER}" --inventory "${WORK}/candidate/inventory.json" \
  >"${WORK}/candidate/inventory_check.json"
"${PYTHON}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${LAUNCH_RELEASE}" \
  "${WORK}/candidate/m1336_receipt.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,release,out=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1336_c2_mapped_production_activity_vcs_candidate_receipt_r1','status':'PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER','created_utc':datetime.now(timezone.utc).isoformat(),'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),'launch_release_sha256':sha(release)},'one_shot':{'attempt_consumed':True,'vcs_compiles':2,'simv_runs':10,'automatic_retry':False},'workloads':{'k8_cycles':[51,131,486,1231,14],'k1x8_cycles':[53,133,499,1246,14]},'claim_boundary':{'functional_vcs_verified':False,'production_saif':False,'ptpx':False,'power':False,'energy':False,'performance':False,'system_speedup':False,'paper_ppa_ready':False,'headline':False}}
out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf '%s\n' PASS_M1336_C2_MAPPED_PRODUCTION_ACTIVITY_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER \
  >"${WORK}/candidate/RUN_COMPLETE.txt"

phase="SUCCESS_STAGE"
mkdir -- "${RESULT_STAGE}"
cp -- "${WORK}/candidate"/* "${RESULT_STAGE}/"
cp -- "${WORK}/build/k8/compile.log" "${RESULT_STAGE}/k8.compile.log"
cp -- "${WORK}/build/k1x8/compile.log" "${RESULT_STAGE}/k1x8.compile.log"
seal_dir "${RESULT_STAGE}"
verify_recursive_seal "${RESULT_STAGE}"
"${PYTHON}" -I "${SOURCE_CHECKER}" --inventory "${RESULT_STAGE}/inventory.json" >/dev/null
phase="PRIVATE_ARCHIVE"
publish_no_replace "${WORK}" "${PRIVATE}"
phase="SUCCESS_PUBLISH"
publish_no_replace "${RESULT_STAGE}" "${RESULT}"
complete=1
trap - EXIT INT TERM HUP
echo PASS_M1336_C2_MAPPED_PRODUCTION_ACTIVITY_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER
