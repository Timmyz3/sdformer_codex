#!/usr/bin/env bash
set -euo pipefail
umask 002

# ABANDONED_UNSEALED_NUMBER_COLLISION: decoder owns M950.  This unsealed C1
# draft is retained only as a tombstone and is intentionally impossible to
# launch.  The sole formal C1 runner/contract identity is M951.

echo "ERROR: abandoned unsealed M950 number-collision draft; use formal M951 identity" >&2
exit 98

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
[[ -n "${M950_EXPECTED_RELEASE_SHA256:-}" \
   && -n "${M950_EXPECTED_HAMMER_REVIEW_SHA256:-}" \
   && -n "${M950_EXPECTED_HAMMER_OUTER_SHA256:-}" ]] || {
  echo "ERROR: exact release/hammer SHA environment absent" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
RTL="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
EXEC_SVA="${HW_ROOT}/verif_m935_c1_match_pipeline/m935_m912_inherited_execution_assertions_r1.sv"
MATCH_SVA="${HW_ROOT}/verif_m935_c1_match_pipeline/m938_three_stage_exact_match_assertions_r2.sv"
TB="${HW_ROOT}/verif_m935_c1_match_pipeline/tb_m948_three_stage_match_pipeline_unit_delay_r3.sv"
STATIC_CHECK="${HW_ROOT}/verif_m935_c1_match_pipeline/static_check_m948_causal_dual_enqueue_tb.py"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m948_m947_m943_m938_c1_causal_dual_enqueue_tb_source_contract_DRAFT_r0_20260829.json"
RUN_CONTRACT="${HW_ROOT}/contracts/m950_m949_m948_m935_c1_causal_dual_enqueue_vcs_source_contract_r1_20260829.json"
M949_DIR="${HW_ROOT}/reviews/m949_m948_m947_causal_dual_enqueue_source_hammer_r1_20260829"
HAMMER_DIR="${HW_ROOT}/reviews/m951_m950_m949_m948_c1_vcs_runner_source_hammer_r1_20260829"
RELEASE="${HW_ROOT}/contracts/m952_m951_m950_m948_c1_vcs_launch_release_r1_20260829.json"
M943_ATTEMPT="${HW_ROOT}/results/.m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_r1_attempt_consumed"
M943_QUARANTINE="${HW_ROOT}/results/m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_r1_20260829.failed_or_incomplete.2807848.quarantine"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
RESULT="${HW_ROOT}/results/m950_m948_m935_c1_causal_dual_enqueue_unit_delay_vcs_r1_20260829"
ATTEMPT="${HW_ROOT}/results/.m950_m948_m935_c1_causal_dual_enqueue_unit_delay_vcs_r1_attempt_consumed"
WORK="${HW_ROOT}/results/.m950_m948_m935_c1_causal_dual_enqueue_unit_delay_vcs_r1_work.$$"
TOP="tb_m948_three_stage_match_pipeline_unit_delay_r3"
PASS_TOKEN="PASS_M948_C1_THREE_STAGE_EXACT_MATCH_PIPELINE_UNIT_DELAY_CANDIDATE"

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
  [[ -d "${dir}" && ! -L "${dir}" \
      && -f "${dir}/SHA256SUMS" \
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || {
    echo "ERROR: recursive seal absent ${dir}" >&2; exit 3; }
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null \
    && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  python3 -I - "${dir}" <<'PY'
import os, stat, sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    if line.strip(): listed.add(line.split(None,1)[1].lstrip('*'))
actual=set()
for root,dirs,files in os.walk(d,followlinks=False):
    rp=Path(root); dirs[:]=[n for n in dirs if not (rp/n).is_symlink()]
    for name in files:
        p=rp/name
        if name in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(str(p.relative_to(d)))
assert listed == actual, (listed-actual,actual-listed)
PY
}

WORK_ACTIVE=0
on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE\nexit_code=%s\nfunctional_vcs_verified=false\n' \
      "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    local q="${RESULT}.failed_or_incomplete.$$.quarantine"
    [[ ! -e "${q}" ]] && mv -- "${WORK}" "${q}" || true
  fi
}
trap on_exit EXIT

# Exact design/tool identities.
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact ad89adc7e9aefd350a225e58e85540ec579bbbe1ce9730891633f311de4eb4f5 "${EXEC_SVA}"
sha_exact eb20ffb5f910d0e3b8eebf836194298d38c719f512b207f38d15e75fc2df9f07 "${MATCH_SVA}"
sha_exact ab4b4d41ae1daedced757b9682f9b005776921eff4f2f1b9ae2dc40e654388e3 "${TB}"
sha_exact 6e82968900c328d3d81bdc0e6a30cd17760351236b56dccd6236fd048e1114e9 "${STATIC_CHECK}"
sha_exact 9efa7c5426e18e2b06eeb825ee43e35c96dd4358f5de879260cb96c86e8ce978 "${SOURCE_CONTRACT}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

# The M948 DRAFT and its independent M949 source hammer are live and sealed.
[[ -f "${SOURCE_CONTRACT}.sha256" \
   && -f "${SOURCE_CONTRACT}.sha256.seal.sha256" ]] || {
  echo "ERROR: M948 source contract seals absent" >&2; exit 3; }
(cd -- "$(dirname -- "${SOURCE_CONTRACT}")" &&
  sha256sum -c "$(basename -- "${SOURCE_CONTRACT}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${SOURCE_CONTRACT}.sha256.seal.sha256")" >/dev/null)
verify_recursive_seal "${M949_DIR}"
sha_exact cc097be4c5d9fbb6a2a5ad19770127ff9af04f681e556dfe49e9ecc24949b492 "${M949_DIR}/review.json"
sha_exact 78866111d3af2d5b62ed255f4747c5f999ee33da5e5ba51395e8652b65b25af4 "${M949_DIR}/SHA256SUMS"
sha_exact 6c417c3f85bfb93f12b5868e1edd6d36a89e83517587489827271d486737fb40 "${M949_DIR}/SHA256SUMS.seal.sha256"
python3 -I "${STATIC_CHECK}"
python3 -I - "${M949_DIR}/review.json" <<'PY'
import json,sys
from pathlib import Path
h=json.loads(Path(sys.argv[1]).read_text())
assert h['review_status']=='PASS_M949_M948_CAUSAL_DUAL_SOURCE_HAMMER'
assert h['verdict']=='GO' and h['score']>=95
assert h['issue_counts']['P0']==0 and h['issue_counts']['P1']==0
assert h['checks']['coverage_minima_unchanged'] is True
assert h['checks']['public_sink_force_only'] is True
assert h['checks']['phase_macro_read'] is True
assert h['checks']['phase_response_plus_forward'] is True
assert h['checks']['phase_delayed_debug_and_cleanroom'] is True
PY

# The consumed M943 identity remains immutable evidence and can never be reused.
sha_exact 4452edf87ffd4d7b31c49a9ca8dff04d1b2f288853f6fd41f96f2bf8c9fed6a6 "${M943_ATTEMPT}/identity.txt"
verify_recursive_seal "${M943_QUARANTINE}"
sha_exact c85fea3d0384eb8a804520cb493e9688c3a4406178a6b7b054a82c367693a908 "${M943_QUARANTINE}/SHA256SUMS"
sha_exact 5c3887456efc1609ef3f911310e49f1eaf1a64d24279e640c84ba7b761e51b70 "${M943_QUARANTINE}/SHA256SUMS.seal.sha256"

# M950's own source contract must be double sealed before any future release.
[[ -f "${RUN_CONTRACT}.sha256" && -f "${RUN_CONTRACT}.sha256.seal.sha256" ]] || {
  echo "ERROR: M950 run source contract seals absent" >&2; exit 3; }
(cd -- "$(dirname -- "${RUN_CONTRACT}")" &&
  sha256sum -c "$(basename -- "${RUN_CONTRACT}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${RUN_CONTRACT}.sha256.seal.sha256")" >/dev/null)

# A distinct future M951 hammer and M952 release are both mandatory.
verify_recursive_seal "${HAMMER_DIR}"
sha_exact "${M950_EXPECTED_HAMMER_REVIEW_SHA256}" "${HAMMER_DIR}/review.json"
sha_exact "${M950_EXPECTED_HAMMER_OUTER_SHA256}" "${HAMMER_DIR}/SHA256SUMS.seal.sha256"
sha_exact "${M950_EXPECTED_RELEASE_SHA256}" "${RELEASE}"
[[ -f "${RELEASE}.sha256" && -f "${RELEASE}.sha256.seal.sha256" ]] || {
  echo "ERROR: launch release seals absent" >&2; exit 3; }
(cd -- "$(dirname -- "${RELEASE}")" &&
  sha256sum -c "$(basename -- "${RELEASE}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${RELEASE}.sha256.seal.sha256")" >/dev/null)

python3 -I - "${RUN_CONTRACT}" "${RUNNER}" "${HAMMER_DIR}/review.json" \
  "${HAMMER_DIR}/SHA256SUMS" "${HAMMER_DIR}/SHA256SUMS.seal.sha256" \
  "${RELEASE}" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,runner,review,manifest,outer,release=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); h=json.loads(review.read_text()); r=json.loads(release.read_text())
assert c['status']=='SOURCE_READY_AWAIT_M951_RUNNER_HAMMER_AND_M952_RELEASE__NO_VCS_RELEASE'
assert c['identity']['runner_sha256']==sha(runner)
assert h['review_status']=='PASS_M951_M950_VCS_RUNNER_SOURCE_HAMMER'
assert h['verdict']=='GO' and h['score']>=95
assert h['issue_counts']['P0']==0 and h['issue_counts']['P1']==0
assert r['status']=='AUTHORIZE_ONE_M950_FUNCTIONAL_VCS_ATTEMPT'
i=r['identity']
assert i['runner_sha256']==sha(runner) and i['contract_sha256']==sha(contract)
assert i['hammer_review_sha256']==sha(review)
assert i['hammer_manifest_sha256']==sha(manifest)
assert i['hammer_outer_seal_file_sha256']==sha(outer)
assert r['authorization']=={'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0}
for d in (c,r):
    b=d['claim_boundary']
    for key in ('timing_verified','cycles_measured','speedup','ppa','power','energy','system_speedup','paper_citable'):
        assert b[key] is False
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] || {
  echo "ERROR: M950 result/attempt/work identity already exists" >&2; exit 4; }

python3 -I - <<'PY'
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
        if p.stat().st_uid != os.getuid(): continue
        comm=(p/'comm').read_text().strip()
        argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or (blocked & {Path(x).name for x in argv}): hits.append((p.name,comm,argv[:4]))
if hits: raise SystemExit('EDA collision: %r' % hits)
PY

mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]] || {
  echo "ERROR: MemAvailable below 64 GiB" >&2; exit 5; }

mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\ncreated_utc=%s\n' \
  "$(sha256sum -- "${RUNNER}" | awk '{print $1}')" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${ATTEMPT}/identity.txt"
mkdir -- "${WORK}"
WORK_ACTIVE=1
cd -- "${WORK}"

export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"
export SNPSLMD_LICENSE_FILE="27030@ic.ismd-nemo"
export LM_LICENSE_FILE="/opt/synopsys/Synopsys.dat"

"${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  -debug_access+pp +define+UNIT_DELAY +vcs+lic+wait \
  "${FOUNDRY_V}" "${MACRO_RTL}" "${RTL}" "${EXEC_SVA}" \
  "${MATCH_SVA}" "${TB}" -top "${TOP}" -o simv 2>&1 | tee compile.log
compile_rc=("${PIPESTATUS[@]}")
[[ "${compile_rc[0]}" -eq 0 && "${compile_rc[1]}" -eq 0 ]] || exit 20

/usr/bin/timeout --signal=TERM --kill-after=30s 900s ./simv -no_save 2>&1 | tee sim.log
sim_rc=("${PIPESTATUS[@]}")
[[ "${sim_rc[0]}" -eq 0 && "${sim_rc[1]}" -eq 0 ]] || exit 21

[[ "$(rg -c "^${PASS_TOKEN} " sim.log)" -eq 1 ]] || exit 22
[[ "$(rg -c '^COVERAGE_M948_CAUSAL_DUAL_ENQUEUE ' sim.log)" -eq 1 ]] || exit 23
[[ "$(rg -c '^COVERAGE_M938_MATCH_RESET ' sim.log)" -eq 1 ]] || exit 24
[[ "$(rg -c '^COVERAGE_M938_EXACT_MATCH_PIPELINE ' sim.log)" -eq 1 ]] || exit 25
[[ "$(rg -c '^COVERAGE_M912_C1_METADATA_PIPELINE ' sim.log)" -eq 1 ]] || exit 26
[[ "$(rg -c '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)" -eq 1 ]] || exit 27
rg -q '^COVERAGE_M948_CAUSAL_DUAL_ENQUEUE macro_read=1 response_plus_forward=1 delayed_debug=1 cleanroom_cover=[1-9][0-9]* public_sink_window=1 internal_force=0$' sim.log || exit 28
rg -q '^PASS_M948_C1_THREE_STAGE_EXACT_MATCH_PIPELINE_UNIT_DELAY_CANDIDATE .*attacks=6 .*bank_distinct_overlap=[1-9][0-9]* .*reset_F=1 .*reset_G=1 .*reset_R63=1 .*causal_macro=1 .*causal_pair=1 .*causal_debug=1 .*cleanroom_dual=[1-9][0-9]* .*public_sink_window=1 internal_force=0 functional_vcs_only=true .*timing_verified=false .*speedup=false .*ppa=false .*headline=false$' sim.log || exit 29

python3 -I - "${RUNNER}" "${RUN_CONTRACT}" "${RTL}" "${EXEC_SVA}" \
  "${MATCH_SVA}" "${TB}" "${STATIC_CHECK}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,rtl,exec_sva,match_sva,tb,checker=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m950_m948_m935_c1_causal_dual_enqueue_unit_delay_vcs_receipt_v1',
   'status':'PASS_FUNCTIONAL_VCS_ONLY',
   'created_utc':datetime.now(timezone.utc).isoformat(),
   'identity':{'runner_sha256':sha(runner),'contract_sha256':sha(contract),
               'rtl_sha256':sha(rtl),'exec_sva_sha256':sha(exec_sva),
               'match_sva_sha256':sha(match_sva),'tb_sha256':sha(tb),
               'static_checker_sha256':sha(checker)},
   'macro_model':'foundry_UNIT_DELAY_functional',
   'attack_count':6,'reset_stage_tests':3,
   'causal_dual_enqueue':{'macro_read':1,'response_plus_forward':1,
                          'delayed_debug':1,'public_sink_window':True,
                          'internal_force':False},
   'claim_boundary':{'functional_vcs_verified':True,'timing_verified':False,
      'cycles_measured':False,'speedup':False,'ppa':False,'power':False,
      'energy':False,'system_speedup':False,'paper_citable':False}}
Path('m950_m948_m935_c1_causal_dual_enqueue_unit_delay_vcs_receipt_r1.json').write_text(
    json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M950/M948/M935 functional VCS result=%s\n' "${RESULT}"
