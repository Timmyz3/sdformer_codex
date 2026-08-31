#!/usr/bin/env bash
set -euo pipefail
umask 002

# M943: one future-released foundry UNIT_DELAY functional VCS attempt for the
# M938/M935 exact F/G/R parent-match candidate.  This runner never establishes
# clock timing, workload cycles, speedup, PPA, power, energy or system results.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
[[ -n "${M943_EXPECTED_RELEASE_SHA256:-}" \
   && -n "${M943_EXPECTED_HAMMER_REVIEW_SHA256:-}" \
   && -n "${M943_EXPECTED_HAMMER_OUTER_SHA256:-}" ]] || {
  echo "ERROR: exact release/hammer SHA environment absent" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
RTL="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
EXEC_SVA="${HW_ROOT}/verif_m935_c1_match_pipeline/m935_m912_inherited_execution_assertions_r1.sv"
MATCH_SVA="${HW_ROOT}/verif_m935_c1_match_pipeline/m938_three_stage_exact_match_assertions_r2.sv"
TB="${HW_ROOT}/verif_m935_c1_match_pipeline/tb_m938_three_stage_match_pipeline_unit_delay_r2.sv"
STATIC_CHECK="${HW_ROOT}/verif_m935_c1_match_pipeline/static_check_m938_three_stage_exact_match_pipeline.py"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m938_m937_m935_three_stage_exact_match_verification_repair_source_contract_r1_20260829.json"
RUN_CONTRACT="${HW_ROOT}/contracts/m943_m941_m938_m935_c1_three_stage_exact_match_vcs_source_contract_r1_20260829.json"
M941_DIR="${HW_ROOT}/reviews/m941_m938_m937_m935_three_stage_exact_match_source_hammer_r1_20260829"
HAMMER_DIR="${HW_ROOT}/reviews/m944_m943_m941_m938_m935_c1_vcs_source_hammer_r1_20260829"
RELEASE="${HW_ROOT}/contracts/m945_m944_m943_m938_m935_c1_vcs_launch_release_r1_20260829.json"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
RESULT="${HW_ROOT}/results/m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_r1_20260829"
ATTEMPT="${HW_ROOT}/results/.m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_r1_attempt_consumed"
WORK="${HW_ROOT}/results/.m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_r1_work.$$"
TOP="tb_m938_three_stage_match_pipeline_unit_delay_r2"
PASS_TOKEN="PASS_M938_C1_THREE_STAGE_EXACT_MATCH_PIPELINE_UNIT_DELAY_CANDIDATE"

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

sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact ad89adc7e9aefd350a225e58e85540ec579bbbe1ce9730891633f311de4eb4f5 "${EXEC_SVA}"
sha_exact eb20ffb5f910d0e3b8eebf836194298d38c719f512b207f38d15e75fc2df9f07 "${MATCH_SVA}"
sha_exact 6b5d58bd35176b5532c21526c6406eaaf7928693c90d1daea51a23998c260e9e "${TB}"
sha_exact 3265da878693376dcc46154e6b64b912fca89686d65aaf05558582e5cfd437d5 "${STATIC_CHECK}"
sha_exact 5d8c91792666b445f13eb9b5542f5a96e7c396b85887e6f94bdbd718276122d1 "${SOURCE_CONTRACT}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

verify_recursive_seal "${M941_DIR}"
sha_exact 6555fd00ef0cbf27e713dad20c4efdb3c6f3b56e1c12a3983ed6e9c132be2524 "${M941_DIR}/review.json"
sha_exact 3aae910626edbefbdfb3388e6bac1730cdb0c28cc3036ec7032b79b03334a439 "${M941_DIR}/SHA256SUMS"
sha_exact 8bac72dcfdfadf52b8a1d584159c5285e0ef3c85c4d67d0188ae3ca64bc256a2 "${M941_DIR}/SHA256SUMS.seal.sha256"

[[ -f "${RUN_CONTRACT}.sha256" && -f "${RUN_CONTRACT}.sha256.seal.sha256" ]] || {
  echo "ERROR: run source contract seals absent" >&2; exit 3; }
(cd -- "$(dirname -- "${RUN_CONTRACT}")" &&
  sha256sum -c "$(basename -- "${RUN_CONTRACT}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${RUN_CONTRACT}.sha256.seal.sha256")" >/dev/null)
python3 -I "${STATIC_CHECK}"

verify_recursive_seal "${HAMMER_DIR}"
sha_exact "${M943_EXPECTED_HAMMER_REVIEW_SHA256}" "${HAMMER_DIR}/review.json"
sha_exact "${M943_EXPECTED_HAMMER_OUTER_SHA256}" "${HAMMER_DIR}/SHA256SUMS.seal.sha256"
sha_exact "${M943_EXPECTED_RELEASE_SHA256}" "${RELEASE}"
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
assert c['status']=='SOURCE_READY_FOR_FRESH_M944_HAMMER__NO_VCS_RELEASE'
assert c['identity']['runner_sha256']==sha(runner)
assert h['review_status']=='PASS_M944_M943_VCS_SOURCE_HAMMER'
assert h['verdict']=='GO' and h['score']>=95
assert h['issue_counts']['P0']==0 and h['issue_counts']['P1']==0
assert r['status']=='AUTHORIZE_ONE_M943_FUNCTIONAL_VCS_ATTEMPT'
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
  echo "ERROR: result/attempt/work identity already exists" >&2; exit 4; }

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
[[ "$(rg -c '^COVERAGE_M938_MATCH_RESET ' sim.log)" -eq 1 ]] || exit 23
[[ "$(rg -c '^COVERAGE_M938_EXACT_MATCH_PIPELINE ' sim.log)" -eq 1 ]] || exit 24
[[ "$(rg -c '^COVERAGE_M912_C1_METADATA_PIPELINE ' sim.log)" -eq 1 ]] || exit 25
[[ "$(rg -c '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)" -eq 1 ]] || exit 26
rg -q '^PASS_M938_C1_THREE_STAGE_EXACT_MATCH_PIPELINE_UNIT_DELAY_CANDIDATE .*attacks=6 .*bank_distinct_overlap=[1-9][0-9]* .*reset_F=1 .*reset_G=1 .*reset_R63=1 .*functional_vcs_only=true .*timing_verified=false .*speedup=false .*ppa=false .*headline=false$' sim.log || exit 27

python3 -I - "${RUNNER}" "${RUN_CONTRACT}" "${RTL}" "${EXEC_SVA}" \
  "${MATCH_SVA}" "${TB}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,rtl,exec_sva,match_sva,tb=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_receipt_v1',
   'status':'PASS_FUNCTIONAL_VCS_ONLY',
   'created_utc':datetime.now(timezone.utc).isoformat(),
   'identity':{'runner_sha256':sha(runner),'contract_sha256':sha(contract),
               'rtl_sha256':sha(rtl),'exec_sva_sha256':sha(exec_sva),
               'match_sva_sha256':sha(match_sva),'tb_sha256':sha(tb)},
   'macro_model':'foundry_UNIT_DELAY_functional',
   'attack_count':6,'reset_stage_tests':3,
   'claim_boundary':{'functional_vcs_verified':True,'timing_verified':False,
      'cycles_measured':False,'speedup':False,'ppa':False,'power':False,
      'energy':False,'system_speedup':False,'paper_citable':False}}
Path('m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_receipt_r1.json').write_text(
    json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M943/M938/M935 functional VCS result=%s\n' "${RESULT}"
