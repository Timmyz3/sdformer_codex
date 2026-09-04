#!/usr/bin/env bash
set -euo pipefail
umask 002

# Source-only M1168R2 compile repair.  The consumed r1 attempt is immutable and
# cannot be retried.  This runner remains fail-closed until a fresh M1172
# different-author hammer and separately sealed M1173 release authorize exactly
# one new-namespace foundry UNIT_DELAY functional VCS attempt.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
[[ -n "${M1168R2_EXPECTED_RELEASE_SHA256:-}" \
   && -n "${M1168R2_EXPECTED_HAMMER_REVIEW_SHA256:-}" \
   && -n "${M1168R2_EXPECTED_HAMMER_OUTER_SHA256:-}" ]] || {
  echo "ERROR: exact fresh-hammer/release SHA environment absent" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
WRAPPER="${HW_ROOT}/rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
SVA="${HW_ROOT}/verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv"
TB="${HW_ROOT}/verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv"
STATIC_CHECK="${HW_ROOT}/verif_m1168r2_c1_common_charge_protocol/static_check_m1168r2_m1162_vcs_source.py"
FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
M1162_CONTRACT="${HW_ROOT}/contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json"
M1162_AUTHOR="${HW_ROOT}/reviews/m1162_m1160_c1_common_charge_protocol_repair_source_author_receipt_r1_20260830"
M1166_HAMMER="${HW_ROOT}/reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830"
RUN_CONTRACT="${HW_ROOT}/contracts/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_contract_r1_20260830.json"
AUTHOR_DIR="${HW_ROOT}/reviews/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_author_receipt_r1_20260830"
HAMMER_DIR="${HW_ROOT}/reviews/m1172_m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_hammer_r1_20260830"
RELEASE="${HW_ROOT}/contracts/m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_launch_release_r2_20260830.json"
R1_ATTEMPT_ID="${HW_ROOT}/results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed/identity.txt"
R1_QUARANTINE="${HW_ROOT}/results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.3074649.quarantine"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON_BIN="/opt/anaconda3/envs/pytorch310/bin/python3.10"
RESULT="${HW_ROOT}/results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"
ATTEMPT="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"
WORK="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.$$"
TOP="tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2"
PASS_TOKEN="PASS_M1168R2_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE"

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
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null \
    && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  "${PYTHON_BIN}" -I - "${dir}" <<'PY'
import os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set(); actual=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    if line.strip(): listed.add(line.split(None,1)[1].lstrip('*'))
for root,dirs,files in os.walk(d,followlinks=False):
    rp=Path(root); dirs[:]=[n for n in dirs if not (rp/n).is_symlink()]
    for name in files:
        p=rp/name
        if name in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(str(p.relative_to(d)))
assert listed==actual,(listed-actual,actual-listed)
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

sha_exact 639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595 "${WRAPPER}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${M935}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${PARENT}"
sha_exact 59ff9141175159e9043d86dd5932a4113fde88582005487f1eb65e372c6a684f "${SVA}"
sha_exact bd5a2c3ce1ab9f03a7017756c96d5013577116583fc7d007ef3374593272ee35 "${TB}"
sha_exact 022cf2d61d29cb22547db78de3dc8f5dbbbc8e0b03443c7469abd4f56d6beae8 "${STATIC_CHECK}"
sha_exact 96331eb20fb6d4e72e157d23c579841a121103053ed6246f0b76f812399f1411 "${FILELIST}"
sha_exact 5787f3302aa3308485e357c41385e69da93e6b41bfdea92410690af5a95ecbdc "${M1162_CONTRACT}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115 "${PYTHON_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
sha_exact 7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae "${R1_ATTEMPT_ID}"
verify_recursive_seal "${R1_QUARANTINE}"
sha_exact 39765d45f5e53de02a4c9139915253b0d0d8190f042027b70344dea08b0037ff "${R1_QUARANTINE}/compile.log"
sha_exact 6f7d480bc752ea5835c3442de72f8e5e484ae41db3a5377b49e593e13838614c "${R1_QUARANTINE}/SHA256SUMS"
sha_exact 72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7 "${R1_QUARANTINE}/SHA256SUMS.seal.sha256"

[[ -f "${M1162_CONTRACT}.sha256" && -f "${M1162_CONTRACT}.sha256.seal.sha256" ]] || exit 3
(cd -- "$(dirname -- "${M1162_CONTRACT}")" &&
  sha256sum -c "$(basename -- "${M1162_CONTRACT}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${M1162_CONTRACT}.sha256.seal.sha256")" >/dev/null)
verify_recursive_seal "${M1162_AUTHOR}"
sha_exact 734ce901318bcc62951a7b479f3d42d0230fbc7a3be9c39137270858f9ad71a5 "${M1162_AUTHOR}/review.json"
sha_exact da799abfdad2dab521ba90f48b8956a5ddcd1dee95aaf675a184b281fa34f302 "${M1162_AUTHOR}/SHA256SUMS"
sha_exact 67cb13ac317f140f4a042373a1c79640295bb861ffc25905605c65656c5fe18a "${M1162_AUTHOR}/SHA256SUMS.seal.sha256"
verify_recursive_seal "${M1166_HAMMER}"
sha_exact 7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c "${M1166_HAMMER}/review.json"
sha_exact da8daaef6b6832dd2d3278fcbdf61613170f07da5bb65e311915a3c421e76363 "${M1166_HAMMER}/SHA256SUMS"
sha_exact afc25e37fa8b3b5c5bd8e8c1b3582fecc5d2d75450df86b7c48f71e992ea02ef "${M1166_HAMMER}/SHA256SUMS.seal.sha256"

[[ -f "${RUN_CONTRACT}.sha256" && -f "${RUN_CONTRACT}.sha256.seal.sha256" ]] || {
  echo "ERROR: M1168 source contract seals absent" >&2; exit 3; }
(cd -- "$(dirname -- "${RUN_CONTRACT}")" &&
  sha256sum -c "$(basename -- "${RUN_CONTRACT}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${RUN_CONTRACT}.sha256.seal.sha256")" >/dev/null)
verify_recursive_seal "${AUTHOR_DIR}"
"${PYTHON_BIN}" -I "${STATIC_CHECK}" >/dev/null

verify_recursive_seal "${HAMMER_DIR}"
sha_exact "${M1168R2_EXPECTED_HAMMER_REVIEW_SHA256}" "${HAMMER_DIR}/review.json"
sha_exact "${M1168R2_EXPECTED_HAMMER_OUTER_SHA256}" "${HAMMER_DIR}/SHA256SUMS.seal.sha256"
sha_exact "${M1168R2_EXPECTED_RELEASE_SHA256}" "${RELEASE}"
[[ -f "${RELEASE}.sha256" && -f "${RELEASE}.sha256.seal.sha256" ]] || exit 3
(cd -- "$(dirname -- "${RELEASE}")" &&
  sha256sum -c "$(basename -- "${RELEASE}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${RELEASE}.sha256.seal.sha256")" >/dev/null)

"${PYTHON_BIN}" -I - "${RUN_CONTRACT}" "${RUNNER}" "${HAMMER_DIR}/review.json" \
  "${HAMMER_DIR}/SHA256SUMS" "${HAMMER_DIR}/SHA256SUMS.seal.sha256" \
  "${RELEASE}" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,runner,review,manifest,outer,release=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); h=json.loads(review.read_text()); r=json.loads(release.read_text())
assert c['status']=='SOURCE_READY_FOR_FRESH_M1172_HAMMER__NO_VCS_RELEASE'
assert c['identity']['runner_sha256']==sha(runner)
assert h['status']=='PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE'
assert h['verdict']=='GO' and h['score']>=95
assert h['issue_counts']['P0']==0 and h['issue_counts']['P1']==0
assert r['status']=='AUTHORIZE_EXACTLY_ONE_M1168R2_FUNCTIONAL_VCS_ATTEMPT'
i=r['identity']
assert i['runner_sha256']==sha(runner) and i['contract_sha256']==sha(contract)
assert i['hammer_review_sha256']==sha(review)
assert i['hammer_manifest_sha256']==sha(manifest)
assert i['hammer_outer_seal_file_sha256']==sha(outer)
assert r['authorization']=={'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0}
for d in (c,r):
    b=d['claim_boundary']
    for key in ('functional_vcs_verified','timing_verified','cycles_measured',
                'speedup','ppa','power','energy','system_speedup','paper_citable'):
        assert b[key] is False
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] || {
  echo "ERROR: result/attempt/work identity already exists" >&2; exit 4; }

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
        if p.stat().st_uid != os.getuid(): continue
        comm=(p/'comm').read_text().strip()
        argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or (blocked & {Path(x).name for x in argv}):
        hits.append((p.name,comm,argv[:4]))
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
  +define+UNIT_DELAY +vcs+lic+wait -f "${FILELIST}" -top "${TOP}" \
  -o simv 2>&1 | tee compile.log
compile_rc=("${PIPESTATUS[@]}")
[[ "${compile_rc[0]}" -eq 0 && "${compile_rc[1]}" -eq 0 ]] || exit 20

/usr/bin/timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save \
  2>&1 | tee sim.log
sim_rc=("${PIPESTATUS[@]}")
[[ "${sim_rc[0]}" -eq 0 && "${sim_rc[1]}" -eq 0 ]] || exit 21

[[ "$(rg -c "^${PASS_TOKEN} " sim.log)" -eq 1 ]] || exit 22
[[ "$(rg -c '^COVERAGE_M1168R2_PROTOCOL ' sim.log)" -eq 1 ]] || exit 23
[[ "$(rg -c '^COVERAGE_M1168R2_RESETS_ATTACKS ' sim.log)" -eq 1 ]] || exit 24
[[ "$(rg -c '^COVERAGE_M1168R2_SERVICE_ASSUMPTIONS ' sim.log)" -eq 1 ]] || exit 25
[[ "$(rg -c '^COVERAGE_M1168R2_FROZEN_M935 ' sim.log)" -eq 1 ]] || exit 26
rg -q '^PASS_M1168R2_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE directed_random=24 protocol_attacks=7 service_assumption_attacks=2 reset_states=3 ii=2 normal_m935_rows=1 normal_m935_tasks=1 functional_vcs_only=true timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false$' sim.log || exit 27
if rg -qi 'assertion[^\n]*(fail|error)|error-[A-Z0-9]+|\$fatal' sim.log; then
  echo "ERROR: assertion/error signature in sim.log" >&2; exit 28
fi

"${PYTHON_BIN}" -I - "${RUNNER}" "${RUN_CONTRACT}" "${WRAPPER}" "${M935}" \
  "${SVA}" "${TB}" "${FILELIST}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,wrapper,m935,sva,tb,filelist=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_receipt_r2_v1',
   'status':'PASS_FUNCTIONAL_VCS_ONLY',
   'created_utc':datetime.now(timezone.utc).isoformat(),
   'identity':{'runner_sha256':sha(runner),'contract_sha256':sha(contract),
      'wrapper_sha256':sha(wrapper),'m935_sha256':sha(m935),
      'sva_sha256':sha(sva),'tb_sha256':sha(tb),'filelist_sha256':sha(filelist)},
   'macro_model':'foundry_UNIT_DELAY_functional',
   'coverage':{'directed_random_transactions':24,'protocol_attacks':7,
      'service_assumption_attacks':2,'reset_pending_states':3,
      'minimum_completed_issue_ii':2,'normal_m935_rows':1,'normal_m935_tasks':1},
   'claim_boundary':{'functional_vcs_verified':True,'timing_verified':False,
      'cycles_measured':False,'speedup':False,'ppa':False,'power':False,
      'energy':False,'system_speedup':False,'paper_citable':False}}
Path('m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_receipt_r2.json').write_text(
    json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M1168R2/M1162 functional VCS result=%s\n' "${RESULT}"
