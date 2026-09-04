#!/usr/bin/env bash
set -euo pipefail
umask 002

# Inert until a fresh M1214 release hammer supplies three independent digests.
# This runner admits one foundry UNIT_DELAY functional compile and one sim only.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
for name in M1213_EXPECTED_RELEASE_SHA256 \
            M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256 \
            M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256 \
            M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256; do
  value="${!name:-}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: ${name} absent/invalid" >&2; exit 2; }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
WRAPPER="${HW_ROOT}/rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
SVA="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
TB="${HW_ROOT}/verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv"
R8_CHECKER="${HW_ROOT}/verif_m1210r8_c1_common_charge_protocol/static_check_m1210r8_m1162_vcs_source.py"
M1213_CHECKER="${HW_ROOT}/verif_m1213_c1_r8_vcs_release/static_check_m1213_c1_r8_vcs_release_source.py"
FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
M1210_CONTRACT="${HW_ROOT}/contracts/m1210_m1207_m1198_m1162_c1_r8_random_request_quiesce_source_contract_r1_20260830.json"
M1210_AUTHOR="${HW_ROOT}/reviews/m1210_m1207_c1_r8_random_request_quiesce_author_receipt_r1_20260830"
M1212_HAMMER="${HW_ROOT}/reviews/m1212_m1210_c1_r8_random_request_quiesce_source_hammer_r1_20260830"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m1213_m1212_m1210_c1_r8_vcs_launcher_source_contract_r1_20260830.json"
RELEASE="${HW_ROOT}/contracts/m1213_m1212_m1210_c1_r8_vcs_launch_release_r1_20260830.json"
AUTHOR_DIR="${HW_ROOT}/reviews/m1213_m1212_m1210_c1_r8_vcs_release_author_receipt_r1_20260830"
RELEASE_HAMMER="${HW_ROOT}/reviews/m1214_m1213_m1210_c1_r8_vcs_release_source_hammer_r1_20260830"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON_BIN="/opt/anaconda3/envs/pytorch310/bin/python3.10"
ATTEMPT="${HW_ROOT}/results/.m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_attempt_consumed"
RESULT="${HW_ROOT}/results/m1213_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs_r8_20260830"
WORK="${HW_ROOT}/results/.m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_work.$$"
TOP="tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8"
PASS_TOKEN="PASS_M1210R8_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE"

sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "ERROR: missing/nonregular ${path}" >&2; exit 3; }
  got="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${got}" == "${expected}" ]] || { echo "ERROR: SHA mismatch ${path}: ${got}" >&2; exit 3; }
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
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || { echo "ERROR: recursive seal absent ${dir}" >&2; exit 3; }
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
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
        if name in {'SHA256SUMS','SHA256SUMS.seal.sha256'} or p.is_symlink(): continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(str(p.relative_to(d)))
assert listed==actual,(listed-actual,actual-listed)
PY
}

WORK_ACTIVE=0
on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE\nexit_code=%s\nfunctional_vcs_verified=false\nautomatic_retry=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    local q="${RESULT}.failed_or_incomplete.$$.quarantine"
    [[ ! -e "${q}" ]] && mv -- "${WORK}" "${q}" || true
  fi
}
trap on_exit EXIT

sha_exact 639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595 "${WRAPPER}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${M935}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${PARENT}"
sha_exact c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472 "${SVA}"
sha_exact 060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b "${TB}"
sha_exact cce8219a13d7584f1c35e262ac4de3e4a935fddc53652d6ce322e7e5f94daa96 "${R8_CHECKER}"
sha_exact 6a14b6a42236aa22abf0d07bee49b9210ab0d695c6d07b702e5871773f67a10d "${M1213_CHECKER}"
sha_exact 048253d22301df9fb84502ff35f5129459a5b43e4ff9e8d11ea62973f7047af6 "${FILELIST}"
sha_exact 26ca340e8f33ca936b169c638862bc3a76f7233035d680cc14ddb7389bcc5d07 "${M1210_CONTRACT}"
sha_exact d9671bff7efa1e808d5008c23d02df119df4553b60d5782fb2e0ba8bb73efc4a "${M1210_AUTHOR}/review.json"
sha_exact cf9e56adcc15c33ca7663502cdad741c1287dc64d8e2f79df55b9120d986cc5a "${M1210_AUTHOR}/SHA256SUMS"
sha_exact 28a209d39c1211a0c9c20b43b471cea68d1e5492d516c332120cc1098a773826 "${M1210_AUTHOR}/SHA256SUMS.seal.sha256"
sha_exact 550d4459ce34f0b01c43ac913123e247270b66c7bd83678d01228b227839fe4d "${M1212_HAMMER}/review.json"
sha_exact 349306f94a43c93acbee71e926ef36474d2bdf0bb1c12f597037fd8b597165a7 "${M1212_HAMMER}/SHA256SUMS"
sha_exact 92e1640e01288841a768d165dc66bbb5bd87fa3f0385bfc88f5843099ece9909 "${M1212_HAMMER}/SHA256SUMS.seal.sha256"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115 "${PYTHON_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

for artifact in "${M1210_CONTRACT}" "${SOURCE_CONTRACT}" "${RELEASE}"; do
  [[ -f "${artifact}.sha256" && -f "${artifact}.sha256.seal.sha256" ]] || exit 3
  (cd -- "$(dirname -- "${artifact}")" && sha256sum -c "$(basename -- "${artifact}.sha256")" >/dev/null && sha256sum -c "$(basename -- "${artifact}.sha256.seal.sha256")" >/dev/null)
done
verify_recursive_seal "${M1210_AUTHOR}"
verify_recursive_seal "${M1212_HAMMER}"
verify_recursive_seal "${AUTHOR_DIR}"
verify_recursive_seal "${RELEASE_HAMMER}"
sha_exact "${M1213_EXPECTED_RELEASE_SHA256}" "${RELEASE}"
sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" "${RELEASE_HAMMER}/review.json"
sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" "${RELEASE_HAMMER}/SHA256SUMS"
sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}" "${RELEASE_HAMMER}/SHA256SUMS.seal.sha256"

"${PYTHON_BIN}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${RELEASE}" \
  "${RELEASE_HAMMER}/review.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
runner,contract,release,review=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); r=json.loads(release.read_text()); x=json.loads(review.read_text())
assert c['status']=='M1213_R8_ACYCLIC_RELEASE_SOURCE_READY__FRESH_RELEASE_HAMMER_REQUIRED__NO_VCS_NO_EDA'
assert c['identity']['runner_sha256']==sha(runner)
assert r['status']=='AUTHORIZE_ONE_M1213_R8_FUNCTIONAL_VCS_ATTEMPT_AFTER_FRESH_RELEASE_HAMMER'
assert r['identity']['runner_sha256']==sha(runner) and r['identity']['source_contract_sha256']==sha(contract)
assert x['schema']=='m1214_m1213_m1210_c1_r8_vcs_release_source_hammer_review_r1_v1'
assert x['status']=='PASS_M1214_M1213_C1_R8_ACYCLIC_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH'
assert x['verdict']=='GO' and x['score']>=95 and x['issue_counts']['P0']==0 and x['issue_counts']['P1']==0
assert x['identity']['runner_sha256']==sha(runner)
assert x['identity']['source_contract_sha256']==sha(contract)
assert x['identity']['release_sha256']==sha(release)
for forbidden in ('hammer_manifest_sha256','hammer_outer_seal_file_sha256','manifest_sha256','outer_seal_file_sha256'):
    assert forbidden not in x.get('identity',{}), 'self-reference forbidden: '+forbidden
for env_name in ('M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256','M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256'):
    assert os.environ[env_name] not in review.read_text(), 'self-digest embedded in sealed review'
assert x['authorization']=={'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0}
for d in (c,r,x):
    for key in ('functional_vcs_verified','timing_verified','cycles_measured','speedup','ppa','power','energy','system_speedup','paper_citable'):
        assert d['claim_boundary'][key] is False
PY

"${PYTHON_BIN}" -I "${R8_CHECKER}" >/dev/null
"${PYTHON_BIN}" -I "${M1213_CHECKER}" >/dev/null

[[ ! -e "${ATTEMPT}" && ! -e "${RESULT}" && ! -e "${WORK}" ]] || { echo "ERROR: M1213 namespace not fresh" >&2; exit 4; }
compgen -G "${HW_ROOT}/results/.m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_work.*" >/dev/null && { echo "ERROR: stale M1213 work" >&2; exit 4; } || true
compgen -G "${HW_ROOT}/results/m1213_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs_r8_20260830.failed_or_incomplete.*" >/dev/null && { echo "ERROR: prior M1213 quarantine" >&2; exit 4; } || true

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
        comm=(p/'comm').read_text().strip(); argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or (blocked & {Path(x).name for x in argv}): hits.append((p.name,comm,argv[:4]))
if hits: raise SystemExit('EDA collision: %r' % hits)
PY
mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]] || { echo "ERROR: MemAvailable below 64 GiB" >&2; exit 5; }

/bin/mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\nrelease_sha256=%s\nrelease_hammer_review_sha256=%s\nrelease_hammer_manifest_sha256=%s\nrelease_hammer_outer_seal_file_sha256=%s\ncreated_utc=%s\nautomatic_retry=false\n' \
  "$(sha256sum -- "${RUNNER}" | awk '{print $1}')" "${M1213_EXPECTED_RELEASE_SHA256}" \
  "${M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" "${M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
  "${M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${ATTEMPT}/identity.txt"
/bin/mkdir -- "${WORK}"
WORK_ACTIVE=1
cd -- "${WORK}"

export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"
export SNPSLMD_LICENSE_FILE="27030@ic.ismd-nemo"
export LM_LICENSE_FILE="/opt/synopsys/Synopsys.dat"
"${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  +define+UNIT_DELAY +vcs+lic+wait -f "${FILELIST}" -top "${TOP}" -o simv 2>&1 | tee compile.log
compile_rc=("${PIPESTATUS[@]}"); [[ "${compile_rc[0]}" -eq 0 && "${compile_rc[1]}" -eq 0 ]] || exit 20
/usr/bin/timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save 2>&1 | tee sim.log
sim_rc=("${PIPESTATUS[@]}"); [[ "${sim_rc[0]}" -eq 0 && "${sim_rc[1]}" -eq 0 ]] || exit 21

[[ "$(rg -c "^${PASS_TOKEN} " sim.log)" -eq 1 ]] || exit 22
rg -q '^COVERAGE_M1210R8_PROTOCOL weight_first=1 psum_first=1 weight_rsp_first=1 psum_rsp_first=1 long_request=5 long_response=5 nonfirst=1 ii2=1 no_duplicate_request=25 random=24 legal_masks_clear=29 random_request_quiesce=24$' sim.log || exit 23
rg -q '^COVERAGE_M1210R8_RESETS_ATTACKS reset_partial=1 reset_complete=1 reset_skew=1 unsolicited_weight=1 unsolicited_psum=1 same_cycle=1 duplicate_response=1 cancel=1 tuple_mutation=1 nonfirst_psum=1 request_attack_windows=2$' sim.log || exit 24
rg -q '^COVERAGE_M1210R8_SERVICE_ASSUMPTIONS weight_payload_mutation=1 psum_valid_drop=1 weight_windows=1 psum_windows=1 independent_checker=1 race_free_negedge_sample=1 skew_isolated=1 reachable_core_ready_force=0 boundary_fault=0 core_fault=0 dut_fault_claim=0$' sim.log || exit 25
rg -q '^COVERAGE_M1210R8_FROZEN_M935 normal_issues=2 normal_rows=1 normal_tasks=1 epoch=36865$' sim.log || exit 26
rg -q '^PASS_M1210R8_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE directed_random=24 protocol_attacks=7 service_assumption_attacks=2 request_attack_windows=2 legal_masks_clear=29 reset_states=3 ii=2 normal_m935_rows=1 normal_m935_tasks=1 random_request_quiesce=24 exactly_one_random_request_handshake=1 service_skew_isolated=1 reachable_core_ready_force=0 boundary_fault=0 core_fault=0 functional_vcs_only=true timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false$' sim.log || exit 27
if rg -qi 'assertion[^\n]*(fail|error)|error-[A-Z0-9]+|\$fatal' sim.log; then exit 28; fi

"${PYTHON_BIN}" -I - "${RUNNER}" "${SOURCE_CONTRACT}" "${RELEASE}" "${RELEASE_HAMMER}/review.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,release,hammer=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1213_m1210r8_m1162_c1_unit_delay_vcs_receipt_r8_v1','status':'PASS_FUNCTIONAL_VCS_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
   'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),'release_sha256':sha(release),'m1214_release_hammer_review_sha256':sha(hammer)},
   'macro_model':'foundry_UNIT_DELAY_functional','one_shot':{'attempt_consumed':True,'automatic_retry':False},
   'coverage':{'assertions':16,'covers':6,'directed_random_transactions':24,'protocol_attacks':7,'service_assumption_attacks':2,'request_attack_windows':2,'legal_masks_clear':29,'random_request_quiesce':24,'exactly_one_random_request_handshake':True,'reset_pending_states':3,'minimum_completed_issue_ii':2,'normal_m935_rows':1,'normal_m935_tasks':1,'service_skew_isolated':True,'reachable_core_ready_force':False,'boundary_fault':False,'core_fault':False},
   'claim_boundary':{'functional_vcs_verified':True,'timing_verified':False,'cycles_measured':False,'speedup':False,'ppa':False,'power':False,'energy':False,'system_speedup':False,'paper_citable':False}}
Path('m1213_m1210r8_m1162_c1_unit_delay_vcs_receipt_r8.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M1213/R8 functional VCS result=%s\n' "${RESULT}"
