#!/usr/bin/env bash
set -euo pipefail
umask 002

# Additive R4 launch-chain repair. Frozen RTL/TB/SVA remain byte-identical.
# A fresh release-hammer review and its recursive outer seal are mandatory
# runtime inputs and are verified before the new R4 attempt is created.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }
for name in M1187_EXPECTED_RELEASE_SHA256 \
            M1187_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256 \
            M1187_EXPECTED_SOURCE_HAMMER_OUTER_SHA256 \
            M1187_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256 \
            M1187_EXPECTED_RELEASE_HAMMER_OUTER_SHA256; do
  value="${!name:-}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: ${name} absent/invalid" >&2; exit 2; }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
WRAPPER="${HW_ROOT}/rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
SVA="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
TB="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3.sv"
STATIC_CHECK="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/static_check_m1168r3_m1162_vcs_source.py"
PRE_GATE="${HW_ROOT}/verif_m1168r3_c1_common_charge_protocol/validate_m1187_m1168r3_vcs_pre_attempt_gate_r4.py"
FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
CONTRACT="${HW_ROOT}/contracts/m1187_m1168r3_m1162_c1_vcs_launcher_source_contract_r4_20260830.json"
SOURCE_AUTHOR="${HW_ROOT}/reviews/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
SOURCE_HAMMER="${HW_ROOT}/reviews/m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_hammer_r1_20260830"
RELEASE="${HW_ROOT}/contracts/m1187_m1168r3_m1162_c1_vcs_launch_release_r4_20260830.json"
RELEASE_HAMMER="${HW_ROOT}/reviews/m1188_m1187_m1168r3_c1_vcs_release_hammer_r1_20260830"
R2_ATTEMPT_ID="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed/identity.txt"
R2_Q="${HW_ROOT}/results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.3284331.quarantine"
R3_ATTEMPT="${HW_ROOT}/results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed"
R3_RESULT="${HW_ROOT}/results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
PYTHON_BIN="/opt/anaconda3/envs/pytorch310/bin/python3.10"
ATTEMPT="${HW_ROOT}/results/.m1187_m1168r3_m1162_c1_common_charge_protocol_vcs_r4_attempt_consumed"
RESULT="${HW_ROOT}/results/m1187_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r4_20260830"
WORK="${HW_ROOT}/results/.m1187_m1168r3_m1162_c1_common_charge_protocol_vcs_r4_work.$$"
TOP="tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3"
PASS_TOKEN="PASS_M1168R3_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE"

sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "ERROR: missing/nonregular ${path}" >&2; exit 3; }
  got="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${got}" == "${expected}" ]] || { echo "ERROR: SHA mismatch ${path}" >&2; exit 3; }
}

seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

verify_recursive_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" && -f "${dir}/SHA256SUMS.seal.sha256" ]] || exit 3
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
    printf 'status=FAILED_OR_INCOMPLETE\nexit_code=%s\nfunctional_vcs_verified=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    local q="${RESULT}.failed_or_incomplete.$$.quarantine"
    [[ ! -e "${q}" ]] && mv -- "${WORK}" "${q}" || true
  fi
}
trap on_exit EXIT

# Exact frozen source identity; R4 changes only the launch/gating chain.
sha_exact 639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595 "${WRAPPER}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${M935}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${PARENT}"
sha_exact c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472 "${SVA}"
sha_exact b68e0f452cdd7aa87c5408b7d90222f0531faacdf0605a87dedced359b7d5a2d "${TB}"
sha_exact 30a67b1f4b0a12017c09077cbc730de936ee532e76df4445d2957035ee47320e "${STATIC_CHECK}"
sha_exact 792563125a0711cd3e584f12c863d99fd5a9a3770347846b62c903ff154664d4 "${PRE_GATE}"
sha_exact 9030f139e20d301ef9bc558a726c7c524353bb830845a9914d7c738d6e4e50a3 "${FILELIST}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115 "${PYTHON_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
sha_exact dde2eca905affe76a5e5a74966fe2502bc1fb82364493ee314819264d8bd75ca "${R2_ATTEMPT_ID}"
verify_recursive_seal "${R2_Q}"
verify_recursive_seal "${SOURCE_AUTHOR}"
verify_recursive_seal "${SOURCE_HAMMER}"
verify_recursive_seal "${RELEASE_HAMMER}"
[[ -f "${CONTRACT}.sha256" && -f "${CONTRACT}.sha256.seal.sha256" ]] || exit 3
[[ -f "${RELEASE}.sha256" && -f "${RELEASE}.sha256.seal.sha256" ]] || exit 3
(cd -- "$(dirname -- "${CONTRACT}")" && sha256sum -c "$(basename -- "${CONTRACT}.sha256")" >/dev/null && sha256sum -c "$(basename -- "${CONTRACT}.sha256.seal.sha256")" >/dev/null)
(cd -- "$(dirname -- "${RELEASE}")" && sha256sum -c "$(basename -- "${RELEASE}.sha256")" >/dev/null && sha256sum -c "$(basename -- "${RELEASE}.sha256.seal.sha256")" >/dev/null)

# This exact semantic gate is executed before any attempt namespace is created.
"${PYTHON_BIN}" -I "${PRE_GATE}" "${CONTRACT}" "${RUNNER}" "${SOURCE_HAMMER}" \
  "${RELEASE}" "${RELEASE_HAMMER}" "${SOURCE_AUTHOR}" "${R2_Q}" >/dev/null

# Old namespaces are immutable/non-reusable; unconsumed R3 must remain absent.
[[ ! -e "${R3_ATTEMPT}" && ! -e "${R3_RESULT}" ]] || { echo "ERROR: R3 namespace exists/reuse forbidden" >&2; exit 4; }
compgen -G "${HW_ROOT}/results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_work.*" >/dev/null && exit 4 || true
compgen -G "${HW_ROOT}/results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830.failed_or_incomplete.*" >/dev/null && exit 4 || true
compgen -G "${HW_ROOT}/results/*m1168r1*" >/dev/null && { echo "ERROR: R1 namespace reuse forbidden" >&2; exit 4; } || true
[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] || { echo "ERROR: R4 namespace not fresh" >&2; exit 4; }

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
    if comm in blocked or (blocked & {Path(x).name for x in argv}): hits.append((p.name,comm,argv[:4]))
if hits: raise SystemExit('EDA collision: %r' % hits)
PY
mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]] || { echo "ERROR: MemAvailable below 64 GiB" >&2; exit 5; }

mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\nrelease_sha256=%s\ncreated_utc=%s\n' \
  "$(sha256sum -- "${RUNNER}" | awk '{print $1}')" "${M1187_EXPECTED_RELEASE_SHA256}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${ATTEMPT}/identity.txt"
mkdir -- "${WORK}"
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
for token in COVERAGE_M1168R3_PROTOCOL COVERAGE_M1168R3_RESETS_ATTACKS \
             COVERAGE_M1168R3_SERVICE_ASSUMPTIONS COVERAGE_M1168R3_FROZEN_M935; do
  [[ "$(rg -c "^${token} " sim.log)" -eq 1 ]] || exit 23
done
rg -q '^PASS_M1168R3_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE directed_random=24 protocol_attacks=7 service_assumption_attacks=2 request_attack_windows=2 legal_masks_clear=29 reset_states=3 ii=2 normal_m935_rows=1 normal_m935_tasks=1 functional_vcs_only=true timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false$' sim.log || exit 27
if rg -qi 'assertion[^\n]*(fail|error)|error-[A-Z0-9]+|\$fatal' sim.log; then exit 28; fi

"${PYTHON_BIN}" -I - "${RUNNER}" "${CONTRACT}" "${RELEASE}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
runner,contract,release=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m1187_m1168r3_m1162_c1_unit_delay_vcs_receipt_r4_v1','status':'PASS_FUNCTIONAL_VCS_ONLY',
   'created_utc':datetime.now(timezone.utc).isoformat(),
   'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),'release_sha256':sha(release)},
   'macro_model':'foundry_UNIT_DELAY_functional',
   'coverage':{'directed_random_transactions':24,'protocol_attacks':7,'service_assumption_attacks':2,
      'request_attack_windows':2,'legal_masks_clear':29,'reset_pending_states':3,
      'minimum_completed_issue_ii':2,'normal_m935_rows':1,'normal_m935_tasks':1},
   'claim_boundary':{'functional_vcs_verified':True,'timing_verified':False,'cycles_measured':False,
      'speedup':False,'ppa':False,'power':False,'energy':False,'system_speedup':False,'paper_citable':False}}
Path('m1187_m1168r3_m1162_c1_unit_delay_vcs_receipt_r4.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M1187/R4 functional VCS result=%s\n' "${RESULT}"
