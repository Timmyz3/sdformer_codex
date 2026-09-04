#!/usr/bin/env bash
set -euo pipefail
umask 002

# M989 source-only successor to M975.  It adds an atomic launch lock, a
# permanent one-shot attempt identity, a fixed work identity, and no-nesting
# atomic publication.  It cannot execute until the independently sealed
# M990 -> M991 -> M992 review/release chain exists.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SELF="$(readlink -f -- "${BASH_SOURCE[0]}")"
SOURCE="${HW_ROOT}/dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829.failed_or_incomplete.3868703.quarantine"
M975_CONTRACT="${HW_ROOT}/contracts/m975_m962_quarantine_copy_only_canonical_recovery_source_contract_r1_20260829.json"
M975_REVIEW="${HW_ROOT}/reviews/m975_m962_quarantine_rc9_forensic_and_promotion_source_hammer_r1_20260829/review.json"
CONTRACT="${HW_ROOT}/contracts/m989_m975_m962_atomic_one_shot_copy_only_promotion_source_contract_r1_20260829.json"
M990_REVIEW="${HW_ROOT}/reviews/m990_m989_m975_m962_atomic_one_shot_promotion_source_hammer_r1_20260829/review.json"
M991_RELEASE="${HW_ROOT}/contracts/m991_m990_m989_atomic_one_shot_copy_only_promotion_release_r1_20260829.json"
M992_HAMMER="${HW_ROOT}/reviews/m992_m991_m990_m989_atomic_one_shot_promotion_release_hammer_r1_20260829/review.json"
TARGET="${HW_ROOT}/dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
LOCK="${HW_ROOT}/dc_handoff/runs/.m993_m989_m962_copy_promotion_launch_lock"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m993_m989_m962_copy_promotion_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m993_m989_m962_copy_promotion_work"
FAILQ="${HW_ROOT}/dc_handoff/runs/m993_m989_m962_copy_promotion_failed_or_incomplete.quarantine"

SOURCE_MANIFEST_SHA256="9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe"
SOURCE_OUTER_FILE_SHA256="a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997"
LOCK_HELD=0
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }

verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}" && ! -L "${payload}"
      && -f "${payload}.sha256" && ! -L "${payload}.sha256"
      && -f "${payload}.sha256.seal.sha256"
      && ! -L "${payload}.sha256.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null &&
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}

verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS"
      && ! -L "${dir}/SHA256SUMS"
      && -f "${dir}/SHA256SUMS.seal.sha256"
      && ! -L "${dir}/SHA256SUMS.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
  python3 -I - "${dir}" <<'PY'
import os,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set(); actual=set(); links=[]
for line in (d/'SHA256SUMS').read_text().splitlines():
    if line.strip(): listed.add(line.split(None,1)[1].lstrip('*'))
for root,dirs,files in os.walk(d,followlinks=False):
    rp=Path(root)
    for n in list(dirs):
        if (rp/n).is_symlink(): links.append(str((rp/n).relative_to(d)))
    dirs[:]=[n for n in dirs if not (rp/n).is_symlink()]
    for n in files:
        p=rp/n
        if p.is_symlink(): links.append(str(p.relative_to(d))); continue
        if n in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(str(p.relative_to(d)))
assert not links,links
assert listed==actual,(listed-actual,actual-listed)
PY
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

on_exit() {
  local rc=$?
  set +e
  trap - EXIT INT TERM
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=M989_COPY_PROMOTION_FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nattempt_consumed=true\ntarget_published=false\n' \
      "${rc}" >"${WORK}/M989_PROMOTION_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    if [[ ! -e "${FAILQ}" ]]; then
      mv -T -- "${WORK}" "${FAILQ}" || true
    fi
  fi
  if [[ ${LOCK_HELD} -eq 1 ]]; then rmdir -- "${LOCK}" 2>/dev/null || true; fi
  exit "${rc}"
}
trap on_exit EXIT INT TERM

# All evidence and future authorization are verified before lock acquisition.
# Two contenders may verify concurrently, but only one may cross mkdir LOCK.
verify_dir_seal "${SOURCE}"
[[ "$(sha_file "${SOURCE}/SHA256SUMS")" == "${SOURCE_MANIFEST_SHA256}" ]] || exit 3
[[ "$(sha_file "${SOURCE}/SHA256SUMS.seal.sha256")" == "${SOURCE_OUTER_FILE_SHA256}" ]] || exit 3
verify_file_seal "${M975_CONTRACT}"
verify_dir_seal "$(dirname -- "${M975_REVIEW}")"
verify_file_seal "${CONTRACT}"
verify_dir_seal "$(dirname -- "${M990_REVIEW}")"
verify_file_seal "${M991_RELEASE}"
verify_dir_seal "$(dirname -- "${M992_HAMMER}")"

python3 -I - "${M975_CONTRACT}" "${M975_REVIEW}" "${CONTRACT}" \
  "${M990_REVIEW}" "${M991_RELEASE}" "${M992_HAMMER}" "${SELF}" <<'PY'
import hashlib,json,sys
from pathlib import Path
m975c,m975v,contract_path,v990,r991,v992,script=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(m975c.read_text()); b=json.loads(m975v.read_text())
c=json.loads(contract_path.read_text()); d=json.loads(v990.read_text())
e=json.loads(r991.read_text()); f=json.loads(v992.read_text())
assert a['status']=='SOURCE_READY__PROMOTION_NOT_AUTHORIZED_NOW'
assert b['status']=='PASS_M975_M962_QUARANTINE_FORENSIC_RECOVERY__GO_COPY_ONLY_PROMOTION_SOURCE'
assert b['decision']['source_is_synthesis_complete'] is True
assert c['status']=='SOURCE_READY__M989_PROMOTION_NOT_AUTHORIZED_NOW'
assert c['authorization']=={'promotion_runs_now':0,'future_copy_only_promotions_max':1,'eda_runs':0}
assert c['identity']['promotion_script_sha256']==sha(script)
assert c['identity']['m975_contract_sha256']==sha(m975c)
assert c['identity']['m975_review_sha256']==sha(m975v)
assert d['status']=='PASS_M990_M989_COPY_ONLY_PROMOTION_SOURCE_HAMMER'
assert d['p0_count']==0 and d['decision']['concurrency_protocol_admitted'] is True
assert e['status']=='AUTHORIZE_ONE_M993_M989_COPY_ONLY_CANONICAL_RECOVERY'
assert e['authorization']=={'copy_only_promotions':1,'eda_runs':0}
assert e['identity']['promotion_script_sha256']==sha(script)
assert e['identity']['source_contract_sha256']==sha(contract_path)
assert e['identity']['m990_review_sha256']==sha(v990)
assert f['status']=='PASS_M992_M991_M990_M989_PROMOTION_RELEASE_HAMMER'
assert f['p0_count']==0 and f['decision']['authorize_m993_execution'] is True
assert f['identity']['m991_release_sha256']==sha(r991)
PY

# Cheap preflight only.  The same conditions are checked again under LOCK.
[[ ! -e "${TARGET}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${FAILQ}" ]] || {
  echo "ERROR: M993 target/attempt/work/failure identity already exists" >&2; exit 4; }

if ! mkdir -- "${LOCK}"; then
  echo "ERROR: another M993 copy promotion owns the atomic launch lock" >&2
  exit 4
fi
LOCK_HELD=1

# Lock-serialized rechecks close the cooperating-runner TOCTOU window.  The
# permanent attempt directory is atomically consumed before any payload copy.
[[ ! -e "${TARGET}" && ! -e "${WORK}" && ! -e "${FAILQ}" ]] || exit 4
if ! mkdir -- "${ATTEMPT}"; then
  echo "ERROR: M993 one-shot attempt already consumed" >&2
  exit 4
fi
printf 'status=M993_M989_COPY_PROMOTION_ATTEMPT_CONSUMED\nmax_promotions=1\nretry=false\ncopy_may_start_only_after_this_seal=true\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
printf 'promotion_script_sha256=%s\nsource_manifest_sha256=%s\n' \
  "$(sha_file "${SELF}")" "${SOURCE_MANIFEST_SHA256}" >"${ATTEMPT}/IDENTITY.txt"
seal_dir "${ATTEMPT}"

mkdir -- "${WORK}"
WORK_ACTIVE=1
mkdir -- "${WORK}/original_quarantine"
cp -a --no-dereference "${SOURCE}/." "${WORK}/original_quarantine/"
verify_dir_seal "${WORK}/original_quarantine"
[[ "$(sha_file "${WORK}/original_quarantine/SHA256SUMS")" == "${SOURCE_MANIFEST_SHA256}" ]] || exit 5
[[ "$(sha_file "${WORK}/original_quarantine/SHA256SUMS.seal.sha256")" == "${SOURCE_OUTER_FILE_SHA256}" ]] || exit 5

python3 -I - "${WORK}" "${SOURCE}" "${M975_REVIEW}" "${CONTRACT}" \
  "${M990_REVIEW}" "${M991_RELEASE}" "${M992_HAMMER}" "${SELF}" <<'PY'
import hashlib,json,sys
from pathlib import Path
work,source,m975,contract,v990,r991,v992,script=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
provenance={
 'schema':'m993_m989_atomic_one_shot_copy_only_recovery_provenance_v1',
 'status':'COPY_ONLY_RECOVERY_OF_SYNTHESIS_COMPLETE_M962_QUARANTINE',
 'source_quarantine':str(source),
 'source_manifest_sha256':'9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe',
 'source_outer_seal_file_sha256':'a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997',
 'original_failure_label_preserved_at':'original_quarantine/RUN_FAILED_OR_INCOMPLETE.txt',
 'runner_bug':{'runner_exit_code':9,'dc_shell_exit_code':0,'log_hit_line':32,
   'cause':'env -i omitted HOME; nonfatal Design Vision startup Tcl error matched over-broad grep'},
 'concurrency':{'atomic_launch_lock':True,'one_shot_attempt_consumed_before_copy':True,
   'fixed_work_identity':True,'atomic_no_nesting_publish':True},
 'identity':{'promotion_script_sha256':sha(script),'m975_review_sha256':sha(m975),
   'm989_contract_sha256':sha(contract),'m990_review_sha256':sha(v990),
   'm991_release_sha256':sha(r991),'m992_hammer_sha256':sha(v992)},
 'mutation':{'source_modified':False,'runner_modified':False,'eda_rerun':False,
   'copied_payload_changed':False},
}
receipt={
 'schema':'m993_recovered_m962_macro_aware_dc_receipt_v1',
 'status':'PASS_RECOVERED_RAW_M962_3NS_SETUP_AREA_COMPONENT_CANDIDATE',
 'clock_period_ns':3.0,'ideal_clock':True,'wireload':'ZeroWireload',
 'macro_cell':'TS1N28HPCPHVTB128X128M4S','macro_count':9,
 'total_cell_area_um2_dc_reported':147246.392090,
 'setup':{'met':True,'wns_ns':0.001795,'tns_ns':0.0,'violating_paths':0,
          'top100_reported_paths':100},
 'claim_boundary':{'setup_area_component_candidate':True,'hold_signoff':False,
   'power':False,'energy':False,'rtl_cycles_measured':False,'speedup':False,
   'system_speedup':False,'full_213376B_storage_integrated':False,
   'paper_ppa_ready':False,'headline':False},
}
(work/'M993_PROMOTION_PROVENANCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
(work/'m993_recovered_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
(work/'RUN_COMPLETE_RECOVERED.txt').write_text(
 'status=PASS_RECOVERED_RAW_M962_3NS_SETUP_AREA_COMPONENT_CANDIDATE\n'
 'source_failure_label_preserved=true\nsetup_met=true\nhold_signoff=false\n'
 'power=false\nenergy=false\nrtl_cycles_measured=false\nspeedup=false\n'
 'system_speedup=false\npaper_ppa_ready=false\n')
PY

seal_dir "${WORK}"
verify_dir_seal "${WORK}"

# Second publication check occurs after the complete copy and seal.  -T makes
# TARGET a literal destination and forbids accidental TARGET/WORK nesting.
[[ ! -e "${TARGET}" ]] || exit 6
mv -T -- "${WORK}" "${TARGET}"
WORK_ACTIVE=0
verify_dir_seal "${TARGET}"
rmdir -- "${LOCK}"
LOCK_HELD=0
trap - EXIT INT TERM
echo "M993 atomic one-shot copy-only recovery published: ${TARGET}"
