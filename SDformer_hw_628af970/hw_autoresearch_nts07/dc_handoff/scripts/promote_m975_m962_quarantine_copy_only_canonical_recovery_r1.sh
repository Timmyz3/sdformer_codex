#!/usr/bin/env bash
set -euo pipefail
umask 002

# Source-only M975 forensic recovery promoter.  This script never reruns EDA,
# never edits the M962 quarantine, and cannot run until a separately sealed
# M976 release exists.  Promotion is additive: the original quarantine is
# copied byte-for-byte below original_quarantine/, including its explicit
# FAILED_OR_INCOMPLETE marker, then recovery provenance is added at the new
# identity root.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SELF="$(readlink -f -- "${BASH_SOURCE[0]}")"
SOURCE="${HW_ROOT}/dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829.failed_or_incomplete.3868703.quarantine"
CONTRACT="${HW_ROOT}/contracts/m975_m962_quarantine_copy_only_canonical_recovery_source_contract_r1_20260829.json"
REVIEW="${HW_ROOT}/reviews/m975_m962_quarantine_rc9_forensic_and_promotion_source_hammer_r1_20260829/review.json"
RELEASE="${HW_ROOT}/contracts/m976_m975_m962_quarantine_copy_only_canonical_recovery_release_r1_20260829.json"
TARGET="${HW_ROOT}/dc_handoff/runs/m975_m962_m935_three_stage_match_macro_aware_dc_3p000ns_recovered_canonical_r1_20260829"
WORK="${TARGET}.copy_work.$$"

SOURCE_MANIFEST_SHA256="9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe"
SOURCE_OUTER_FILE_SHA256="a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997"

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
import os, stat, sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    if line.strip(): listed.add(line.split(None,1)[1].lstrip('*'))
actual=set(); links=[]
for root, dirs, files in os.walk(d, followlinks=False):
    rp=Path(root)
    for name in list(dirs):
        if (rp/name).is_symlink(): links.append(str((rp/name).relative_to(d)))
    dirs[:]=[n for n in dirs if not (rp/n).is_symlink()]
    for name in files:
        p=rp/name
        if p.is_symlink(): links.append(str(p.relative_to(d))); continue
        if name in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(str(p.relative_to(d)))
assert not links, links
assert listed == actual, (listed-actual, actual-listed)
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

# The source and source-review identities are immutable.  A future release is
# the only authorization to execute this otherwise inert copy plan.
verify_dir_seal "${SOURCE}"
[[ "$(sha_file "${SOURCE}/SHA256SUMS")" == "${SOURCE_MANIFEST_SHA256}" ]] || exit 3
[[ "$(sha_file "${SOURCE}/SHA256SUMS.seal.sha256")" == "${SOURCE_OUTER_FILE_SHA256}" ]] || exit 3
verify_file_seal "${CONTRACT}"
verify_dir_seal "$(dirname -- "${REVIEW}")"
verify_file_seal "${RELEASE}"

python3 -I - "${CONTRACT}" "${REVIEW}" "${RELEASE}" "${SELF}" <<'PY'
import hashlib, json, sys
from pathlib import Path
contract,review,release,script=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); v=json.loads(review.read_text()); r=json.loads(release.read_text())
assert c['status']=='SOURCE_READY__PROMOTION_NOT_AUTHORIZED_NOW'
assert c['authorization']=={'promotion_runs_now':0,'future_copy_only_promotions_max':1,'eda_runs':0}
assert c['identity']['promotion_script_sha256']==sha(script)
assert v['status']=='PASS_M975_M962_QUARANTINE_FORENSIC_RECOVERY__GO_COPY_ONLY_PROMOTION_SOURCE'
assert v['p0_count']==0 and v['decision']['source_is_synthesis_complete'] is True
assert r['status']=='AUTHORIZE_ONE_M975_COPY_ONLY_CANONICAL_RECOVERY'
assert r['authorization']=={'copy_only_promotions':1,'eda_runs':0}
assert r['identity']['promotion_script_sha256']==sha(script)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['m975_review_sha256']==sha(review)
PY

[[ ! -e "${TARGET}" && ! -e "${WORK}" ]] || {
  echo "ERROR: recovery target/work identity already exists" >&2; exit 4; }

# Recompute the forensic admission before any copy.  The only runner-regex hit
# allowed is the pinned HOME-less Design Vision startup error at dc.log:32.
python3 -I - "${SOURCE}" <<'PY'
import math,re,sys
from pathlib import Path
d=Path(sys.argv[1])
assert (d/'dc.rc').read_text().strip()=='0'
assert (d/'RUN_FAILED_OR_INCOMPLETE.txt').read_text().splitlines()==[
 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE','exit_code=9','setup_admitted=false']
terminal=dict(x.split('=',1) for x in (d/'TCL_PASS_TERMINAL.txt').read_text().splitlines())
assert terminal=={'status':'PASS_M962_DC_EXECUTION_AND_REPORT_CLOSURE','setup_status':'MET',
 'TIM-209':'0','OPT-150':'0','macro_count_pre':'9','macro_count_post':'9',
 'hold_signoff':'false','power_measured':'false'}
summary=dict(x.split('=',1) for x in (d/'reports/setup_summary_machine.txt').read_text().splitlines())
assert summary=={'status':'MET','setup_wns_ns':'0.001795','setup_tns_ns':'0.000000',
 'setup_violating_paths':'0','clock_period_ns':'3.000'}
macro=dict(x.split('=',1) for x in (d/'reports/macro_binding_audit.txt').read_text().splitlines())
assert macro['status']=='PASS_M962_RESOLVED_LIBRARY_MACRO_STRUCTURE'
assert macro['macro_count_pre']=='9' and macro['macro_count_post']=='9'
area=(d/'reports/area_hierarchy.rpt').read_text(errors='replace')
def val(label):
    m=re.search(r'^'+re.escape(label)+r':\s+([0-9.]+)\s*$',area,re.M); assert m; return float(m.group(1))
comb=val('Combinational area'); seq=val('Noncombinational area')
mac=val('Macro/Black Box area'); total=val('Total cell area')
assert all(math.isfinite(x) and x>0 for x in (comb,seq,mac,total))
assert abs(comb+seq+mac-total)<0.01
top=(d/'reports/timing_setup_top100.rpt').read_text(errors='replace')
assert len(re.findall(r'^  Startpoint:',top,re.M))==100
assert len(re.findall(r'^  Endpoint:',top,re.M))==100
assert len(re.findall(r'^  slack \(MET\)',top,re.M))==100
assert 'slack (VIOLATED)' not in top
mapped=(d/'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v').read_text(errors='replace')
assert len(re.findall(r'\bTS1N28HPCPHVTB128X128M4S\b',mapped))==9
log=(d/'dc.log').read_text(errors='replace')
pat=re.compile(r'(^|[^A-Za-z])(Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+)|\((TIM-209|OPT-150)\)',re.I|re.M)
hits=[]
for m in pat.finditer(log):
    pos=m.start(2) if m.group(2) is not None else m.start(3)
    line_no=log.count('\n',0,pos)+1
    hits.append((line_no,log.splitlines()[line_no-1]))
assert hits==[(32,'Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl')],hits
assert 'Optimization Complete' in log and 'Thank you...' in log
PY

mkdir -- "${WORK}"
mkdir -- "${WORK}/original_quarantine"
cp -a --no-dereference "${SOURCE}/." "${WORK}/original_quarantine/"
verify_dir_seal "${WORK}/original_quarantine"
[[ "$(sha_file "${WORK}/original_quarantine/SHA256SUMS")" == "${SOURCE_MANIFEST_SHA256}" ]] || exit 5
[[ "$(sha_file "${WORK}/original_quarantine/SHA256SUMS.seal.sha256")" == "${SOURCE_OUTER_FILE_SHA256}" ]] || exit 5

python3 -I - "${WORK}" "${SOURCE}" "${CONTRACT}" "${REVIEW}" "${RELEASE}" "${SELF}" <<'PY'
import hashlib,json,sys
from pathlib import Path
work,source,contract,review,release,script=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
provenance={
 'schema':'m975_m962_quarantine_copy_only_recovery_provenance_v1',
 'status':'COPY_ONLY_RECOVERY_OF_SYNTHESIS_COMPLETE_M962_QUARANTINE',
 'source_quarantine':str(source),
 'source_manifest_sha256':'9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe',
 'source_outer_seal_file_sha256':'a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997',
 'original_failure_label_preserved_at':'original_quarantine/RUN_FAILED_OR_INCOMPLETE.txt',
 'runner_bug':{'runner_exit_code':9,'dc_shell_exit_code':0,'log_hit_line':32,
   'cause':'env -i omitted HOME; nonfatal Design Vision startup Tcl error was matched by an over-broad post-run grep'},
 'identity':{'promotion_script_sha256':sha(script),'source_contract_sha256':sha(contract),
   'm975_review_sha256':sha(review),'m976_release_sha256':sha(release)},
 'mutation':{'source_modified':False,'runner_modified':False,'eda_rerun':False,'copied_payload_changed':False},
}
receipt={
 'schema':'m975_recovered_m962_macro_aware_dc_receipt_v1',
 'status':'PASS_RECOVERED_RAW_M962_3NS_SETUP_AREA_COMPONENT_CANDIDATE',
 'clock_period_ns':3.0,'ideal_clock':True,'wireload':'ZeroWireload',
 'macro_cell':'TS1N28HPCPHVTB128X128M4S','macro_count':9,
 'total_cell_area_um2_dc_reported':147246.392090,
 'area':{'combinational_um2':41912.009793,'noncombinational_um2':26509.139132,
         'macro_black_box_um2':78825.243164},
 'setup':{'met':True,'wns_ns':0.001795,'tns_ns':0.0,'violating_paths':0,
          'top100_reported_paths':100},
 'claim_boundary':{'setup_area_component_candidate':True,'hold_signoff':False,
   'power':False,'energy':False,'rtl_cycles_measured':False,'speedup':False,
   'system_speedup':False,'full_213376B_storage_integrated':False,
   'paper_ppa_ready':False,'headline':False},
}
(work/'M975_PROMOTION_PROVENANCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
(work/'m975_recovered_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
(work/'RUN_COMPLETE_RECOVERED.txt').write_text(
 'status=PASS_RECOVERED_RAW_M962_3NS_SETUP_AREA_COMPONENT_CANDIDATE\n'
 'source_failure_label_preserved=true\nsetup_met=true\nhold_signoff=false\n'
 'power=false\nenergy=false\nrtl_cycles_measured=false\nspeedup=false\n'
 'system_speedup=false\npaper_ppa_ready=false\n')
PY

seal_dir "${WORK}"
mv -- "${WORK}" "${TARGET}"
echo "M975 copy-only recovery published: ${TARGET}"
