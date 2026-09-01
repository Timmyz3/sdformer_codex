#!/usr/bin/env bash
set -euo pipefail
umask 002

# M1659 source-only, atomic, copy-only recovery plan for the exact sealed
# PID519344 M1649 quarantine.  It never runs EDA and cannot copy a byte until
# an independently sealed M1660 review and separately sealed M1664 release are
# present and caller-pinned.  The original quarantine remains immutable and is
# preserved verbatim under original_quarantine/ in a fresh recovered identity.

[[ $# -eq 0 ]] || { echo "ERROR: M1659 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SELF="$(readlink -f -- "${BASH_SOURCE[0]}")"
SOURCE="${HW_ROOT}/dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_dc_r1_20260901.failed_or_incomplete.519344.quarantine"
SOURCE_ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_dc_attempt_consumed"
M1649_RUNNER="${HW_ROOT}/dc_handoff/scripts/run_dc_m1649_m1630_c1_resource_gate_successor_exact_sha_r1.sh"
M1649_CONTRACT="${HW_ROOT}/contracts/m1649_m1630_c1_resource_gate_successor_dc_source_contract_r1_20260901.json"
M1650_REVIEW_DIR="${HW_ROOT}/reviews/m1650_m1649_m1630_c1_resource_gate_successor_dc_source_hammer_r1_20260901"
M1651_RELEASE="${HW_ROOT}/contracts/m1651_m1650_m1649_m1630_c1_resource_gate_successor_dc_launch_release_r1_20260901.json"
M1655_REVIEW_DIR="${HW_ROOT}/reviews/m1655_m1649_c1_quarantine_forensic_recovery_review_r1_20260901"
CONTRACT="${HW_ROOT}/contracts/m1659_m1649_c1_atomic_canonical_recovery_source_contract_r1_20260901.json"
FUTURE_REVIEW_DIR="${HW_ROOT}/reviews/m1660_m1659_c1_canonical_recovery_source_independent_review_r1_20260901"
FUTURE_RELEASE="${HW_ROOT}/contracts/m1664_m1660_m1659_c1_canonical_recovery_release_r1_20260901.json"

TARGET="${HW_ROOT}/dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_launch_lock"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_work"
FAILQ="${HW_ROOT}/dc_handoff/runs/m1665_m1659_c1_canonical_recovery_failed_or_incomplete.quarantine"

SOURCE_MANIFEST_SHA256="e94ffc3680513cb2f374676037cc7c3b14b77a7bc47b9d35edb812f17a9ae843"
SOURCE_OUTER_FILE_SHA256="c221bb79e4950780c6db04ef54ed1ea809ac880ad054f9316f7bba702a49ff44"
SOURCE_ATTEMPT_MANIFEST_SHA256="53556c6a16f00c0529702e17b6eb52b4f2cc5bf17ca55399b11347c57d72310a"
SOURCE_ATTEMPT_OUTER_FILE_SHA256="13ae47cdfc544f2fd84d505499d434930ff4bfe8d88cef0f8547062543e5f985"
M1649_RUNNER_SHA256="8a1688206acf75ee0942c7bf6acb20b16c3017c7bf54451ab11d84953a4474e3"
M1649_CONTRACT_SHA256="5ca134044f1e100c925785db8025b8a7dce3e23daf5c3964608ca039ace84fb3"
M1650_REVIEW_SHA256="1ed6522019a7c34109ce44e0a7f5a959343e61f28151f08ec27dbb66546589bb"
M1650_MANIFEST_SHA256="91bf3b68191dbe31557c5610a8beb91aeeb98ab7a640959b9e4f0e24c0d4845b"
M1650_OUTER_FILE_SHA256="b0f92fb708c68badba280661c8daa5da070f0571bd794b298fae41ea7a75338e"
M1651_RELEASE_SHA256="5e68e99c49a5e7ab04b0883b06537398b5cf41c76d6812d08b9c87fc988771ef"
M1655_REVIEW_SHA256="4d6f3e2cb238fbe77038cfc213d31ce061e17d49f43badcbc6b30ee8ffb825b2"
M1655_MANIFEST_SHA256="349a78db9de8d138445889f1566ff1764a66ce3aa28d6599788979e20a8b2268"
M1655_OUTER_FILE_SHA256="5c3e1346ac3e4ecd9935190be6f8e4acf5fa9435941f2ed0a21c66512b9534f7"
LOCK_HELD=0
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
fail() { echo "ERROR: M1659 $*" >&2; exit 3; }

sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/nonregular ${path}"
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || fail "SHA mismatch ${path}: ${got}"
}

verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}" && ! -L "${payload}"
      && -f "${payload}.sha256" && ! -L "${payload}.sha256"
      && -f "${payload}.sha256.seal.sha256"
      && ! -L "${payload}.sha256.seal.sha256" ]] ||
    fail "file seal absent/nonregular ${payload}"
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null &&
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) ||
    fail "file seal invalid ${payload}"
}

verify_dir_seal() {
  local dir="$1" expected_members="$2"
  [[ -d "${dir}" && ! -L "${dir}"
      && -f "${dir}/SHA256SUMS" && ! -L "${dir}/SHA256SUMS"
      && -f "${dir}/SHA256SUMS.seal.sha256"
      && ! -L "${dir}/SHA256SUMS.seal.sha256" ]] ||
    fail "directory seal absent/nonregular ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) ||
    fail "directory seal invalid ${dir}"
  /usr/bin/python3.6 -I - "${dir}" "${expected_members}" <<'PY'
import os,re,stat,sys
from pathlib import Path
d=Path(sys.argv[1]); expected=int(sys.argv[2]); listed={}; actual=set(); links=[]
for row in (d/'SHA256SUMS').read_text().splitlines():
    assert re.match(r'^[0-9a-f]{64}  [^/\n][^\n]*$',row),row
    digest,name=row.split('  ',1)
    assert name not in listed and not Path(name).is_absolute()
    assert all(part not in ('','.','..') for part in Path(name).parts)
    listed[name]=digest
for root,dirs,files in os.walk(str(d),followlinks=False):
    rp=Path(root)
    for name in list(dirs):
        if (rp/name).is_symlink(): links.append(str((rp/name).relative_to(d)))
    dirs[:]=[name for name in dirs if not (rp/name).is_symlink()]
    for name in files:
        path=rp/name
        if path.is_symlink(): links.append(str(path.relative_to(d))); continue
        rel=path.relative_to(d).as_posix()
        if path.name in ('SHA256SUMS','SHA256SUMS.seal.sha256'): continue
        assert stat.S_ISREG(path.lstat().st_mode),rel
        actual.add(rel)
assert not links,links
assert len(listed)==expected and set(listed)==actual,(len(listed),listed.keys()-actual,actual-listed.keys())
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

forensic_gate() {
  local tree="$1"
  /usr/bin/python3.6 -I - "${tree}" <<'PY'
import hashlib,math,os,re,stat,sys
from pathlib import Path
d=Path(sys.argv[1])
expected={
'dc.log':'a02a10adf0de69ad863445290ac95554399b8401842542868b11191a0e2d1b4a',
'dc.rc':'9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
'TCL_INTERNAL_COMPLETE.txt':'07ed11af7c64167f0054f119350ae6d798c3c00cfe7c331041316fa6dba30649',
'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc':'2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0',
'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.svf':'7c15c1a30827df74c0da35f24f7e88723484c2a211edd3d6c049f52e21dec274',
'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc':'5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198',
'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v':'842d100f6a3fc26684e13a8065191028af7840685aaf4b7cfa77a4fe998c46ee',
'reports/setup_posthold_summary_machine.txt':'123d8653bf0800934857325fa77e6759fdff93f78e099c9411b4c689d4d0647d',
'reports/hold_posthold_summary_machine.txt':'db11b098828b57fd61b6a4ef8bff2b3302b332bca78f04c7ea442c41b46d519f',
'reports/area_posthold.rpt':'66f18b4890ec68ec9c4b7e69e004cc326063efe4b6b62d6f95d544228ee60333',
'reports/qor_posthold.rpt':'268909e6433b799bf59909f670c28f2697a1b8fcfbcdcb8d96cff2b06fbd872a',
'reports/macro_binding_audit.txt':'2e21f34b7263596729746460c27663ed469b410178a9753b791ef4429fc08742'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def kv(path):
    out={}
    for line in path.read_text().splitlines():
        if '=' in line:
            key,value=line.split('=',1); assert key not in out; out[key]=value
    return out
for name,digest in expected.items():
    p=d/name; assert p.is_file() and not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode)
    assert p.stat().st_size>0 and sha(p)==digest,(name,sha(p))
assert (d/'dc.rc').read_text()=='0\n'
assert (d/'RUN_FAILED_OR_INCOMPLETE.txt').read_text().splitlines()==[
 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE','exit_code=3','retry=false']
terminal=kv(d/'TCL_INTERNAL_COMPLETE.txt')
assert terminal['status']=='M1630_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED'
assert terminal['input_generation']=='original_m993_m1006_admitted_ddc'
assert terminal['failed_m1614_output_used']=='false'
assert terminal['set_fix_hold_count']=='1' and terminal['hold_only_incremental_mapping_count']=='1'
assert terminal['mapped_identity_modified']=='true' and terminal['formality_required']=='true'
assert terminal['independent_pt_required']=='true' and terminal['paper_citable']=='false'
flow=kv(d/'reports/flow_contract.rpt')
assert flow['clock_period_ns']=='3.000' and flow['setup_uncertainty_ns']=='0.200'
assert flow['optimization_hold_guardband_ns']=='0.051' and flow['reported_hold_uncertainty_ns']=='0.050'
assert flow['all_compile_command_count']=='1' and flow['hold_only_incremental_mapping_count']=='1'
for name in ('false_path_count','multicycle_path_count','min_delay_exception_count',
             'max_delay_exception_count','disabled_timing_arc_count','case_analysis_count'):
    assert flow[name]=='0',name
def timing(name,kind,wns):
    row=kv(d/'reports'/name)
    assert row=={'phase':'POST_RESTORE_REPORTED','delay_type':kind,'status':'MET',
      'wns_ns':wns,'tns_ns':'0.000000000','violating_paths':'0',
      'negative_path_ceiling':'200000'},row
timing('setup_posthold_summary_machine.txt','max','0.002221110')
timing('hold_posthold_summary_machine.txt','min','0.000999451')
area=(d/'reports/area_posthold.rpt').read_text(errors='replace')
m=re.search(r'Total cell area:\s*([0-9.]+)',area); assert m
value=float(m.group(1)); assert math.isfinite(value) and value==152898.625984
assert value<=154608.7116945
macro=kv(d/'reports/macro_binding_audit.txt')
assert macro['status']=='PASS_M1630_RESOLVED_LIBRARY_MACRO_STRUCTURE'
assert macro['macro_count_pre']==macro['macro_count_post']==macro['expected_macro_count']=='9'
assert macro['behavioral_macro_verilog_read_by_dc']=='false'
qor=(d/'reports/qor_posthold.rpt').read_text(errors='replace')
assert re.search(r'Nets With Violations:\s+0(?:\.00)?\s*$',qor,re.M)
log=(d/'dc.log').read_text(errors='replace'); lines=log.splitlines()
errors=[(i+1,line) for i,line in enumerate(lines) if re.match(r'^(Error|Fatal):',line)]
assert errors==[(32,'Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl')],errors
assert 'no such variable\n    (read trace on "::env(HOME)")' in log
start=next(i for i,line in enumerate(lines) if line.startswith('Current time:'))
assert start>31 and not any(re.match(r'^(Error|Fatal):',line) for line in lines[start+1:])
assert "Writing verilog file '" in log and "Writing ddc file '" in log
assert 'set_svf -off' in log and 'Thank you...' in log
v=(d/'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v').read_text(errors='replace')
assert v.rstrip().endswith('endmodule') and len(re.findall(r'\bTS1N28HPCPHVTB128X128M4S\b',v))==9
s=(d/'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc').read_text(errors='replace')
assert len(re.findall(r'(?m)^create_clock\b',s))==1
assert re.search(r'create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)',s)
assert re.search(r'set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b',s)
assert re.search(r'set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b',s)
assert not re.search(r'set_clock_uncertainty\s+-hold\s+0?\.051(?:0+)?\b',s)
for command in ('set_false_path','set_multicycle_path','set_min_delay','set_max_delay',
                'set_disable_timing','set_case_analysis'):
    assert not re.search(r'(?m)^\s*'+command+r'\b',s),command
PY
}

on_exit() {
  local rc=$?
  set +e
  trap - EXIT INT TERM
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=M1659_COPY_RECOVERY_FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nattempt_consumed=true\ntarget_published=false\nretry=false\n' \
      "${rc}" >"${WORK}/M1659_RECOVERY_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    if [[ ! -e "${FAILQ}" ]]; then mv -T -- "${WORK}" "${FAILQ}" || true; fi
  fi
  if [[ ${LOCK_HELD} -eq 1 ]]; then rmdir -- "${LOCK}" 2>/dev/null || true; fi
  exit "${rc}"
}
trap on_exit EXIT INT TERM

# Immutable source evidence and full launch/review provenance are checked before
# any future lock or copy.  M1659 changes no physical artifact.
verify_dir_seal "${SOURCE}" 39
sha_exact "${SOURCE_MANIFEST_SHA256}" "${SOURCE}/SHA256SUMS"
sha_exact "${SOURCE_OUTER_FILE_SHA256}" "${SOURCE}/SHA256SUMS.seal.sha256"
verify_dir_seal "${SOURCE_ATTEMPT}" 1
sha_exact "${SOURCE_ATTEMPT_MANIFEST_SHA256}" "${SOURCE_ATTEMPT}/SHA256SUMS"
sha_exact "${SOURCE_ATTEMPT_OUTER_FILE_SHA256}" "${SOURCE_ATTEMPT}/SHA256SUMS.seal.sha256"
sha_exact "${M1649_RUNNER_SHA256}" "${M1649_RUNNER}"
verify_file_seal "${M1649_CONTRACT}"
sha_exact "${M1649_CONTRACT_SHA256}" "${M1649_CONTRACT}"
verify_dir_seal "${M1650_REVIEW_DIR}" 9
sha_exact "${M1650_REVIEW_SHA256}" "${M1650_REVIEW_DIR}/review.json"
sha_exact "${M1650_MANIFEST_SHA256}" "${M1650_REVIEW_DIR}/SHA256SUMS"
sha_exact "${M1650_OUTER_FILE_SHA256}" "${M1650_REVIEW_DIR}/SHA256SUMS.seal.sha256"
verify_file_seal "${M1651_RELEASE}"
sha_exact "${M1651_RELEASE_SHA256}" "${M1651_RELEASE}"
verify_dir_seal "${M1655_REVIEW_DIR}" 7
sha_exact "${M1655_REVIEW_SHA256}" "${M1655_REVIEW_DIR}/review.json"
sha_exact "${M1655_MANIFEST_SHA256}" "${M1655_REVIEW_DIR}/SHA256SUMS"
sha_exact "${M1655_OUTER_FILE_SHA256}" "${M1655_REVIEW_DIR}/SHA256SUMS.seal.sha256"
verify_file_seal "${CONTRACT}"

# A different-author review and separately sealed release are mandatory.  The
# expected status and every identity are checked before any copy namespace.
verify_dir_seal "${FUTURE_REVIEW_DIR}" 7
verify_file_seal "${FUTURE_RELEASE}"
/usr/bin/python3.6 -I - "${CONTRACT}" "${M1655_REVIEW_DIR}/review.json" \
  "${FUTURE_REVIEW_DIR}/review.json" "${FUTURE_RELEASE}" "${SELF}" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,m1655_path,review_path,release_path,source=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); f=json.loads(m1655_path.read_text())
v=json.loads(review_path.read_text()); r=json.loads(release_path.read_text())
assert c['status']=='SOURCE_ONLY_M1659_C1_ATOMIC_CANONICAL_RECOVERY__NO_COPY_NO_EDA'
assert c['authorization']=={'recovery_runs_now':0,'future_copy_only_recoveries_max':1,'all_eda_runs':0}
assert c['identity']['source_sha256']==sha(source)
assert c['identity']['m1655_review_sha256']==sha(m1655_path)
assert f['status']=='PASS_M1655_M1649_C1_SEALED_QUARANTINE_FORENSIC__AUTHORIZE_SOURCE_ONLY_CANONICAL_RECOVERY__NO_EDA'
assert f['authorization']['source_only_canonical_recovery_authoring'] is True
assert v['status']=='PASS_M1660_M1659_C1_CANONICAL_RECOVERY_SOURCE__AUTHORIZE_M1664_RELEASE_ONLY'
assert v['p0_count']==0 and v['p1_count']==0
assert v['authorization']=={'m1664_release_authoring':True,'recovery_now':False,'all_eda':False}
assert r['status']=='AUTHORIZE_ONE_M1665_M1659_C1_COPY_ONLY_CANONICAL_RECOVERY'
assert r['authorization']=={'copy_only_recoveries':1,'all_eda_runs':0}
assert r['identity']['source_sha256']==sha(source)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['m1660_review_sha256']==sha(review_path)
PY

[[ -n "${M1659_EXPECTED_SOURCE_SHA256:-}" &&
    "$(sha_file "${SELF}")" == "${M1659_EXPECTED_SOURCE_SHA256}" ]] ||
  fail "caller must pin reviewed M1659 source SHA"
[[ -n "${M1659_EXPECTED_RELEASE_SHA256:-}" &&
    "$(sha_file "${FUTURE_RELEASE}")" == "${M1659_EXPECTED_RELEASE_SHA256}" ]] ||
  fail "caller must pin reviewed M1664 release SHA"

# Re-derive every forensic positive gate before any mutation.
forensic_gate "${SOURCE}"
[[ ! -e "${TARGET}" && ! -e "${ATTEMPT}" && ! -e "${WORK}"
    && ! -e "${FAILQ}" ]] || fail "target/attempt/work/failure namespace collision"

if ! mkdir -- "${LOCK}"; then fail "atomic recovery lock collision"; fi
LOCK_HELD=1
[[ ! -e "${TARGET}" && ! -e "${WORK}" && ! -e "${FAILQ}" ]] ||
  fail "post-lock target/work/failure namespace collision"
if ! mkdir -- "${ATTEMPT}"; then fail "one-shot recovery attempt consumed"; fi
printf 'status=M1665_M1659_COPY_ONLY_RECOVERY_ATTEMPT_CONSUMED\nmax_recoveries=1\nretry=false\ncopy_may_start_only_after_this_seal=true\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
printf 'source_sha256=%s\nrelease_sha256=%s\nsource_quarantine_manifest_sha256=%s\n' \
  "$(sha_file "${SELF}")" "$(sha_file "${FUTURE_RELEASE}")" \
  "${SOURCE_MANIFEST_SHA256}" >"${ATTEMPT}/IDENTITY.txt"
seal_dir "${ATTEMPT}"

mkdir -- "${WORK}"
WORK_ACTIVE=1
mkdir -- "${WORK}/original_quarantine"
cp -a --no-dereference "${SOURCE}/." "${WORK}/original_quarantine/"
verify_dir_seal "${WORK}/original_quarantine" 39
sha_exact "${SOURCE_MANIFEST_SHA256}" "${WORK}/original_quarantine/SHA256SUMS"
sha_exact "${SOURCE_OUTER_FILE_SHA256}" "${WORK}/original_quarantine/SHA256SUMS.seal.sha256"
forensic_gate "${WORK}/original_quarantine"

/usr/bin/python3.6 -I - "${WORK}" "${SOURCE}" "${SOURCE_ATTEMPT}" \
  "${M1649_RUNNER}" "${M1649_CONTRACT}" "${M1650_REVIEW_DIR}/review.json" \
  "${M1651_RELEASE}" "${M1655_REVIEW_DIR}/review.json" "${CONTRACT}" \
  "${FUTURE_REVIEW_DIR}/review.json" "${FUTURE_RELEASE}" "${SELF}" <<'PY'
import hashlib,json,sys
from pathlib import Path
(work,source,source_attempt,m1649_runner,m1649_contract,m1650,m1651,m1655,
 contract,m1660,m1664,script)=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity={'m1659_source_sha256':sha(script),'m1659_contract_sha256':sha(contract),
 'm1660_review_sha256':sha(m1660),'m1664_release_sha256':sha(m1664),
 'm1649_runner_sha256':sha(m1649_runner),'m1649_contract_sha256':sha(m1649_contract),
 'm1650_review_sha256':sha(m1650),'m1651_release_sha256':sha(m1651),
 'm1655_review_sha256':sha(m1655)}
provenance={
 'schema':'m1665_m1659_m1649_c1_copy_only_recovery_provenance_r1_v1',
 'status':'COPY_ONLY_RECOVERY_OF_DC_COMPLETE_SEALED_M1649_QUARANTINE',
 'source_quarantine':str(source),'source_attempt':str(source_attempt),
 'source_manifest_sha256':'e94ffc3680513cb2f374676037cc7c3b14b77a7bc47b9d35edb812f17a9ae843',
 'source_outer_seal_file_sha256':'c221bb79e4950780c6db04ef54ed1ea809ac880ad054f9316f7bba702a49ff44',
 'source_members':39,'original_failure_marker_preserved_at':'original_quarantine/RUN_FAILED_OR_INCOMPLETE.txt',
 'runner_classification':{'runner_exit_code':3,'dc_process_return_code':0,
   'only_error_line':32,'only_error_phase':'pre_flow_gui_startup',
   'cause':'env -i omitted HOME; dv.tcl startup error matched an over-broad post-run grep',
   'in_flow_error_or_fatal_count':0},
 'concurrency':{'atomic_launch_lock':True,'one_shot_attempt_before_copy':True,
   'fixed_work_identity':True,'no_replace_atomic_publish':True,'retry':False},
 'identity':identity,
 'mutation':{'source_quarantine_modified':False,'dc_rerun':False,
   'copied_artifact_bytes_changed':False,'eda_run':False}}
receipt={
 'schema':'m1665_recovered_m1649_c1_residual_hold_closed_dc_receipt_r1_v1',
 'status':'PASS_RECOVERED_M1649_C1_RESIDUAL_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_POWER',
 'technology_nm':28,'clock_period_ns':3.0,'ideal_clock':True,'wireload':'ZeroWireload',
 'uncertainty_ns':{'setup':0.2,'reported_hold':0.05,'optimization_hold_guardband':0.051},
 'setup':{'met':True,'wns_ns':0.002221110,'tns_ns':0.0,'violating_paths':0},
 'hold':{'met':True,'wns_ns':0.000999451,'tns_ns':0.0,'violating_paths':0},
 'area':{'baseline_um2':147246.392090,'recovered_um2':152898.625984,
   'overhead_percent':3.8386230139650923,'within_five_percent':True},
 'macros':{'cell':'TS1N28HPCPHVTB128X128M4S','count':9},
 'drc_violating_nets':0,'input_generation':'original_m993_m1006_admitted_ddc',
 'failed_m1614_output_used':False,'hold_only_incremental_mapping_count':1,
 'identity':identity,
 'next_gates':{'gate_to_gate_formality':True,'direct_or_transitive_rtl_formality':True,
   'independent_prime_time_max_min':True,'power':True},
 'claim_boundary':{'dc_setup_hold_area_macro_drc_candidate':True,
   'formality':False,'independent_pt':False,'power':False,'energy':False,
   'cycle_speedup':False,'system_speedup':False,'paper_ppa_ready':False,
   'paper_citable':False,'headline':False}}
(work/'M1665_RECOVERY_PROVENANCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True,allow_nan=False)+'\n')
(work/'m1665_recovered_c1_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
(work/'RUN_COMPLETE_RECOVERED.txt').write_text(
 'status=PASS_RECOVERED_M1649_C1_RESIDUAL_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_POWER\n'
 'source_failure_marker_preserved=true\ndc_rerun=false\nformality=false\nindependent_pt=false\n'
 'power=false\nenergy=false\ncycle_speedup=false\nsystem_speedup=false\n'
 'paper_ppa_ready=false\npaper_citable=false\n')
PY

seal_dir "${WORK}"
verify_dir_seal "${WORK}" 42
[[ ! -e "${TARGET}" ]] || fail "target appeared before publication"
mv -T -- "${WORK}" "${TARGET}"
WORK_ACTIVE=0
verify_dir_seal "${TARGET}" 42
rmdir -- "${LOCK}"
LOCK_HELD=0
trap - EXIT INT TERM
echo "M1659 copy-only recovery published: ${TARGET}"
