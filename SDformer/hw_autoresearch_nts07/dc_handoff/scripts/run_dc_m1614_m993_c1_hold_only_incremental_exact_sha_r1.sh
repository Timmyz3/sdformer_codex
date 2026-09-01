#!/usr/bin/env bash
set -euo pipefail
umask 002

# M1614 is source-only until a fresh M1615 different-author source hammer and
# M1616 release exist and are caller-pinned.  One consumed attempt permits one
# DC process and exactly one hold-only incremental mapping command in the
# frozen Tcl.  Failure, OOM, a negative timing/area gate, or partial output can
# never be retried under this identity.

[[ $# -eq 0 ]] || { echo "ERROR: M1614 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
CONTRACT="${HW_ROOT}/contracts/m1614_m993_c1_hold_only_incremental_dc_source_contract_r1_20260901.json"
HAMMER_DIR="${HW_ROOT}/reviews/m1615_m1614_c1_hold_only_incremental_dc_source_hammer_r1_20260901"
HAMMER_REVIEW="${HAMMER_DIR}/review.json"
RELEASE="${HW_ROOT}/contracts/m1616_m1615_m1614_c1_hold_only_incremental_dc_launch_release_r1_20260901.json"
M993_DIR="${HW_ROOT}/dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
M993_ORIGINAL="${M993_DIR}/original_quarantine"
M1006_DIR="${HW_ROOT}/reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1612_DIR="${HW_ROOT}/reviews/m1612_m993_c1_hold_closure_first_principles_readonly_review_r1_20260901"
INPUT_DDC="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc"
INPUT_SDC="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc"
INPUT_MAPPED_V="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v"
INPUT_SVF="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf"
RTL="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m1614_m993_c1_hold_only_incremental_candidate.tcl"
DOC359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
DC_SHELL="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
DC_ACTUAL="/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"
LMUTIL="/opt/synopsys/scl/2025.03/linux64/bin/lmutil"
LICENSE_FILE="/opt/synopsys/Synopsys.dat"
STD_SLOW="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
STD_FAST="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
MACRO_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
MACRO_SLOW="${MACRO_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
MACRO_FAST="${MACRO_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"
MACRO_MANIFEST="${MACRO_ROOT}/SHA256SUMS"
RESULT="${HW_ROOT}/dc_handoff/runs/m1614_m993_c1_macro_aware_hold_only_incremental_dc_r1_20260901"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1614_m993_c1_macro_aware_hold_only_incremental_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1614_m993_c1_macro_aware_hold_only_incremental_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1614_m993_c1_macro_aware_hold_only_incremental_dc_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "ERROR: missing/nonregular ${path}" >&2; exit 3; }
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || {
    echo "ERROR: SHA mismatch ${path}: ${got}" >&2; exit 3; }
}
sha_tool() {
  local expected="$1" path="$2" got
  [[ -f "${path}" ]] || { echo "ERROR: missing tool ${path}" >&2; exit 3; }
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || {
    echo "ERROR: tool SHA mismatch ${path}: ${got}" >&2; exit 3; }
}
verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}.sha256" && ! -L "${payload}.sha256"
      && -f "${payload}.sha256.seal.sha256"
      && ! -L "${payload}.sha256.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null &&
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS"
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
  python3 -I - "${dir}" <<'PY'
import os, stat, sys
from pathlib import Path
d=Path(sys.argv[1]); listed=set()
for line in (d/'SHA256SUMS').read_text().splitlines():
    if line.strip(): listed.add(line.split(None,1)[1].lstrip('*'))
actual=set()
for root, dirs, files in os.walk(d, followlinks=False):
    rp=Path(root); dirs[:]=[n for n in dirs if not (rp/n).is_symlink()]
    for name in files:
        p=rp/name
        if name in {'SHA256SUMS','SHA256SUMS.seal.sha256'}: continue
        if stat.S_ISREG(os.lstat(p).st_mode): actual.add(str(p.relative_to(d)))
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
on_exit() {
  local rc=$?
  set +e
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' \
      "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM

# Immutable evidence, design, technology and tool identity.
verify_dir_seal "${M993_DIR}"
verify_dir_seal "${M993_ORIGINAL}"
verify_dir_seal "${M1006_DIR}"
verify_dir_seal "${M1612_DIR}"
sha_exact d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56 "${INPUT_DDC}"
sha_exact cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5 "${INPUT_SDC}"
sha_exact 9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf "${INPUT_MAPPED_V}"
sha_exact 8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7 "${INPUT_SVF}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
sha_exact 7baba71a21be61842be8c76bddfa40abf8d2c0b0736e06aa44a80d53556cef72 "${M1612_DIR}/review.json"
sha_tool 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2 "${DC_SHELL}"
sha_tool bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391 "${DC_ACTUAL}"
sha_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
sha_exact fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490 "${LICENSE_FILE}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${STD_SLOW}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${STD_FAST}"
sha_exact cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${MACRO_SLOW}"
sha_exact 8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f "${MACRO_FAST}"
sha_exact c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${MACRO_MANIFEST}"
(cd -- "${MACRO_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)

verify_file_seal "${CONTRACT}"
python3 -I - "${CONTRACT}" "${RUNNER}" "${TCL}" "${M1612_DIR}/review.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,runner,tcl,m1612=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); r=json.loads(m1612.read_text())
assert c['status']=='SOURCE_ONLY_M1614_C1_HOLD_PACKAGE__NO_EDA_AUTHORIZED'
assert c['authorization']['dc_runs_now']==0
assert c['authorization']['future_dc_runs_max']==1
assert c['identity']['runner_sha256']==sha(runner)
assert c['identity']['tcl_sha256']==sha(tcl)
assert c['identity']['m1612_review_sha256']==sha(m1612)
assert r['authorization']['all_eda_now']==0
PY

# Static constraint and single-command hygiene before any release gate.
python3 -I - "${TCL}" "${INPUT_SDC}" <<'PY'
import re,sys
from pathlib import Path
tcl,sdc=(Path(x).read_text() for x in sys.argv[1:])
strip=lambda s:'\n'.join(x.split('#',1)[0] for x in s.splitlines())
t=strip(tcl); s=strip(sdc)
assert len(re.findall(r'(?m)^\s*set_fix_hold\b',t))==1
assert len(re.findall(r'(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$',t))==1
assert not re.search(r'(?m)^\s*compile_ultra\b',t)
assert len(re.findall(r'(?m)^\s*compile\b',t))==1
for term in ('set_false_path','set_multicycle_path','set_min_delay','set_max_delay',
             'set_disable_timing','set_case_analysis'):
    assert not re.search(r'(?m)^\s*'+term+r'\b',t+'\n'+s), term
assert len(re.findall(r'(?m)^\s*create_clock\b',s))==1
assert re.search(r'create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)',s)
assert re.search(r'set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b',s)
assert re.search(r'set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b',s)
PY

# M1614 cannot launch until a fresh different-author hammer and release bind
# this exact source package.  Their absence is the intentional hard stop now.
verify_dir_seal "${HAMMER_DIR}"
verify_file_seal "${RELEASE}"
python3 -I - "${HAMMER_REVIEW}" "${RELEASE}" "${RUNNER}" "${CONTRACT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
hammer,release,runner,contract=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads(hammer.read_text()); r=json.loads(release.read_text())
assert h['status']=='PASS_M1615_M1614_C1_HOLD_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT'
assert h['score']>=95 and h['p0_count']==0 and h['p1_count']==0
assert r['status']=='AUTHORIZE_ONE_M1614_C1_HOLD_ONLY_INCREMENTAL_DC_ATTEMPT'
assert r['authorization']=={'dc_runs':1,'all_other_eda_runs':0}
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['hammer_review_sha256']==sha(hammer)
PY

[[ -n "${M1614_EXPECTED_DC_RUNNER_SHA256:-}" &&
    "$(sha_file "${RUNNER}")" == "${M1614_EXPECTED_DC_RUNNER_SHA256}" ]] || {
  echo "ERROR: caller must pin reviewed M1614 runner SHA" >&2; exit 3; }
[[ -n "${M1614_EXPECTED_DC_RELEASE_SHA256:-}" &&
    "$(sha_file "${RELEASE}")" == "${M1614_EXPECTED_DC_RELEASE_SHA256}" ]] || {
  echo "ERROR: caller must pin reviewed M1616 release SHA" >&2; exit 3; }

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] || {
  echo "ERROR: M1614 result identity already consumed or colliding" >&2; exit 4; }
python3 -I - <<'PY'
import os
from pathlib import Path
blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}
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
        argv={Path(x.decode(errors='replace')).name for x in
              (p/'cmdline').read_bytes().split(b'\0') if x}
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or blocked & argv: hits.append((p.name,comm,sorted(argv)))
if hits: raise SystemExit('same-UID DC collision: %r' % hits)
PY
mkdir -- "${LOCK}" || exit 4
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
headroom=$((commit_limit-committed))
[[ "${mem_available}" -ge 100663296 && "${swap_free}" -ge 16777216
    && "${headroom}" -ge 67108864 ]] || {
  echo "ERROR: M1614 memory/commit gate not met" >&2; exit 4; }
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null

mkdir -- "${ATTEMPT}"
printf 'status=M1614_ATTEMPT_CONSUMED\nmax_dc_runs=1\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -- "${WORK}"
WORK_ACTIVE=1
printf 'status=M1614_DC_ATTEMPT_ADMITTED\nclock_period_ns=3.000\nhold_only_passes=1\nmacro_count=9\nretry=false\n' \
  >"${WORK}/admission.txt"

set +e
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
  M1614_INPUT_DDC="${INPUT_DDC}" M1614_INPUT_SDC="${INPUT_SDC}" \
  M1614_INPUT_MAPPED_V="${INPUT_MAPPED_V}" M1614_INPUT_SVF="${INPUT_SVF}" \
  M1614_STD_SLOW_DB="${STD_SLOW}" M1614_STD_FAST_DB="${STD_FAST}" \
  M1614_MACRO_SLOW_DB="${MACRO_SLOW}" M1614_MACRO_FAST_DB="${MACRO_FAST}" \
  M1614_OUTPUT_DIR="${WORK}" \
  "${DC_SHELL}" -f "${TCL}" >"${WORK}/dc.log" 2>&1
dc_rc=$?
set -e
printf '%s\n' "${dc_rc}" >"${WORK}/dc.rc"
[[ "${dc_rc}" -eq 0 ]] || exit "${dc_rc}"

if rg -ni '(^|[^A-Za-z])(Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+)|\((TIM-209|OPT-150)\)' \
    "${WORK}/dc.log" >/dev/null; then
  echo "ERROR: DC log contains fatal/link/loop evidence" >&2; exit 9
fi

required=(
  admission.txt dc.log dc.rc TCL_INTERNAL_COMPLETE.txt
  reports/link.rpt reports/flow_contract.rpt reports/macro_binding_audit.txt
  reports/check_design_prehold.rpt reports/check_timing_prehold.rpt
  reports/qor_prehold.rpt reports/area_prehold.rpt reports/clocks_prehold.rpt
  reports/references_prehold.rpt reports/timing_setup_prehold_top100.rpt
  reports/timing_hold_prehold_top100.rpt reports/constraint_setup_prehold_all.rpt
  reports/constraint_hold_prehold_all.rpt reports/setup_prehold_summary_machine.txt
  reports/hold_prehold_summary_machine.txt reports/qor_posthold.rpt
  reports/area_posthold.rpt reports/hierarchy_posthold.rpt
  reports/resources_posthold.rpt reports/references_posthold.rpt
  reports/clocks_posthold.rpt reports/timing_setup_posthold_top100.rpt
  reports/timing_hold_posthold_top100.rpt reports/constraint_setup_posthold_all.rpt
  reports/constraint_hold_posthold_all.rpt reports/constraint_design_rules_posthold.rpt
  reports/check_design_posthold.rpt reports/check_timing_posthold.rpt
  reports/setup_posthold_summary_machine.txt reports/hold_posthold_summary_machine.txt
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired_mapped.v
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired_mapped.sdc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired.ddc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired.svf
)
for artifact in "${required[@]}"; do
  [[ -s "${WORK}/${artifact}" && ! -L "${WORK}/${artifact}" ]] || exit 6
done
grep -Fxq 'status=PASS_M1614_RESOLVED_LIBRARY_MACRO_STRUCTURE' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_pre=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_post=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'hold_only_incremental_mapping_count=1' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'compile_ultra_incremental_count=0' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'generic_incremental_mapping_count=0' "${WORK}/reports/flow_contract.rpt"

output_v="${WORK}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired_mapped.v"
output_sdc="${WORK}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired_mapped.sdc"
[[ "$(rg -o 'TS1N28HPCPHVTB128X128M4S' "${output_v}" | wc -l)" -eq 9 ]] || exit 9

python3 -I - "${WORK}" "${RUNNER}" "${CONTRACT}" "${RELEASE}" \
  "${INPUT_DDC}" "${INPUT_SDC}" "${INPUT_MAPPED_V}" "${INPUT_SVF}" <<'PY'
import hashlib,json,math,re,sys
from pathlib import Path
root,runner,contract,release,input_ddc,input_sdc,input_v,input_svf=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def kv(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
def timing(name):
    row=kv(root/'reports'/name)
    out={'status':row['status'],'wns_ns':float(row['wns_ns']),
         'tns_ns':float(row['tns_ns']),'violating_paths':int(row['violating_paths'])}
    if not all(math.isfinite(out[x]) for x in ('wns_ns','tns_ns')): raise SystemExit('nonfinite timing')
    met=out['status']=='MET'
    if met != (out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0):
        raise SystemExit('timing summary inconsistency '+name)
    return out
pre_setup=timing('setup_prehold_summary_machine.txt')
pre_hold=timing('hold_prehold_summary_machine.txt')
post_setup=timing('setup_posthold_summary_machine.txt')
post_hold=timing('hold_posthold_summary_machine.txt')
area_text=(root/'reports/area_posthold.rpt').read_text(errors='replace')
m=re.search(r'Total cell area:\s*([0-9.]+)',area_text)
if not m: raise SystemExit('missing total cell area')
area=float(m.group(1)); baseline=147246.392090; ceiling=154608.7116945
if not math.isfinite(area) or area<=0: raise SystemExit('invalid area')
overhead=(area/baseline-1.0)*100.0
qor=(root/'reports/qor_posthold.rpt').read_text(errors='replace')
drc=re.search(r'Nets With Violations:\s*([0-9.]+)',qor)
if not drc: raise SystemExit('missing DRC population')
drc_count=int(float(drc.group(1)))
sdc=(root/'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1614_hold_repaired_mapped.sdc').read_text(errors='replace')
for term in ('set_false_path','set_multicycle_path','set_min_delay','set_max_delay','set_disable_timing','set_case_analysis'):
    if re.search(r'(?m)^\s*'+term+r'\b',sdc): raise SystemExit('forbidden output SDC '+term)
if len(re.findall(r'(?m)^\s*create_clock\b',sdc))!=1: raise SystemExit('clock population')
if not re.search(r'create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)',sdc): raise SystemExit('period drift')
if not re.search(r'set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b',sdc): raise SystemExit('setup uncertainty drift')
if not re.search(r'set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b',sdc): raise SystemExit('hold uncertainty drift')
macro=kv(root/'reports/macro_binding_audit.txt')
macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)
timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')
area_ok=area<=ceiling
positive=timing_ok and area_ok and macro_ok and drc_count==0
status=('PASS_RAW_M1614_C1_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_RESULT_HAMMER'
        if positive else 'SEALED_NEGATIVE_M1614_C1_HOLD_OR_AREA_GATE_FAILED__NO_RETRY')
receipt={
 'schema':'m1614_m993_c1_hold_only_incremental_dc_receipt_v1','status':status,
 'positive_dc_candidate':positive,'clock_period_ns':3.0,'setup_uncertainty_ns':0.2,
 'hold_uncertainty_ns':0.05,'ideal_clock':True,'wireload':'ZeroWireload',
 'pre':{'setup':pre_setup,'hold':pre_hold},'post':{'setup':post_setup,'hold':post_hold},
 'area':{'baseline_um2':baseline,'post_um2':area,'overhead_percent':overhead,
         'ceiling_um2':ceiling,'within_5_percent':area_ok},
 'macros':{'cell':'TS1N28HPCPHVTB128X128M4S','pre':9,'post':9,'passed':macro_ok},
 'design_rule_violating_nets':drc_count,
 'flow':{'set_fix_hold_count':1,'hold_only_incremental_mapping_count':1,
         'compile_ultra_incremental_count':0,'generic_incremental_mapping_count':0,
         'retry':False},
 'identity':{'runner_sha256':sha(runner),'contract_sha256':sha(contract),
             'release_sha256':sha(release),'input_ddc_sha256':sha(input_ddc),
             'input_sdc_sha256':sha(input_sdc),'input_mapped_v_sha256':sha(input_v),
             'input_svf_sha256':sha(input_svf)},
 'next_gates':{'different_author_result_hammer':True,'formality_gate_to_gate':True,
               'formality_direct_rtl_for_complete_c1':True,'independent_pt':True},
 'claim_boundary':{'dc_hold_closed_candidate':positive,'formality':False,'pt':False,
    'power':False,'energy':False,'cycle_speedup':False,'system_speedup':False,
    'paper_ppa_ready':False,'headline':False}}
(root/'m1614_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
(root/'RUN_COMPLETE.txt').write_text(
    'status='+status+'\nretry=false\nformality=false\nindependent_pt=false\npaper_citable=false\n')
PY
python3 -m json.tool "${WORK}/m1614_dc_receipt.json" >/dev/null

run_status="$(python3 -I - "${WORK}/m1614_dc_receipt.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)"
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM
rmdir -- "${LOCK}"
echo "M1614 completed: ${RESULT} (${run_status})"
