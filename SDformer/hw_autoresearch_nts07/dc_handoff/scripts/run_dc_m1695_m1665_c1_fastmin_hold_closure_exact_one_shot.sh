#!/usr/bin/env bash
set -euo pipefail
umask 002

# Inert until a different-author M1696 source hammer and separately sealed
# M1697 release exist and are caller-pinned.  A released identity permits one
# dc_shell process only.  No RTL, Formality, PrimeTime, VCS or PTPX is run.

[[ $# -eq 0 ]] || { echo "ERROR: M1695 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_candidate.tcl"
TEST="${HW_ROOT}/system_simulator/tests/test_m1695_c1_fastmin_hold_closure_source.py"
CONTRACT="${HW_ROOT}/contracts/m1695_m1665_c1_fastmin_hold_closure_source_contract_r1_20260901.json"
AUTHOR_DIR="${HW_ROOT}/reviews/m1695_m1665_c1_fastmin_hold_closure_source_author_receipt_r1_20260901"
HAMMER_DIR="${HW_ROOT}/reviews/m1696_m1695_c1_fastmin_hold_closure_source_hammer_r1_20260901"
HAMMER_REVIEW="${HAMMER_DIR}/review.json"
RELEASE="${HW_ROOT}/contracts/m1697_m1696_m1695_c1_fastmin_hold_closure_launch_release_r1_20260901.json"

M1665_DIR="${HW_ROOT}/dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
M1665_ORIGINAL="${M1665_DIR}/original_quarantine"
INPUT_DDC="${M1665_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc"
INPUT_SDC="${M1665_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc"
M1678_NEGATIVE="${HW_ROOT}/dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_r1_20260901.failed_or_incomplete.1991841.quarantine"
M1678_RTL_FM="${M1678_NEGATIVE}/rtl_to_m993/FORMALITY_INTERNAL_COMPLETE.txt"
M1678_GATE_FM="${M1678_NEGATIVE}/m993_to_m1665/FORMALITY_INTERNAL_COMPLETE.txt"
M1678_GLOBAL="${M1678_NEGATIVE}/ptsta/reports/global_timing.rpt"
M1678_SETUP="${M1678_NEGATIVE}/ptsta/reports/timing_setup_slow.rpt"
M1678_HOLD="${M1678_NEGATIVE}/ptsta/reports/timing_hold_fast.rpt"
M1678_CONSTRAINTS="${M1678_NEGATIVE}/ptsta/reports/constraint_violators.rpt"
M1665_DC_HOLD="${M1665_ORIGINAL}/reports/timing_hold_posthold_top100.rpt"
DOC359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

DC_SHELL="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
DC_ACTUAL="/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"
LMUTIL="/opt/synopsys/scl/2025.03/linux64/bin/lmutil"
FLOCK="/usr/bin/flock"
LICENSE_FILE="/opt/synopsys/Synopsys.dat"
STD_SLOW="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
STD_FAST="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
MACRO_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
MACRO_SLOW="${MACRO_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
MACRO_FAST="${MACRO_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"
MACRO_MANIFEST="${MACRO_ROOT}/SHA256SUMS"

RESULT="${HW_ROOT}/dc_handoff/runs/m1695_m1665_c1_fastmin_hold_closure_dc_r1_20260901"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1695_m1665_c1_fastmin_hold_closure_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1695_m1665_c1_fastmin_hold_closure_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1695_m1665_c1_fastmin_hold_closure_dc_launch_lock"
SHARED_QUEUE="/tmp/date_dual_synopsys_same_uid_eda_queue.lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
fail() { echo "ERROR: M1695 $*" >&2; exit 3; }
sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/nonregular ${path}"
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || fail "SHA mismatch ${path}: ${got}"
}
verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}.sha256" && ! -L "${payload}.sha256" && \
      -f "${payload}.sha256.seal.sha256" && ! -L "${payload}.sha256.seal.sha256" ]] ||
    fail "file seal absent ${payload}"
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null &&
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) ||
    fail "file seal invalid ${payload}"
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" && \
      -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "directory seal absent ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) ||
    fail "directory seal invalid ${dir}"
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
same_uid_dc() {
  /usr/bin/python3.6 -I - <<'PY'
from __future__ import print_function
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
if hits: print(repr(hits))
PY
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

# Exact immutable inputs and M1678 calibration/failure evidence.
verify_dir_seal "${M1665_DIR}"
verify_dir_seal "${M1665_ORIGINAL}"
verify_dir_seal "${M1678_NEGATIVE}"
sha_exact a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72 "${M1665_DIR}/SHA256SUMS"
sha_exact 12d87acb439b0cc171d3f42cd4f169fa6a531946c9c3c120cc9babc9c36fbc08 "${M1665_DIR}/SHA256SUMS.seal.sha256"
sha_exact 9556e3bfab30af74326473f6cb9e492d41d3b782d0f23fabb6564626ce6fc675 "${M1678_NEGATIVE}/SHA256SUMS"
sha_exact 7b90352dd62288415f12903cbc4c2745cf2f2fa574080b37f63871015bc77602 "${M1678_NEGATIVE}/SHA256SUMS.seal.sha256"
sha_exact 2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0 "${INPUT_DDC}"
sha_exact 5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198 "${INPUT_SDC}"
sha_exact 9eee52aa958d835e9b682d99e5b52cfed515bacee74854fb8f0a4a8ddfab7eb9 "${M1678_RTL_FM}"
sha_exact b27aeb9e49081c6fbc238a082dfe7c364270e25ca11579e7ee73c717d0a12fd8 "${M1678_GATE_FM}"
sha_exact c323bdd22a6f9137ee02f85aba0ed9c7792cf1febd6d8c3b11fb2650d41f7557 "${M1678_GLOBAL}"
sha_exact c0dc0bce139cdf1f8be3058c43bc40ed5b67fa8c2c82292b7265f0f232f35495 "${M1678_SETUP}"
sha_exact eeacd609124059018fdc1bbdafd460342adcc524473d0769c4d43daa43aa3445 "${M1678_HOLD}"
sha_exact d974d269d592fe02ea04db0c062c8061bba1f8d6e67fd479bb929a1da97526eb "${M1678_CONSTRAINTS}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
sha_exact 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2 "${DC_SHELL}"
sha_exact bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391 "${DC_ACTUAL}"
sha_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
sha_exact 54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435 "${FLOCK}"
sha_exact fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490 "${LICENSE_FILE}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${STD_SLOW}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${STD_FAST}"
sha_exact cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${MACRO_SLOW}"
sha_exact 8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f "${MACRO_FAST}"
sha_exact c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${MACRO_MANIFEST}"
(cd -- "${MACRO_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)

/usr/bin/python3.6 -I - "${M1678_RTL_FM}" "${M1678_GATE_FM}" \
  "${M1678_GLOBAL}" "${M1678_SETUP}" "${M1678_HOLD}" \
  "${M1678_CONSTRAINTS}" "${M1665_DC_HOLD}" <<'PY'
from __future__ import print_function
import re,sys
from pathlib import Path
rtl,gate,glob,setup,hold,constraints,dc=map(Path,sys.argv[1:])
assert 'INTERNAL_COMPLETE=PASS' in rtl.read_text()
assert 'INTERNAL_COMPLETE=PASS' in gate.read_text()
g=glob.read_text(errors='replace')
assert re.search(r'TNS\s+-40\.24\s+-40\.24',g)
assert re.search(r'NUM\s+10610\s+10610',g)
s=setup.read_text(errors='replace'); h=hold.read_text(errors='replace')
assert re.search(r'slack \(MET\)\s+0\.002221',s)
assert re.search(r'slack \(VIOLATED\)\s+-0\.028168',h)
assert re.search(r'library hold time\s+0\.12685[89]',h)
assert 'slack (VIOLATED)' in constraints.read_text(errors='replace')
d=dc.read_text(errors='replace')
assert re.search(r'library hold time\s+0\.097685',d)
assert abs(0.126859-0.097685-0.029174)<1e-12
PY

# The exact mapped SDC remains the sole constraint input.
[[ "$(grep -Ec '^[[:space:]]*create_clock .* -period 3([.]0+)?([[:space:]]|$)' "${INPUT_SDC}")" -eq 1 ]] || fail "3 ns clock identity"
[[ "$(grep -Ec '^[[:space:]]*set_clock_uncertainty -setup 0[.]2([[:space:]]|$)' "${INPUT_SDC}")" -eq 1 ]] || fail "setup uncertainty identity"
[[ "$(grep -Ec '^[[:space:]]*set_clock_uncertainty -hold 0[.]05([[:space:]]|$)' "${INPUT_SDC}")" -eq 1 ]] || fail "hold uncertainty identity"
! grep -Eq '^[[:space:]]*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis)([[:space:]]|$)' "${INPUT_SDC}" || fail "forbidden timing exception"

# Source authority: double-sealed author package, different-author M1696, and
# separately sealed M1697 release.  Missing any item leaves this runner inert.
verify_file_seal "${CONTRACT}"
verify_dir_seal "${AUTHOR_DIR}"
verify_dir_seal "${HAMMER_DIR}"
verify_file_seal "${RELEASE}"
/usr/bin/python3.6 -I - "${CONTRACT}" "${AUTHOR_DIR}/author_receipt.json" \
  "${HAMMER_REVIEW}" "${RELEASE}" "${RUNNER}" "${TCL}" "${TEST}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
c,a,h,r,runner,tcl,test=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
C=json.loads(c.read_text()); A=json.loads(a.read_text())
H=json.loads(h.read_text()); R=json.loads(r.read_text())
assert C['status']=='SOURCE_ONLY_M1695_C1_FASTMIN_HOLD_CLOSURE__NO_EDA_AUTHORIZED'
assert C['authorization']['dc_runs_now']==0 and C['authorization']['future_dc_runs_max']==1
assert C['identity']['runner_sha256']==sha(runner)
assert C['identity']['tcl_sha256']==sha(tcl)
assert C['identity']['author_test_sha256']==sha(test)
assert A['status']=='PASS_M1695_C1_FASTMIN_HOLD_CLOSURE_SOURCE_AUTHOR_HANDOFF__NO_EDA'
assert H['status']=='PASS_M1696_M1695_C1_FASTMIN_HOLD_CLOSURE_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT'
assert H['score']>=95 and H['p0_count']==0 and H['p1_count']==0
assert R['status']=='AUTHORIZE_ONE_M1695_C1_FASTMIN_HOLD_CLOSURE_DC_ATTEMPT'
assert R['authorization']=={'dc_runs':1,'all_other_eda_runs':0}
assert R['identity']['runner_sha256']==sha(runner)
assert R['identity']['source_contract_sha256']==sha(c)
assert R['identity']['hammer_review_sha256']==sha(h)
PY

[[ -n "${M1695_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(sha_file "${RUNNER}")" == "${M1695_EXPECTED_DC_RUNNER_SHA256}" ]] ||
  fail "caller must pin reviewed M1695 runner SHA"
[[ -n "${M1695_EXPECTED_DC_RELEASE_SHA256:-}" && \
   "$(sha_file "${RELEASE}")" == "${M1695_EXPECTED_DC_RELEASE_SHA256}" ]] ||
  fail "caller must pin reviewed M1697 release SHA"

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] ||
  fail "result identity already consumed or colliding"
# Serialize every Synopsys campaign on this host with the shared C1/C3 queue.
# FD 9 stays open (and the exclusive lock stays held) through dc_shell exit,
# result sealing and publication.  Recheck ancestry-aware same-UID state both
# after taking the lock and immediately before launch.
exec 9>"${SHARED_QUEUE}"
"${FLOCK}" -x 9
[[ -z "$(same_uid_dc)" ]] || fail "same-UID DC collision after shared lock"
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
headroom=$((commit_limit-committed))
[[ "${mem_available}" -ge 16777216 && "${swap_free}" -ge 8388608 && \
    "${headroom}" -ge 25165824 ]] || fail "24 GiB commit/memory gate not met"
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null
[[ -z "$(same_uid_dc)" ]] || fail "same-UID DC collision after license probe"

mkdir -- "${LOCK}"
mkdir -- "${ATTEMPT}"
printf 'status=M1695_ATTEMPT_CONSUMED_BEFORE_DC\nmax_dc_runs=1\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -- "${WORK}"
WORK_ACTIVE=1
printf 'status=M1695_DC_ATTEMPT_ADMITTED\nclock_period_ns=3.000\nsetup_uncertainty_ns=0.200\noptimization_hold_uncertainty_ns=0.081\nreported_hold_uncertainty_ns=0.050\nmacro_count=9\ncommit_headroom_min_kib=25165824\nretry=false\n' \
  >"${WORK}/admission.txt"

[[ -z "$(same_uid_dc)" ]] || fail "same-UID DC collision immediately before launch"
set +e
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
  M1695_INPUT_DDC="${INPUT_DDC}" M1695_INPUT_SDC="${INPUT_SDC}" \
  M1695_STD_SLOW_DB="${STD_SLOW}" M1695_STD_FAST_DB="${STD_FAST}" \
  M1695_MACRO_SLOW_DB="${MACRO_SLOW}" M1695_MACRO_FAST_DB="${MACRO_FAST}" \
  M1695_OUTPUT_DIR="${WORK}" \
  "${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}" \
  >"${WORK}/dc.log" 2>&1
dc_rc=$?
set -e
printf '%s\n' "${dc_rc}" >"${WORK}/dc.rc"
[[ "${dc_rc}" -eq 0 ]] || exit "${dc_rc}"

required=(
  admission.txt dc.log dc.rc TCL_INTERNAL_COMPLETE.txt
  reports/link.rpt reports/flow_contract.rpt reports/macro_binding_audit.txt
  reports/check_design_prehold.rpt reports/check_timing_prehold.rpt
  reports/qor_prehold.rpt reports/area_prehold.rpt reports/references_prehold.rpt
  reports/timing_setup_prehold_top100.rpt reports/timing_hold_prehold_top100.rpt
  reports/setup_prehold_summary_machine.txt reports/hold_prehold_summary_machine.txt
  reports/qor_posthold.rpt reports/area_posthold.rpt reports/hierarchy_posthold.rpt
  reports/resources_posthold.rpt reports/references_posthold.rpt reports/clocks_posthold.rpt
  reports/timing_setup_posthold_top100.rpt reports/timing_hold_posthold_top100.rpt
  reports/constraint_setup_posthold_all.rpt reports/constraint_hold_posthold_all.rpt
  reports/constraint_design_rules_posthold.rpt reports/check_design_posthold.rpt
  reports/check_timing_posthold.rpt reports/setup_posthold_summary_machine.txt
  reports/hold_posthold_summary_machine.txt
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1695_fastmin_hold_closed_mapped.v
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1695_fastmin_hold_closed_mapped.sdc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1695_fastmin_hold_closed.ddc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1695_fastmin_hold_closed.svf
)
for artifact in "${required[@]}"; do
  [[ -s "${WORK}/${artifact}" && ! -L "${WORK}/${artifact}" ]] ||
    fail "missing result artifact ${artifact}"
done

if grep -Eiq '(^|[[:space:]])(Error|Fatal):|LINK-[0-9]+|unresolved (reference|design|cell)|unable to resolve|combinational[ _-]*loop|timing[ _-]*loop|\((TIM-209|OPT-150)\)' \
    "${WORK}/dc.log" "${WORK}/reports/link.rpt" \
    "${WORK}/reports/check_design_prehold.rpt" "${WORK}/reports/check_timing_prehold.rpt" \
    "${WORK}/reports/check_design_posthold.rpt" "${WORK}/reports/check_timing_posthold.rpt"; then
  fail "Error/Fatal/link/loop evidence found"
fi
grep -Fxq 'status=PASS_M1695_RESOLVED_LIBRARY_MACRO_STRUCTURE' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_pre=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_post=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'optimization_hold_uncertainty_ns=0.081' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'reported_hold_uncertainty_ns=0.050' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'hold_only_incremental_mapping_count=1' "${WORK}/reports/flow_contract.rpt"

/usr/bin/python3.6 -I - "${WORK}" "${RUNNER}" "${CONTRACT}" "${RELEASE}" \
  "${INPUT_DDC}" "${INPUT_SDC}" <<'PY'
from __future__ import print_function
import hashlib,json,math,re,sys
from pathlib import Path
root,runner,contract,release,input_ddc,input_sdc=map(Path,sys.argv[1:])
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
pre_setup=timing('setup_prehold_summary_machine.txt'); pre_hold=timing('hold_prehold_summary_machine.txt')
post_setup=timing('setup_posthold_summary_machine.txt'); post_hold=timing('hold_posthold_summary_machine.txt')
area_text=(root/'reports/area_posthold.rpt').read_text(errors='replace')
m=re.search(r'Total cell area:\s*([0-9.]+)',area_text)
if not m: raise SystemExit('missing total cell area')
area=float(m.group(1)); baseline=152898.625984; ceiling=168188.4885824
qor=(root/'reports/qor_posthold.rpt').read_text(errors='replace')
drc=re.search(r'Nets With Violations:\s*([0-9.]+)',qor)
if not drc: raise SystemExit('missing DRC population')
drc_count=int(float(drc.group(1)))
sdc_path=root/'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1695_fastmin_hold_closed_mapped.sdc'
sdc=sdc_path.read_text(errors='replace')
for term in ('set_false_path','set_multicycle_path','set_min_delay','set_max_delay','set_disable_timing','set_case_analysis'):
    if re.search(r'(?m)^\s*'+term+r'\b',sdc): raise SystemExit('forbidden output SDC '+term)
if len(re.findall(r'(?m)^\s*create_clock\b',sdc))!=1: raise SystemExit('clock population')
if not re.search(r'create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)',sdc): raise SystemExit('period drift')
if not re.search(r'set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b',sdc): raise SystemExit('setup uncertainty drift')
if not re.search(r'set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b',sdc): raise SystemExit('hold uncertainty drift')
if re.search(r'set_clock_uncertainty\s+-hold\s+0?\.081(?:0+)?\b',sdc): raise SystemExit('optimization uncertainty leaked')
macro=kv(root/'reports/macro_binding_audit.txt')
macro_ok=int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9
timing_ok=post_setup['status']=='MET' and post_hold['status']=='MET'
area_ok=math.isfinite(area) and area>0 and area<=ceiling
positive=timing_ok and area_ok and macro_ok and drc_count==0
status=('PASS_RAW_M1695_C1_FASTMIN_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_RESULT_HAMMER'
        if positive else 'SEALED_NEGATIVE_M1695_C1_FASTMIN_HOLD_OR_AREA_GATE_FAILED__NO_RETRY')
receipt={'schema':'m1695_m1665_c1_fastmin_hold_closure_dc_receipt_v1','status':status,
 'positive_dc_candidate':positive,'clock_period_ns':3.0,'setup_uncertainty_ns':0.2,
 'optimization_hold_uncertainty_ns':0.081,'reported_hold_uncertainty_ns':0.05,
 'ideal_clock':True,'wireload':'ZeroWireload','pre':{'setup':pre_setup,'hold':pre_hold},
 'post':{'setup':post_setup,'hold':post_hold},
 'area':{'baseline_um2':baseline,'post_um2':area,'overhead_percent':(area/baseline-1)*100,
         'ceiling_um2':ceiling,'within_10_percent':area_ok},
 'macros':{'cell':'TS1N28HPCPHVTB128X128M4S','pre':9,'post':9,'passed':macro_ok},
 'design_rule_violating_nets':drc_count,
 'calibration':{'m1678_pt_hold_wns_ns':-0.028168,'m1678_hold_violating_paths':10610,
  'm1678_hold_tns_ns':-40.24,'dc_macro_hold_check_ns':0.097685,
  'pt_macro_hold_check_ns':0.126859,'delta_ns':0.029174},
 'flow':{'input_generation':'frozen_m1665_ddc_only','set_fix_hold_count':1,
  'hold_only_incremental_mapping_count':1,'retry':False,'functional_rtl_modified':False},
 'identity':{'runner_sha256':sha(runner),'contract_sha256':sha(contract),
  'release_sha256':sha(release),'input_ddc_sha256':sha(input_ddc),'input_sdc_sha256':sha(input_sdc)},
 'next_gates':{'different_author_result_hammer':True,'formality_gate_to_gate':True,
  'independent_pt_slowmax_fastmin':True},
 'claim_boundary':{'dc_hold_closed_candidate':positive,'formality_new_identity':False,
  'prime_time_new_identity':False,'power':False,'energy':False,'cycle_speedup':False,
  'system_speedup':False,'paper_ppa_ready':False,'headline':False}}
(root/'m1695_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
(root/'RUN_COMPLETE.txt').write_text('status='+status+'\nretry=false\nformality=false\nindependent_pt=false\npaper_citable=false\n')
PY
/usr/bin/python3.6 -m json.tool "${WORK}/m1695_dc_receipt.json" >/dev/null

run_status="$(/usr/bin/python3.6 -I - "${WORK}/m1695_dc_receipt.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)"
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM
rmdir -- "${LOCK}"
echo "M1695 completed: ${RESULT} (${run_status})"
