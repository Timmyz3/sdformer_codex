#!/usr/bin/env bash
set -euo pipefail
umask 002

# M962 source-only runner for one future macro-aware M935 setup/area DC point.
# M959 is deliberately not accepted as a clean zero-assertion gate.  This
# runner cannot reach resource, license, attempt, or dc_shell work unless a
# separately sealed M963 hammer and M964 release bind one of the two explicit
# M960 repairs: a clean-negative-test identity or a superseding admission that
# knowingly accepts the single expected M923 attack assertion.

[[ $# -eq 0 ]] || { echo "ERROR: M962 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
CONTRACT="${HW_ROOT}/contracts/m962_m960_m959_m935_three_stage_match_macro_aware_dc_source_contract_r1_20260829.json"
HAMMER_DIR="${HW_ROOT}/reviews/m963_m962_m960_m935_c1_macro_aware_dc_source_hammer_r1_20260829"
HAMMER_REVIEW="${HAMMER_DIR}/review.json"
RELEASE="${HW_ROOT}/contracts/m964_m963_m962_m960_m935_c1_macro_aware_dc_launch_release_r1_20260829.json"
RTL="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m962_m935_three_stage_match_macro_aware_dc.f"
SDC="${HW_ROOT}/dc_handoff/constraints/date_m962_m935_three_stage_match_macro_aware_3ns.sdc"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m962_m935_three_stage_match_macro_aware_candidate.tcl"
DOC359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M959_DIR="${HW_ROOT}/reviews/m959_m955_m948_m935_c1_causal_dual_enqueue_vcs_result_hammer_r1_20260829"
M959_REVIEW="${M959_DIR}/review.json"
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
RESULT="${HW_ROOT}/dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m962_m935_three_stage_match_macro_aware_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m962_m935_three_stage_match_macro_aware_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m962_m935_three_stage_match_macro_aware_dc_launch_lock"
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
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nsetup_admitted=false\n' \
      "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM

# Source/tool/foundry identities are immutable.  No release can override them.
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
sha_exact c697628149446c66b9a14aab0b9aeeb69efee99b05ba8d1d12b92e3179a89114 "${M959_REVIEW}"
sha_exact c79fea228aa7b7bb1b44bc2f0a6007d57112d4c6459ac3765e661a925c34df43 "${M959_DIR}/SHA256SUMS"
sha_exact 255ea3dcc20828ad2bb9caa57ca7d4ca3c2cc34faba60f2bd18fcd0195c84ef4 "${M959_DIR}/SHA256SUMS.seal.sha256"
verify_dir_seal "${M959_DIR}"
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
python3 -I - "${CONTRACT}" "${RUNNER}" "${FILELIST}" "${SDC}" "${TCL}" \
  "${M959_REVIEW}" <<'PY'
import hashlib, json, sys
from pathlib import Path
contract,runner,filelist,sdc,tcl,m959=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(contract.read_text()); v=json.loads(m959.read_text())
assert d['status']=='STOP_M960_FUNCTIONAL_GATE_UNSATISFIED__SOURCE_ONLY__NO_DC_AUTHORIZED'
assert d['authorization']['dc_runs_now']==0 and d['authorization']['future_dc_runs_max']==1
assert d['identity']['runner_sha256']==sha(runner)
assert d['identity']['filelist_sha256']==sha(filelist)
assert d['identity']['sdc_sha256']==sha(sdc)
assert d['identity']['tcl_sha256']==sha(tcl)
assert v['claim_boundary']['negative_attack_assertion_expected'] is True
assert v['claim_boundary']['zero_assertion_failure_claim'] is False
assert d['m960_blocker']['m959_is_clean_gate'] is False
assert d['m960_blocker']['launch_now'] is False
assert d['claim_boundary']['timing_verified'] is False
PY

# Constraint hygiene is checked independently of the JSON declaration.
if rg -ni '\b(set_false_path|set_multicycle_path|set_max_delay|set_min_delay|set_disable_timing|set_case_analysis)\b' \
    "${SDC}" "${TCL}" >/dev/null; then
  echo "ERROR: forbidden timing exception found" >&2; exit 3
fi
[[ "$(grep -cvE '^[[:space:]]*(#|$)' "${FILELIST}")" -eq 2 ]] || exit 3
! rg -n '\.v($|[[:space:]])' "${FILELIST}" >/dev/null || exit 3

# The following fixed-path pair does not exist at M962 source authoring.  Its
# absence is the intentional hard STOP.  A future author may not bypass it by
# supplying arbitrary paths or merely asserting that M959 was expected.
verify_dir_seal "${HAMMER_DIR}"
verify_file_seal "${RELEASE}"
python3 -I - "${HAMMER_REVIEW}" "${RELEASE}" "${RUNNER}" "${CONTRACT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
hammer,release,runner,contract=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads(hammer.read_text()); r=json.loads(release.read_text())
assert h['status']=='PASS_M963_M962_SOURCE_HAMMER_AFTER_M960_REPAIR'
assert h['score']>=95 and h['p0_count']==0 and h['p1_count']==0
assert r['status']=='AUTHORIZE_ONE_M962_M935_MACRO_AWARE_SETUP_AREA_DC_ATTEMPT'
assert r['authorization']=={'dc_runs':1,'all_other_eda_runs':0}
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['hammer_review_sha256']==sha(hammer)
gate_path=Path(r['functional_gate']['path'])
if not gate_path.is_absolute(): gate_path=contract.parents[1]/gate_path
assert gate_path.is_file() and not gate_path.is_symlink()
assert sha(gate_path)==r['functional_gate']['sha256']
g=json.loads(gate_path.read_text())
kind=r['functional_gate']['kind']
assert kind in {'CLEAN_NEGATIVE_TEST_IDENTITY','SUPERSEDING_ADMISSION'}
if kind=='CLEAN_NEGATIVE_TEST_IDENTITY':
    assert g['claim_boundary']['fault_activation_verified'] is True
    assert g['claim_boundary']['zero_unexpected_assertion_failures'] is True
    assert g['claim_boundary']['attack_phase_target_sva_isolated'] is True
else:
    assert g['decision']['supersedes_m934_zero_assertion_admission'] is True
    assert g['decision']['accepts_exactly_one_expected_m923_attack_assertion'] is True
    assert g['decision']['authorizes_m962_source_hammer'] is True
PY

[[ -n "${M962_EXPECTED_DC_RUNNER_SHA256:-}" &&
    "$(sha_file "${RUNNER}")" == "${M962_EXPECTED_DC_RUNNER_SHA256}" ]] || {
  echo "ERROR: caller must pin the independently reviewed M962 runner SHA" >&2; exit 3; }
[[ -n "${M962_EXPECTED_DC_RELEASE_SHA256:-}" &&
    "$(sha_file "${RELEASE}")" == "${M962_EXPECTED_DC_RELEASE_SHA256}" ]] || {
  echo "ERROR: caller must pin the independently reviewed M964 release SHA" >&2; exit 3; }

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] || {
  echo "ERROR: M962 result identity already consumed or colliding" >&2; exit 4; }
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
  echo "ERROR: M962 memory/commit gate not met" >&2; exit 4; }
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f DC-Ultra-Opt >/dev/null

mkdir -- "${ATTEMPT}"
printf 'status=M962_ATTEMPT_CONSUMED\nmax_dc_runs=1\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -- "${WORK}"
WORK_ACTIVE=1
printf 'status=M962_DC_ATTEMPT_ADMITTED\nclock_period_ns=3.000\nmacro_count=9\nfalse_paths=0\n' \
  >"${WORK}/admission.txt"

set +e
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
  HW_ROOT="${HW_ROOT}" RTL_FILELIST="${FILELIST}" SDC_FILE="${SDC}" \
  OUTPUT_DIR="${WORK}" STD_SLOW_DB="${STD_SLOW}" STD_FAST_DB="${STD_FAST}" \
  MACRO_SLOW_DB="${MACRO_SLOW}" MACRO_FAST_DB="${MACRO_FAST}" \
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
  reports/link.rpt reports/macro_binding_audit.txt
  reports/check_design_precompile.rpt reports/check_design_postcompile.rpt
  reports/check_timing_precompile.rpt reports/check_timing_postcompile.rpt
  reports/resources_precompile.rpt reports/resources_postcompile.rpt
  reports/references_precompile.rpt reports/references_postcompile.rpt
  reports/hierarchy_postcompile.rpt reports/qor.rpt reports/area_hierarchy.rpt
  reports/clocks.rpt reports/timing_setup_top100.rpt
  reports/constraint_setup_all.rpt reports/constraint_max_capacitance.rpt
  reports/constraint_max_transition.rpt reports/constraint_max_fanout.rpt
  reports/flow_contract.rpt reports/precompile_loop_gate.rpt
  reports/setup_summary_machine.txt TCL_PASS_TERMINAL.txt
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf
)
for artifact in "${required[@]}"; do
  [[ -s "${WORK}/${artifact}" && ! -L "${WORK}/${artifact}" ]] || exit 6
done
grep -Fxq 'status=PASS_M962_RESOLVED_LIBRARY_MACRO_STRUCTURE' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_pre=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_post=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'TIM-209=0' "${WORK}/reports/precompile_loop_gate.rpt"
grep -Fxq 'OPT-150=0' "${WORK}/reports/precompile_loop_gate.rpt"
netlist="${WORK}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v"
[[ "$(rg -o 'TS1N28HPCPHVTB128X128M4S' "${netlist}" | wc -l)" -eq 9 ]] || exit 9

python3 -I - "${WORK}" "${RUNNER}" "${CONTRACT}" "${RELEASE}" <<'PY'
import hashlib,json,math,re,sys
from pathlib import Path
root,runner,contract,release=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
area_text=(root/'reports/area_hierarchy.rpt').read_text(errors='replace')
summary=dict(line.split('=',1) for line in
             (root/'reports/setup_summary_machine.txt').read_text().splitlines()
             if '=' in line)
m=re.search(r'Total cell area:\s*([0-9.]+)',area_text)
if not m: raise SystemExit('missing total cell area')
area=float(m.group(1)); wns=float(summary['setup_wns_ns']); tns=float(summary['setup_tns_ns'])
viol=int(summary['setup_violating_paths']); met=summary['status']=='MET'
if not all(math.isfinite(x) for x in (area,wns,tns)) or area<=0: raise SystemExit('invalid metrics')
if met != (viol==0 and wns>=0 and tns==0): raise SystemExit('setup summary inconsistency')
status=('PASS_RAW_M962_3NS_SETUP_AREA_CANDIDATE_PENDING_RESULT_HAMMER' if met else
        'SEALED_NEGATIVE_M962_3NS_SETUP_VIOLATION_PENDING_RESULT_HAMMER')
receipt={
 'schema':'m962_m935_three_stage_match_macro_aware_dc_receipt_v1',
 'status':status,'clock_period_ns':3.0,'ideal_clock':True,'wireload':'ZeroWireload',
 'macro_cell':'TS1N28HPCPHVTB128X128M4S','macro_count':9,
 'total_cell_area_um2_dc_reported':area,
 'setup':{'met':met,'wns_ns':wns,'tns_ns':tns,'violating_paths':viol,
          'top100_report_preserved':True},
 'identity':{'runner_sha256':sha(runner),'source_contract_sha256':sha(contract),
             'release_sha256':sha(release)},
 'claim_boundary':{'setup_area_component_candidate':True,'hold_signoff':False,
    'power':False,'energy':False,'speedup':False,'system_speedup':False,
    'paper_ppa_ready':False,'headline':False}}
(root/'m962_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
PY
python3 -m json.tool "${WORK}/m962_dc_receipt.json" >/dev/null
setup_status="$(jq -er '.setup.met' "${WORK}/m962_dc_receipt.json")"
if [[ "${setup_status}" == true ]]; then
  run_status=PASS_RAW_M962_3NS_SETUP_AREA_CANDIDATE_PENDING_RESULT_HAMMER
else
  run_status=SEALED_NEGATIVE_M962_3NS_SETUP_VIOLATION_PENDING_RESULT_HAMMER
fi
printf 'status=%s\nsetup_met=%s\nhold_signoff=false\npower=false\nspeedup=false\npaper_ppa_ready=false\n' \
  "${run_status}" "${setup_status}" >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM
rmdir -- "${LOCK}"
echo "M962 completed: ${RESULT} (${run_status})"
