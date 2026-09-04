#!/usr/bin/env bash
set -euo pipefail
umask 002

# Source-only until M1631 different-author review plus M1632 release exist and
# are caller-pinned.  A released identity permits exactly one DC process from
# the original M993 DDC.  It never consumes the failed M1614 mapped output.

[[ $# -eq 0 ]] || { echo "ERROR: M1630 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_candidate.tcl"
CONTRACT="${HW_ROOT}/contracts/m1630_m993_c1_residual_hold_guardband_dc_source_contract_r1_20260901.json"
HAMMER_DIR="${HW_ROOT}/reviews/m1631_m1630_c1_residual_hold_guardband_dc_source_hammer_r1_20260901"
HAMMER_REVIEW="${HAMMER_DIR}/review.json"
RELEASE="${HW_ROOT}/contracts/m1632_m1631_m1630_c1_residual_hold_guardband_dc_launch_release_r1_20260901.json"

M993_DIR="${HW_ROOT}/dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
M993_ORIGINAL="${M993_DIR}/original_quarantine"
M1006_DIR="${HW_ROOT}/reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1614_NEGATIVE="${HW_ROOT}/dc_handoff/runs/m1614_m993_c1_macro_aware_hold_only_incremental_dc_r1_20260901.failed_or_incomplete.4065447.quarantine"
INPUT_DDC="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc"
INPUT_SDC="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc"
INPUT_MAPPED_V="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v"
INPUT_SVF="${M993_ORIGINAL}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf"
RTL="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
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

RESULT="${HW_ROOT}/dc_handoff/runs/m1630_m993_c1_residual_hold_guardband_dc_r1_20260901"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1630_m993_c1_residual_hold_guardband_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1630_m993_c1_residual_hold_guardband_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1630_m993_c1_residual_hold_guardband_dc_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
fail() { echo "ERROR: M1630 $*" >&2; exit 3; }
sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/nonregular ${path}"
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || fail "SHA mismatch ${path}: ${got}"
}
sha_tool() {
  local expected="$1" path="$2" got
  [[ -f "${path}" ]] || fail "missing tool ${path}"
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || fail "tool SHA mismatch ${path}: ${got}"
}
verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}.sha256" && ! -L "${payload}.sha256"
      && -f "${payload}.sha256.seal.sha256"
      && ! -L "${payload}.sha256.seal.sha256" ]] || fail "file seal absent ${payload}"
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null &&
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) ||
    fail "file seal invalid ${payload}"
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS"
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "directory seal absent ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
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

# Immutable admitted input and negative motivation identities.
verify_dir_seal "${M993_DIR}"
verify_dir_seal "${M993_ORIGINAL}"
verify_dir_seal "${M1006_DIR}"
verify_dir_seal "${M1614_NEGATIVE}"
sha_exact 8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093 "${M993_DIR}/SHA256SUMS"
sha_exact 0cc3b953342d6f149183e5fdf55b97174f69f97701574b0a79f05a5068ff6689 "${M993_DIR}/SHA256SUMS.seal.sha256"
sha_exact 9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe "${M993_ORIGINAL}/SHA256SUMS"
sha_exact a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997 "${M993_ORIGINAL}/SHA256SUMS.seal.sha256"
sha_exact d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea "${M1006_DIR}/review.json"
sha_exact a550e8b25f735daf1a25a57679b6cdae2a427388bfa9851bd38359766fdf920f "${M1006_DIR}/SHA256SUMS"
sha_exact 4d599019ec7132d9208280bbb37a172dfc84291f3a55b8328ad04bc3219638a4 "${M1006_DIR}/SHA256SUMS.seal.sha256"
sha_exact cef7b0bb2cbcfbc0e723068e54018fbca5acf708f3cb0850e3d2a59677875d13 "${M1614_NEGATIVE}/SHA256SUMS"
sha_exact 374d45c922dac46e307f5ffa1220c11c4cb917ff2b6f8310c3c052c2ee4914cb "${M1614_NEGATIVE}/SHA256SUMS.seal.sha256"
sha_exact ff8f206815233c222c781a507d3ce504571148885ff7b113c5f94cb8824f639b "${M1614_NEGATIVE}/reports/setup_posthold_summary_machine.txt"
sha_exact fb12725e8bc76cce0c9f8198cb8b915dc91cf8c0fbb4e185b4fae2da2414da8c "${M1614_NEGATIVE}/reports/hold_posthold_summary_machine.txt"
sha_exact aa109ac641cbee88d6617e4b4f3008f6669a91d51e055bb151d7b3c324ec655e "${M1614_NEGATIVE}/reports/area_posthold.rpt"

sha_exact d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56 "${INPUT_DDC}"
sha_exact cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5 "${INPUT_SDC}"
sha_exact 9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf "${INPUT_MAPPED_V}"
sha_exact 8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7 "${INPUT_SVF}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
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
/usr/bin/python3.6 -I - "${CONTRACT}" "${RUNNER}" "${TCL}" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,runner,tcl=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text())
assert c['status']=='SOURCE_ONLY_M1630_C1_RESIDUAL_HOLD_GUARDBAND__NO_EDA_AUTHORIZED'
assert c['authorization']['dc_runs_now']==0
assert c['authorization']['future_dc_runs_max']==1
assert c['identity']['runner_sha256']==sha(runner)
assert c['identity']['tcl_sha256']==sha(tcl)
assert c['input_policy']['only_original_m993_ddc'] is True
assert c['input_policy']['failed_m1614_output_is_input'] is False
PY

# Static command/constraint hygiene before any future release gate.
/usr/bin/python3.6 -I - "${TCL}" "${INPUT_SDC}" "${RUNNER}" <<'PY'
import re,sys
from pathlib import Path
tcl,sdc,runner=(Path(x).read_text() for x in sys.argv[1:])
strip=lambda s:'\n'.join(x.split('#',1)[0] for x in s.splitlines())
t=strip(tcl); s=strip(sdc)
assert len(re.findall(r'(?m)^\s*read_ddc\b',t))==1
assert len(re.findall(r'(?m)^\s*set_fix_hold\b',t))==1
assert len(re.findall(r'(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$',t))==1
assert len(re.findall(r'(?m)^\s*compile\b',t))==1
assert not re.search(r'(?m)^\s*compile_ultra\b',t)
assert len(re.findall(r'(?m)^\s*set_clock_uncertainty\s+-hold\b',t))==2
assert 'set_clock_uncertainty -hold $optimization_hold_guardband_ns $core_clock' in t
assert 'set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock' in t
for term in ('set_false_path','set_multicycle_path','set_min_delay','set_max_delay',
             'set_disable_timing','set_case_analysis'):
    assert not re.search(r'(?m)^\s*'+term+r'\b',t+'\n'+s), term
assert len(re.findall(r'(?m)^\s*create_clock\b',s))==1
assert re.search(r'create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)',s)
assert re.search(r'set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b',s)
assert re.search(r'set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b',s)
assert 'get_object_name $design_collection' in t
assert '-no_home_init -no_local_init -no_gui -f "${TCL}"' in runner
assert not re.search(r'(?m)^\s*(?:export\s+)?HOME=',runner)
assert not re.search(r'(?m)^INPUT_DDC=.*m1614',runner)
assert 'm1614_hold_repaired.ddc' not in t
PY

# M1630 remains inert until a different author seals M1631 and M1632.
verify_dir_seal "${HAMMER_DIR}"
verify_file_seal "${RELEASE}"
/usr/bin/python3.6 -I - "${HAMMER_REVIEW}" "${RELEASE}" "${RUNNER}" "${CONTRACT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
hammer,release,runner,contract=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads(hammer.read_text()); r=json.loads(release.read_text())
assert h['status']=='PASS_M1631_M1630_C1_RESIDUAL_HOLD_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT'
assert h['score']>=95 and h['p0_count']==0 and h['p1_count']==0
assert r['status']=='AUTHORIZE_ONE_M1630_C1_RESIDUAL_HOLD_GUARDBAND_DC_ATTEMPT'
assert r['authorization']=={'dc_runs':1,'all_other_eda_runs':0}
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['hammer_review_sha256']==sha(hammer)
PY

[[ -n "${M1630_EXPECTED_DC_RUNNER_SHA256:-}" &&
    "$(sha_file "${RUNNER}")" == "${M1630_EXPECTED_DC_RUNNER_SHA256}" ]] ||
  fail "caller must pin reviewed M1630 runner SHA"
[[ -n "${M1630_EXPECTED_DC_RELEASE_SHA256:-}" &&
    "$(sha_file "${RELEASE}")" == "${M1630_EXPECTED_DC_RELEASE_SHA256}" ]] ||
  fail "caller must pin reviewed M1632 release SHA"

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] ||
  fail "result identity already consumed or colliding"
/usr/bin/python3.6 -I - <<'PY'
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
mkdir -- "${LOCK}" || fail "launch lock collision"
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
headroom=$((commit_limit-committed))
[[ "${mem_available}" -ge 100663296 && "${swap_free}" -ge 16777216
    && "${headroom}" -ge 67108864 ]] || fail "memory/commit gate not met"
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null

mkdir -- "${ATTEMPT}"
printf 'status=M1630_ATTEMPT_CONSUMED\nmax_dc_runs=1\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -- "${WORK}"
WORK_ACTIVE=1
printf 'status=M1630_DC_ATTEMPT_ADMITTED\nclock_period_ns=3.000\nsetup_uncertainty_ns=0.200\noptimization_hold_guardband_ns=0.051\nreported_hold_uncertainty_ns=0.050\nhold_only_passes=1\nmacro_count=9\nretry=false\n' \
  >"${WORK}/admission.txt"

set +e
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
  M1630_INPUT_DDC="${INPUT_DDC}" M1630_INPUT_SDC="${INPUT_SDC}" \
  M1630_INPUT_MAPPED_V="${INPUT_MAPPED_V}" M1630_INPUT_SVF="${INPUT_SVF}" \
  M1630_STD_SLOW_DB="${STD_SLOW}" M1630_STD_FAST_DB="${STD_FAST}" \
  M1630_MACRO_SLOW_DB="${MACRO_SLOW}" M1630_MACRO_FAST_DB="${MACRO_FAST}" \
  M1630_OUTPUT_DIR="${WORK}" \
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
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc
  netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.svf
)
for artifact in "${required[@]}"; do
  [[ -s "${WORK}/${artifact}" && ! -L "${WORK}/${artifact}" ]] ||
    fail "missing result artifact ${artifact}"
done

# Any tool/link/loop evidence is fatal even when dc_shell returned zero.
if grep -Eiq '(^|[[:space:]])(Error|Fatal):|LINK-[0-9]+|link([^[:alnum:]]|.*)(fail|error|unresolved)|unresolved (reference|design|cell)|unable to resolve|combinational[ _-]*loop|timing[ _-]*loop|loop[ _-]*(detected|breaking|broken)|\((TIM-209|OPT-150)\)' \
    "${WORK}/dc.log" "${WORK}/reports/link.rpt" \
    "${WORK}/reports/check_design_prehold.rpt" \
    "${WORK}/reports/check_timing_prehold.rpt" \
    "${WORK}/reports/check_design_posthold.rpt" \
    "${WORK}/reports/check_timing_posthold.rpt"; then
  fail "Error/Fatal/link/loop evidence found"
fi

grep -Fxq 'status=PASS_M1630_RESOLVED_LIBRARY_MACRO_STRUCTURE' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_pre=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_post=9' "${WORK}/reports/macro_binding_audit.txt"
grep -Fxq 'input_generation=original_m993_m1006_admitted_ddc' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'failed_m1614_output_used=false' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'optimization_hold_guardband_ns=0.051' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'reported_hold_uncertainty_ns=0.050' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'hold_only_incremental_mapping_count=1' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'all_compile_command_count=1' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'compile_ultra_incremental_count=0' "${WORK}/reports/flow_contract.rpt"
grep -Fxq 'generic_incremental_mapping_count=0' "${WORK}/reports/flow_contract.rpt"

output_v="${WORK}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v"
output_sdc="${WORK}/netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc"
[[ "$(grep -o 'TS1N28HPCPHVTB128X128M4S' "${output_v}" | wc -l)" -eq 9 ]] ||
  fail "output macro count not nine"

/usr/bin/python3.6 -I - "${WORK}" "${RUNNER}" "${CONTRACT}" "${RELEASE}" \
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
    if not all(math.isfinite(out[x]) for x in ('wns_ns','tns_ns')):
        raise SystemExit('nonfinite timing')
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
sdc=(root/'netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc').read_text(errors='replace')
for term in ('set_false_path','set_multicycle_path','set_min_delay','set_max_delay','set_disable_timing','set_case_analysis'):
    if re.search(r'(?m)^\s*'+term+r'\b',sdc): raise SystemExit('forbidden output SDC '+term)
if len(re.findall(r'(?m)^\s*create_clock\b',sdc))!=1: raise SystemExit('clock population')
if not re.search(r'create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)',sdc): raise SystemExit('period drift')
if not re.search(r'set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b',sdc): raise SystemExit('setup uncertainty drift')
if not re.search(r'set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b',sdc): raise SystemExit('reported hold uncertainty drift')
if re.search(r'set_clock_uncertainty\s+-hold\s+0?\.051(?:0+)?\b',sdc): raise SystemExit('optimization guardband leaked into output SDC')
macro=kv(root/'reports/macro_binding_audit.txt')
macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)
timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')
area_ok=area<=ceiling
positive=timing_ok and area_ok and macro_ok and drc_count==0
status=('PASS_RAW_M1630_C1_RESIDUAL_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_RESULT_HAMMER'
        if positive else 'SEALED_NEGATIVE_M1630_C1_RESIDUAL_HOLD_OR_AREA_GATE_FAILED__NO_RETRY')
receipt={
 'schema':'m1630_m993_c1_residual_hold_guardband_dc_receipt_v1','status':status,
 'positive_dc_candidate':positive,'clock_period_ns':3.0,'setup_uncertainty_ns':0.2,
 'optimization_hold_guardband_ns':0.051,'reported_hold_uncertainty_ns':0.05,
 'ideal_clock':True,'wireload':'ZeroWireload',
 'pre':{'setup':pre_setup,'hold':pre_hold},'post':{'setup':post_setup,'hold':post_hold},
 'area':{'baseline_um2':baseline,'post_um2':area,'overhead_percent':overhead,
         'ceiling_um2':ceiling,'within_5_percent':area_ok},
 'macros':{'cell':'TS1N28HPCPHVTB128X128M4S','pre':9,'post':9,'passed':macro_ok},
 'design_rule_violating_nets':drc_count,
 'flow':{'input_generation':'original_m993_m1006_admitted_ddc',
         'failed_m1614_output_used':False,'set_fix_hold_count':1,
         'hold_only_incremental_mapping_count':1,'all_compile_command_count':1,
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
(root/'m1630_dc_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
(root/'RUN_COMPLETE.txt').write_text(
    'status='+status+'\nretry=false\nformality=false\nindependent_pt=false\npaper_citable=false\n')
PY
/usr/bin/python3.6 -m json.tool "${WORK}/m1630_dc_receipt.json" >/dev/null

run_status="$(/usr/bin/python3.6 -I - "${WORK}/m1630_dc_receipt.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)"
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM
rmdir -- "${LOCK}"
echo "M1630 completed: ${RESULT} (${run_status})"
