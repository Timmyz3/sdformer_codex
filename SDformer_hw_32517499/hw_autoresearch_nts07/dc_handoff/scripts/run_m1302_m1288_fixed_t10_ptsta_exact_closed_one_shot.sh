#!/usr/bin/env bash
set -euo pipefail

# M1302 is an additive exact-closed launch/adjudication wrapper around the
# frozen M1288 source DAG.  It does not alter M1288, its Tcl, netlist, SDC, or
# timing reports.  The legacy M1289 admission pathname is retained because it
# is part of the exact M1288 source identity reviewed by M1299.
m1302_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m1302_hw_root="$(cd "${m1302_dc_root}/.." && pwd)"
m1302_wrapper="$(realpath "${BASH_SOURCE[0]}")"
m1302_admission=contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json
m1302_contract=contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json
m1302_m1288_runner=dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh
m1302_m1288_contract=contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json
m1302_m1299=reviews/m1299_m1288_c3_m917_fixed_t10_ptsta_receipt_blind_hammer_r1_20260830
m1302_m917=dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829
m1302_m928=reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829
m1302_m1285=reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830
m1302_netlist="${m1302_m917}/fixed/netlist/m518_matched_fixed_t10_atlif_mapped.v"
m1302_sdc="${m1302_m917}/fixed/netlist/m518_matched_fixed_t10_atlif_mapped.sdc"
m1302_pt=/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell
m1302_lmutil=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
m1302_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m1302_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m1302_m1288_canonical="${m1302_dc_root}/runs/m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830"
m1302_m1288_work="${m1302_m1288_canonical}.work"
m1302_m1288_attempt="${m1302_dc_root}/runs/.m1288_m917_fixed_t10_ptsta_attempt_consumed"
m1302_canonical="${m1302_dc_root}/runs/m1302_m1288_fixed_t10_ptsta_adjudication_r1_20260830"
m1302_work="${m1302_canonical}.work"
m1302_attempt="${m1302_dc_root}/runs/.m1302_m1288_fixed_t10_ptsta_attempt_consumed"
m1302_uid="$(id -u)"
m1302_attempted=0
m1302_done=0
m1302_tmp=

m1302_sha() { sha256sum "$1" | awk '{print $1}'; }
m1302_expect() { [[ -f "$1" && ! -L "$1" && "$(m1302_sha "$1")" == "$2" ]]; }
m1302_sealed_payload_ok() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    [[ -f "${payload}" && ! -L "${payload}" && \
       -f "${payload}.sha256" && ! -L "${payload}.sha256" && \
       -f "${payload}.sha256.seal.sha256" && ! -L "${payload}.sha256.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}
m1302_double_seal_ok() {
    local dir=$1
    [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" && \
       -f "${dir}/SHA256SUMS.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
m1302_seal_dir() {
    local dir=$1
    (cd "${dir}" && \
      find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS && \
      sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
      sha256sum -c SHA256SUMS >/dev/null && \
      sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

m1302_collisions() {
    python3 - "${m1302_uid}" <<'PY'
import os,sys
uid=int(sys.argv[1])
eda={"pt_shell","dc_shell","dc_shell-t","fm_shell","vcs","vcs1","vlogan",
     "simv","common_shell_exec","common_shell_exe"}
hits=[]
for name in os.listdir('/proc'):
    if not name.isdigit(): continue
    try:
        pid=int(name); status=open('/proc/%d/status'%pid).read().splitlines()
        puid=int(next(x for x in status if x.startswith('Uid:')).split()[1])
        stat=open('/proc/%d/stat'%pid).read(); rest=stat[stat.rfind(')')+2:].split()
        state,start=rest[0],int(rest[19])
        comm=open('/proc/%d/comm'%pid).read().strip()
        exe=os.path.basename(os.path.realpath('/proc/%d/exe'%pid))
    except (OSError,StopIteration,ValueError):
        continue
    if puid==uid and state!='Z' and (comm in eda or exe in eda):
        hits.append('%d:%d:%s:%s'%(pid,start,comm,exe))
print(','.join(sorted(hits)))
PY
}

m1302_fail() {
    local rc=$?
    [[ -z "${m1302_tmp}" ]] || rm -rf -- "${m1302_tmp}" 2>/dev/null || true
    if [[ "${m1302_attempted}" -eq 1 && "${m1302_done}" -ne 1 ]]; then
        if [[ ! -e "${m1302_work}" ]]; then mkdir -p "${m1302_work}"; fi
        printf '{\n  "schema": "m1302_m1288_fixed_t10_ptsta_adjudication_receipt_v1",\n  "status": "STOP_M1302_LAUNCH_OR_ADJUDICATION_FAILED",\n  "runner_exit_code": %s,\n  "claim_boundary": {"timing_gate_pass": false, "power": false, "energy": false, "speedup": false, "system": false, "paper_ppa_ready": false, "headline": false}\n}\n' "${rc}" \
            >"${m1302_work}/m1302_adjudication_receipt_r1.json"
        printf 'STOP_M1302_LAUNCH_OR_ADJUDICATION_FAILED\n' >"${m1302_work}/RUN_COMPLETE.txt"
        m1302_seal_dir "${m1302_work}" || true
        mv -T "${m1302_work}" "${m1302_canonical}.failed_or_incomplete.$$.quarantine" || true
    fi
    exit "${rc}"
}
trap m1302_fail EXIT INT TERM

[[ -n "${M1302_EXPECTED_WRAPPER_SHA256:-}" && \
   "$(m1302_sha "${m1302_wrapper}")" == "${M1302_EXPECTED_WRAPPER_SHA256}" ]] || exit 3
[[ -n "${M1302_EXPECTED_ADMISSION_SHA256:-}" ]] || exit 3
cd "${m1302_hw_root}"
m1302_expect "${m1302_admission}" "${M1302_EXPECTED_ADMISSION_SHA256}" || exit 3
m1302_sealed_payload_ok "${m1302_admission}" || exit 3

# The admission parser is exact-closed at every authority-bearing object.  It
# intentionally validates the legacy M1288 field names required by the frozen
# runner as well as the additive M1302 wrapper/adjudication fields.
python3 - "${m1302_admission}" "${M1302_EXPECTED_WRAPPER_SHA256}" <<'PY'
import json,re,sys
p=sys.argv[1]; wrapper_sha=sys.argv[2]
d=json.load(open(p))
def exact(obj,keys,name):
    if type(obj) is not dict or set(obj)!=set(keys): raise SystemExit(name+' keyset')
def b(obj,key,value):
    if type(obj[key]) is not bool or obj[key] is not value: raise SystemExit('bool '+key)
exact(d,('schema','date','milestone','status','objective','identity','exact_files',
         'tool','preflight','authorization','result_adjudication','claim_boundary'),'top')
if d['schema']!='m1302_m1288_c3_fixed_t10_ptsta_exact_closed_launch_admission_v1': raise SystemExit('schema')
if d['status']!='AUTHORIZED_ONE_M1288_M917_FIXED_T10_PTSTA_ATTEMPT': raise SystemExit('status')
exact(d['identity'],('admission_path','wrapper_path','wrapper_sha256','runner_path',
      'runner_sha256','contract_path','contract_sha256','m1302_contract_path',
      'm1302_contract_sha256','m1299_review_dir','m1299_outer_seal_sha256',
      'mapped_netlist_path','mapped_sdc_path','canonical_result','attempt',
      'adjudication_result','adjudication_attempt'),'identity')
if d['identity']['wrapper_sha256']!=wrapper_sha: raise SystemExit('wrapper SHA')
if d['identity']['admission_path']!='contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json': raise SystemExit('admission path')
if d['identity']['wrapper_path']!='dc_handoff/scripts/run_m1302_m1288_fixed_t10_ptsta_exact_closed_one_shot.sh': raise SystemExit('wrapper path')
if d['identity']['runner_path']!='dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh': raise SystemExit('runner path')
exact(d['authorization'],('launch_now','launch_after_independent_hammer',
      'max_attempts','run_pt','run_dc','run_vcs','run_formality','run_ptpx',
      'run_remote','query_license','result_adjudication'),'authorization')
for k,v in {'launch_now':False,'launch_after_independent_hammer':True,
            'run_pt':True,'run_dc':False,'run_vcs':False,'run_formality':False,
            'run_ptpx':False,'run_remote':False,'query_license':True,
            'result_adjudication':True}.items(): b(d['authorization'],k,v)
if type(d['authorization']['max_attempts']) is not int or d['authorization']['max_attempts']!=1: raise SystemExit('attempts')
exact(d['claim_boundary'],('launch_admission_only','pt_executed','setup_completed',
      'hold_closed','coverage_closed','unconstrained_paths_zero','automatic_hold_fix',
      'power','energy','speedup','system','paper_ppa_ready','headline'),'claims')
for k in d['claim_boundary']:
    b(d['claim_boundary'],k,k=='launch_admission_only')
exact(d['result_adjudication'],('setup_state','setup_slack_min_ns','hold_state',
      'hold_slack_min_ns','constraint_violated_paths','unconstrained_paths',
      'required_coverage_rows','coverage_rule','negative_terminal','pass_terminal',
      'fresh_result_hammer_required'),'result')
if d['result_adjudication']['setup_state']!='MET' or d['result_adjudication']['hold_state']!='MET': raise SystemExit('state')
if d['result_adjudication']['setup_slack_min_ns']!=0.0 or d['result_adjudication']['hold_slack_min_ns']!=0.0: raise SystemExit('slack')
if d['result_adjudication']['constraint_violated_paths']!=0 or d['result_adjudication']['unconstrained_paths']!=0: raise SystemExit('path counts')
if d['result_adjudication']['required_coverage_rows']!=['setup','hold','out_setup','out_hold']: raise SystemExit('coverage rows')
if d['result_adjudication']['coverage_rule']!='each total>0 and met==total and violated==0 and untested==0': raise SystemExit('coverage rule')
if type(d['result_adjudication']['fresh_result_hammer_required']) is not bool or not d['result_adjudication']['fresh_result_hammer_required']: raise SystemExit('result hammer')
exact(d['preflight'],('same_uid_collision_gate','resource_gate','license_gate',
      'fresh_namespace_gate','order'),'preflight')
if d['preflight']['same_uid_collision_gate'] is not True: raise SystemExit('same uid gate')
if d['preflight']['fresh_namespace_gate']!=['M1288 canonical','M1288 work','M1288 attempt','M1302 canonical','M1302 work','M1302 attempt']: raise SystemExit('fresh names')
if d['preflight']['order']!=['exact admission and seals','same-UID collision','resource','license','repeat collision and freshness','consume M1302 attempt','invoke M1288']: raise SystemExit('order')
exact(d['preflight']['resource_gate'],('mem_available_min_kib','commit_headroom_min_kib',
      'filesystem_available_min_kib'),'resource')
if d['preflight']['resource_gate']!={'mem_available_min_kib':8388608,
      'commit_headroom_min_kib':8388608,'filesystem_available_min_kib':4194304}: raise SystemExit('resource values')
exact(d['preflight']['license_gate'],('feature','server','query_before_attempt',
      'issued_gt_in_use_required'),'license')
if d['preflight']['license_gate']!={'feature':'PrimeTime','server':'27030@ic.ismd-nemo',
      'query_before_attempt':True,'issued_gt_in_use_required':True}: raise SystemExit('license values')
expected_files={
 'dc_handoff/scripts/run_m1302_m1288_fixed_t10_ptsta_exact_closed_one_shot.sh',
 'tests/test_m1302_m1288_fixed_t10_ptsta_launch_source_static.py',
 'dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh',
 'dc_handoff/scripts/run_ptsta_m1288_m917_fixed_t10_slowmax_fastmin_inert.tcl',
 'contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json',
 'contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json.sha256',
 'contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json.sha256.seal.sha256',
 'contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json',
 'contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json.sha256',
 'contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json.sha256.seal.sha256',
 'reviews/m1299_m1288_c3_m917_fixed_t10_ptsta_receipt_blind_hammer_r1_20260830/SHA256SUMS.seal.sha256',
 'dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829/SHA256SUMS.seal.sha256',
 'reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829/SHA256SUMS.seal.sha256',
 'reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830/SHA256SUMS.seal.sha256',
 'dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829/fixed/netlist/m518_matched_fixed_t10_atlif_mapped.v',
 'dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829/fixed/netlist/m518_matched_fixed_t10_atlif_mapped.sdc',
 'docs/359_DATE终局冻结_20260813.md'}
if type(d['exact_files']) is not dict or set(d['exact_files'])!=expected_files: raise SystemExit('exact files')
for k,v in d['exact_files'].items():
    if type(k) is not str or re.match(r'^[0-9a-f]{64}$',v or '') is None: raise SystemExit('file SHA')
exact(d['tool'],('pt_shell_path','pt_shell_sha256','lmutil_path','lmutil_sha256',
      'python3_path','python3_sha256','bash_path','bash_sha256','setsid_path',
      'setsid_sha256','slow_db_path','slow_db_sha256','fast_db_path','fast_db_sha256'),'tool')
if d['tool']['pt_shell_path']!='/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell': raise SystemExit('pt path')
if d['tool']['lmutil_path']!='/opt/synopsys/scl/2025.03/linux64/bin/lmutil': raise SystemExit('lmutil path')
if d['tool']['python3_path']!='/usr/bin/python3' or d['tool']['bash_path']!='/usr/bin/bash' or d['tool']['setsid_path']!='/usr/bin/setsid': raise SystemExit('base tool path')
PY

m1302_expect "${m1302_m1288_runner}" "$(jq -er '.identity.runner_sha256' "${m1302_admission}")" || exit 3
m1302_expect "${m1302_m1288_contract}" "$(jq -er '.identity.contract_sha256' "${m1302_admission}")" || exit 3
m1302_expect "${m1302_contract}" "$(jq -er '.identity.m1302_contract_sha256' "${m1302_admission}")" || exit 3
m1302_sealed_payload_ok "${m1302_m1288_contract}" || exit 3
m1302_sealed_payload_ok "${m1302_contract}" || exit 3
while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m1302_expect "${path}" "${expected}" || exit 3
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "${m1302_admission}")
for dir in "${m1302_m1299}" "${m1302_m917}" "${m1302_m928}" "${m1302_m1285}"; do
    m1302_double_seal_ok "${dir}" || exit 3
done
m1302_expect "${m1302_m1299}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.m1299_outer_seal_sha256' "${m1302_admission}")" || exit 3
m1302_expect "${m1302_pt}" "$(jq -er '.tool.pt_shell_sha256' "${m1302_admission}")" || exit 3
m1302_expect "${m1302_lmutil}" "$(jq -er '.tool.lmutil_sha256' "${m1302_admission}")" || exit 3
m1302_expect /usr/bin/python3 "$(jq -er '.tool.python3_sha256' "${m1302_admission}")" || exit 3
m1302_expect /usr/bin/bash "$(jq -er '.tool.bash_sha256' "${m1302_admission}")" || exit 3
m1302_expect /usr/bin/setsid "$(jq -er '.tool.setsid_sha256' "${m1302_admission}")" || exit 3
m1302_expect "${m1302_slow}" "$(jq -er '.tool.slow_db_sha256' "${m1302_admission}")" || exit 3
m1302_expect "${m1302_fast}" "$(jq -er '.tool.fast_db_sha256' "${m1302_admission}")" || exit 3

[[ "${PATH:-}" == /usr/bin:/bin && "${LANG:-}" == C.UTF-8 && \
   "${LC_ALL:-}" == C.UTF-8 && -z "${HOME:-}" ]] || exit 3
[[ "${SNPSLMD_LICENSE_FILE:-}" == 27030@ic.ismd-nemo && \
   "${LM_LICENSE_FILE:-}" == /opt/synopsys/Synopsys.dat ]] || exit 3
[[ -z "${M1288_RUN_DIR:-}" && ! -e "${m1302_m1288_canonical}" && \
   ! -e "${m1302_m1288_work}" && ! -e "${m1302_m1288_attempt}" && \
   ! -e "${m1302_canonical}" && ! -e "${m1302_work}" && ! -e "${m1302_attempt}" ]] || exit 5
[[ -z "$(m1302_collisions)" ]] || exit 4

# All live resource and license probes remain before either one-shot attempt.
m1302_mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
m1302_commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
m1302_committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
m1302_commit_headroom=$((m1302_commit_limit-m1302_committed))
m1302_disk_available="$(df -Pk "${m1302_dc_root}" | awk 'NR==2 {print $4}')"
[[ "${m1302_mem_available}" -ge 8388608 && \
   "${m1302_commit_headroom}" -ge 8388608 && \
   "${m1302_disk_available}" -ge 4194304 ]] || exit 6

m1302_tmp="$(mktemp -d /tmp/m1302_pt_license.XXXXXX)"
chmod 0700 "${m1302_tmp}"
if ! "${m1302_lmutil}" lmstat -c "${SNPSLMD_LICENSE_FILE}" -f PrimeTime \
    >"${m1302_tmp}/lmstat.txt" 2>&1; then exit 7; fi
read -r m1302_issued m1302_in_use < <(python3 - "${m1302_tmp}/lmstat.txt" <<'PY'
import re,sys
t=open(sys.argv[1],errors='replace').read()
m=re.search(r'Users of PrimeTime:.*?Total of\s+(\d+)\s+licenses? issued;\s+Total of\s+(\d+)\s+licenses? in use',t,re.S)
if not m: raise SystemExit(1)
print(m.group(1),m.group(2))
PY
)
[[ "${m1302_issued}" -gt "${m1302_in_use}" ]] || exit 7
rm -rf -- "${m1302_tmp}"; m1302_tmp=
[[ -z "$(m1302_collisions)" ]] || exit 4
[[ ! -e "${m1302_m1288_canonical}" && ! -e "${m1302_m1288_work}" && \
   ! -e "${m1302_m1288_attempt}" && ! -e "${m1302_canonical}" && \
   ! -e "${m1302_work}" && ! -e "${m1302_attempt}" ]] || exit 5

mkdir "${m1302_attempt}"
m1302_attempted=1
printf 'status=M1302_ONE_SHOT_ATTEMPT_CONSUMED\n' >"${m1302_attempt}/attempt.txt"
mkdir -p "${m1302_work}"
cp "${m1302_admission}" "${m1302_work}/launch_admission.json"
cp "${m1302_contract}" "${m1302_work}/source_contract.json"
printf '{"mem_available_kib":%s,"commit_headroom_kib":%s,"filesystem_available_kib":%s,"license_feature":"PrimeTime","licenses_issued":%s,"licenses_in_use":%s,"same_uid_collision_count":0}\n' \
    "${m1302_mem_available}" "${m1302_commit_headroom}" "${m1302_disk_available}" \
    "${m1302_issued}" "${m1302_in_use}" >"${m1302_work}/preflight_summary.json"
sha256sum "${m1302_wrapper}" "${m1302_admission}" "${m1302_contract}" \
    "${m1302_m1288_runner}" "${m1302_m1288_contract}" "${m1302_netlist}" \
    "${m1302_sdc}" "${m1302_m1299}/SHA256SUMS.seal.sha256" \
    "${m1302_m917}/SHA256SUMS.seal.sha256" "${m1302_m928}/SHA256SUMS.seal.sha256" \
    "${m1302_m1285}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md >"${m1302_work}/input_sha256.txt"

env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
    LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat \
    M1288_EXPECTED_RUNNER_SHA256="$(jq -er '.identity.runner_sha256' "${m1302_admission}")" \
    M1288_EXPECTED_ADMISSION_SHA256="${M1302_EXPECTED_ADMISSION_SHA256}" \
    /usr/bin/bash "${m1302_m1288_runner}"

m1302_double_seal_ok "${m1302_m1288_canonical}" || exit 30
cp "${m1302_m1288_canonical}/SHA256SUMS" "${m1302_work}/m1288_result_SHA256SUMS"
cp "${m1302_m1288_canonical}/SHA256SUMS.seal.sha256" "${m1302_work}/m1288_result_outer_seal.sha256"

# This adjudication is stricter than M1288's source receipt.  It requires zero
# unconstrained-path diagnostics and complete setup/hold/out coverage, so a
# report-complete or merely nonnegative-slack run cannot be promoted alone.
python3 - "${m1302_m1288_canonical}" "${m1302_work}" <<'PY'
import json,re,sys
from pathlib import Path
src=Path(sys.argv[1]); out=Path(sys.argv[2]); reports=src/'reports'
def text(name): return (reports/name).read_text(encoding='utf-8',errors='replace')
def slack(name):
    m=re.search(r'slack \((MET|VIOLATED)\)\s+(-?\d+(?:\.\d+)?)',text(name))
    if not m: raise RuntimeError('missing slack '+name)
    return m.group(1),float(m.group(2))
def coverage_row(name,cov):
    m=re.search(r'^'+re.escape(name)+r'\s+(\d+)\s+(\d+) \([^\n]+?\)\s+(\d+) \([^\n]+?\)\s+(\d+) \(',cov,re.M)
    if not m: raise RuntimeError('missing coverage '+name)
    return tuple(map(int,m.groups()))
setup_state,setup=slack('timing_setup_slow.rpt')
hold_state,hold=slack('timing_hold_fast.rpt')
cov_text=text('analysis_coverage.rpt')
coverage={n:coverage_row(n,cov_text) for n in ('setup','hold','out_setup','out_hold')}
coverage_gate=all(total>0 and met==total and violated==0 and untested==0
                  for total,met,violated,untested in coverage.values())
check=text('check_timing.rpt')
counts=[]
for pat in (r'There (?:is|are)\s+(\d+)\s+input ports?.{0,240}?will be unconstrained',
            r'There (?:is|are)\s+(\d+)\s+endpoints?.{0,240}?unconstrained'):
    counts += [int(x) for x in re.findall(pat,check,re.I|re.S)]
unconstrained=sum(counts)
constraint_violations=text('constraint_violators.rpt').count('slack (VIOLATED)')
m1288=json.loads((src/'m1288_m917_fixed_t10_prelayout_ptsta_receipt_r1.json').read_text())
gate=(setup_state=='MET' and setup>=0.0 and hold_state=='MET' and hold>=0.0
      and constraint_violations==0 and unconstrained==0 and coverage_gate)
status=('PASS_M1302_M1288_FIXED_T10_PRELAYOUT_PTSTA_STRICT_TIMING_GATE'
        if gate else 'STOP_M1302_M1288_FIXED_T10_PRELAYOUT_PTSTA_STRICT_TIMING_GATE')
receipt={
 'schema':'m1302_m1288_fixed_t10_ptsta_adjudication_receipt_v1','status':status,
 'm1288_status':m1288.get('status'),
 'setup':{'state':setup_state,'worst_slack_ns':setup,'closed':setup_state=='MET' and setup>=0.0},
 'hold':{'state':hold_state,'worst_slack_ns':hold,'closed':hold_state=='MET' and hold>=0.0,
         'automatic_fix_performed':False},
 'constraint_violated_paths':constraint_violations,
 'unconstrained_paths':unconstrained,
 'analysis_coverage':{n:{'total':r[0],'met':r[1],'violated':r[2],'untested':r[3]}
                      for n,r in coverage.items()},
 'coverage_gate_pass':coverage_gate,'strict_timing_gate_pass':gate,
 'scope':{'single_fixed_t10_component':True,'prelayout':True,'spef':False,
          'ideal_clock':True,'zero_wireload':True,'macro_count':0,
          'mapped_identity_mutated':False},
 'claim_boundary':{'fresh_result_hammer_required':True,'power':False,'energy':False,
                   'speedup':False,'system':False,'paper_ppa_ready':False,'headline':False}}
(out/'m1302_adjudication_receipt_r1.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
(out/'RUN_COMPLETE.txt').write_text(status+'\n')
(out/'GATE_EXIT_CODE.txt').write_text(('0' if gate else '10')+'\n')
PY

[[ "$(m1302_sha docs/359_DATE终局冻结_20260813.md)" == \
   "$(jq -er '.exact_files["docs/359_DATE终局冻结_20260813.md"]' "${m1302_admission}")" ]] || exit 31
m1302_seal_dir "${m1302_attempt}"
m1302_seal_dir "${m1302_work}"
mv -T "${m1302_work}" "${m1302_canonical}"
m1302_done=1
trap - EXIT INT TERM
m1302_gate_rc="$(tr -d '\n' <"${m1302_canonical}/GATE_EXIT_CODE.txt")"
exit "${m1302_gate_rc}"
