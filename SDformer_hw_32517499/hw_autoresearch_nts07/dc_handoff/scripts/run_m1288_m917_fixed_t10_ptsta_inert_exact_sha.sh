#!/usr/bin/env bash
set -euo pipefail

# Source-only until a separately sealed M1289 launch admission exists.  This
# runner reports M917 setup/hold; it never changes the netlist or fixes hold.
m1288_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m1288_hw_root="$(cd "${m1288_dc_root}/.." && pwd)"
m1288_runner="$(realpath "${BASH_SOURCE[0]}")"
m1288_pt=/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell
m1288_setsid=/usr/bin/setsid
m1288_bash=/usr/bin/bash
m1288_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m1288_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m1288_m917=dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829
m1288_m928=reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829
m1288_m1285=reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830
m1288_netlist="${m1288_m917}/fixed/netlist/m518_matched_fixed_t10_atlif_mapped.v"
m1288_sdc="${m1288_m917}/fixed/netlist/m518_matched_fixed_t10_atlif_mapped.sdc"
m1288_tcl=dc_handoff/scripts/run_ptsta_m1288_m917_fixed_t10_slowmax_fastmin_inert.tcl
m1288_contract=contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json
m1288_admission=contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json
m1288_canonical="${m1288_dc_root}/runs/m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830"
m1288_work="${m1288_canonical}.work"
m1288_attempt="${m1288_dc_root}/runs/.m1288_m917_fixed_t10_ptsta_attempt_consumed"
m1288_uid="$(id -u)"

m1288_sha() { sha256sum "$1" | awk '{print $1}'; }
m1288_expect() { [[ -f "$1" && ! -L "$1" && "$(m1288_sha "$1")" == "$2" ]]; }
m1288_double_seal_ok() {
    local dir=$1
    [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" && \
       -f "${dir}/SHA256SUMS.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
m1288_sealed_payload_ok() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    [[ -f "${payload}" && ! -L "${payload}" && \
       -f "${payload}.sha256" && ! -L "${payload}.sha256" && \
       -f "${payload}.sha256.seal.sha256" && ! -L "${payload}.sha256.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}
m1288_seal_dir() {
    local dir=$1
    (cd "${dir}" && \
      find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS && \
      sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
      sha256sum -c SHA256SUMS >/dev/null && \
      sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

# Print same-UID EDA processes outside an optional exact isolated PT job.
# Exclusion is by PID/starttime/UID plus pgrp+session, never textual argv.
m1288_collisions() {
    local root_pid=${1:-0} root_start=${2:-0} root_pgrp=${3:-0} root_session=${4:-0}
    python3 - "${m1288_uid}" "${root_pid}" "${root_start}" \
        "${root_pgrp}" "${root_session}" <<'PY'
import os, sys
uid, root_pid, root_start, root_pgrp, root_session = map(int, sys.argv[1:])
eda = {"pt_shell", "dc_shell", "dc_shell-t", "fm_shell", "vcs", "vcs1",
       "vlogan", "simv", "common_shell_exec", "common_shell_exe"}
hits=[]
for name in os.listdir('/proc'):
    if not name.isdigit(): continue
    pid=int(name)
    try:
        status=open(f'/proc/{pid}/status').read().splitlines()
        puid=int(next(x for x in status if x.startswith('Uid:')).split()[1])
        stat=open(f'/proc/{pid}/stat').read(); rest=stat[stat.rfind(')')+2:].split()
        state, pgrp, session, start = rest[0], int(rest[2]), int(rest[3]), int(rest[19])
        comm=open(f'/proc/{pid}/comm').read().strip()
        exe=os.path.basename(os.path.realpath(f'/proc/{pid}/exe'))
    except (OSError, StopIteration, ValueError):
        continue
    if puid != uid or state == 'Z' or (comm not in eda and exe not in eda): continue
    own=(root_pid and pgrp==root_pgrp and session==root_session and start>=root_start)
    if not own: hits.append(f'{pid}:{start}:{comm}:{exe}:{pgrp}:{session}')
print(','.join(sorted(hits)))
PY
}
m1288_job_members() {
    local pgrp=$1 session=$2 min_start=$3
    python3 - "${m1288_uid}" "${pgrp}" "${session}" "${min_start}" <<'PY'
import os, sys
uid,pgrp,session,min_start=map(int,sys.argv[1:])
out=[]
for name in os.listdir('/proc'):
    if not name.isdigit(): continue
    try:
        pid=int(name); st=open(f'/proc/{pid}/status').read().splitlines()
        puid=int(next(x for x in st if x.startswith('Uid:')).split()[1])
        raw=open(f'/proc/{pid}/stat').read(); r=raw[raw.rfind(')')+2:].split()
        state,pg,se,start=r[0],int(r[2]),int(r[3]),int(r[19])
    except (OSError,StopIteration,ValueError): continue
    if puid==uid and state!='Z' and pg==pgrp and se==session and start>=min_start:
        out.append(f'{pid}:{start}')
print(','.join(sorted(out)))
PY
}
m1288_wait_job_empty() {
    local pgrp=$1 session=$2 start=$3 i
    for i in $(seq 1 200); do
        [[ -z "$(m1288_job_members "${pgrp}" "${session}" "${start}")" ]] && return 0
        sleep 0.1
    done
    return 1
}
m1288_terminate_job() {
    local pgrp=$1 session=$2 start=$3
    [[ -n "$(m1288_job_members "${pgrp}" "${session}" "${start}")" ]] || return 0
    kill -TERM -- "-${pgrp}" 2>/dev/null || true
    m1288_wait_job_empty "${pgrp}" "${session}" "${start}" && return 0
    kill -KILL -- "-${pgrp}" 2>/dev/null || true
    m1288_wait_job_empty "${pgrp}" "${session}" "${start}"
}

# Inert source gate.  An independent reviewer must create and seal M1289;
# absent that exact admission, the runner exits before namespace mutation,
# collision inspection, tool/version/license calls, or any PT process.
[[ -n "${M1288_EXPECTED_RUNNER_SHA256:-}" && \
   "$(m1288_sha "${m1288_runner}")" == "${M1288_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M1288_EXPECTED_ADMISSION_SHA256:-}" ]] || exit 3
cd "${m1288_hw_root}"
m1288_expect "${m1288_admission}" "${M1288_EXPECTED_ADMISSION_SHA256}" || exit 3
m1288_sealed_payload_ok "${m1288_admission}" || exit 3
jq -e '.status == "AUTHORIZED_ONE_M1288_M917_FIXED_T10_PTSTA_ATTEMPT"
       and .authorization.max_attempts == 1
       and .authorization.run_pt == true
       and .authorization.run_dc == false
       and .authorization.run_vcs == false
       and .authorization.run_formality == false
       and .authorization.run_ptpx == false
       and .authorization.run_remote == false' "${m1288_admission}" >/dev/null || exit 3
m1288_expect "${m1288_contract}" "$(jq -er '.identity.contract_sha256' "${m1288_admission}")" || exit 3
m1288_sealed_payload_ok "${m1288_contract}" || exit 3
[[ "$(jq -er '.identity.runner_sha256' "${m1288_admission}")" == \
   "${M1288_EXPECTED_RUNNER_SHA256}" ]] || exit 3
jq -e '.status == "M1288_SOURCE_ONLY__NO_PT_EDA_AUTHORIZED"
       and .authorization.launch_now == false
       and .authorization.run_pt == false' "${m1288_contract}" >/dev/null || exit 3

while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m1288_expect "${path}" "${expected}" || exit 3
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "${m1288_contract}")
m1288_expect "${m1288_pt}" "$(jq -er '.tool.pt_shell_sha256' "${m1288_contract}")" || exit 3
m1288_expect "${m1288_setsid}" "$(jq -er '.tool.setsid_sha256' "${m1288_contract}")" || exit 3
m1288_expect "${m1288_bash}" "$(jq -er '.tool.bash_sha256' "${m1288_contract}")" || exit 3
m1288_expect "${m1288_slow}" "$(jq -er '.tool.slow_db_sha256' "${m1288_contract}")" || exit 3
m1288_expect "${m1288_fast}" "$(jq -er '.tool.fast_db_sha256' "${m1288_contract}")" || exit 3
m1288_double_seal_ok "${m1288_m917}" || exit 3
m1288_double_seal_ok "${m1288_m928}" || exit 3
m1288_double_seal_ok "${m1288_m1285}" || exit 3

[[ "${PATH:-}" == /usr/bin:/bin && "${LANG:-}" == C.UTF-8 && \
   "${LC_ALL:-}" == C.UTF-8 && -z "${HOME:-}" ]] || exit 3
[[ "${SNPSLMD_LICENSE_FILE:-}" == 27030@ic.ismd-nemo && \
   "${LM_LICENSE_FILE:-}" == /opt/synopsys/Synopsys.dat ]] || exit 3
[[ -z "${M1288_RUN_DIR:-}" && ! -e "${m1288_canonical}" && \
   ! -e "${m1288_work}" && ! -e "${m1288_attempt}" ]] || exit 5

m1288_collision="$(m1288_collisions)"
[[ -z "${m1288_collision}" ]] || exit 4
mkdir "${m1288_attempt}"
printf 'status=M1288_ONE_SHOT_ATTEMPT_CONSUMED\n' >"${m1288_attempt}/attempt.txt"
mkdir -p "${m1288_work}/reports" "${m1288_work}/safe_home"
chmod 0700 "${m1288_work}/safe_home"
cp "${m1288_contract}" "${m1288_work}/contract.json"
sha256sum "${m1288_runner}" "${m1288_tcl}" "${m1288_netlist}" "${m1288_sdc}" \
    "${m1288_slow}" "${m1288_fast}" "${m1288_contract}" \
    "${m1288_m917}/SHA256SUMS.seal.sha256" \
    "${m1288_m928}/SHA256SUMS.seal.sha256" \
    "${m1288_m1285}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md >"${m1288_work}/input_sha256.txt"

m1288_done=0; m1288_pid=; m1288_start=; m1288_pgrp=; m1288_session=
m1288_fail() {
    local rc=$?
    if [[ -n "${m1288_pgrp}" && -n "${m1288_start}" ]]; then
        m1288_terminate_job "${m1288_pgrp}" "${m1288_session}" "${m1288_start}" || true
    fi
    if [[ "${m1288_done}" -ne 1 && -d "${m1288_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' "${rc}" \
            >"${m1288_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m1288_seal_dir "${m1288_work}" || true
        mv -T "${m1288_work}" "${m1288_canonical}.failed_or_incomplete.$$.quarantine" || true
    fi
    exit "${rc}"
}
trap m1288_fail EXIT INT TERM

export M1288_SLOW_DB="${m1288_slow}" M1288_FAST_DB="${m1288_fast}"
export M1288_MAPPED_NETLIST="${m1288_hw_root}/${m1288_netlist}"
export M1288_MAPPED_SDC="${m1288_hw_root}/${m1288_sdc}"
export M1288_PT_OUTPUT_DIR="${m1288_work}"

"${m1288_setsid}" env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    HOME="${m1288_work}/safe_home" \
    SNPSLMD_LICENSE_FILE="${SNPSLMD_LICENSE_FILE}" LM_LICENSE_FILE="${LM_LICENSE_FILE}" \
    M1288_SLOW_DB="${M1288_SLOW_DB}" M1288_FAST_DB="${M1288_FAST_DB}" \
    M1288_MAPPED_NETLIST="${M1288_MAPPED_NETLIST}" M1288_MAPPED_SDC="${M1288_MAPPED_SDC}" \
    M1288_PT_OUTPUT_DIR="${M1288_PT_OUTPUT_DIR}" \
    "${m1288_pt}" -f "${m1288_hw_root}/${m1288_tcl}" \
    >"${m1288_work}/pt.raw.log" 2>&1 &
m1288_pid=$!
for _ in $(seq 1 100); do
    [[ -r "/proc/${m1288_pid}/stat" ]] || break
    m1288_stat="$(cat "/proc/${m1288_pid}/stat")"; m1288_rest=${m1288_stat##*) }
    set -- ${m1288_rest}; m1288_start=${20}; m1288_pgrp=$3; m1288_session=$4
    [[ "${m1288_pgrp}" == "${m1288_pid}" && "${m1288_session}" == "${m1288_pid}" ]] && break
    sleep 0.05
done
[[ "${m1288_pgrp}" == "${m1288_pid}" && "${m1288_session}" == "${m1288_pid}" ]] || exit 20

m1288_monitor_rc=0
while kill -0 "${m1288_pid}" 2>/dev/null; do
    m1288_collision="$(m1288_collisions "${m1288_pid}" "${m1288_start}" \
        "${m1288_pgrp}" "${m1288_session}")"
    if [[ -n "${m1288_collision}" ]]; then
        printf '%s\n' "${m1288_collision}" >>"${m1288_work}/runtime_external_collisions.log"
        m1288_monitor_rc=1; m1288_terminate_job "${m1288_pgrp}" "${m1288_session}" "${m1288_start}"; break
    fi
    sleep 2
done
set +e; wait "${m1288_pid}"; m1288_pt_rc=$?; set -e
printf '%s\n' "${m1288_pt_rc}" >"${m1288_work}/pt.rc"
printf '%s\n' "${m1288_monitor_rc}" >"${m1288_work}/runtime_monitor.rc"
m1288_wait_job_empty "${m1288_pgrp}" "${m1288_session}" "${m1288_start}" || exit 21
[[ "${m1288_pt_rc}" -eq 0 && "${m1288_monitor_rc}" -eq 0 ]] || exit 22
[[ "$(stat -c '%a' "${m1288_work}/safe_home")" == 700 ]] || exit 23
[[ "$(m1288_sha docs/359_DATE终局冻结_20260813.md)" == \
   "$(jq -er '.exact_files["docs/359_DATE终局冻结_20260813.md"]' "${m1288_contract}")" ]] || exit 24
grep -Fqx 'M1288_M917_FIXED_T10_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS' \
    "${m1288_work}/PTSTA_INTERNAL_COMPLETE.txt" || exit 25
for report in check_timing.rpt analysis_coverage.rpt global_timing.rpt \
    timing_setup_slow.rpt timing_hold_fast.rpt constraint_violators.rpt \
    clock.rpt exceptions.rpt design.rpt wire_load.rpt libraries.rpt runtime_scope.rpt; do
    [[ -s "${m1288_work}/reports/${report}" ]] || exit 25
done
if grep -Eiq '^(Error|Fatal):|fix_eco_timing|set_fix_hold' "${m1288_work}/pt.raw.log"; then exit 26; fi

python3 - "${m1288_work}" <<'PY'
import json,re,sys
from pathlib import Path
run=Path(sys.argv[1])
def slack(name):
    text=(run/'reports'/name).read_text(errors='replace')
    m=re.search(r'slack \((MET|VIOLATED)\)\s+(-?\d+(?:\.\d+)?)',text)
    if not m: raise SystemExit(f'cannot parse {name}')
    return m.group(1),float(m.group(2))
setup_state,setup=slack('timing_setup_slow.rpt')
hold_state,hold=slack('timing_hold_fast.rpt')
hold_closed=hold_state=='MET' and hold>=0
setup_closed=setup_state=='MET' and setup>=0
if setup_closed and hold_closed:
    status='PASS_M1288_M917_FIXED_T10_PRELAYOUT_PTSTA_SETUP_HOLD_MET'
elif setup_closed and not hold_closed:
    status='DIAGNOSTIC_STOP_M1288_M917_FIXED_T10_HOLD_NEGATIVE__NEW_NETLIST_HOLD_FIX_REQUIRED'
else:
    status='DIAGNOSTIC_STOP_M1288_M917_FIXED_T10_SETUP_OR_HOLD_NOT_CLOSED'
receipt={
 'schema':'m1288_m917_fixed_t10_prelayout_ptsta_receipt_v1',
 'status':status,
 'setup':{'corner':'ssg0p9v125c','state':setup_state,'worst_slack_ns':setup},
 'hold':{'corner':'ffg1p05vm40c','state':hold_state,'worst_slack_ns':hold,
         'closed':hold_closed,'automatic_fix_performed':False,
         'successor_required':not hold_closed,
         'successor':'new netlist-only hold-fix identity plus Formality and repeated PT'},
 'scope':{'prelayout':True,'spef':False,'ideal_clock':True,'zero_wireload':True,
          'macro_count':0,'mapped_identity_mutated':False},
 'claim_boundary':{'diagnostic_if_negative':True,'power':False,'energy':False,
                   'speedup':False,'system':False,'paper_ppa_ready':False,'headline':False}}
(run/'m1288_m917_fixed_t10_prelayout_ptsta_receipt_r1.json').write_text(
 json.dumps(receipt,indent=2,sort_keys=True)+'\n')
(run/'RUN_COMPLETE.txt').write_text(status+'\n')
PY

m1288_seal_dir "${m1288_attempt}"
m1288_seal_dir "${m1288_work}"
mv -T "${m1288_work}" "${m1288_canonical}"
m1288_done=1
trap - EXIT INT TERM
