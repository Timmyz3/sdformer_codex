#!/usr/bin/env bash
set -euo pipefail

m518_r3_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m518_r3_hw_root="$(cd "${m518_r3_dc_root}/.." && pwd)"
m518_r3_runner="$(realpath "${BASH_SOURCE[0]}")"
m518_r3_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m518_r3_dc_wrapper=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
m518_r3_dc_actual=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m518_r3_dc_install_root=/opt/synopsys/syn/V-2023.12-SP3
m518_r3_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m518_r3_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m518_r3_filelist=dc_handoff/filelists/date_m518_matched_fixed_rank3_logic_only_dc.f
m518_r3_sdc=dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc
m518_r3_tcl=dc_handoff/scripts/run_dc_m518_r3_per_point_setup_area.tcl
m518_r3_contract=contracts/m518_r3_per_point_setup_area_dc_contract_r1_20260828.json
m518_r3_m555=reviews/m555_m518_r2_failure_receipt_blind_hammer_r1_20260828
m518_r3_r2_quarantine=dc_handoff/runs/m518_matched_fixed_rank3_logic_only_dc_3p000ns_r2_20260827.failed_or_incomplete.1433205.quarantine
m518_r3_r2_attempt=dc_handoff/runs/.m518_matched_fixed_rank3_logic_only_dc_r2_attempt_consumed
m518_r3_uid="$(id -u)"

# KiB, matching /proc/meminfo.  Runtime thresholds are a new r3 policy and do
# not reclassify the sealed r2 failure.
m518_r3_preflight_commit_kib=67108864
m518_r3_runtime_soft_commit_kib=50331648
m518_r3_runtime_hard_commit_kib=41943040
m518_r3_mem_available_kib=134217728
m518_r3_swap_free_kib=33554432

m518_r3_sha() { sha256sum "$1" | awk '{print $1}'; }
m518_r3_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m518_r3_sha "${path}")" == "${expected}" ]] || {
        echo "M518 r3 identity mismatch: ${path}" >&2
        exit 3
    }
}
m518_r3_double_seal_ok() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    [[ -f "${payload}.sha256" && -f "${payload}.sha256.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}
m518_r3_seal_dir() {
    local dir=$1
    (
        cd "${dir}"
        find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
            -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}

[[ "${M518_R3_POINT:-}" == fixed || "${M518_R3_POINT:-}" == rank3 ]] || {
    echo "M518 r3 requires M518_R3_POINT=fixed or rank3" >&2
    exit 3
}
m518_r3_point=${M518_R3_POINT}
case "${m518_r3_point}" in
    fixed)
        m518_r3_top=m518_matched_fixed_t10_atlif
        m518_r3_admission=contracts/m518_r3_fixed_setup_area_dc_launch_admission_r1_20260828.json
        ;;
    rank3)
        m518_r3_top=m273_integrated_rank3_atlif
        m518_r3_admission=contracts/m518_r3_rank3_setup_area_dc_launch_admission_r1_20260828.json
        ;;
esac
m518_r3_canonical="${m518_r3_dc_root}/runs/m518_r3_${m518_r3_point}_setup_area_logic_only_dc_3p000ns_r1_20260828"
m518_r3_work="${m518_r3_dc_root}/runs/.m518_r3_${m518_r3_point}_setup_area_work.$$"
m518_r3_attempt="${m518_r3_dc_root}/runs/.m518_r3_${m518_r3_point}_setup_area_attempt_consumed"
m518_r3_preflight="${m518_r3_dc_root}/runs/.m518_r3_${m518_r3_point}_preflight.$$.staging"
m518_r3_preflight_reject="${m518_r3_canonical}.preflight_rejected.$$.quarantine"
m518_r3_quarantine="${m518_r3_canonical}.failed_or_incomplete.$$.quarantine"

[[ -n "${M518_R3_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m518_r3_sha "${m518_r3_runner}")" == \
   "${M518_R3_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M518 r3 caller must pin independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M518_R3_EXPECTED_POINT_ADMISSION_SHA256:-}" ]] || {
    echo "M518 r3 source package has no implicit per-point authorization" >&2
    exit 3
}
[[ -z "${M518_R3_DC_RUN_DIR:-}" ]] || {
    echo "M518 r3 canonical override is forbidden" >&2
    exit 5
}
[[ ! -e "${m518_r3_canonical}" && ! -e "${m518_r3_work}" && \
   ! -e "${m518_r3_attempt}" && ! -e "${m518_r3_preflight}" ]] || {
    echo "M518 r3 ${m518_r3_point} identity is consumed or colliding" >&2
    exit 5
}

cd "${m518_r3_hw_root}"
m518_r3_expect "${m518_r3_admission}" \
    "${M518_R3_EXPECTED_POINT_ADMISSION_SHA256}"
m518_r3_double_seal_ok "${m518_r3_admission}" || exit 3
jq -e --arg point "${m518_r3_point}" \
    '.status == ("AUTHORIZED_ONE_M518_R3_" + ($point|ascii_upcase) + "_SETUP_AREA_DC_ATTEMPT")
     and .point == $point
     and .authorization.max_attempts == 1
     and .authorization.run_dc == true
     and .authorization.run_vcs == false
     and .authorization.run_formality == false
     and .authorization.run_pt == false
     and .authorization.run_ptpx == false
     and .authorization.run_remote == false
     and .authorization.run_paired_comparison == false' \
    "${m518_r3_admission}" >/dev/null || exit 3

m518_r3_expect "${m518_r3_contract}" \
    "$(jq -er '.identity.contract_sha256' "${m518_r3_admission}")"
m518_r3_double_seal_ok "${m518_r3_contract}" || exit 3
jq -e '.status == "AUTHOR_SOURCE_ONLY__FRESH_STATIC_REVIEW_REQUIRED__NO_LAUNCH_ADMISSION"
       and .authorization.launch_now == false
       and .authorization.run_dc == false
       and .authorization.run_vcs == false
       and .authorization.run_formality == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false
       and .authorization.run_remote == false' \
    "${m518_r3_contract}" >/dev/null || exit 3
[[ "$(jq -er '.identity.runner_sha256' "${m518_r3_admission}")" == \
   "${M518_R3_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.identity.point' "${m518_r3_admission}")" == \
   "${m518_r3_point}" ]] || exit 3

while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m518_r3_expect "${path}" "${expected}"
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' \
    "${m518_r3_contract}")
m518_r3_expect "${m518_r3_dc}" \
    "$(jq -er '.tool.dc_shell_sha256' "${m518_r3_contract}")"
m518_r3_expect "${m518_r3_dc_wrapper}" \
    "$(jq -er '.tool.dc_wrapper_sha256' "${m518_r3_contract}")"
m518_r3_expect "${m518_r3_dc_actual}" \
    "$(jq -er '.tool.dc_actual_executable_sha256' "${m518_r3_contract}")"
m518_r3_expect "${m518_r3_slow}" \
    "$(jq -er '.tool.slow_db_sha256' "${m518_r3_contract}")"
m518_r3_expect "${m518_r3_fast}" \
    "$(jq -er '.tool.fast_db_sha256' "${m518_r3_contract}")"
[[ "$(realpath "${m518_r3_dc}")" == "${m518_r3_dc_wrapper}" ]] || exit 3

# Keep the source declaration tuple and DC bit-port namespaces separate.
python3 - <<'PY'
import re
from pathlib import Path

def signature(path, module):
    text = Path(path).read_text()
    start = text.index("module " + module)
    body = text[text.index(") (", start) + 3:text.index(");", start)]
    result = []
    skip = False
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("`ifdef"):
            skip = True
            continue
        if line.startswith("`endif"):
            skip = False
            continue
        if skip:
            continue
        line = line.split("//", 1)[0].strip().lstrip(",").rstrip(",").strip()
        if not line:
            continue
        match = re.fullmatch(
            r"(input|output)\s+logic\s*(\[[^]]+\])?\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)", line)
        if not match:
            raise SystemExit("unparsed source port declaration: " + line)
        result.append((match.group(1),
                       re.sub(r"\s+", "", match.group(2) or ""),
                       match.group(3)))
    return result

fixed = signature("rtl_m518/m518_matched_fixed_t10_atlif.sv",
                  "m518_matched_fixed_t10_atlif")
rank3 = signature("rtl_m273/m273_integrated_rank3_atlif.sv",
                  "m273_integrated_rank3_atlif")
if len(fixed) != 50 or len(rank3) != 50 or fixed != rank3:
    raise SystemExit("M518 r3 source declaration tuple mismatch")
PY

m518_r3_proc_identity() {
    local pid=$1 stat rest
    [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/status" ]] || return 1
    stat="$(cat "/proc/${pid}/stat")" || return 1
    rest=${stat##*) }
    set -- ${rest}
    [[ $# -ge 20 ]] || return 1
    M518_R3_PROC_STATE=$1
    M518_R3_PROC_PPID=$2
    M518_R3_PROC_STARTTIME=${20}
    M518_R3_PROC_UID="$(awk '/^Uid:/ {print $2; exit}' "/proc/${pid}/status")"
    M518_R3_PROC_EXE="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    M518_R3_PROC_CMDLINE_HEX="$(od -An -tx1 -v "/proc/${pid}/cmdline" 2>/dev/null | tr -d ' \n')"
    M518_R3_PROC_PID=${pid}
    [[ -n "${M518_R3_PROC_UID}" && -n "${M518_R3_PROC_EXE}" ]] || return 1
}
m518_r3_root_state() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=$5 cmdhex=$6
    [[ -e "/proc/${pid}" ]] || return 1
    m518_r3_proc_identity "${pid}" || return 2
    [[ "${M518_R3_PROC_STARTTIME}" == "${start}" && \
       "${M518_R3_PROC_UID}" == "${uid}" && \
       "${M518_R3_PROC_EXE}" == "${exe}" && \
       "${M518_R3_PROC_PPID}" == "${parent}" && \
       "${M518_R3_PROC_CMDLINE_HEX}" == "${cmdhex}" ]] || return 2
    [[ "${M518_R3_PROC_STATE}" != Z ]] || return 1
}
m518_r3_term_exact() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=$5 cmdhex=$6 signal=$7
    m518_r3_root_state "${pid}" "${start}" "${uid}" "${exe}" \
        "${parent}" "${cmdhex}" || return $?
    kill -s "${signal}" "${pid}"
}
m518_r3_external_collisions() {
    local root=${1:-} log=$2 label=$3 proc pid comm exe_base first=1
    : >"${log}.tmp"
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m518_r3_proc_identity "${pid}" || continue
        [[ "${M518_R3_PROC_UID}" == "${m518_r3_uid}" && \
           "${M518_R3_PROC_STATE}" != Z && "${pid}" != "${root}" ]] || continue
        IFS= read -r comm <"/proc/${pid}/comm" 2>/dev/null || continue
        exe_base=${M518_R3_PROC_EXE##*/}
        case "${comm}:${exe_base}" in
            dc_shell:*|dc_shell-t:*|fm_shell:*|pt_shell:*|vcs:*|vcs1:*|vlogan:*|simv:*|common_shell_ex*:common_shell_exec) ;;
            *) continue ;;
        esac
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${pid}" \
            "${M518_R3_PROC_PPID}" "${M518_R3_PROC_UID}" \
            "${M518_R3_PROC_STARTTIME}" "${M518_R3_PROC_EXE}" \
            "${M518_R3_PROC_CMDLINE_HEX}" >>"${log}.tmp"
        [[ "${first}" -eq 1 ]] || printf ','
        printf '%s:%s' "${pid}" "${M518_R3_PROC_STARTTIME}"
        first=0
    done
    if [[ -s "${log}.tmp" ]]; then
        cat "${log}.tmp" >>"${log}"
    fi
    rm -f "${log}.tmp"
}
m518_r3_snapshot() {
    local label=$1 log=$2 root=${3:-} collisions limit committed
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    M518_R3_HEADROOM=$((limit - committed))
    M518_R3_AVAILABLE="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    M518_R3_SWAP="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    M518_R3_FAILCNT="$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)"
    M518_R3_UNDER="$(awk '/^under_oom / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    M518_R3_OOMKILL="$(awk '/^oom_kill / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    collisions="$(m518_r3_external_collisions "${root}" \
        "${log%.log}_external_collisions.tsv" "${label}")"
    M518_R3_COLLISIONS=${collisions:-none}
    printf 'timestamp=%s label=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s external_eda_collision=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${M518_R3_HEADROOM}" \
        "${M518_R3_AVAILABLE}" "${M518_R3_SWAP}" "${M518_R3_FAILCNT}" \
        "${M518_R3_UNDER}" "${M518_R3_OOMKILL}" \
        "${M518_R3_COLLISIONS}" >>"${log}"
}

mkdir "${m518_r3_preflight}"
: >"${m518_r3_preflight}/resource_preflight.log"
: >"${m518_r3_preflight}/resource_preflight_external_collisions.tsv"
m518_r3_preflight_pass=1
for m518_r3_sample in 1 2 3; do
    m518_r3_snapshot "preflight_${m518_r3_sample}" \
        "${m518_r3_preflight}/resource_preflight.log"
    if [[ "${M518_R3_HEADROOM}" -lt "${m518_r3_preflight_commit_kib}" || \
          "${M518_R3_AVAILABLE}" -lt "${m518_r3_mem_available_kib}" || \
          "${M518_R3_SWAP}" -lt "${m518_r3_swap_free_kib}" || \
          "${M518_R3_FAILCNT}" -ne 0 || "${M518_R3_UNDER}" -ne 0 || \
          "${M518_R3_OOMKILL}" -ne 0 || \
          "${M518_R3_COLLISIONS}" != none ]]; then
        m518_r3_preflight_pass=0
    fi
    [[ "${m518_r3_sample}" -eq 3 ]] || sleep 10
done
printf 'point=%s\nsamples=3\nsample_interval_seconds=10\ncommit_headroom_gate_kib=%s\nmem_available_gate_kib=%s\nswap_free_gate_kib=%s\nexternal_eda_required_none=true\nstatus=%s\n' \
    "${m518_r3_point}" "${m518_r3_preflight_commit_kib}" \
    "${m518_r3_mem_available_kib}" "${m518_r3_swap_free_kib}" \
    "$([[ "${m518_r3_preflight_pass}" -eq 1 ]] && echo PASS || echo FAIL)" \
    >"${m518_r3_preflight}/preflight_receipt.txt"
m518_r3_seal_dir "${m518_r3_preflight}"
if [[ "${m518_r3_preflight_pass}" -ne 1 ]]; then
    printf 'status=PREFLIGHT_REJECTED_NO_POINT_ATTEMPT_CONSUMED\npoint=%s\n' \
        "${m518_r3_point}" >"${m518_r3_preflight}/PREFLIGHT_REJECTED.txt"
    m518_r3_seal_dir "${m518_r3_preflight}"
    mv -T "${m518_r3_preflight}" "${m518_r3_preflight_reject}"
    exit 40
fi

mkdir "${m518_r3_work}"
mkdir "${m518_r3_work}/preflight"
mv -T "${m518_r3_preflight}" "${m518_r3_work}/preflight/${m518_r3_point}"
m518_r3_complete=0
m518_r3_child_pid=""
m518_r3_child_start=""
m518_r3_child_uid=""
m518_r3_child_parent=""
m518_r3_child_exe=""
m518_r3_child_cmdhex=""
m518_r3_monitor_pid=""
m518_r3_child_rc=not_started
m518_r3_monitor_rc=not_started
m518_r3_signal=none

m518_r3_failure_cleanup() {
    local rc=$? state=1 term_rc=not_needed
    set +e
    if [[ -n "${m518_r3_child_pid}" && -n "${m518_r3_child_start}" ]]; then
        m518_r3_root_state "${m518_r3_child_pid}" "${m518_r3_child_start}" \
            "${m518_r3_child_uid}" "${m518_r3_child_exe}" \
            "${m518_r3_child_parent}" "${m518_r3_child_cmdhex}"
        state=$?
        if [[ "${state}" -eq 0 ]]; then
            m518_r3_term_exact "${m518_r3_child_pid}" \
                "${m518_r3_child_start}" "${m518_r3_child_uid}" \
                "${m518_r3_child_exe}" "${m518_r3_child_parent}" \
                "${m518_r3_child_cmdhex}" TERM
            term_rc=$?
        elif [[ "${state}" -eq 2 ]]; then
            term_rc=pid_identity_mismatch_no_signal
        fi
    fi
    [[ -z "${m518_r3_child_pid}" ]] || wait "${m518_r3_child_pid}" 2>/dev/null
    [[ -z "${m518_r3_monitor_pid}" ]] || wait "${m518_r3_monitor_pid}" 2>/dev/null
    if [[ "${m518_r3_complete}" -ne 1 && -d "${m518_r3_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\npoint=%s\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\nexact_term_status=%s\n' \
            "${m518_r3_point}" "${rc}" "${m518_r3_child_rc}" \
            "${m518_r3_monitor_rc}" "${m518_r3_signal}" "${term_rc}" \
            >"${m518_r3_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m518_r3_seal_dir "${m518_r3_work}"
        mv -T "${m518_r3_work}" "${m518_r3_quarantine}"
    fi
    return "${rc}"
}
trap m518_r3_failure_cleanup EXIT

mkdir "${m518_r3_work}/.attempt_staging"
printf 'status=CONSUMED_BEFORE_EXACT_POINT_DC_LAUNCH\npoint=%s\ntimestamp=%s\ncanonical=%s\n' \
    "${m518_r3_point}" "$(date --iso-8601=seconds)" \
    "${m518_r3_canonical}" \
    >"${m518_r3_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m518_r3_runner}" "${m518_r3_contract}" \
    "${m518_r3_admission}" \
    >"${m518_r3_work}/.attempt_staging/identity.sha256"
m518_r3_seal_dir "${m518_r3_work}/.attempt_staging"
mv -T "${m518_r3_work}/.attempt_staging" "${m518_r3_attempt}"

sha256sum "${m518_r3_runner}" "${m518_r3_contract}" \
    "${m518_r3_admission}" "${m518_r3_tcl}" "${m518_r3_filelist}" \
    "${m518_r3_sdc}" "${m518_r3_dc}" "${m518_r3_dc_wrapper}" \
    "${m518_r3_dc_actual}" "${m518_r3_slow}" "${m518_r3_fast}" \
    rtl_m518/m518_matched_fixed_t10_atlif.sv \
    rtl_m273/m273_integrated_rank3_atlif.sv \
    "${m518_r3_m555}/SHA256SUMS.seal.sha256" \
    "${m518_r3_r2_quarantine}/SHA256SUMS.seal.sha256" \
    "${m518_r3_r2_attempt}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md \
    >"${m518_r3_work}/input_sha256.txt"
cp "${m518_r3_contract}" "${m518_r3_work}/contract.json"
cp "${m518_r3_admission}" "${m518_r3_work}/point_launch_admission.json"

export HW_ROOT="${m518_r3_hw_root}"
export LIB_DB="${m518_r3_slow}"
export MIN_LIB_DB="${m518_r3_fast}"
export RTL_FILELIST="${m518_r3_hw_root}/${m518_r3_filelist}"
export SDC_FILE="${m518_r3_hw_root}/${m518_r3_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000
export DESIGN_NAME="${m518_r3_top}"
export OUTPUT_DIR="${m518_r3_work}/${m518_r3_point}"
mkdir "${OUTPUT_DIR}"

m518_r3_dc_cmdline_matches() {
    local pid=$1 exact_tcl="${m518_r3_hw_root}/${m518_r3_tcl}"
    local -a argv=()
    mapfile -d '' -t argv <"/proc/${pid}/cmdline" || return 1
    [[ "${#argv[@]}" -eq 7 && "${argv[0]}" == "${m518_r3_dc_actual}" && \
       "${argv[1]}" == -shell && "${argv[2]}" == dc_shell && \
       "${argv[3]}" == -r && "${argv[4]}" == "${m518_r3_dc_install_root}" && \
       "${argv[5]}" == -f && "${argv[6]}" == "${exact_tcl}" ]]
}
m518_r3_capture_child() {
    local pid=$1 try birth_start= birth_uid= birth_parent=
    for try in $(seq 1 200); do
        m518_r3_proc_identity "${pid}" || return 1
        if [[ -z "${birth_start}" ]]; then
            birth_start=${M518_R3_PROC_STARTTIME}
            birth_uid=${M518_R3_PROC_UID}
            birth_parent=${M518_R3_PROC_PPID}
            m518_r3_child_start=${birth_start}
            m518_r3_child_uid=${birth_uid}
            m518_r3_child_parent=${birth_parent}
        fi
        [[ "${M518_R3_PROC_STARTTIME}" == "${birth_start}" && \
           "${M518_R3_PROC_UID}" == "${birth_uid}" && \
           "${M518_R3_PROC_PPID}" == "${birth_parent}" && \
           "${birth_uid}" == "${m518_r3_uid}" && \
           "${birth_parent}" == "$$" ]] || return 1
        if [[ "${M518_R3_PROC_EXE}" == "${m518_r3_dc_actual}" ]]; then
            m518_r3_dc_cmdline_matches "${pid}" || return 1
            m518_r3_proc_identity "${pid}" || return 1
            [[ "${M518_R3_PROC_STARTTIME}" == "${birth_start}" && \
               "${M518_R3_PROC_UID}" == "${birth_uid}" && \
               "${M518_R3_PROC_PPID}" == "${birth_parent}" && \
               "${M518_R3_PROC_EXE}" == "${m518_r3_dc_actual}" ]] || return 1
            m518_r3_dc_cmdline_matches "${pid}" || return 1
            m518_r3_child_exe=${M518_R3_PROC_EXE}
            m518_r3_child_cmdhex=${M518_R3_PROC_CMDLINE_HEX}
            return 0
        fi
        sleep 0.01
    done
    return 1
}
m518_r3_runtime_monitor() {
    local child=$1 start=$2 uid=$3 exe=$4 parent=$5 cmdhex=$6 point=$7
    local state=0 sample=0 soft_bad=0 failed=0 reason=none gate=none
    : >"${point}/resource_runtime.log"
    : >"${point}/resource_runtime_external_collisions.tsv"
    while true; do
        set +e
        m518_r3_root_state "${child}" "${start}" "${uid}" "${exe}" \
            "${parent}" "${cmdhex}"
        state=$?
        set -e
        [[ "${state}" -eq 0 ]] || break
        sample=$((sample + 1))
        m518_r3_snapshot "runtime_${sample}" \
            "${point}/resource_runtime.log" "${child}"
        if [[ "${M518_R3_HEADROOM}" -lt \
              "${m518_r3_runtime_soft_commit_kib}" ]]; then
            soft_bad=$((soft_bad + 1))
        else
            soft_bad=0
        fi
        gate=none
        if [[ "${M518_R3_HEADROOM}" -lt \
              "${m518_r3_runtime_hard_commit_kib}" ]]; then
            gate=commit_headroom_below_40gib_immediate
        elif [[ "${soft_bad}" -ge 3 ]]; then
            gate=commit_headroom_below_48gib_three_consecutive
        elif [[ "${M518_R3_AVAILABLE}" -lt \
                "${m518_r3_mem_available_kib}" ]]; then
            gate=mem_available_below_128gib_immediate
        elif [[ "${M518_R3_SWAP}" -lt "${m518_r3_swap_free_kib}" ]]; then
            gate=swap_free_below_32gib_immediate
        elif [[ "${M518_R3_FAILCNT}" -ne 0 || "${M518_R3_UNDER}" -ne 0 || \
                "${M518_R3_OOMKILL}" -ne 0 ]]; then
            gate=cgroup_oom_counter_nonzero_immediate
        elif [[ "${M518_R3_COLLISIONS}" != none ]]; then
            gate=external_eda_collision_immediate
        fi
        printf 'timestamp=%s sample=%s soft_low_consecutive=%s gate=%s\n' \
            "$(date --iso-8601=seconds)" "${sample}" "${soft_bad}" \
            "${gate}" >>"${point}/runtime_gate_every_snapshot.log"
        if [[ "${gate}" != none ]]; then
            failed=1
            reason=${gate}
            set +e
            m518_r3_term_exact "${child}" "${start}" "${uid}" "${exe}" \
                "${parent}" "${cmdhex}" TERM
            set -e
            break
        fi
        sleep 10
    done
    [[ "${state}" -ne 2 ]] || { failed=1; reason=child_identity_mismatch; }
    sample=$((sample + 1))
    m518_r3_snapshot runtime_final "${point}/resource_runtime.log"
    gate=none
    if [[ "${M518_R3_HEADROOM}" -lt \
          "${m518_r3_runtime_hard_commit_kib}" ]]; then
        gate=runtime_final_commit_below_40gib
    elif [[ "${M518_R3_AVAILABLE}" -lt \
            "${m518_r3_mem_available_kib}" ]]; then
        gate=runtime_final_mem_available_below_128gib
    elif [[ "${M518_R3_SWAP}" -lt "${m518_r3_swap_free_kib}" ]]; then
        gate=runtime_final_swap_free_below_32gib
    elif [[ "${M518_R3_FAILCNT}" -ne 0 || "${M518_R3_UNDER}" -ne 0 || \
            "${M518_R3_OOMKILL}" -ne 0 || \
            "${M518_R3_COLLISIONS}" != none ]]; then
        gate=runtime_final_oom_or_collision
    fi
    [[ "${gate}" == none ]] || { failed=1; reason=${gate}; }
    printf 'timestamp=%s final_gate_applied=true samples_including_final=%s soft_low_consecutive=%s runtime_resource_latch=%s reason=%s status=%s\n' \
        "$(date --iso-8601=seconds)" "${sample}" "${soft_bad}" \
        "${failed}" "${reason}" \
        "$([[ "${failed}" -eq 0 ]] && echo PASS_FINAL_GATE_ACK || echo FAIL_FINAL_GATE_ACK)" \
        >"${point}/runtime_final_gate_ack.txt"
    [[ "${failed}" -eq 0 ]]
}

set +e
"${m518_r3_dc}" -f "${m518_r3_hw_root}/${m518_r3_tcl}" \
    >"${OUTPUT_DIR}/dc.log" 2>&1 &
m518_r3_child_pid=$!
set -e
if ! m518_r3_capture_child "${m518_r3_child_pid}"; then
    printf 'status=FAIL_EXACT_DC_CHILD_CAPTURE\npid=%s\nstarttime=%s\nuid=%s\nparent=%s\n' \
        "${m518_r3_child_pid}" "${m518_r3_child_start:-unknown}" \
        "${m518_r3_child_uid:-unknown}" "${m518_r3_child_parent:-unknown}" \
        >"${OUTPUT_DIR}/dc_identity_capture_failure.txt"
    if [[ -n "${m518_r3_child_start}" ]] && \
            m518_r3_proc_identity "${m518_r3_child_pid}" && \
            [[ "${M518_R3_PROC_STARTTIME}" == "${m518_r3_child_start}" && \
               "${M518_R3_PROC_UID}" == "${m518_r3_child_uid}" && \
               "${M518_R3_PROC_PPID}" == "${m518_r3_child_parent}" ]]; then
        kill -TERM "${m518_r3_child_pid}" 2>/dev/null || true
    fi
    set +e; wait "${m518_r3_child_pid}"; m518_r3_child_rc=$?; set -e
    exit 41
fi
printf 'pid=%s\nstarttime=%s\nuid=%s\nparent=%s\nexe=%s\ncmdline_nul_hex=%s\n' \
    "${m518_r3_child_pid}" "${m518_r3_child_start}" \
    "${m518_r3_child_uid}" "${m518_r3_child_parent}" \
    "${m518_r3_child_exe}" "${m518_r3_child_cmdhex}" \
    >"${OUTPUT_DIR}/dc_child_identity.txt"
m518_r3_runtime_monitor "${m518_r3_child_pid}" "${m518_r3_child_start}" \
    "${m518_r3_child_uid}" "${m518_r3_child_exe}" \
    "${m518_r3_child_parent}" "${m518_r3_child_cmdhex}" "${OUTPUT_DIR}" &
m518_r3_monitor_pid=$!
set +e
wait "${m518_r3_child_pid}"
m518_r3_child_rc=$?
wait "${m518_r3_monitor_pid}"
m518_r3_monitor_rc=$?
set -e
printf '%s\n' "${m518_r3_child_rc}" >"${OUTPUT_DIR}/dc.rc"
printf '%s\n' "${m518_r3_monitor_rc}" >"${OUTPUT_DIR}/runtime_monitor.rc"
m518_r3_child_pid=""
m518_r3_monitor_pid=""
[[ "${m518_r3_child_rc}" -eq 0 ]] || exit "${m518_r3_child_rc}"
[[ "${m518_r3_monitor_rc}" -eq 0 ]] || exit 42
grep -Fxq 'final_gate_applied=true' \
    <(tr ' ' '\n' <"${OUTPUT_DIR}/runtime_final_gate_ack.txt") || exit 42
grep -Fq 'status=PASS_FINAL_GATE_ACK' \
    "${OUTPUT_DIR}/runtime_final_gate_ack.txt" || exit 42
[[ -s "${OUTPUT_DIR}/TCL_PASS_TERMINAL.txt" && \
   ! -e "${OUTPUT_DIR}/TCL_EXPLICIT_FAILURE.txt" ]] || exit 43
grep -Fxq 'status=PASS_M518_R3_PER_POINT_SETUP_AREA_DC_TCL_TERMINAL' \
    "${OUTPUT_DIR}/TCL_PASS_TERMINAL.txt" || exit 43
grep -Fxq "design=${m518_r3_top}" "${OUTPUT_DIR}/TCL_PASS_TERMINAL.txt" || exit 43
grep -Fxq 'compile_ultra_count=1' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'incremental_compile_count=0' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'hold_fix_command_count=0' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'hold_only_optimization_count=0' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'hold_not_closed_at_dc=true' "${OUTPUT_DIR}/reports/flow_contract.rpt"
for m518_r3_report in area.rpt qor.rpt timing_setup.rpt \
        constraint_setup.rpt constraint_max_capacitance.rpt \
        constraint_max_transition.rpt constraint_max_fanout.rpt \
        check_design_postcompile.rpt check_timing_postcompile.rpt \
        structured_postcompile_gate.rpt dc_bit_port_count.txt \
        flow_contract.rpt compile_receipt.rpt; do
    [[ -s "${OUTPUT_DIR}/reports/${m518_r3_report}" ]] || exit 45
done
grep -Fxq 'check_design_ok=1' \
    "${OUTPUT_DIR}/reports/structured_postcompile_gate.rpt" || exit 47
grep -Fxq 'check_timing_ok=1' \
    "${OUTPUT_DIR}/reports/structured_postcompile_gate.rpt" || exit 47
grep -Fxq 'dc_bit_level_port_count=1175' \
    "${OUTPUT_DIR}/reports/structured_postcompile_gate.rpt" || exit 47
[[ "$(tr -d '[:space:]' <"${OUTPUT_DIR}/reports/dc_bit_port_count.txt")" \
   -eq 1175 ]] || exit 47
grep -Fq 'Number of macros/black boxes:               0' \
    "${OUTPUT_DIR}/reports/area.rpt" || exit 48
grep -Fq 'slack (MET)' "${OUTPUT_DIR}/reports/timing_setup.rpt" || exit 46
! grep -Fq 'slack (VIOLATED)' "${OUTPUT_DIR}/reports/timing_setup.rpt" || exit 46
for m518_r3_constraint in constraint_setup.rpt \
        constraint_max_capacitance.rpt constraint_max_transition.rpt \
        constraint_max_fanout.rpt; do
    grep -Fq 'This design has no violated constraints.' \
        "${OUTPUT_DIR}/reports/${m518_r3_constraint}" || exit 46
done
[[ -s "${OUTPUT_DIR}/netlist/${m518_r3_top}_mapped.v" && \
   -s "${OUTPUT_DIR}/netlist/${m518_r3_top}_mapped.sdc" && \
   -s "${OUTPUT_DIR}/netlist/${m518_r3_top}.ddc" && \
   -s "${OUTPUT_DIR}/netlist/${m518_r3_top}.svf" ]] || exit 45

m518_r3_area="$(awk '/Total cell area:/ {print $4; exit}' "${OUTPUT_DIR}/reports/area.rpt")"
m518_r3_cells="$(awk '/Number of cells:/ {print $4; exit}' "${OUTPUT_DIR}/reports/area.rpt")"
m518_r3_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "${OUTPUT_DIR}/reports/timing_setup.rpt")"
[[ -n "${m518_r3_area}" && -n "${m518_r3_cells}" && \
   -n "${m518_r3_setup}" ]] || exit 49
awk -v x="${m518_r3_area}" 'BEGIN{exit !(x>0 && x<500000)}' || exit 49
awk -v x="${m518_r3_setup}" 'BEGIN{exit !(x>=0)}' || exit 49
printf 'status=PASS_M518_R3_%s_RAW_SETUP_AREA_DC__AWAITING_INDEPENDENT_POINT_RECEIPT_REVIEW\npoint=%s\ndesign=%s\ncell_area_um2=%s\ncell_count=%s\nsetup_worst_slack_ns=%s\nsource_declaration_tuple_count=50\ndc_bit_level_port_count=1175\ncompile_ultra_count=1\nincremental_compile_count=0\nhold_optimization_count=0\nhold_not_closed_at_dc=true\nlogic_only=true\nmacro_count=0\npoint_receipt_reviewed=false\npaired_comparison_admitted=false\nsta_completed=false\npower=false\nenergy=false\nsystem_speedup=false\npaper_ppa_ready=false\nheadline=false\n' \
    "${m518_r3_point^^}" "${m518_r3_point}" "${m518_r3_top}" \
    "${m518_r3_area}" "${m518_r3_cells}" "${m518_r3_setup}" \
    >"${m518_r3_work}/RUN_COMPLETE.txt"
m518_r3_seal_dir "${m518_r3_work}"
mv -T "${m518_r3_work}" "${m518_r3_canonical}"
m518_r3_complete=1
trap - EXIT
echo "PASS M518 r3 ${m518_r3_point} raw setup/area point sealed"
