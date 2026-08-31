#!/usr/bin/env bash
set -euo pipefail

# M917 is a Fixed-only corrective successor to the consumed M518 r4 point.
# It changes process-tree containment and HOME isolation only; RTL, Tcl,
# constraints, libraries and the setup/area-only admission gates stay frozen.
m917_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m917_hw_root="$(cd "${m917_dc_root}/.." && pwd)"
m917_runner="$(realpath "${BASH_SOURCE[0]}")"
m917_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m917_dc_wrapper=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
m917_dc_actual=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m917_dc_install_root=/opt/synopsys/syn/V-2023.12-SP3
m917_setsid=/usr/bin/setsid
m917_bash=/usr/bin/bash
m917_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m917_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m917_filelist=dc_handoff/filelists/date_m518_matched_fixed_rank3_logic_only_dc.f
m917_sdc=dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc
m917_tcl=dc_handoff/scripts/run_dc_m518_r3_per_point_setup_area.tcl
m917_contract=contracts/m916_m518_r5_fixed_descendant_safe_setup_area_dc_contract_r1_20260829.json
m917_admission=contracts/m917_m518_r5_fixed_descendant_safe_setup_area_dc_launch_admission_r1_20260829.json
m917_forensic=reviews/m915_m518_r4_fixed_descendant_collision_quarantine_forensic_r1_20260829
m917_r4_quarantine=dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828.failed_or_incomplete.2923446.quarantine
m917_r4_attempt=dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed
m917_uid="$(id -u)"
m917_top=m518_matched_fixed_t10_atlif

m917_canonical="${m917_dc_root}/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829"
m917_work="${m917_dc_root}/runs/.m917_m518_r5_fixed_descendant_safe_setup_area_work.$$"
m917_attempt="${m917_dc_root}/runs/.m917_m518_r5_fixed_descendant_safe_setup_area_attempt_consumed"
m917_preflight="${m917_dc_root}/runs/.m917_m518_r5_fixed_descendant_safe_preflight.$$.staging"
m917_preflight_reject="${m917_canonical}.preflight_rejected.$$.quarantine"
m917_quarantine="${m917_canonical}.failed_or_incomplete.$$.quarantine"

m917_preflight_commit_kib=67108864
m917_runtime_soft_commit_kib=50331648
m917_runtime_hard_commit_kib=41943040
m917_mem_available_kib=134217728
m917_swap_free_kib=33554432

m917_sha() { sha256sum "$1" | awk '{print $1}'; }
m917_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && ! -L "${path}" && "$(m917_sha "${path}")" == "${expected}" ]] || {
        echo "M917 identity mismatch: ${path}" >&2
        exit 3
    }
}
m917_expect_frozen_tool_link() {
    local path=$1 expected=$2 expected_target=$3
    [[ -L "${path}" && "$(realpath "${path}")" == "${expected_target}" && \
       "$(m917_sha "${path}")" == "${expected}" ]] || {
        echo "M917 frozen tool link mismatch: ${path}" >&2
        exit 3
    }
}
m917_double_seal_ok() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    [[ -f "${payload}.sha256" && ! -L "${payload}.sha256" && \
       -f "${payload}.sha256.seal.sha256" && ! -L "${payload}.sha256.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}
m917_recursive_sealed_dir_ok() {
    local root=$1 seal dir found=0
    [[ -d "${root}" && ! -L "${root}" ]] || return 1
    while IFS= read -r -d '' seal; do
        found=1; dir="$(dirname "${seal}")"
        [[ -f "${dir}/SHA256SUMS" && ! -L "${dir}/SHA256SUMS" ]] || return 1
        (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null && \
            sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || return 1
    done < <(find "${root}" -type f -name SHA256SUMS.seal.sha256 -print0)
    [[ "${found}" -eq 1 ]]
}
m917_seal_dir() {
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

[[ "${PATH:-}" == /usr/bin:/bin && "${LANG:-}" == C.UTF-8 && \
   "${LC_ALL:-}" == C.UTF-8 && -z "${HOME:-}" ]] || {
    echo "M917 requires exact clean environment with incoming HOME absent" >&2
    exit 3
}
[[ "${SNPSLMD_LICENSE_FILE:-}" == 27030@ic.ismd-nemo && \
   "${LM_LICENSE_FILE:-}" == /opt/synopsys/Synopsys.dat ]] || {
    echo "M917 license environment mismatch" >&2
    exit 3
}
[[ -n "${M917_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m917_sha "${m917_runner}")" == "${M917_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M917 caller must pin runner SHA" >&2
    exit 3
}
[[ -n "${M917_EXPECTED_ADMISSION_SHA256:-}" ]] || exit 3
[[ -z "${M917_DC_RUN_DIR:-}" ]] || exit 5
[[ ! -e "${m917_canonical}" && ! -e "${m917_work}" && \
   ! -e "${m917_attempt}" && ! -e "${m917_preflight}" ]] || exit 5

cd "${m917_hw_root}"
m917_expect "${m917_admission}" "${M917_EXPECTED_ADMISSION_SHA256}"
m917_double_seal_ok "${m917_admission}" || exit 3
jq -e '.status == "AUTHORIZED_ONE_M917_M518_R5_FIXED_DESCENDANT_SAFE_SETUP_AREA_DC_ATTEMPT"
       and .authorization.max_attempts == 1
       and .authorization.run_dc == true
       and .authorization.run_vcs == false
       and .authorization.run_formality == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false
       and .authorization.run_remote == false
       and .authorization.run_rank3 == false' "${m917_admission}" >/dev/null || exit 3
m917_expect "${m917_contract}" "$(jq -er '.identity.contract_sha256' "${m917_admission}")"
m917_double_seal_ok "${m917_contract}" || exit 3
[[ "$(jq -er '.identity.runner_sha256' "${m917_contract}")" == "${M917_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
jq -e '.status == "M916_SOURCE_ONLY__NO_EDA_AUTHORIZED"
       and .authorization.launch_now == false
       and .authorization.run_dc == false' "${m917_contract}" >/dev/null || exit 3
while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m917_expect "${path}" "${expected}"
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "${m917_contract}")
m917_expect_frozen_tool_link "${m917_dc}" \
    "$(jq -er '.tool.dc_shell_sha256' "${m917_contract}")" "${m917_dc_wrapper}"
m917_expect "${m917_dc_wrapper}" "$(jq -er '.tool.dc_wrapper_sha256' "${m917_contract}")"
m917_expect "${m917_dc_actual}" "$(jq -er '.tool.dc_actual_executable_sha256' "${m917_contract}")"
m917_expect "${m917_setsid}" "$(jq -er '.tool.setsid_sha256' "${m917_contract}")"
m917_expect "${m917_bash}" "$(jq -er '.tool.bash_sha256' "${m917_contract}")"
m917_expect "${m917_slow}" "$(jq -er '.tool.slow_db_sha256' "${m917_contract}")"
m917_expect "${m917_fast}" "$(jq -er '.tool.fast_db_sha256' "${m917_contract}")"
[[ "$(realpath "${m917_dc}")" == "${m917_dc_wrapper}" ]] || exit 3

m917_expect "${m917_forensic}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.forensic.m915_outer_seal_file_sha256' "${m917_contract}")"
m917_recursive_sealed_dir_ok "${m917_forensic}" || exit 3
m917_expect "${m917_r4_attempt}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.forensic.r4_attempt_outer_seal_file_sha256' "${m917_contract}")"
m917_recursive_sealed_dir_ok "${m917_r4_attempt}" || exit 3
[[ -d "${m917_r4_quarantine}" && ! -L "${m917_r4_quarantine}" ]] || exit 3
(cd "${m917_r4_quarantine}" && \
 sha256sum -c "${m917_hw_root}/${m917_forensic}/quarantine_current_manifest.sha256" >/dev/null) || exit 3
[[ "$(find "${m917_r4_quarantine}" -type f | wc -l)" -eq \
   "$(wc -l <"${m917_forensic}/quarantine_current_manifest.sha256")" ]] || exit 3
grep -Fxq 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE' \
    "${m917_r4_quarantine}/RUN_FAILED_OR_INCOMPLETE.txt" || exit 3

# Parse an exact process tuple from /proc without using textual argv matching.
m917_proc_identity() {
    local pid=$1 stat rest
    [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/status" ]] || return 1
    stat="$(cat "/proc/${pid}/stat")" || return 1
    rest=${stat##*) }; set -- ${rest}; [[ $# -ge 20 ]] || return 1
    M917_P_STATE=$1; M917_P_PPID=$2; M917_P_PGRP=$3; M917_P_SESSION=$4
    M917_P_START=${20}
    M917_P_UID="$(awk '/^Uid:/ {print $2; exit}' "/proc/${pid}/status")"
    M917_P_EXE="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    M917_P_CMDHEX="$(od -An -tx1 -v "/proc/${pid}/cmdline" 2>/dev/null | tr -d ' \n')"
    M917_P_PID=${pid}
    [[ -n "${M917_P_UID}" && -n "${M917_P_EXE}" ]]
}
m917_root_state() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=$5 pgrp=$6 session=$7 cmdhex=$8
    m917_proc_identity "${pid}" || return 1
    [[ "${M917_P_START}" == "${start}" && "${M917_P_UID}" == "${uid}" && \
       "${M917_P_EXE}" == "${exe}" && "${M917_P_PPID}" == "${parent}" && \
       "${M917_P_PGRP}" == "${pgrp}" && "${M917_P_SESSION}" == "${session}" && \
       "${M917_P_CMDHEX}" == "${cmdhex}" ]] || return 2
    [[ "${M917_P_STATE}" != Z ]]
}
m917_is_exact_root_descendant() {
    local candidate=$1 root=$2 start=$3 uid=$4 exe=$5 parent=$6 pgrp=$7 session=$8 cmdhex=$9
    local cursor=${candidate} next hops=0
    [[ "${candidate}" != "${root}" ]] || {
        m917_root_state "${root}" "${start}" "${uid}" "${exe}" "${parent}" "${pgrp}" "${session}" "${cmdhex}"
        return $?
    }
    while [[ "${hops}" -lt 128 ]]; do
        m917_proc_identity "${cursor}" || return 1
        [[ "${M917_P_UID}" == "${uid}" ]] || return 1
        next=${M917_P_PPID}
        if [[ "${next}" == "${root}" ]]; then
            m917_root_state "${root}" "${start}" "${uid}" "${exe}" "${parent}" "${pgrp}" "${session}" "${cmdhex}"
            return $?
        fi
        [[ "${next}" =~ ^[0-9]+$ && "${next}" -gt 1 && "${next}" != "${cursor}" ]] || return 1
        cursor=${next}; hops=$((hops + 1))
    done
    return 1
}
m917_external_collisions() {
    local log=$1 label=$2 root=${3:-} start=${4:-} uid=${5:-} exe=${6:-}
    local parent=${7:-} pgrp=${8:-} session=${9:-} cmdhex=${10:-}
    local proc pid comm exe_base first=1
    : >"${log}.tmp"
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}; m917_proc_identity "${pid}" || continue
        [[ "${M917_P_UID}" == "${m917_uid}" && "${M917_P_STATE}" != Z ]] || continue
        IFS= read -r comm <"/proc/${pid}/comm" 2>/dev/null || continue
        exe_base=${M917_P_EXE##*/}
        case "${comm}:${exe_base}" in
            dc_shell:*|dc_shell-t:*|fm_shell:*|pt_shell:*|vcs:*|vcs1:*|vlogan:*|simv:*|common_shell_ex*:common_shell_exec) ;;
            *) continue ;;
        esac
        if [[ -n "${root}" ]] && m917_is_exact_root_descendant "${pid}" "${root}" \
                "${start}" "${uid}" "${exe}" "${parent}" "${pgrp}" "${session}" "${cmdhex}"; then
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${pid}" "${M917_P_PPID}" \
            "${M917_P_PGRP}" "${M917_P_SESSION}" "${M917_P_UID}" \
            "${M917_P_START}" "${M917_P_EXE}" "${M917_P_CMDHEX}" >>"${log}.tmp"
        [[ "${first}" -eq 1 ]] || printf ','; printf '%s:%s' "${pid}" "${M917_P_START}"; first=0
    done
    [[ ! -s "${log}.tmp" ]] || cat "${log}.tmp" >>"${log}"
    rm -f "${log}.tmp"
}
m917_snapshot() {
    local label=$1 log=$2 root=${3:-} start=${4:-} uid=${5:-} exe=${6:-}
    local parent=${7:-} pgrp=${8:-} session=${9:-} cmdhex=${10:-} collisions limit committed
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    M917_HEADROOM=$((limit - committed))
    M917_AVAILABLE="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    M917_SWAP="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    M917_FAILCNT="$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)"
    M917_UNDER="$(awk '/^under_oom / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    M917_OOMKILL="$(awk '/^oom_kill / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    collisions="$(m917_external_collisions "${log%.log}_external_collisions.tsv" "${label}" \
        "${root}" "${start}" "${uid}" "${exe}" "${parent}" "${pgrp}" "${session}" "${cmdhex}")"
    M917_COLLISIONS=${collisions:-none}
    printf 'timestamp=%s label=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s external_eda_collision=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${M917_HEADROOM}" "${M917_AVAILABLE}" \
        "${M917_SWAP}" "${M917_FAILCNT}" "${M917_UNDER}" "${M917_OOMKILL}" \
        "${M917_COLLISIONS}" >>"${log}"
}

mkdir "${m917_preflight}"
: >"${m917_preflight}/resource_preflight.log"
: >"${m917_preflight}/resource_preflight_external_collisions.tsv"
m917_preflight_pass=1
for m917_sample in 1 2 3; do
    m917_snapshot "preflight_${m917_sample}" "${m917_preflight}/resource_preflight.log"
    if [[ "${M917_HEADROOM}" -lt "${m917_preflight_commit_kib}" || \
          "${M917_AVAILABLE}" -lt "${m917_mem_available_kib}" || \
          "${M917_SWAP}" -lt "${m917_swap_free_kib}" || \
          "${M917_FAILCNT}" -ne 0 || "${M917_UNDER}" -ne 0 || \
          "${M917_OOMKILL}" -ne 0 || "${M917_COLLISIONS}" != none ]]; then
        m917_preflight_pass=0
    fi
    [[ "${m917_sample}" -eq 3 ]] || sleep 10
done
printf 'status=%s\nsamples=3\nsample_interval_seconds=10\ncommit_headroom_gate_kib=%s\nmem_available_gate_kib=%s\nswap_free_gate_kib=%s\nexternal_eda_required_none=true\n' \
    "$([[ "${m917_preflight_pass}" -eq 1 ]] && echo PASS || echo FAIL)" \
    "${m917_preflight_commit_kib}" "${m917_mem_available_kib}" "${m917_swap_free_kib}" \
    >"${m917_preflight}/preflight_receipt.txt"
m917_seal_dir "${m917_preflight}"
if [[ "${m917_preflight_pass}" -ne 1 ]]; then
    printf 'status=PREFLIGHT_REJECTED_NO_ATTEMPT_CONSUMED\n' >"${m917_preflight}/PREFLIGHT_REJECTED.txt"
    m917_seal_dir "${m917_preflight}"; mv -T "${m917_preflight}" "${m917_preflight_reject}"; exit 40
fi

mkdir "${m917_work}" "${m917_work}/preflight"
mv -T "${m917_preflight}" "${m917_work}/preflight/fixed"
m917_complete=0; m917_child_pid=; m917_child_start=; m917_child_uid=; m917_child_parent=
m917_child_pgrp=; m917_child_session=; m917_child_exe=; m917_child_cmdhex=; m917_monitor_pid=
m917_child_rc=not_started; m917_monitor_rc=not_started; m917_signal=none

m917_job_members() {
    local pgrp=$1 session=$2 uid=$3 min_start=$4 proc pid
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}; m917_proc_identity "${pid}" || continue
        [[ "${M917_P_UID}" == "${uid}" && "${M917_P_PGRP}" == "${pgrp}" && \
           "${M917_P_SESSION}" == "${session}" && "${M917_P_STATE}" != Z && \
           "${M917_P_START}" -ge "${min_start}" ]] || continue
        printf '%s:%s\n' "${pid}" "${M917_P_START}"
    done
}
m917_wait_job_empty() {
    local pgrp=$1 session=$2 uid=$3 min_start=$4 loops=${5:-100} i
    for i in $(seq 1 "${loops}"); do
        [[ -z "$(m917_job_members "${pgrp}" "${session}" "${uid}" "${min_start}")" ]] && return 0
        sleep 0.1
    done
    return 1
}
m917_terminate_job() {
    local pgrp=$1 session=$2 uid=$3 min_start=$4 members
    members="$(m917_job_members "${pgrp}" "${session}" "${uid}" "${min_start}")"
    [[ -n "${members}" ]] || return 0
    kill -TERM -- "-${pgrp}" 2>/dev/null || true
    m917_wait_job_empty "${pgrp}" "${session}" "${uid}" "${min_start}" 100 && return 0
    kill -KILL -- "-${pgrp}" 2>/dev/null || true
    m917_wait_job_empty "${pgrp}" "${session}" "${uid}" "${min_start}" 100
}
m917_failure_cleanup() {
    local rc=$?
    set +e
    if [[ -n "${m917_child_pgrp}" && -n "${m917_child_session}" && -n "${m917_child_start}" ]]; then
        m917_terminate_job "${m917_child_pgrp}" "${m917_child_session}" \
            "${m917_child_uid}" "${m917_child_start}"
    fi
    [[ -z "${m917_child_pid}" ]] || wait "${m917_child_pid}" 2>/dev/null
    [[ -z "${m917_monitor_pid}" ]] || wait "${m917_monitor_pid}" 2>/dev/null
    if [[ "${m917_complete}" -ne 1 && -d "${m917_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\njob_tree_drained_before_seal=true\n' \
            "${rc}" "${m917_child_rc}" "${m917_monitor_rc}" "${m917_signal}" \
            >"${m917_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m917_seal_dir "${m917_work}"; mv -T "${m917_work}" "${m917_quarantine}"
    fi
    return "${rc}"
}
trap m917_failure_cleanup EXIT

mkdir "${m917_work}/.attempt_staging"
printf 'status=CONSUMED_BEFORE_EXACT_FIXED_DC_LAUNCH\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m917_canonical}" \
    >"${m917_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m917_runner}" "${m917_contract}" "${m917_admission}" \
    >"${m917_work}/.attempt_staging/identity.sha256"
m917_seal_dir "${m917_work}/.attempt_staging"
mv -T "${m917_work}/.attempt_staging" "${m917_attempt}"

sha256sum "${m917_runner}" "${m917_contract}" "${m917_admission}" \
    "${m917_tcl}" "${m917_filelist}" "${m917_sdc}" "${m917_dc}" \
    "${m917_dc_wrapper}" "${m917_dc_actual}" "${m917_setsid}" \
    "${m917_slow}" "${m917_fast}" rtl_m518/m518_matched_fixed_t10_atlif.sv \
    docs/359_DATE终局冻结_20260813.md "${m917_forensic}/SHA256SUMS.seal.sha256" \
    "${m917_r4_attempt}/SHA256SUMS.seal.sha256" >"${m917_work}/input_sha256.txt"
cp "${m917_contract}" "${m917_work}/contract.json"
cp "${m917_admission}" "${m917_work}/launch_admission.json"

export HW_ROOT="${m917_hw_root}" LIB_DB="${m917_slow}" MIN_LIB_DB="${m917_fast}"
export RTL_FILELIST="${m917_hw_root}/${m917_filelist}" SDC_FILE="${m917_hw_root}/${m917_sdc}"
export OPERATING_CONDITION=ssg0p9v125c CLOCK_PERIOD_NS=3.000 DESIGN_NAME="${m917_top}"
export OUTPUT_DIR="${m917_work}/fixed"
mkdir "${OUTPUT_DIR}" "${m917_work}/safe_home"
chmod 700 "${m917_work}/safe_home"
export HOME="${m917_work}/safe_home"
printf 'home=%s\nmode=0700\ninside_work=true\n' "${HOME}" >"${m917_work}/safe_home_contract.txt"

m917_dc_cmdline_matches() {
    local pid=$1 exact_tcl="${m917_hw_root}/${m917_tcl}"; local -a argv=()
    mapfile -d '' -t argv <"/proc/${pid}/cmdline" || return 1
    [[ "${#argv[@]}" -eq 7 && "${argv[0]}" == "${m917_dc_actual}" && \
       "${argv[1]}" == -shell && "${argv[2]}" == dc_shell && \
       "${argv[3]}" == -r && "${argv[4]}" == "${m917_dc_install_root}" && \
       "${argv[5]}" == -f && "${argv[6]}" == "${exact_tcl}" ]]
}
m917_capture_child() {
    local pid=$1 try birth_start= birth_uid= birth_parent=
    for try in $(seq 1 400); do
        m917_proc_identity "${pid}" || return 1
        if [[ -z "${birth_start}" ]]; then
            birth_start=${M917_P_START}; birth_uid=${M917_P_UID}; birth_parent=${M917_P_PPID}
            m917_child_start=${birth_start}; m917_child_uid=${birth_uid}; m917_child_parent=${birth_parent}
        fi
        [[ "${M917_P_START}" == "${birth_start}" && "${M917_P_UID}" == "${birth_uid}" && \
           "${M917_P_PPID}" == "${birth_parent}" && "${birth_uid}" == "${m917_uid}" && \
           "${birth_parent}" == "$$" ]] || return 1
        if [[ "${M917_P_EXE}" == "${m917_dc_actual}" ]]; then
            m917_dc_cmdline_matches "${pid}" || return 1
            [[ "${M917_P_PGRP}" == "${pid}" && "${M917_P_SESSION}" == "${pid}" ]] || return 1
            m917_child_pgrp=${M917_P_PGRP}; m917_child_session=${M917_P_SESSION}
            m917_child_exe=${M917_P_EXE}; m917_child_cmdhex=${M917_P_CMDHEX}; return 0
        fi
        sleep 0.01
    done
    return 1
}
m917_runtime_monitor() {
    local child=$1 start=$2 uid=$3 exe=$4 parent=$5 pgrp=$6 session=$7 cmdhex=$8 point=$9
    local state=0 sample=0 soft_bad=0 failed=0 reason=none gate=none
    : >"${point}/resource_runtime.log"; : >"${point}/resource_runtime_external_collisions.tsv"
    : >"${point}/runtime_descendant_exclusions.tsv"
    while true; do
        set +e; m917_root_state "${child}" "${start}" "${uid}" "${exe}" \
            "${parent}" "${pgrp}" "${session}" "${cmdhex}"; state=$?; set -e
        [[ "${state}" -eq 0 ]] || break
        sample=$((sample + 1))
        m917_snapshot "runtime_${sample}" "${point}/resource_runtime.log" \
            "${child}" "${start}" "${uid}" "${exe}" "${parent}" "${pgrp}" "${session}" "${cmdhex}"
        soft_bad=$(( M917_HEADROOM < m917_runtime_soft_commit_kib ? soft_bad + 1 : 0 ))
        gate=none
        if [[ "${M917_HEADROOM}" -lt "${m917_runtime_hard_commit_kib}" ]]; then gate=commit_headroom_below_40gib_immediate
        elif [[ "${soft_bad}" -ge 3 ]]; then gate=commit_headroom_below_48gib_three_consecutive
        elif [[ "${M917_AVAILABLE}" -lt "${m917_mem_available_kib}" ]]; then gate=mem_available_below_128gib_immediate
        elif [[ "${M917_SWAP}" -lt "${m917_swap_free_kib}" ]]; then gate=swap_free_below_32gib_immediate
        elif [[ "${M917_FAILCNT}" -ne 0 || "${M917_UNDER}" -ne 0 || "${M917_OOMKILL}" -ne 0 ]]; then gate=cgroup_oom_counter_nonzero_immediate
        elif [[ "${M917_COLLISIONS}" != none ]]; then gate=external_eda_collision_immediate; fi
        printf 'timestamp=%s sample=%s soft_low_consecutive=%s gate=%s\n' \
            "$(date --iso-8601=seconds)" "${sample}" "${soft_bad}" "${gate}" \
            >>"${point}/runtime_gate_every_snapshot.log"
        if [[ "${gate}" != none ]]; then
            failed=1; reason=${gate}; m917_terminate_job "${pgrp}" "${session}" "${uid}" "${start}" || true; break
        fi
        sleep 10
    done
    [[ "${state}" -ne 2 ]] || { failed=1; reason=child_identity_mismatch; }
    if ! m917_wait_job_empty "${pgrp}" "${session}" "${uid}" "${start}" 300; then
        failed=1; reason=dc_descendant_linger_after_root_exit
        m917_terminate_job "${pgrp}" "${session}" "${uid}" "${start}" || true
    fi
    sample=$((sample + 1)); m917_snapshot runtime_final "${point}/resource_runtime.log"
    soft_bad=$(( M917_HEADROOM < m917_runtime_soft_commit_kib ? soft_bad + 1 : 0 )); gate=none
    if [[ "${M917_HEADROOM}" -lt "${m917_runtime_hard_commit_kib}" ]]; then gate=runtime_final_commit_below_40gib
    elif [[ "${soft_bad}" -ge 3 ]]; then gate=runtime_final_commit_below_48gib_three_consecutive
    elif [[ "${M917_AVAILABLE}" -lt "${m917_mem_available_kib}" ]]; then gate=runtime_final_mem_available_below_128gib
    elif [[ "${M917_SWAP}" -lt "${m917_swap_free_kib}" ]]; then gate=runtime_final_swap_free_below_32gib
    elif [[ "${M917_FAILCNT}" -ne 0 || "${M917_UNDER}" -ne 0 || "${M917_OOMKILL}" -ne 0 || "${M917_COLLISIONS}" != none ]]; then gate=runtime_final_oom_or_collision; fi
    [[ "${gate}" == none ]] || { failed=1; reason=${gate}; }
    printf 'timestamp=%s final_gate_applied=true samples_including_final=%s runtime_resource_latch=%s reason=%s job_tree_empty_before_ack=%s status=%s\n' \
        "$(date --iso-8601=seconds)" "${sample}" "${failed}" "${reason}" \
        "$([[ -z "$(m917_job_members "${pgrp}" "${session}" "${uid}" "${start}")" ]] && echo true || echo false)" \
        "$([[ "${failed}" -eq 0 ]] && echo PASS_FINAL_GATE_ACK || echo FAIL_FINAL_GATE_ACK)" \
        >"${point}/runtime_final_gate_ack.txt"
    [[ "${failed}" -eq 0 ]]
}

set +e
"${m917_setsid}" "${m917_dc}" -f "${m917_hw_root}/${m917_tcl}" >"${OUTPUT_DIR}/dc.log" 2>&1 &
m917_child_pid=$!
set -e
if ! m917_capture_child "${m917_child_pid}"; then
    printf 'status=FAIL_EXACT_SETSID_DC_CHILD_CAPTURE\npid=%s\n' "${m917_child_pid}" \
        >"${OUTPUT_DIR}/dc_identity_capture_failure.txt"
    [[ -z "${m917_child_start}" ]] || m917_terminate_job "${m917_child_pid}" "${m917_child_pid}" \
        "${m917_child_uid}" "${m917_child_start}" || true
    set +e; wait "${m917_child_pid}"; m917_child_rc=$?; set -e; exit 41
fi
printf 'pid=%s\nstarttime=%s\nuid=%s\nparent=%s\npgrp=%s\nsession=%s\nexe=%s\ncmdline_nul_hex=%s\n' \
    "${m917_child_pid}" "${m917_child_start}" "${m917_child_uid}" "${m917_child_parent}" \
    "${m917_child_pgrp}" "${m917_child_session}" "${m917_child_exe}" "${m917_child_cmdhex}" \
    >"${OUTPUT_DIR}/dc_child_identity.txt"
m917_runtime_monitor "${m917_child_pid}" "${m917_child_start}" "${m917_child_uid}" \
    "${m917_child_exe}" "${m917_child_parent}" "${m917_child_pgrp}" \
    "${m917_child_session}" "${m917_child_cmdhex}" "${OUTPUT_DIR}" &
m917_monitor_pid=$!
set +e; wait "${m917_child_pid}"; m917_child_rc=$?; wait "${m917_monitor_pid}"; m917_monitor_rc=$?; set -e
printf '%s\n' "${m917_child_rc}" >"${OUTPUT_DIR}/dc.rc"
printf '%s\n' "${m917_monitor_rc}" >"${OUTPUT_DIR}/runtime_monitor.rc"
m917_child_pid=; m917_monitor_pid=
[[ "${m917_child_rc}" -eq 0 ]] || exit "${m917_child_rc}"
[[ "${m917_monitor_rc}" -eq 0 ]] || exit 42
grep -Fq 'status=PASS_FINAL_GATE_ACK' "${OUTPUT_DIR}/runtime_final_gate_ack.txt" || exit 42
grep -Fq 'job_tree_empty_before_ack=true' "${OUTPUT_DIR}/runtime_final_gate_ack.txt" || exit 42
[[ -s "${OUTPUT_DIR}/TCL_PASS_TERMINAL.txt" && ! -e "${OUTPUT_DIR}/TCL_EXPLICIT_FAILURE.txt" ]] || exit 43
grep -Fxq 'status=PASS_M518_R3_PER_POINT_SETUP_AREA_DC_TCL_TERMINAL' "${OUTPUT_DIR}/TCL_PASS_TERMINAL.txt" || exit 43
grep -Fxq "design=${m917_top}" "${OUTPUT_DIR}/TCL_PASS_TERMINAL.txt" || exit 43
grep -Fxq 'compile_ultra_count=1' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'incremental_compile_count=0' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'hold_fix_command_count=0' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'hold_only_optimization_count=0' "${OUTPUT_DIR}/reports/flow_contract.rpt"
grep -Fxq 'hold_not_closed_at_dc=true' "${OUTPUT_DIR}/reports/flow_contract.rpt"
for report in area.rpt qor.rpt timing_setup.rpt constraint_setup.rpt \
    constraint_max_capacitance.rpt constraint_max_transition.rpt constraint_max_fanout.rpt \
    check_design_postcompile.rpt check_timing_postcompile.rpt structured_postcompile_gate.rpt \
    dc_bit_port_count.txt flow_contract.rpt compile_receipt.rpt; do
    [[ -s "${OUTPUT_DIR}/reports/${report}" ]] || exit 45
done
grep -Fxq 'check_design_ok=1' "${OUTPUT_DIR}/reports/structured_postcompile_gate.rpt" || exit 47
grep -Fxq 'check_timing_ok=1' "${OUTPUT_DIR}/reports/structured_postcompile_gate.rpt" || exit 47
grep -Fxq 'dc_bit_level_port_count=1175' "${OUTPUT_DIR}/reports/structured_postcompile_gate.rpt" || exit 47
grep -Fq 'Number of macros/black boxes:               0' "${OUTPUT_DIR}/reports/area.rpt" || exit 48
grep -Fq 'slack (MET)' "${OUTPUT_DIR}/reports/timing_setup.rpt" || exit 46
! grep -Fq 'slack (VIOLATED)' "${OUTPUT_DIR}/reports/timing_setup.rpt" || exit 46
for c in constraint_setup.rpt constraint_max_capacitance.rpt constraint_max_transition.rpt constraint_max_fanout.rpt; do
    grep -Fq 'This design has no violated constraints.' "${OUTPUT_DIR}/reports/${c}" || exit 46
done
[[ -s "${OUTPUT_DIR}/netlist/${m917_top}_mapped.v" && \
   -s "${OUTPUT_DIR}/netlist/${m917_top}_mapped.sdc" && \
   -s "${OUTPUT_DIR}/netlist/${m917_top}.ddc" && \
   -s "${OUTPUT_DIR}/netlist/${m917_top}.svf" ]] || exit 45

m917_area="$(awk '/Total cell area:/ {print $4; exit}' "${OUTPUT_DIR}/reports/area.rpt")"
m917_cells="$(awk '/Number of cells:/ {print $4; exit}' "${OUTPUT_DIR}/reports/area.rpt")"
m917_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "${OUTPUT_DIR}/reports/timing_setup.rpt")"
[[ -n "${m917_area}" && -n "${m917_cells}" && -n "${m917_setup}" ]] || exit 49
awk -v x="${m917_area}" 'BEGIN{exit !(x>0 && x<500000)}' || exit 49
awk -v x="${m917_setup}" 'BEGIN{exit !(x>=0)}' || exit 49
! grep -Fq 'no such variable' "${OUTPUT_DIR}/dc.log" || exit 50
! grep -Fq '::env(HOME)' "${OUTPUT_DIR}/dc.log" || exit 50
printf 'status=PASS_M917_M518_R5_FIXED_RAW_SETUP_AREA_DC__AWAITING_INDEPENDENT_RESULT_REVIEW\ndesign=%s\ncell_area_um2=%s\ncell_count=%s\nsetup_worst_slack_ns=%s\ncompile_ultra_count=1\nincremental_compile_count=0\nhold_optimization_count=0\nhold_not_closed_at_dc=true\nlogic_only=true\nmacro_count=0\ndescendant_aware_monitor=true\nsetsid_job_tree_drained=true\nsafe_private_home=true\nresult_reviewed=false\nsta_completed=false\npower=false\nenergy=false\nsystem_speedup=false\npaper_ppa_ready=false\nheadline=false\n' \
    "${m917_top}" "${m917_area}" "${m917_cells}" "${m917_setup}" >"${m917_work}/RUN_COMPLETE.txt"
m917_seal_dir "${m917_work}"; mv -T "${m917_work}" "${m917_canonical}"
m917_complete=1; trap - EXIT
echo 'PASS M917 M518 r5 Fixed raw setup/area point sealed'
