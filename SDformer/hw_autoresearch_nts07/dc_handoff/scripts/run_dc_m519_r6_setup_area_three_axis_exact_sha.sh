#!/usr/bin/env bash
set -euo pipefail

m519_r6_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m519_r6_hw_root="$(cd "${m519_r6_dc_root}/.." && pwd)"
m519_r6_runner="$(realpath "${BASH_SOURCE[0]}")"
m519_r6_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m519_r6_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m519_r6_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m519_r6_filelist=dc_handoff/filelists/date_m519_r5_channel_local_fault_three_axis_logic_only_dc.f
m519_r6_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
m519_r6_tcl=dc_handoff/scripts/run_dc_m519_r6_setup_area_three_axis.tcl
m519_r6_contract=contracts/m519_r6_setup_area_three_axis_recovery_contract_r1_20260827.json
m519_r6_admission=contracts/m519_r6_setup_area_three_axis_dc_launch_admission_r1_20260827.json
m519_r6_r5_static=reviews/m519_r5_channel_local_fault_static_hammer_r1_20260827
m519_r6_r5_vcs=results/m519_r5_channel_local_fault_vcs_r1_20260827
m519_r6_r5_vcs_review=reviews/m519_r5_channel_local_fault_vcs_receipt_blind_hammer_r1_20260827
m519_r6_r5_failure=reviews/m519_r5_final_failure_receipt_hammer_r1_20260827
m519_r6_r5_quarantine=dc_handoff/runs/m519_r5_channel_local_fault_three_axis_logic_only_dc_3p000ns_r1_20260827.failed_or_incomplete.4165439.quarantine
m519_r6_canonical="${m519_r6_dc_root}/runs/m519_r6_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260827"
m519_r6_work="${m519_r6_dc_root}/runs/.m519_r6_channel_local_fault_dc_work.$$"
m519_r6_attempt="${m519_r6_dc_root}/runs/.m519_r6_channel_local_fault_dc_attempt_consumed"
m519_r6_quarantine="${m519_r6_canonical}.failed_or_incomplete.$$.quarantine"
m519_r6_preflight_staging="${m519_r6_dc_root}/runs/.m519_r6_preflight.$$.staging"
m519_r6_preflight_reject="${m519_r6_canonical}.preflight_rejected.$$.quarantine"
m519_r6_uid="$(id -u)"

# All memory units below are KiB, matching /proc/meminfo.
m519_r6_preflight_commit_kib=67108864
m519_r6_runtime_commit_kib=33554432
m519_r6_mem_available_kib=134217728
m519_r6_swap_free_kib=33554432

m519_r6_sha() { sha256sum "$1" | awk '{print $1}'; }
m519_r6_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m519_r6_sha "${path}")" == "${expected}" ]] || {
        echo "M519 R6 identity mismatch: ${path}" >&2
        exit 3
    }
}

[[ -n "${M519_R6_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m519_r6_sha "${m519_r6_runner}")" == \
   "${M519_R6_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M519 R6 caller must pin independently reviewed DC runner SHA" >&2
    exit 3
}
[[ -n "${M519_R6_EXPECTED_DC_LAUNCH_ADMISSION_SHA256:-}" ]] || {
    echo "M519 R6 source-only package has no implicit launch authorization" >&2
    exit 3
}
[[ ! -e "${m519_r6_canonical}" && ! -e "${m519_r6_work}" && \
   ! -e "${m519_r6_attempt}" && ! -e "${m519_r6_quarantine}" && \
   ! -e "${m519_r6_preflight_staging}" ]] || {
    echo "M519 R6 refuses consumed or colliding result identity" >&2
    exit 5
}
[[ -z "${M519_R6_DC_RUN:-}" ]] || {
    echo "M519 R6 canonical path override is forbidden" >&2
    exit 5
}

cd "${m519_r6_hw_root}"
m519_r6_expect "${m519_r6_admission}" \
    "${M519_R6_EXPECTED_DC_LAUNCH_ADMISSION_SHA256}"
jq -e '.status == "AUTHORIZED_ONE_M519_R6_THREE_AXIS_SETUP_AREA_DC_ATTEMPT"
       and .authorization.run_dc == true
       and .authorization.max_attempts == 1
       and .authorization.run_vcs == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false
       and .authorization.run_formality == false' \
    "${m519_r6_admission}" >/dev/null || exit 3
for key_path in \
    recovery_contract_sha256 \
    r5_static_review_outer_seal_file_sha256 \
    r5_vcs_result_outer_seal_file_sha256 \
    r5_vcs_review_outer_seal_file_sha256 \
    r5_final_failure_review_outer_seal_file_sha256 \
    r5_quarantine_outer_seal_file_sha256 \
    dc_runner_sha256 dc_tcl_sha256 dc_filelist_sha256 sdc_sha256 \
    slow_lib_sha256 fast_lib_sha256 dc_shell_sha256; do
    value="$(jq -er ".identity.${key_path}" "${m519_r6_admission}")"
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done
[[ "$(jq -er '.identity.dc_runner_sha256' "${m519_r6_admission}")" == \
   "${M519_R6_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.docs359_sha256' "${m519_r6_admission}")" == \
   dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 ]] \
    || exit 3
m519_r6_expect "${m519_r6_contract}" \
    "$(jq -er '.identity.recovery_contract_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_r5_static}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.r5_static_review_outer_seal_file_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_r5_vcs}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.r5_vcs_result_outer_seal_file_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_r5_vcs_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.r5_vcs_review_outer_seal_file_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_r5_failure}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.r5_final_failure_review_outer_seal_file_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_r5_quarantine}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.r5_quarantine_outer_seal_file_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_tcl}" \
    "$(jq -er '.identity.dc_tcl_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_filelist}" \
    "$(jq -er '.identity.dc_filelist_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_sdc}" \
    "$(jq -er '.identity.sdc_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_slow}" \
    "$(jq -er '.identity.slow_lib_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_fast}" \
    "$(jq -er '.identity.fast_lib_sha256' "${m519_r6_admission}")"
m519_r6_expect "${m519_r6_dc}" \
    "$(jq -er '.identity.dc_shell_sha256' "${m519_r6_admission}")"
m519_r6_expect docs/359_DATE终局冻结_20260813.md \
    dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
for sealed in "${m519_r6_r5_static}" "${m519_r6_r5_vcs}" \
        "${m519_r6_r5_vcs_review}" "${m519_r6_r5_failure}" \
        "${m519_r6_r5_quarantine}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done

m519_r6_pid_is_descendant() {
    local pid=$1 root=$2 parent guard=0
    [[ -n "${root}" ]] || return 1
    while [[ "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 && \
             "${guard}" -lt 64 ]]; do
        [[ "${pid}" -eq "${root}" ]] && return 0
        [[ -r "/proc/${pid}/stat" ]] || return 1
        parent="$(awk '{print $4}' "/proc/${pid}/stat" 2>/dev/null || true)"
        [[ "${parent}" =~ ^[0-9]+$ && "${parent}" -ne "${pid}" ]] || return 1
        pid=${parent}
        guard=$((guard + 1))
    done
    return 1
}

m519_r6_external_eda_pids() {
    local allowed_root=${1:-} pid comm first=1
    while read -r pid comm; do
        case "${comm}" in
            dc_shell|dc_shell-t|fm_shell|pt_shell|vcs|vcs1|vlogan|simv)
                if [[ -n "${allowed_root}" ]] && \
                        m519_r6_pid_is_descendant "${pid}" "${allowed_root}"; then
                    continue
                fi
                [[ "${first}" -eq 1 ]] || printf ','
                printf '%s:%s' "${pid}" "${comm}"
                first=0
                ;;
        esac
    done < <(ps -u "${m519_r6_uid}" -o pid=,comm=)
}

m519_r6_read_cgroup() {
    M519_R6_CGROUP_FAILCNT="$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)"
    M519_R6_CGROUP_UNDER_OOM="$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    M519_R6_CGROUP_OOM_KILL="$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
}

m519_r6_resource_snapshot() {
    local label=$1 log=$2 h0=${3:-NA} allowed_root=${4:-}
    local limit committed delta
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    M519_R6_HEADROOM_KIB=$((limit - committed))
    M519_R6_MEM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    M519_R6_SWAP_FREE_KIB="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    m519_r6_read_cgroup
    M519_R6_COLLISION="$(m519_r6_external_eda_pids "${allowed_root}")"
    if [[ "${h0}" =~ ^[0-9]+$ ]]; then
        delta=$((h0 - M519_R6_HEADROOM_KIB))
    else
        delta=NA
    fi
    printf 'timestamp=%s label=%s h0_commit_headroom_kib=%s commit_headroom_kib=%s h0_minus_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s external_eda_collision=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${h0}" \
        "${M519_R6_HEADROOM_KIB}" "${delta}" \
        "${M519_R6_MEM_AVAILABLE_KIB}" "${M519_R6_SWAP_FREE_KIB}" \
        "${M519_R6_CGROUP_FAILCNT}" "${M519_R6_CGROUP_UNDER_OOM}" \
        "${M519_R6_CGROUP_OOM_KILL}" "${M519_R6_COLLISION:-none}" \
        >>"${log}"
}

m519_r6_pid_tree_snapshot() {
    local label=$1 log=$2
    printf 'timestamp=%s label=%s\n' "$(date --iso-8601=seconds)" \
        "${label}" >>"${log}"
    ps -eo pid=,ppid=,uid=,etimes=,stat=,comm= --sort pid >>"${log}"
}

m519_r6_seal_dir() {
    local dir=$1
    (
        cd "${dir}"
        find . -type f ! -path './SHA256SUMS' \
            ! -path './SHA256SUMS.seal.sha256' \
            -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}

m519_r6_axis_preflight() {
    local axis=$1 dir=$2 sample pass=1 h0=0
    mkdir -p "${dir}"
    : >"${dir}/resource_preflight.log"
    : >"${dir}/pid_tree_preflight.log"
    for sample in 1 2 3; do
        m519_r6_resource_snapshot "${axis}_preflight_${sample}" \
            "${dir}/resource_preflight.log" NA ""
        m519_r6_pid_tree_snapshot "${axis}_preflight_${sample}" \
            "${dir}/pid_tree_preflight.log"
        if [[ "${sample}" -eq 1 || "${M519_R6_HEADROOM_KIB}" -lt "${h0}" ]]; then
            h0=${M519_R6_HEADROOM_KIB}
        fi
        if [[ "${M519_R6_HEADROOM_KIB}" -lt "${m519_r6_preflight_commit_kib}" || \
              "${M519_R6_MEM_AVAILABLE_KIB}" -lt "${m519_r6_mem_available_kib}" || \
              "${M519_R6_SWAP_FREE_KIB}" -lt "${m519_r6_swap_free_kib}" || \
              "${M519_R6_CGROUP_FAILCNT}" -ne 0 || \
              "${M519_R6_CGROUP_UNDER_OOM}" -ne 0 || \
              "${M519_R6_CGROUP_OOM_KILL}" -ne 0 || \
              -n "${M519_R6_COLLISION}" ]]; then
            pass=0
        fi
        [[ "${sample}" -eq 3 ]] || sleep 10
    done
    printf 'axis=%s\nh0_commit_headroom_kib=%s\nsamples=3\nsample_interval_seconds=10\ncommit_headroom_gate_kib=%s\nmem_available_gate_kib=%s\nswap_free_gate_kib=%s\ncgroup_required_zero=true\nsame_uid_external_eda_required_none=true\nstatus=%s\n' \
        "${axis}" "${h0}" "${m519_r6_preflight_commit_kib}" \
        "${m519_r6_mem_available_kib}" "${m519_r6_swap_free_kib}" \
        "$([[ "${pass}" -eq 1 ]] && echo PASS || echo FAIL)" \
        >"${dir}/preflight_receipt.txt"
    m519_r6_seal_dir "${dir}"
    [[ "${pass}" -eq 1 ]]
}

# A rejected first-axis preflight does not consume the single DC attempt, but
# the complete three-sample/PID-tree evidence is still independently sealed.
if ! m519_r6_axis_preflight k1 "${m519_r6_preflight_staging}"; then
    printf 'status=PREFLIGHT_REJECTED_NO_DC_ATTEMPT_CONSUMED\n' \
        >"${m519_r6_preflight_staging}/PREFLIGHT_REJECTED.txt"
    m519_r6_seal_dir "${m519_r6_preflight_staging}"
    mv -T "${m519_r6_preflight_staging}" "${m519_r6_preflight_reject}"
    exit 40
fi

mkdir "${m519_r6_work}"
mkdir "${m519_r6_work}/preflight"
mv -T "${m519_r6_preflight_staging}" "${m519_r6_work}/preflight/k1"
m519_r6_run_created=1
m519_r6_complete=0
m519_r6_child_pid=""
m519_r6_monitor_pid=""
m519_r6_child_rc=not_started
m519_r6_monitor_rc=not_started
m519_r6_signal=none
m519_r6_runtime_latch=0
m519_r6_runtime_latch_reason=none

m519_r6_signal_handler() {
    local signal_name=$1
    m519_r6_signal="${signal_name}"
    printf 'timestamp=%s signal=%s child_pid=%s monitor_pid=%s\n' \
        "$(date --iso-8601=seconds)" "${signal_name}" \
        "${m519_r6_child_pid:-none}" "${m519_r6_monitor_pid:-none}" \
        >>"${m519_r6_work}/signal_provenance.txt"
    if [[ -n "${m519_r6_child_pid}" ]] && \
            kill -0 "${m519_r6_child_pid}" 2>/dev/null; then
        kill -s "${signal_name}" "${m519_r6_child_pid}" 2>/dev/null || true
    fi
    if [[ -n "${m519_r6_monitor_pid}" ]] && \
            kill -0 "${m519_r6_monitor_pid}" 2>/dev/null; then
        kill -TERM "${m519_r6_monitor_pid}" 2>/dev/null || true
    fi
}
trap 'm519_r6_signal_handler INT' INT
trap 'm519_r6_signal_handler TERM' TERM

m519_r6_failure_cleanup() {
    local rc=$?
    set +e
    if [[ -n "${m519_r6_child_pid}" ]] && \
            kill -0 "${m519_r6_child_pid}" 2>/dev/null; then
        m519_r6_pid_tree_snapshot failure_before_term \
            "${m519_r6_work}/failure_pid_tree.log"
        kill -TERM "${m519_r6_child_pid}" 2>/dev/null
        wait "${m519_r6_child_pid}"
        m519_r6_child_rc=$?
    fi
    if [[ -n "${m519_r6_monitor_pid}" ]] && \
            kill -0 "${m519_r6_monitor_pid}" 2>/dev/null; then
        kill -TERM "${m519_r6_monitor_pid}" 2>/dev/null
        wait "${m519_r6_monitor_pid}"
        m519_r6_monitor_rc=$?
    fi
    if [[ "${m519_r6_run_created}" -eq 1 && \
          "${m519_r6_complete}" -ne 1 && -d "${m519_r6_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\nruntime_resource_latch=%s\nruntime_latch_reason=%s\n' \
            "${rc}" "${m519_r6_child_rc}" "${m519_r6_monitor_rc}" \
            "${m519_r6_signal}" "${m519_r6_runtime_latch}" \
            "${m519_r6_runtime_latch_reason}" \
            >"${m519_r6_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m519_r6_seal_dir "${m519_r6_work}"
        mv -T "${m519_r6_work}" "${m519_r6_quarantine}"
        m519_r6_run_created=0
    fi
    return "${rc}"
}
trap m519_r6_failure_cleanup EXIT

mkdir "${m519_r6_work}/.attempt_staging"
printf 'status=CONSUMED_AT_FIRST_DC_LAUNCH\ntimestamp=%s\ncanonical=%s\npreflight_k1_outer_seal_sha256=%s\n' \
    "$(date --iso-8601=seconds)" "${m519_r6_canonical}" \
    "$(m519_r6_sha "${m519_r6_work}/preflight/k1/SHA256SUMS.seal.sha256")" \
    >"${m519_r6_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m519_r6_runner}" "${m519_r6_contract}" \
    "${m519_r6_admission}" \
    >"${m519_r6_work}/.attempt_staging/identity.sha256"
m519_r6_seal_dir "${m519_r6_work}/.attempt_staging"
mv -T "${m519_r6_work}/.attempt_staging" "${m519_r6_attempt}"

sha256sum "${m519_r6_runner}" "${m519_r6_contract}" \
    "${m519_r6_admission}" "${m519_r6_tcl}" "${m519_r6_filelist}" \
    "${m519_r6_sdc}" "${m519_r6_dc}" "${m519_r6_slow}" \
    "${m519_r6_fast}" \
    "${m519_r6_r5_failure}/SHA256SUMS.seal.sha256" \
    "${m519_r6_r5_quarantine}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md \
    >"${m519_r6_work}/input_sha256.txt"
cp "${m519_r6_contract}" "${m519_r6_work}/contract.json"

export HW_ROOT="${m519_r6_hw_root}"
export LIB_DB="${m519_r6_slow}"
export MIN_LIB_DB="${m519_r6_fast}"
export SDC_FILE="${m519_r6_hw_root}/${m519_r6_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m519_r6_descendant_pids() {
    local root=$1 pid
    [[ -n "${root}" ]] || return 0
    for pid in /proc/[0-9]*; do
        pid=${pid#/proc/}
        m519_r6_pid_is_descendant "${pid}" "${root}" && printf '%s\n' "${pid}"
    done
}

m519_r6_runtime_monitor() {
    local child=$1 h0=$2 point=$3
    local failed=0 commit_bad_count=0 reason=none sample=0 pid status
    local vmpeak vmsize vmrss vmswap comm key
    declare -A high_peak=() high_size=() high_rss=() high_swap=() high_comm=()
    : >"${point}/resource_runtime.log"
    printf 'timestamp\tsample\tpid\tppid\tcomm\tVmPeak_kib\tVmSize_kib\tVmRSS_kib\tVmSwap_kib\n' \
        >"${point}/descendant_memory_runtime.tsv"
    while kill -0 "${child}" 2>/dev/null; do
        sample=$((sample + 1))
        m519_r6_resource_snapshot runtime_${sample} \
            "${point}/resource_runtime.log" "${h0}" "${child}"
        while read -r pid; do
            status="/proc/${pid}/status"
            [[ -r "${status}" ]] || continue
            comm="$(awk '/^Name:/ {print $2}' "${status}")"
            vmpeak="$(awk '/^VmPeak:/ {print $2}' "${status}")"; vmpeak=${vmpeak:-0}
            vmsize="$(awk '/^VmSize:/ {print $2}' "${status}")"; vmsize=${vmsize:-0}
            vmrss="$(awk '/^VmRSS:/ {print $2}' "${status}")"; vmrss=${vmrss:-0}
            vmswap="$(awk '/^VmSwap:/ {print $2}' "${status}")"; vmswap=${vmswap:-0}
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date --iso-8601=seconds)" "${sample}" "${pid}" \
                "$(awk '{print $4}' "/proc/${pid}/stat" 2>/dev/null || echo -1)" \
                "${comm}" "${vmpeak}" "${vmsize}" "${vmrss}" "${vmswap}" \
                >>"${point}/descendant_memory_runtime.tsv"
            key=${pid}
            high_comm[${key}]="${comm}"
            [[ "${vmpeak}" -le "${high_peak[${key}]:-0}" ]] || high_peak[${key}]=${vmpeak}
            [[ "${vmsize}" -le "${high_size[${key}]:-0}" ]] || high_size[${key}]=${vmsize}
            [[ "${vmrss}" -le "${high_rss[${key}]:-0}" ]] || high_rss[${key}]=${vmrss}
            [[ "${vmswap}" -le "${high_swap[${key}]:-0}" ]] || high_swap[${key}]=${vmswap}
        done < <(m519_r6_descendant_pids "${child}")

        if [[ "${M519_R6_HEADROOM_KIB}" -lt "${m519_r6_runtime_commit_kib}" ]]; then
            commit_bad_count=$((commit_bad_count + 1))
        else
            commit_bad_count=0
        fi
        if [[ "${commit_bad_count}" -ge 3 ]]; then
            reason=commit_headroom_below_32gib_for_three_consecutive_samples
        elif [[ "${M519_R6_MEM_AVAILABLE_KIB}" -lt "${m519_r6_mem_available_kib}" ]]; then
            reason=mem_available_below_128gib
        elif [[ "${M519_R6_SWAP_FREE_KIB}" -lt "${m519_r6_swap_free_kib}" ]]; then
            reason=swap_free_below_32gib
        elif [[ "${M519_R6_CGROUP_FAILCNT}" -ne 0 || \
                "${M519_R6_CGROUP_UNDER_OOM}" -ne 0 || \
                "${M519_R6_CGROUP_OOM_KILL}" -ne 0 ]]; then
            reason=cgroup_or_oom_counter_nonzero
        elif [[ -n "${M519_R6_COLLISION}" ]]; then
            reason=new_external_same_uid_eda_collision
        fi
        if [[ "${reason}" != none ]]; then
            failed=1
            printf 'timestamp=%s status=RUNTIME_RESOURCE_LATCH reason=%s sample=%s commit_bad_consecutive=%s\n' \
                "$(date --iso-8601=seconds)" "${reason}" "${sample}" \
                "${commit_bad_count}" >"${point}/runtime_latch.txt"
            kill -TERM "${child}" 2>/dev/null || true
            break
        fi
        sleep 10
    done

    if [[ "${failed}" -eq 0 ]]; then
        m519_r6_resource_snapshot runtime_final \
            "${point}/resource_runtime.log" "${h0}" "${child}"
    fi
    printf 'pid\tcomm\tVmPeak_kib\tVmSize_kib\tVmRSS_kib\tVmSwap_kib\n' \
        >"${point}/descendant_memory_highwater.tsv"
    for key in "${!high_comm[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${key}" \
            "${high_comm[${key}]}" "${high_peak[${key}]:-0}" \
            "${high_size[${key}]:-0}" "${high_rss[${key}]:-0}" \
            "${high_swap[${key}]:-0}"
    done | sort -n >>"${point}/descendant_memory_highwater.tsv"
    printf 'runtime_resource_latch=%s\nreason=%s\ncommit_below_32gib_consecutive_final=%s\n' \
        "${failed}" "${reason}" "${commit_bad_count}" \
        >>"${point}/resource_runtime.log"
    return "${failed}"
}

m519_r6_run_point() {
    local id=$1 mode=$2
    local point="${m519_r6_work}/${id}" h0
    h0="$(awk -F= '/^h0_commit_headroom_kib=/ {print $2}' \
        "${m519_r6_work}/preflight/${id}/preflight_receipt.txt")"
    mkdir "${point}"
    export DESIGN_NAME=m519_fc2_registered_release_matched_8bank_raw4_acc24
    export RTL_FILELIST="${m519_r6_hw_root}/${m519_r6_filelist}"
    export OUTPUT_DIR="${point}"
    export ELAB_PARAMETERS="ARCH_MODE=${mode}"
    m519_r6_child_pid=""
    m519_r6_monitor_pid=""
    m519_r6_child_rc=running
    m519_r6_monitor_rc=running
    set +e
    "${m519_r6_dc}" -f "${m519_r6_hw_root}/${m519_r6_tcl}" \
        >"${point}/dc.log" 2>&1 &
    m519_r6_child_pid=$!
    printf 'timestamp=%s axis=%s child_pid=%s runner_pid=%s h0_commit_headroom_kib=%s\n' \
        "$(date --iso-8601=seconds)" "${id}" "${m519_r6_child_pid}" \
        "$$" "${h0}" >"${point}/launch_pid_tree_root.txt"
    m519_r6_runtime_monitor "${m519_r6_child_pid}" "${h0}" "${point}" &
    m519_r6_monitor_pid=$!
    wait "${m519_r6_child_pid}"
    m519_r6_child_rc=$?
    wait "${m519_r6_monitor_pid}"
    m519_r6_monitor_rc=$?
    set -e
    echo "${m519_r6_child_rc}" >"${point}/dc.rc"
    echo "${m519_r6_monitor_rc}" >"${point}/runtime_monitor.rc"
    m519_r6_child_pid=""
    m519_r6_monitor_pid=""

    [[ "${m519_r6_signal}" == none ]] || return 130
    [[ "${m519_r6_monitor_rc}" -eq 0 ]] || {
        m519_r6_runtime_latch=1
        m519_r6_runtime_latch_reason="$(awk -F= '/^reason=/ {print $2}' \
            "${point}/resource_runtime.log" | tail -1)"
        return 42
    }
    [[ "${m519_r6_child_rc}" -eq 0 ]] || return "${m519_r6_child_rc}"
    [[ -s "${point}/TCL_PASS_TERMINAL.txt" ]] || return 43
    grep -Fxq 'status=PASS_M519_R6_SETUP_AREA_DC_TCL_TERMINAL' \
        "${point}/TCL_PASS_TERMINAL.txt" || return 43
    grep -Fxq 'compile_ultra_count=1' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'incremental_compile_count=0' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'hold_optimization_count=0' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'hold_not_closed_at_dc=true' "${point}/TCL_PASS_TERMINAL.txt"
    [[ ! -e "${point}/TCL_EXPLICIT_FAILURE.txt" ]] || return 43
    grep -Fxq 'TIM-209=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'OPT-150=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
        "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'flow=m519_r6_setup_area_only' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'compile_ultra_count=1' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'incremental_compile_count=0' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'hold_fix_command_count=0' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'hold_only_optimization_count=0' "${point}/reports/flow_contract.rpt"
    ! grep -Eq '^(Warning|Information):.*\((TIM-209|OPT-150)\)|^Error:|^Fatal:' \
        "${point}/dc.log" || return 44
    for report in area.rpt qor.rpt timing_setup.rpt \
            timing_hold_diagnostic.rpt constraint_setup.rpt \
            constraint_hold_diagnostic.rpt constraint_max_capacitance.rpt \
            constraint_max_transition.rpt constraint_max_fanout.rpt \
            check_design_postcompile.rpt check_timing_postcompile.rpt \
            flow_contract.rpt compile_receipt.rpt; do
        [[ -s "${point}/reports/${report}" ]] || return 45
    done
    [[ -s "${point}/netlist/m519_fc2_registered_release_matched_8bank_raw4_acc24_mapped.v" ]] \
        || return 45
    ! grep -Fq 'slack (VIOLATED)' "${point}/reports/timing_setup.rpt" \
        || return 46
    for setup_area_constraint in constraint_setup.rpt \
            constraint_max_capacitance.rpt constraint_max_transition.rpt \
            constraint_max_fanout.rpt; do
        grep -Fq 'This design has no violated constraints.' \
            "${point}/reports/${setup_area_constraint}" || return 46
    done
    printf 'status=PASS_M519_R6_%s_SETUP_AREA_LOGIC_ONLY_DC_3NS_PENDING_RECEIPT_REVIEW\nmacro_count=0\nhold_not_closed_at_dc=true\npaper_ppa_ready=false\nsystem_speedup=false\nheadline=false\n' \
        "${id^^}" >"${point}/RUN_COMPLETE.txt"
}

m519_r6_run_point k1 0
m519_r6_axis_preflight k8 "${m519_r6_work}/preflight/k8" || exit 40
m519_r6_run_point k8 1
m519_r6_axis_preflight k1x8 "${m519_r6_work}/preflight/k1x8" || exit 40
m519_r6_run_point k1x8 2
m519_r6_axis_preflight post_k1x8_recovery \
    "${m519_r6_work}/preflight/post_k1x8_recovery" || exit 40

printf 'status=PASS_M519_R6_THREE_AXIS_SETUP_AREA_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RECEIPT_REVIEW\nhold_not_closed_at_dc=true\npaper_ppa_ready=false\nsystem_speedup=false\nheadline=false\n' \
    >"${m519_r6_work}/RUN_COMPLETE.txt"
m519_r6_seal_dir "${m519_r6_work}"
mv -T "${m519_r6_work}" "${m519_r6_canonical}"
m519_r6_run_created=0
m519_r6_complete=1
trap - EXIT INT TERM
echo "PASS M519 R6 raw setup/area DC result sealed at ${m519_r6_canonical}"
