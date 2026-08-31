#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M925 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m925_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m925_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m925_repo_root="$(cd "${m925_hw_root}/.." && pwd)"
m925_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m925_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m925_setsid=/usr/bin/setsid
m925_setsid_sha=827259531e3511bcc704143690d8a3afec043d24a7922bf3ebfacf917cd7e100
m925_runner_uid="$(id -u)"
m925_driver="${m925_hw_root}/system_simulator/scripts/execute_m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2.py"
m925_contract="${m925_hw_root}/contracts/m925_m896_decoder_run_gtls_full_first_row_exact_scalability_source_contract_r1_20260829.json"
m925_release="${m925_hw_root}/contracts/m927_m925_decoder_run_gtls_full_first_row_exact_scalability_release_r1_20260829.json"
m925_result="${m925_hw_root}/results/m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_20260829"
m925_attempt="${m925_hw_root}/results/.m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_attempt_consumed"
m925_attempt_stage="${m925_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m925_stage="${m925_result}.stage.$$.${RANDOM}.${RANDOM}"
m925_quarantine="${m925_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m925_partial="${m925_quarantine}.partial_artifact"
m925_stdout="${m925_result}.worker_stdout.$$.${RANDOM}.${RANDOM}.log"
m925_stderr="${m925_result}.worker_stderr.$$.${RANDOM}.${RANDOM}.log"
m925_snapshots="${m925_result}.runtime_resource_snapshots.$$.${RANDOM}.${RANDOM}.tsv"
m925_worker_identity="${m925_result}.worker_identity.$$.${RANDOM}.${RANDOM}.txt"
m925_drain_receipt="${m925_result}.job_tree_drain.$$.${RANDOM}.${RANDOM}.txt"
m925_started=0
m925_success=0
m925_phase=SOURCE_ONLY_PREFLIGHT
m925_signal=none
m925_worker_pid=
m925_worker_start=
m925_worker_uid=
m925_worker_parent=
m925_worker_pgrp=
m925_worker_session=
m925_worker_exe=
m925_worker_cmdhex=
m925_worker_captured=0
m925_worker_rc=not_started
m925_tree_drained=0
m925_monitor_reason=none

m925_sha() { sha256sum "$1" | awk '{print $1}'; }

m925_driver_env() {
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        PYTHONDONTWRITEBYTECODE=1 "${m925_python}" "${m925_driver}" "$@"
}

m925_resources() {
    local free_kib mem_available commit_limit committed commit_headroom
    free_kib="$(df -Pk "$(dirname "${m925_result}")" | awk 'NR==2 {print $4}')"
    mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    commit_headroom=$((commit_limit - committed))
    printf '%s\t%s\t%s\t%s\n' "${free_kib}" "${mem_available}" \
        "${commit_headroom}" "${committed}"
}

m925_resource_gate() {
    local row free_kib mem_available commit_headroom committed
    row="$(m925_resources)"
    IFS=$'\t' read -r free_kib mem_available commit_headroom committed <<<"${row}"
    [[ "${free_kib}" -ge 2097152 && \
       "${mem_available}" -ge 100663296 && \
       "${commit_headroom}" -ge 100663296 ]] || {
        echo "M925 launch requires 2 GiB disk and 96 GiB memory/commit headroom" >&2
        return 40
    }
}

m925_empty_log() {
    local path=$1
    if [[ ! -e "${path}" && ! -L "${path}" ]]; then
        (umask 077; : >"${path}")
    fi
    [[ -f "${path}" && ! -L "${path}" ]]
}

m925_proc_identity() {
    local pid=$1 stat rest
    [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/status" ]] || return 1
    stat="$(cat "/proc/${pid}/stat")" || return 1
    rest=${stat##*) }; set -- ${rest}; [[ $# -ge 20 ]] || return 1
    M925_P_STATE=$1; M925_P_PPID=$2; M925_P_PGRP=$3; M925_P_SESSION=$4
    M925_P_START=${20}
    M925_P_UID="$(awk '/^Uid:/ {print $2; exit}' "/proc/${pid}/status")"
    M925_P_EXE="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    M925_P_CMDHEX="$(od -An -tx1 -v "/proc/${pid}/cmdline" 2>/dev/null | tr -d ' \n')"
    M925_P_RSS="$(awk '/^VmRSS:/ {print $2; exit}' "/proc/${pid}/status")"
    M925_P_RSS=${M925_P_RSS:-0}
    [[ -n "${M925_P_UID}" && -n "${M925_P_EXE}" ]]
}

m925_root_running() {
    local pid=$1
    m925_proc_identity "${pid}" || return 1
    [[ "${M925_P_START}" == "${m925_worker_start}" && \
       "${M925_P_UID}" == "${m925_worker_uid}" && \
       "${M925_P_PGRP}" == "${m925_worker_pgrp}" && \
       "${M925_P_SESSION}" == "${m925_worker_session}" && \
       "${M925_P_STATE}" != Z ]]
}

m925_job_members() {
    local proc pid
    [[ -n "${m925_worker_pgrp}" && -n "${m925_worker_session}" && \
       -n "${m925_worker_uid}" && -n "${m925_worker_start}" ]] || return 0
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}; m925_proc_identity "${pid}" || continue
        [[ "${M925_P_UID}" == "${m925_worker_uid}" && \
           "${M925_P_PGRP}" == "${m925_worker_pgrp}" && \
           "${M925_P_SESSION}" == "${m925_worker_session}" && \
           "${M925_P_STATE}" != Z && \
           "${M925_P_START}" -ge "${m925_worker_start}" ]] || continue
        printf '%s:%s\n' "${pid}" "${M925_P_START}"
    done
}

m925_group_rss_kib() {
    local proc pid total=0
    [[ -n "${m925_worker_pgrp}" ]] || { printf '0\n'; return; }
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}; m925_proc_identity "${pid}" || continue
        [[ "${M925_P_UID}" == "${m925_worker_uid}" && \
           "${M925_P_PGRP}" == "${m925_worker_pgrp}" && \
           "${M925_P_SESSION}" == "${m925_worker_session}" && \
           "${M925_P_STATE}" != Z && \
           "${M925_P_START}" -ge "${m925_worker_start}" ]] || continue
        total=$((total + M925_P_RSS))
    done
    printf '%s\n' "${total}"
}

m925_wait_group_empty() {
    local loops=${1:-100} i
    for i in $(seq 1 "${loops}"); do
        [[ -z "$(m925_job_members)" ]] && return 0
        sleep 0.1
    done
    return 1
}

m925_reap_root() {
    if [[ -n "${m925_worker_pid}" ]]; then
        set +e
        wait "${m925_worker_pid}"
        m925_worker_rc=$?
        set -e
        m925_worker_pid=
    fi
}

m925_drain_job() {
    local reason=${1:-unspecified}
    [[ "${m925_tree_drained}" -eq 0 ]] || return 0
    if [[ -n "${m925_worker_pgrp}" && -n "$(m925_job_members)" ]]; then
        kill -TERM -- "-${m925_worker_pgrp}" 2>/dev/null || true
        if ! m925_wait_group_empty 100; then
            kill -KILL -- "-${m925_worker_pgrp}" 2>/dev/null || true
            m925_wait_group_empty 100 || return 1
        fi
    fi
    m925_reap_root
    m925_wait_group_empty 20 || return 1
    (umask 077; printf 'status=PASS_JOB_TREE_DRAINED_BEFORE_RENAME_OR_SEAL\nreason=%s\nworker_start=%s\nworker_pgrp=%s\nworker_session=%s\nworker_rc=%s\nprocess_group_empty=true\nroot_reaped=true\n' \
        "${reason}" "${m925_worker_start:-none}" "${m925_worker_pgrp:-none}" \
        "${m925_worker_session:-none}" "${m925_worker_rc}" >"${m925_drain_receipt}")
    m925_tree_drained=1
}

m925_fail_closed() {
    local rc=$?
    trap - EXIT TERM INT HUP
    set +e
    if [[ "${m925_started}" -eq 1 && "${m925_success}" -ne 1 ]]; then
        [[ "${rc}" -ne 0 ]] || rc=98
        m925_drain_job "failure_or_signal_${m925_signal}_${m925_monitor_reason}" || exit 99
        [[ "${m925_tree_drained}" -eq 1 ]] || exit 99
        local partial=""
        if [[ -e "${m925_stage}" || -L "${m925_stage}" ]]; then
            mv -T --no-clobber -- "${m925_stage}" "${m925_partial}" || exit 99
            partial="${m925_partial}"
        fi
        m925_empty_log "${m925_stdout}" || exit 99
        m925_empty_log "${m925_stderr}" || exit 99
        m925_empty_log "${m925_snapshots}" || exit 99
        m925_empty_log "${m925_worker_identity}" || exit 99
        [[ -f "${m925_drain_receipt}" && ! -L "${m925_drain_receipt}" ]] || exit 99
        m925_driver_env --write-failure-receipt \
            --release "${m925_release}" --runner "${m925_runner}" \
            --expected-release-sha256 "${M925_EXPECTED_RELEASE_SHA256}" \
            --expected-runner-sha256 "${M925_EXPECTED_RUNNER_SHA256}" \
            --hammer-review-sha256 "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
            --hammer-outer-sha256 "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
            --stdout-log "${m925_stdout}" --stderr-log "${m925_stderr}" \
            --snapshot-log "${m925_snapshots}" --worker-identity "${m925_worker_identity}" \
            --drain-receipt "${m925_drain_receipt}" \
            --output "${m925_quarantine}" --return-code "${rc}" \
            --phase "${m925_phase}" --partial-artifact "${partial}" >/dev/null || exit 99
        rm -f -- "${m925_stdout}" "${m925_stderr}" "${m925_snapshots}" \
            "${m925_worker_identity}" "${m925_drain_receipt}"
    fi
    exit "${rc}"
}

trap m925_fail_closed EXIT
trap 'm925_signal=HUP; exit 129' HUP
trap 'm925_signal=INT; exit 130' INT
trap 'm925_signal=TERM; exit 143' TERM

[[ "${m925_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m925_runner}" == "${m925_hw_root}/system_simulator/scripts/run_m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_one_shot.sh" ]] || {
    echo "M925 canonical path drift" >&2
    exit 3
}
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--dry-run-no-work" ) ]] || {
    echo "M925 accepts no arguments or --dry-run-no-work only" >&2
    exit 3
}
[[ -n "${M925_EXPECTED_RUNNER_SHA256:-}" && \
   "${M925_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(m925_sha "${m925_runner}")" == "${M925_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M925 caller must pin reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M925_EXPECTED_CONTRACT_SHA256:-}" && \
   "${M925_EXPECTED_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m925_contract}" && ! -L "${m925_contract}" && \
   "$(m925_sha "${m925_contract}")" == "${M925_EXPECTED_CONTRACT_SHA256}" ]] || {
    echo "M925 caller must pin reviewed source contract SHA" >&2
    exit 3
}
[[ -x "${m925_python}" && ! -L "${m925_python}" && \
   "$(m925_sha "${m925_python}")" == "${m925_python_sha}" && \
   "$(${m925_python} -c 'import platform; print(platform.python_version())')" == 3.10.18 && \
   -x "${m925_setsid}" && ! -L "${m925_setsid}" && \
   "$(m925_sha "${m925_setsid}")" == "${m925_setsid_sha}" ]] || {
    echo "M925 Python/setsid identity drift" >&2
    exit 4
}
[[ -f "${m925_driver}" && ! -L "${m925_driver}" ]] || exit 4
m925_driver_env --validate-source-contract --source-contract "${m925_contract}" >/dev/null

if [[ "$#" -eq 1 ]]; then
    m925_driver_env --dry-run-no-work --source-contract "${m925_contract}"
    m925_resource_gate
    trap - EXIT TERM INT HUP
    echo "PASS_M925_NO_WORK_DRY_RUN__NO_FILES_NO_ATTEMPT"
    exit 0
fi

[[ -n "${M925_EXPECTED_RELEASE_SHA256:-}" && \
   "${M925_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -n "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256:-}" && \
   "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -n "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256:-}" && \
   "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "M925 formal diagnostic requires future M927/M928 caller pins" >&2
    exit 3
}

m925_driver_env --validate-formal-preflight --release "${m925_release}" \
    --runner "${m925_runner}" \
    --expected-release-sha256 "${M925_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M925_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" >/dev/null
m925_resource_gate

m925_phase=CONSUME_FRESH_R2_ATTEMPT
m925_driver_env --consume-attempt --release "${m925_release}" \
    --runner "${m925_runner}" \
    --expected-release-sha256 "${M925_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M925_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
    --stage-basename "$(basename "${m925_attempt_stage}")" >/dev/null
m925_started=1

m925_empty_log "${m925_stdout}"
m925_empty_log "${m925_stderr}"
m925_empty_log "${m925_snapshots}"
printf 'epoch_s\telapsed_s\tfree_kib\tmem_available_kib\tcommit_headroom_kib\tcommitted_as_kib\tprocess_group_rss_kib\tgroup_member_count\theartbeat_present\tover_timeout\tover_resource\n' >"${m925_snapshots}"

m925_phase=RUN_ONE_R2_EXACT_SCALABILITY_DIAGNOSTIC
m925_t0="$(date +%s%N)"
set +e
"${m925_setsid}" --wait /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
    "${m925_python}" "${m925_driver}" --run-full-first-row \
    --release "${m925_release}" --runner "${m925_runner}" \
    --expected-release-sha256 "${M925_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M925_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
    --output "${m925_stage}" >>"${m925_stdout}" 2>>"${m925_stderr}" &
m925_worker_pid=$!
set -e

for m925_capture_try in $(seq 1 400); do
    if m925_proc_identity "${m925_worker_pid}"; then
        if [[ -z "${m925_worker_start}" && \
              "${M925_P_UID}" == "${m925_runner_uid}" && \
              "${M925_P_PGRP}" == "${m925_worker_pid}" && \
              "${M925_P_SESSION}" == "${m925_worker_pid}" && \
              "${M925_P_PPID}" == "$$" ]]; then
            m925_worker_start=${M925_P_START}
            m925_worker_uid=${M925_P_UID}
            m925_worker_parent=${M925_P_PPID}
            m925_worker_pgrp=${M925_P_PGRP}
            m925_worker_session=${M925_P_SESSION}
        fi
        if [[ -n "${m925_worker_start}" && \
              "${M925_P_START}" == "${m925_worker_start}" && \
              "${M925_P_UID}" == "${m925_worker_uid}" && \
              "${M925_P_PGRP}" == "${m925_worker_pgrp}" && \
              "${M925_P_SESSION}" == "${m925_worker_session}" && \
              "${M925_P_PPID}" == "${m925_worker_parent}" && \
              "${M925_P_EXE}" == "${m925_python}" ]]; then
            m925_worker_exe=${M925_P_EXE}
            m925_worker_cmdhex=${M925_P_CMDHEX}
            m925_worker_captured=1
            break
        fi
    fi
    sleep 0.01
done
[[ "${m925_worker_captured}" -eq 1 ]] || {
    m925_monitor_reason=worker_identity_capture_failed
    exit 41
}
(umask 077; printf 'pid=%s\nstarttime=%s\nuid=%s\nparent=%s\npgrp=%s\nsession=%s\nexe=%s\ncmdline_nul_hex=%s\nsetsid_wait_private_group=true\nresource_observation_scope=actual_worker_process_group\n' \
    "${m925_worker_pid}" "${m925_worker_start}" "${m925_worker_uid}" \
    "${m925_worker_parent}" "${m925_worker_pgrp}" "${m925_worker_session}" \
    "${m925_worker_exe}" "${m925_worker_cmdhex}" >"${m925_worker_identity}")

m925_over_timeout=0
m925_over_resource=0
m925_forced_rc=0
while m925_root_running "${m925_worker_pid}"; do
    sleep 1
    m925_now="$(date +%s%N)"
    m925_elapsed_ms=$(((m925_now - m925_t0) / 1000000))
    m925_row="$(m925_resources)"
    IFS=$'\t' read -r m925_free m925_mem m925_commit m925_committed <<<"${m925_row}"
    m925_group_rss="$(m925_group_rss_kib)"
    m925_members="$(m925_job_members | wc -l)"
    m925_heartbeat=no
    [[ -f "${m925_stage}/runtime_heartbeat.json" && \
       ! -L "${m925_stage}/runtime_heartbeat.json" ]] && m925_heartbeat=yes
    if [[ "${m925_elapsed_ms}" -gt 2715000 ]]; then
        m925_over_timeout=$((m925_over_timeout + 1))
    else
        m925_over_timeout=0
    fi
    if [[ "${m925_mem}" -lt 8388608 || "${m925_commit}" -lt 8388608 || \
          "${m925_free}" -lt 1048576 ]]; then
        m925_over_resource=$((m925_over_resource + 1))
    else
        m925_over_resource=0
    fi
    printf '%s\t%s.%03d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$((m925_now / 1000000000))" "$((m925_elapsed_ms / 1000))" \
        "$((m925_elapsed_ms % 1000))" "${m925_free}" "${m925_mem}" \
        "${m925_commit}" "${m925_committed}" "${m925_group_rss}" \
        "${m925_members}" "${m925_heartbeat}" "${m925_over_timeout}" \
        "${m925_over_resource}" >>"${m925_snapshots}"
    if [[ "${m925_over_timeout}" -ge 3 ]]; then
        m925_monitor_reason=operational_safety_timeout_three_consecutive
        m925_forced_rc=124
        m925_drain_job "${m925_monitor_reason}" || exit 99
        break
    fi
    if [[ "${m925_over_resource}" -ge 3 ]]; then
        m925_monitor_reason=emergency_resource_floor_three_consecutive
        m925_forced_rc=125
        m925_drain_job "${m925_monitor_reason}" || exit 99
        break
    fi
done

if [[ "${m925_tree_drained}" -eq 0 ]]; then
    m925_reap_root
    if ! m925_wait_group_empty 30; then
        m925_monitor_reason=descendant_linger_after_normal_root_exit
        m925_forced_rc=126
        m925_drain_job "${m925_monitor_reason}" || exit 99
    else
        (umask 077; printf 'status=PASS_JOB_TREE_DRAINED_BEFORE_RENAME_OR_SEAL\nreason=normal_worker_exit\nworker_start=%s\nworker_pgrp=%s\nworker_session=%s\nworker_rc=%s\nprocess_group_empty=true\nroot_reaped=true\n' \
            "${m925_worker_start}" "${m925_worker_pgrp}" "${m925_worker_session}" \
            "${m925_worker_rc}" >"${m925_drain_receipt}")
        m925_tree_drained=1
    fi
fi
[[ "${m925_forced_rc}" -eq 0 ]] || exit "${m925_forced_rc}"
[[ "${m925_worker_rc}" -eq 0 ]] || exit "${m925_worker_rc}"
[[ "${m925_tree_drained}" -eq 1 && -z "$(m925_job_members)" ]] || exit 99

m925_phase=SEAL_AND_PUBLISH_R2_DIAGNOSTIC_AFTER_REAP
[[ -d "${m925_stage}" && ! -L "${m925_stage}" && \
   -f "${m925_stage}/diagnostic.json" && \
   -f "${m925_stage}/runtime_heartbeat.json" ]] || exit 50
mv -T --no-clobber -- "${m925_worker_identity}" \
    "${m925_stage}/worker_identity.txt"
cp --no-clobber -- "${m925_snapshots}" \
    "${m925_stage}/runtime_resource_snapshots.tsv"
cp --no-clobber -- "${m925_drain_receipt}" \
    "${m925_stage}/job_tree_drain_receipt.txt"
(cd "${m925_stage}" && \
    sha256sum diagnostic.json runtime_heartbeat.json \
        runtime_resource_snapshots.tsv worker_identity.txt \
        job_tree_drain_receipt.txt >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m925_driver_env --publish-no-replace --release "${m925_release}" \
    --runner "${m925_runner}" \
    --expected-release-sha256 "${M925_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M925_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
    --output "${m925_stage}" --publish-to "${m925_result}" >/dev/null
m925_success=1
trap - EXIT TERM INT HUP
rm -f -- "${m925_stdout}" "${m925_stderr}" "${m925_snapshots}" \
    "${m925_worker_identity}" "${m925_drain_receipt}"
echo "PASS_M925_ONE_R2_EXACT_SCALABILITY_DIAGNOSTIC__FRESH_RESULT_HAMMER_REQUIRED"
