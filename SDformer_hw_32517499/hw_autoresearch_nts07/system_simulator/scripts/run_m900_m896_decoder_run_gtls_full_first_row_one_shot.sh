#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M900 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m900_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m900_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m900_repo_root="$(cd "${m900_hw_root}/.." && pwd)"
m900_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m900_python_sha="9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
m900_driver="${m900_hw_root}/system_simulator/scripts/execute_m900_m896_decoder_run_gtls_full_first_row_runtime_gate.py"
m900_release="${m900_hw_root}/contracts/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_release_r1_20260829.json"
m900_result="${m900_hw_root}/results/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829"
m900_attempt="${m900_hw_root}/results/.m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_attempt_consumed"
m900_attempt_stage="${m900_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m900_stage="${m900_result}.stage.$$.${RANDOM}.${RANDOM}"
m900_quarantine="${m900_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m900_partial="${m900_quarantine}.partial_artifact"
m900_stdout="${m900_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m900_stderr="${m900_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m900_snapshots="${m900_result}.runtime_resource_snapshots.$$.${RANDOM}.${RANDOM}.tsv"
m900_started=0
m900_published=0
m900_success=0
m900_phase="PRE_ATTEMPT"

m900_driver_env() {
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        "${m900_python}" "${m900_driver}" "$@"
}

m900_resources() {
    local free_kib mem_available commit_limit committed commit_headroom
    free_kib="$(df -Pk "$(dirname "${m900_result}")" | awk 'NR==2 {print $4}')"
    mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    commit_headroom=$((commit_limit - committed))
    printf '%s\t%s\t%s\t%s\n' "${free_kib}" "${mem_available}" \
        "${commit_headroom}" "${committed}"
}

m900_resource_gate() {
    local row free_kib mem_available commit_headroom committed
    row="$(m900_resources)"
    IFS=$'\t' read -r free_kib mem_available commit_headroom committed <<<"${row}"
    [[ "${free_kib}" -ge 2097152 && \
       "${mem_available}" -ge 100663296 && \
       "${commit_headroom}" -ge 100663296 ]] || {
        echo "M900 launch requires 2 GiB disk and 96 GiB memory/commit headroom" >&2
        return 40
    }
}

m900_empty_log() {
    local path="$1"
    if [[ ! -e "${path}" && ! -L "${path}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m900_python}" - "${path}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${path}" && ! -L "${path}" ]]
}

m900_fail_closed() {
    local rc=$?
    trap - EXIT
    if [[ "${m900_started}" -eq 1 && "${m900_success}" -ne 1 ]]; then
        [[ "${rc}" -ne 0 ]] || rc=98
        local partial=""
        if [[ "${m900_published}" -eq 1 && \
              ( -e "${m900_result}" || -L "${m900_result}" ) ]]; then
            mv -T --no-clobber -- "${m900_result}" "${m900_partial}" || exit 99
            partial="${m900_partial}"
        elif [[ -e "${m900_stage}" || -L "${m900_stage}" ]]; then
            mv -T --no-clobber -- "${m900_stage}" "${m900_partial}" || exit 99
            partial="${m900_partial}"
        fi
        m900_empty_log "${m900_stdout}" || exit 99
        m900_empty_log "${m900_stderr}" || exit 99
        m900_empty_log "${m900_snapshots}" || exit 99
        m900_driver_env --write-failure-receipt \
            --release "${m900_release}" --runner "${m900_runner}" \
            --expected-release-sha256 "${M900_EXPECTED_RELEASE_SHA256}" \
            --expected-runner-sha256 "${M900_EXPECTED_RUNNER_SHA256}" \
            --hammer-review-sha256 "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
            --hammer-outer-sha256 "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
            --stdout-log "${m900_stdout}" --stderr-log "${m900_stderr}" \
            --snapshot-log "${m900_snapshots}" --output "${m900_quarantine}" \
            --return-code "${rc}" --phase "${m900_phase}" \
            --partial-artifact "${partial}" >/dev/null || exit 99
        rm -f -- "${m900_stdout}" "${m900_stderr}" "${m900_snapshots}"
    fi
    exit "${rc}"
}
trap m900_fail_closed EXIT

[[ "${m900_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m900_runner}" == "${m900_hw_root}/system_simulator/scripts/run_m900_m896_decoder_run_gtls_full_first_row_one_shot.sh" ]] || {
    echo "M900 canonical path drift" >&2
    exit 3
}
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--dry-run-no-work" ) ]] || {
    echo "M900 accepts no arguments or --dry-run-no-work only" >&2
    exit 3
}
[[ -n "${M900_EXPECTED_RUNNER_SHA256:-}" && \
   "${M900_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m900_runner}" | awk '{print $1}')" == \
   "${M900_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M900 caller must pin the reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M900_EXPECTED_RELEASE_SHA256:-}" && \
   "${M900_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m900_release}" && ! -L "${m900_release}" && \
   "$(sha256sum "${m900_release}" | awk '{print $1}')" == \
   "${M900_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M900 caller must pin the reviewed release SHA" >&2
    exit 3
}
[[ -x "${m900_python}" && ! -L "${m900_python}" && \
   "$(sha256sum "${m900_python}" | awk '{print $1}')" == "${m900_python_sha}" && \
   "$("${m900_python}" -c 'import platform; print(platform.python_version())')" == "3.10.18" ]] || {
    echo "M900 Python-3.10 identity drift" >&2
    exit 4
}
[[ -f "${m900_driver}" && ! -L "${m900_driver}" ]] || {
    echo "M900 driver absent or nonregular" >&2
    exit 4
}

if [[ "$#" -eq 1 ]]; then
    m900_driver_env --dry-run-no-work --release "${m900_release}"
    m900_resource_gate
    trap - EXIT
    echo "PASS_M900_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT"
    exit 0
fi

[[ -n "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256:-}" && \
   "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -n "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256:-}" && \
   "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "M900 formal run requires final-hammer caller pins" >&2
    exit 3
}

m900_driver_env --validate-formal-preflight --release "${m900_release}" \
    --expected-release-sha256 "${M900_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M900_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" >/dev/null
m900_resource_gate

m900_phase="CONSUME_ONE_WAY_ATTEMPT"
m900_driver_env --consume-attempt --release "${m900_release}" \
    --runner "${m900_runner}" \
    --expected-release-sha256 "${M900_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M900_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
    --stage-basename "$(basename "${m900_attempt_stage}")" >/dev/null
m900_started=1

m900_empty_log "${m900_stdout}"
m900_empty_log "${m900_stderr}"
m900_empty_log "${m900_snapshots}"
printf 'epoch_s\telapsed_s\tfree_kib\tmem_available_kib\tcommit_headroom_kib\tcommitted_as_kib\tchild_rss_kib\theartbeat_phase\tcounted_state_bytes\tover_runtime\tover_resource\tover_counted_state\n' >"${m900_snapshots}"

m900_phase="RUN_ONE_FULL_ROW_WITH_RUNTIME_MONITOR"
set +e
m900_driver_env --run-full-first-row --release "${m900_release}" \
    --runner "${m900_runner}" \
    --expected-release-sha256 "${M900_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M900_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
    --output "${m900_stage}" >>"${m900_stdout}" 2>>"${m900_stderr}" &
m900_child=$!
m900_t0="$(date +%s%N)"
m900_over_runtime=0
m900_over_resource=0
m900_over_state=0
m900_monitor_killed=0
while kill -0 "${m900_child}" 2>/dev/null; do
    sleep 1
    m900_now="$(date +%s%N)"
    m900_elapsed_ms=$(((m900_now - m900_t0) / 1000000))
    m900_row="$(m900_resources)"
    IFS=$'\t' read -r m900_free m900_mem m900_commit m900_committed <<<"${m900_row}"
    m900_rss=0
    if [[ -r "/proc/${m900_child}/status" ]]; then
        m900_rss="$(awk '/^VmRSS:/ {print $2}' "/proc/${m900_child}/status")"
        m900_rss="${m900_rss:-0}"
    fi
    m900_hb_phase="ABSENT"
    m900_counted="NA"
    if [[ -f "${m900_stage}/runtime_heartbeat.json" && \
          ! -L "${m900_stage}/runtime_heartbeat.json" ]]; then
        m900_hb="$(${m900_python} - "${m900_stage}/runtime_heartbeat.json" <<'PY' 2>/dev/null
import json, pathlib, sys
row = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(str(row.get("phase", "INVALID")) + "\t" +
      ("NA" if row.get("counted_live_scheduler_state_bytes") is None
       else str(int(row["counted_live_scheduler_state_bytes"]))))
PY
)"
        if [[ -n "${m900_hb}" ]]; then
            IFS=$'\t' read -r m900_hb_phase m900_counted <<<"${m900_hb}"
        fi
    fi
    if [[ "${m900_elapsed_ms}" -gt 9321 ]]; then
        m900_over_runtime=$((m900_over_runtime + 1))
    else
        m900_over_runtime=0
    fi
    if [[ "${m900_mem}" -lt 8388608 || "${m900_commit}" -lt 8388608 || \
          "${m900_free}" -lt 1048576 ]]; then
        m900_over_resource=$((m900_over_resource + 1))
    else
        m900_over_resource=0
    fi
    if [[ "${m900_counted}" =~ ^[0-9]+$ && \
          "${m900_counted}" -gt 536870912 ]]; then
        m900_over_state=$((m900_over_state + 1))
    else
        m900_over_state=0
    fi
    printf '%s\t%s.%03d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$((m900_now / 1000000000))" "$((m900_elapsed_ms / 1000))" \
        "$((m900_elapsed_ms % 1000))" "${m900_free}" "${m900_mem}" \
        "${m900_commit}" "${m900_committed}" "${m900_rss}" \
        "${m900_hb_phase}" "${m900_counted}" "${m900_over_runtime}" \
        "${m900_over_resource}" "${m900_over_state}" >>"${m900_snapshots}"
    if [[ "${m900_over_runtime}" -ge 3 || "${m900_over_resource}" -ge 3 || \
          "${m900_over_state}" -ge 3 ]]; then
        kill -TERM "${m900_child}" 2>/dev/null || true
        m900_monitor_killed=1
        break
    fi
done
wait "${m900_child}"
m900_driver_rc=$?
set -e
if [[ "${m900_monitor_killed}" -eq 1 && "${m900_driver_rc}" -eq 0 ]]; then
    m900_driver_rc=124
fi
[[ "${m900_driver_rc}" -eq 0 ]] || exit "${m900_driver_rc}"

m900_phase="SEAL_AND_PUBLISH_DIAGNOSTIC"
[[ -d "${m900_stage}" && ! -L "${m900_stage}" && \
   -f "${m900_stage}/diagnostic.json" && \
   -f "${m900_stage}/runtime_heartbeat.json" ]] || exit 50
cp --no-clobber -- "${m900_snapshots}" \
    "${m900_stage}/runtime_resource_snapshots.tsv"
(cd "${m900_stage}" && \
    sha256sum diagnostic.json runtime_heartbeat.json \
        runtime_resource_snapshots.tsv >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m900_driver_env --publish-no-replace --release "${m900_release}" \
    --runner "${m900_runner}" \
    --expected-release-sha256 "${M900_EXPECTED_RELEASE_SHA256}" \
    --expected-runner-sha256 "${M900_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M900_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M900_EXPECTED_FINAL_HAMMER_OUTER_SHA256}" \
    --output "${m900_stage}" --publish-to "${m900_result}" >/dev/null
m900_published=1
m900_success=1
trap - EXIT
rm -f -- "${m900_stdout}" "${m900_stderr}" "${m900_snapshots}"
echo "PASS_M900_ONE_FULL_ROW_RUNTIME_GATE__FRESH_RESULT_HAMMER_REQUIRED__NONCITABLE"
