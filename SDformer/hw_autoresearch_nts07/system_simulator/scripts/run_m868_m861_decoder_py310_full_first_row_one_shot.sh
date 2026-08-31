#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M868 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m868_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m868_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m868_repo_root="$(cd "${m868_hw_root}/.." && pwd)"
m868_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m868_python_sha="9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
m868_driver="${m868_hw_root}/system_simulator/scripts/execute_m868_m861_decoder_py310_full_first_row_diagnostic.py"
m868_candidate="${m868_hw_root}/contracts/m868_m861_decoder_py310_full_first_row_diagnostic_candidate_r1_20260829.json"
m868_result="${m868_hw_root}/results/m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829"
m868_attempt="${m868_hw_root}/results/.m868_m861_decoder_py310_full_first_row_diagnostic_r1_attempt_consumed"
m868_attempt_stage="${m868_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m868_stage="${m868_result}.stage.$$.${RANDOM}.${RANDOM}"
m868_quarantine="${m868_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m868_partial_artifact="${m868_quarantine}.partial_artifact"
m868_stdout_log="${m868_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m868_stderr_log="${m868_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m868_started=0
m868_published=0
m868_success=0
m868_phase="PRE_ATTEMPT"

m868_driver_env() {
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        "${m868_python}" "${m868_driver}" "$@"
}

m868_resource_gate() {
    local free_kib mem_available commit_limit committed commit_headroom
    free_kib="$(df -Pk "$(dirname "${m868_result}")" | awk 'NR==2 {print $4}')"
    mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    commit_headroom=$((commit_limit - committed))
    [[ "${free_kib}" -ge 2097152 && \
       "${mem_available}" -ge 100663296 && \
       "${commit_headroom}" -ge 100663296 ]] || {
        echo "M868 resource gate requires 2 GiB disk and 96 GiB memory/commit headroom" >&2
        return 40
    }
}

m868_ensure_empty_regular_log() {
    local path="$1"
    if [[ ! -e "${path}" && ! -L "${path}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m868_python}" - "${path}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${path}" && ! -L "${path}" ]]
}

m868_fail_closed() {
    local rc=$?
    trap - EXIT
    if [[ "${m868_started}" -eq 1 && "${m868_success}" -ne 1 ]]; then
        [[ "${rc}" -ne 0 ]] || rc=98
        local partial=""
        if [[ "${m868_published}" -eq 1 && \
              ( -e "${m868_result}" || -L "${m868_result}" ) ]]; then
            mv -T --no-clobber -- "${m868_result}" \
                "${m868_partial_artifact}" || exit 99
            partial="${m868_partial_artifact}"
        elif [[ -e "${m868_stage}" || -L "${m868_stage}" ]]; then
            mv -T --no-clobber -- "${m868_stage}" \
                "${m868_partial_artifact}" || exit 99
            partial="${m868_partial_artifact}"
        fi
        m868_ensure_empty_regular_log "${m868_stdout_log}" || exit 99
        m868_ensure_empty_regular_log "${m868_stderr_log}" || exit 99
        m868_driver_env --write-failure-receipt \
            --candidate "${m868_candidate}" \
            --runner "${m868_runner}" \
            --expected-runner-sha256 "${M868_EXPECTED_RUNNER_SHA256}" \
            --hammer-review-sha256 "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" \
            --hammer-outer-sha256 "${M868_EXPECTED_HAMMER_OUTER_SHA256}" \
            --stdout-log "${m868_stdout_log}" \
            --stderr-log "${m868_stderr_log}" \
            --output "${m868_quarantine}" \
            --return-code "${rc}" --phase "${m868_phase}" \
            --partial-artifact "${partial}" >/dev/null || exit 99
        [[ -d "${m868_quarantine}" && ! -L "${m868_quarantine}" && \
           -f "${m868_quarantine}/failure.json" && \
           -f "${m868_quarantine}/SHA256SUMS" && \
           -f "${m868_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m868_stdout_log}" "${m868_stderr_log}"
    fi
    exit "${rc}"
}
trap m868_fail_closed EXIT

[[ "${m868_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m868_runner}" == \
   "${m868_hw_root}/system_simulator/scripts/run_m868_m861_decoder_py310_full_first_row_one_shot.sh" ]] || {
    echo "M868 canonical path drift" >&2
    exit 3
}
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--dry-run-no-work" ) ]] || {
    echo "M868 accepts no arguments or --dry-run-no-work only" >&2
    exit 3
}
[[ -n "${M868_EXPECTED_RUNNER_SHA256:-}" && \
   "${M868_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m868_runner}" | awk '{print $1}')" == \
   "${M868_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M868 caller must pin the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M868_EXPECTED_CANDIDATE_SHA256:-}" && \
   "${M868_EXPECTED_CANDIDATE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m868_candidate}" && ! -L "${m868_candidate}" && \
   "$(sha256sum "${m868_candidate}" | awk '{print $1}')" == \
   "${M868_EXPECTED_CANDIDATE_SHA256}" ]] || {
    echo "M868 caller must pin the independently reviewed candidate SHA" >&2
    exit 3
}
[[ -x "${m868_python}" && ! -L "${m868_python}" && \
   "$(sha256sum "${m868_python}" | awk '{print $1}')" == \
   "${m868_python_sha}" && \
   "$("${m868_python}" -c 'import platform; print(platform.python_version())')" == \
   "3.10.18" ]] || {
    echo "M868 Python-3.10 interpreter identity drift" >&2
    exit 4
}
[[ -f "${m868_driver}" && ! -L "${m868_driver}" ]] || {
    echo "M868 driver absent or nonregular" >&2
    exit 4
}

if [[ "$#" -eq 1 ]]; then
    m868_driver_env --dry-run-no-work --candidate "${m868_candidate}"
    m868_resource_gate
    trap - EXIT
    echo "PASS_M868_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT"
    exit 0
fi

[[ -n "${M868_EXPECTED_HAMMER_REVIEW_SHA256:-}" && \
   "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -n "${M868_EXPECTED_HAMMER_OUTER_SHA256:-}" && \
   "${M868_EXPECTED_HAMMER_OUTER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "M868 formal run requires independently reviewed hammer pins" >&2
    exit 3
}

m868_driver_env --validate-formal-preflight \
    --candidate "${m868_candidate}" \
    --hammer-review-sha256 "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M868_EXPECTED_HAMMER_OUTER_SHA256}" >/dev/null
m868_resource_gate

m868_phase="CONSUME_ONE_WAY_ATTEMPT"
m868_driver_env --consume-attempt \
    --candidate "${m868_candidate}" \
    --runner "${m868_runner}" \
    --expected-runner-sha256 "${M868_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M868_EXPECTED_HAMMER_OUTER_SHA256}" \
    --stage-basename "$(basename "${m868_attempt_stage}")" >/dev/null
m868_started=1

m868_phase="VALIDATE_CONSUMED_ATTEMPT"
m868_driver_env --validate-attempt \
    --candidate "${m868_candidate}" \
    --runner "${m868_runner}" \
    --expected-runner-sha256 "${M868_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M868_EXPECTED_HAMMER_OUTER_SHA256}" >/dev/null

m868_ensure_empty_regular_log "${m868_stdout_log}"
m868_ensure_empty_regular_log "${m868_stderr_log}"
m868_phase="RUN_EXACT_ONE_FULL_FIRST_ROW_DIAGNOSTIC"
set +e
m868_driver_env --run-full-first-row \
    --candidate "${m868_candidate}" \
    --runner "${m868_runner}" \
    --expected-runner-sha256 "${M868_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M868_EXPECTED_HAMMER_OUTER_SHA256}" \
    --output "${m868_stage}" \
    >>"${m868_stdout_log}" 2>>"${m868_stderr_log}"
m868_driver_rc=$?
set -e
[[ "${m868_driver_rc}" -eq 0 ]] || exit "${m868_driver_rc}"

m868_phase="VERIFY_AND_SEAL_DIAGNOSTIC_STAGE"
[[ -d "${m868_stage}" && ! -L "${m868_stage}" && \
   -f "${m868_stage}/diagnostic.json" && \
   ! -L "${m868_stage}/diagnostic.json" ]] || exit 50
(cd "${m868_stage}" && \
    /usr/bin/sha256sum diagnostic.json >SHA256SUMS && \
    /usr/bin/sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    /usr/bin/sha256sum -c SHA256SUMS >/dev/null && \
    /usr/bin/sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m868_phase="PUBLISH_CANONICAL_NOREPLACE"
m868_driver_env --publish-no-replace \
    --candidate "${m868_candidate}" \
    --runner "${m868_runner}" \
    --expected-runner-sha256 "${M868_EXPECTED_RUNNER_SHA256}" \
    --hammer-review-sha256 "${M868_EXPECTED_HAMMER_REVIEW_SHA256}" \
    --hammer-outer-sha256 "${M868_EXPECTED_HAMMER_OUTER_SHA256}" \
    --output "${m868_stage}" --publish-to "${m868_result}" >/dev/null
m868_published=1
m868_phase="VERIFY_CANONICAL_DIAGNOSTIC"
[[ -d "${m868_result}" && ! -L "${m868_result}" && \
   ! -e "${m868_stage}" && ! -L "${m868_stage}" && \
   -f "${m868_result}/diagnostic.json" && \
   -f "${m868_result}/SHA256SUMS" && \
   -f "${m868_result}/SHA256SUMS.seal.sha256" ]] || exit 51
(cd "${m868_result}" && \
    /usr/bin/sha256sum -c SHA256SUMS >/dev/null && \
    /usr/bin/sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m868_success=1
trap - EXIT
rm -f -- "${m868_stdout_log}" "${m868_stderr_log}"
echo "PASS_M868_ONE_FULL_FIRST_ROW_DIAGNOSTIC__FRESH_RESULT_HAMMER_REQUIRED__NONCITABLE"
