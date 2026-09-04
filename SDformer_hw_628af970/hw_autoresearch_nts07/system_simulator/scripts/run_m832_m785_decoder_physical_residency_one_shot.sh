#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M832 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m832_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m832_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m832_repo_root="$(cd "${m832_hw_root}/.." && pwd)"
m832_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m832_driver="${m832_hw_root}/system_simulator/scripts/execute_m832_m828_decoder_directory_bound_consumption.py"
m832_candidate="${m832_hw_root}/contracts/m832_m785_decoder_directory_bound_consumption_candidate_r1_20260829.json"
m832_release="${m832_hw_root}/contracts/m832_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
m832_result="${m832_hw_root}/results/m832_m785_h67_decoder_physical_residency_cycles_r1_20260829"
m832_attempt="${m832_hw_root}/results/.m832_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m832_attempt_stage="${m832_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m832_attempt_stage_name="$(basename "${m832_attempt_stage}")"
m832_stage="${m832_result}.stage.$$.${RANDOM}.${RANDOM}"
m832_quarantine="${m832_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m832_partial_artifact="${m832_quarantine}.partial_artifact"
m832_stdout_log="${m832_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m832_stderr_log="${m832_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m832_started=0
m832_published=0
m832_success=0
m832_phase="PRE_ATTEMPT"

m832_ensure_empty_regular_log() {
    local m832_log="$1"
    if [[ ! -e "${m832_log}" && ! -L "${m832_log}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m832_python}" - "${m832_log}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${m832_log}" && ! -L "${m832_log}" ]]
}

m832_fail_closed() {
    local m832_rc=$?
    trap - EXIT
    if [[ "${m832_started}" -eq 1 && "${m832_success}" -ne 1 ]]; then
        [[ "${m832_rc}" -ne 0 ]] || m832_rc=98
        local m832_partial=""
        if [[ "${m832_published}" -eq 1 && \
              ( -e "${m832_result}" || -L "${m832_result}" ) ]]; then
            mv -T --no-clobber -- "${m832_result}" \
                "${m832_partial_artifact}" || exit 99
            m832_partial="${m832_partial_artifact}"
        elif [[ -e "${m832_stage}" || -L "${m832_stage}" ]]; then
            mv -T --no-clobber -- "${m832_stage}" \
                "${m832_partial_artifact}" || exit 99
            m832_partial="${m832_partial_artifact}"
        fi
        m832_ensure_empty_regular_log "${m832_stdout_log}" || exit 99
        m832_ensure_empty_regular_log "${m832_stderr_log}" || exit 99
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            M832_EXPECTED_RELEASE_SHA256="${M832_EXPECTED_RELEASE_SHA256}" \
            "${m832_python}" "${m832_driver}" \
            --write-failure-receipt \
            --candidate "${m832_candidate}" \
            --release "${m832_release}" \
            --attempt "${m832_attempt}" \
            --runner "${m832_runner}" \
            --stdout-log "${m832_stdout_log}" \
            --stderr-log "${m832_stderr_log}" \
            --output "${m832_quarantine}" \
            --expected-runner-sha256 "${M832_EXPECTED_RUNNER_SHA256}" \
            --expected-release-sha256 "${M832_EXPECTED_RELEASE_SHA256}" \
            --return-code "${m832_rc}" \
            --phase "${m832_phase}" \
            --partial-artifact "${m832_partial}" >/dev/null || exit 99
        [[ -d "${m832_quarantine}" && ! -L "${m832_quarantine}" && \
           -f "${m832_quarantine}/failure.json" && \
           -f "${m832_quarantine}/driver.log" && \
           -f "${m832_quarantine}/SHA256SUMS" && \
           -f "${m832_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m832_stdout_log}" "${m832_stderr_log}"
    fi
    exit "${m832_rc}"
}
trap m832_fail_closed EXIT

[[ "${m832_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m832_runner}" == \
   "${m832_hw_root}/system_simulator/scripts/run_m832_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M832 canonical path drift" >&2
    exit 3
}
[[ -n "${M832_EXPECTED_RUNNER_SHA256:-}" && \
   "${M832_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m832_runner}" | awk '{print $1}')" == \
   "${M832_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M832 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M832_EXPECTED_RELEASE_SHA256:-}" && \
   "${M832_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m832_release}" && ! -L "${m832_release}" && \
   "$(sha256sum "${m832_release}" | awk '{print $1}')" == \
   "${M832_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M832 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m832_python}" && ! -L "${m832_python}" && \
   "$(sha256sum "${m832_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M832 Python identity drift" >&2
    exit 4
}
[[ -f "${m832_driver}" && ! -L "${m832_driver}" && \
   -f "${m832_candidate}" && ! -L "${m832_candidate}" ]] || {
    echo "M832 source files absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m832_result}" && ! -L "${m832_result}" && \
   ! -e "${m832_attempt}" && ! -L "${m832_attempt}" && \
   ! -e "${m832_attempt_stage}" && ! -L "${m832_attempt_stage}" && \
   ! -e "${m832_stage}" && ! -L "${m832_stage}" && \
   ! -e "${m832_quarantine}" && ! -L "${m832_quarantine}" && \
   ! -e "${m832_partial_artifact}" && ! -L "${m832_partial_artifact}" && \
   ! -e "${m832_stdout_log}" && ! -L "${m832_stdout_log}" && \
   ! -e "${m832_stderr_log}" && ! -L "${m832_stderr_log}" ]] || {
    echo "M832 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M832_EXPECTED_RELEASE_SHA256="${M832_EXPECTED_RELEASE_SHA256}" \
    "${m832_python}" "${m832_driver}" \
    --validate-release-preflight \
    --candidate "${m832_candidate}" \
    --release "${m832_release}" >/dev/null

m832_free_kib=$(df -Pk "$(dirname "${m832_result}")" | awk 'NR==2 {print $4}')
m832_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m832_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m832_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m832_commit_headroom=$((m832_commit_limit - m832_committed))
[[ "${m832_free_kib}" -ge 1048576 && \
   "${m832_mem_available}" -ge 67108864 && \
   "${m832_commit_headroom}" -ge 67108864 ]] || {
    echo "M832 resource gate failed without consuming one-shot" >&2
    exit 40
}

# M831 repair: inspection, pathname rebinding, stage creation, sealed receipt,
# and no-replace attempt publication execute under the same parent/results FDs.
m832_phase="DIRECTORY_FD_BOUND_ATTEMPT_CONSUMPTION"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M832_EXPECTED_RELEASE_SHA256="${M832_EXPECTED_RELEASE_SHA256}" \
    "${m832_python}" "${m832_driver}" \
    --guard-and-consume-attempt \
    --candidate "${m832_candidate}" \
    --release "${m832_release}" \
    --runner "${m832_runner}" \
    --stage-basename "${m832_attempt_stage_name}" \
    --expected-runner-sha256 "${M832_EXPECTED_RUNNER_SHA256}" >/dev/null
m832_started=1
m832_phase="ATTEMPT_PUBLISHED_POSTCHECK"
[[ -d "${m832_attempt}" && ! -L "${m832_attempt}" && \
   ! -e "${m832_attempt_stage}" && ! -L "${m832_attempt_stage}" && \
   -f "${m832_attempt}/attempt.json" && \
   -f "${m832_attempt}/SHA256SUMS" && \
   -f "${m832_attempt}/SHA256SUMS.seal.sha256" ]] || exit 41

m832_phase="ATTEMPT_REQUIRED_PREFLIGHT"
m832_ensure_empty_regular_log "${m832_stdout_log}"
m832_ensure_empty_regular_log "${m832_stderr_log}"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M832_EXPECTED_RELEASE_SHA256="${M832_EXPECTED_RELEASE_SHA256}" \
    "${m832_python}" "${m832_driver}" \
    --validate-consumed-attempt \
    --candidate "${m832_candidate}" \
    --release "${m832_release}" \
    --attempt "${m832_attempt}" \
    >>"${m832_stdout_log}" 2>>"${m832_stderr_log}"
m832_driver_rc=$?
set -e
[[ "${m832_driver_rc}" -eq 0 ]] || exit "${m832_driver_rc}"

m832_phase="RUN_PRODUCTION_DRIVER"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M832_EXPECTED_RELEASE_SHA256="${M832_EXPECTED_RELEASE_SHA256}" \
    "${m832_python}" "${m832_driver}" \
    --run-production \
    --candidate "${m832_candidate}" \
    --release "${m832_release}" \
    --attempt "${m832_attempt}" \
    --output "${m832_stage}" \
    >>"${m832_stdout_log}" 2>>"${m832_stderr_log}"
m832_driver_rc=$?
set -e
[[ "${m832_driver_rc}" -eq 0 ]] || exit "${m832_driver_rc}"

m832_phase="VERIFY_AND_SEAL_STAGE"
[[ -d "${m832_stage}" && ! -L "${m832_stage}" && \
   -f "${m832_stage}/result.json" && ! -L "${m832_stage}/result.json" && \
   -f "${m832_stage}/detailed_rows.json" && \
   ! -L "${m832_stage}/detailed_rows.json" ]] || exit 50
(cd "${m832_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m832_phase="PUBLISH_CANONICAL_NOREPLACE"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m832_python}" "${m832_driver}" \
    --publish-no-replace \
    --candidate "${m832_candidate}" \
    --output "${m832_stage}" \
    --publish-to "${m832_result}" >/dev/null
m832_published=1
m832_phase="VERIFY_CANONICAL_PUBLICATION"
[[ -d "${m832_result}" && ! -L "${m832_result}" && \
   ! -e "${m832_stage}" && ! -L "${m832_stage}" && \
   -f "${m832_result}/result.json" && \
   -f "${m832_result}/detailed_rows.json" && \
   -f "${m832_result}/SHA256SUMS" && \
   -f "${m832_result}/SHA256SUMS.seal.sha256" ]] || exit 51
m832_success=1
trap - EXIT
rm -f -- "${m832_stdout_log}" "${m832_stderr_log}"
echo "PASS_M832_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
