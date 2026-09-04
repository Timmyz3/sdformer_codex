#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M836 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m836_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m836_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m836_repo_root="$(cd "${m836_hw_root}/.." && pwd)"
m836_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m836_driver="${m836_hw_root}/system_simulator/scripts/execute_m836_m832_decoder_publication_boundary_repair.py"
m836_candidate="${m836_hw_root}/contracts/m836_m785_decoder_publication_boundary_repair_candidate_r1_20260829.json"
m836_release="${m836_hw_root}/contracts/m836_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
m836_result="${m836_hw_root}/results/m836_m785_h67_decoder_physical_residency_cycles_r1_20260829"
m836_attempt="${m836_hw_root}/results/.m836_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m836_attempt_stage="${m836_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m836_attempt_stage_name="$(basename "${m836_attempt_stage}")"
m836_stage="${m836_result}.stage.$$.${RANDOM}.${RANDOM}"
m836_quarantine="${m836_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m836_partial_artifact="${m836_quarantine}.partial_artifact"
m836_stdout_log="${m836_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m836_stderr_log="${m836_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m836_started=0
m836_published=0
m836_success=0
m836_phase="PRE_ATTEMPT"

m836_ensure_empty_regular_log() {
    local m836_log="$1"
    if [[ ! -e "${m836_log}" && ! -L "${m836_log}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m836_python}" - "${m836_log}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${m836_log}" && ! -L "${m836_log}" ]]
}

m836_fail_closed() {
    local m836_rc=$?
    trap - EXIT
    if [[ "${m836_started}" -eq 1 && "${m836_success}" -ne 1 ]]; then
        [[ "${m836_rc}" -ne 0 ]] || m836_rc=98
        local m836_partial=""
        if [[ "${m836_published}" -eq 1 && \
              ( -e "${m836_result}" || -L "${m836_result}" ) ]]; then
            mv -T --no-clobber -- "${m836_result}" \
                "${m836_partial_artifact}" || exit 99
            m836_partial="${m836_partial_artifact}"
        elif [[ -e "${m836_stage}" || -L "${m836_stage}" ]]; then
            mv -T --no-clobber -- "${m836_stage}" \
                "${m836_partial_artifact}" || exit 99
            m836_partial="${m836_partial_artifact}"
        fi
        m836_ensure_empty_regular_log "${m836_stdout_log}" || exit 99
        m836_ensure_empty_regular_log "${m836_stderr_log}" || exit 99
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            M836_EXPECTED_RELEASE_SHA256="${M836_EXPECTED_RELEASE_SHA256}" \
            "${m836_python}" "${m836_driver}" \
            --write-failure-receipt \
            --candidate "${m836_candidate}" \
            --release "${m836_release}" \
            --attempt "${m836_attempt}" \
            --runner "${m836_runner}" \
            --stdout-log "${m836_stdout_log}" \
            --stderr-log "${m836_stderr_log}" \
            --output "${m836_quarantine}" \
            --expected-runner-sha256 "${M836_EXPECTED_RUNNER_SHA256}" \
            --expected-release-sha256 "${M836_EXPECTED_RELEASE_SHA256}" \
            --return-code "${m836_rc}" \
            --phase "${m836_phase}" \
            --partial-artifact "${m836_partial}" >/dev/null || exit 99
        [[ -d "${m836_quarantine}" && ! -L "${m836_quarantine}" && \
           -f "${m836_quarantine}/failure.json" && \
           -f "${m836_quarantine}/driver.log" && \
           -f "${m836_quarantine}/SHA256SUMS" && \
           -f "${m836_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m836_stdout_log}" "${m836_stderr_log}"
    fi
    exit "${m836_rc}"
}
trap m836_fail_closed EXIT

[[ "${m836_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m836_runner}" == \
   "${m836_hw_root}/system_simulator/scripts/run_m836_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M836 canonical path drift" >&2
    exit 3
}
[[ -n "${M836_EXPECTED_RUNNER_SHA256:-}" && \
   "${M836_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m836_runner}" | awk '{print $1}')" == \
   "${M836_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M836 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M836_EXPECTED_RELEASE_SHA256:-}" && \
   "${M836_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m836_release}" && ! -L "${m836_release}" && \
   "$(sha256sum "${m836_release}" | awk '{print $1}')" == \
   "${M836_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M836 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m836_python}" && ! -L "${m836_python}" && \
   "$(sha256sum "${m836_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M836 Python identity drift" >&2
    exit 4
}
[[ -f "${m836_driver}" && ! -L "${m836_driver}" && \
   -f "${m836_candidate}" && ! -L "${m836_candidate}" ]] || {
    echo "M836 source files absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m836_result}" && ! -L "${m836_result}" && \
   ! -e "${m836_attempt}" && ! -L "${m836_attempt}" && \
   ! -e "${m836_attempt_stage}" && ! -L "${m836_attempt_stage}" && \
   ! -e "${m836_stage}" && ! -L "${m836_stage}" && \
   ! -e "${m836_quarantine}" && ! -L "${m836_quarantine}" && \
   ! -e "${m836_partial_artifact}" && ! -L "${m836_partial_artifact}" && \
   ! -e "${m836_stdout_log}" && ! -L "${m836_stdout_log}" && \
   ! -e "${m836_stderr_log}" && ! -L "${m836_stderr_log}" ]] || {
    echo "M836 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M836_EXPECTED_RELEASE_SHA256="${M836_EXPECTED_RELEASE_SHA256}" \
    "${m836_python}" "${m836_driver}" \
    --validate-release-preflight \
    --candidate "${m836_candidate}" \
    --release "${m836_release}" >/dev/null

m836_free_kib=$(df -Pk "$(dirname "${m836_result}")" | awk 'NR==2 {print $4}')
m836_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m836_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m836_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m836_commit_headroom=$((m836_commit_limit - m836_committed))
[[ "${m836_free_kib}" -ge 1048576 && \
   "${m836_mem_available}" -ge 67108864 && \
   "${m836_commit_headroom}" -ge 67108864 ]] || {
    echo "M836 resource gate failed without consuming one-shot" >&2
    exit 40
}

# M835 repair: the helper owns scan, sealed-stage validation, publication,
# postpublication current-path binding, and exact-inode rollback.
m836_phase="POSTPUBLICATION_BOUND_ATTEMPT_CONSUMPTION"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M836_EXPECTED_RELEASE_SHA256="${M836_EXPECTED_RELEASE_SHA256}" \
    "${m836_python}" "${m836_driver}" \
    --guard-and-consume-attempt \
    --candidate "${m836_candidate}" \
    --release "${m836_release}" \
    --runner "${m836_runner}" \
    --stage-basename "${m836_attempt_stage_name}" \
    --expected-runner-sha256 "${M836_EXPECTED_RUNNER_SHA256}" >/dev/null
m836_started=1
m836_phase="ATTEMPT_PUBLISHED_POSTCHECK"
[[ -d "${m836_attempt}" && ! -L "${m836_attempt}" && \
   ! -e "${m836_attempt_stage}" && ! -L "${m836_attempt_stage}" && \
   -f "${m836_attempt}/attempt.json" && \
   -f "${m836_attempt}/SHA256SUMS" && \
   -f "${m836_attempt}/SHA256SUMS.seal.sha256" ]] || exit 41

m836_phase="ATTEMPT_REQUIRED_PREFLIGHT"
m836_ensure_empty_regular_log "${m836_stdout_log}"
m836_ensure_empty_regular_log "${m836_stderr_log}"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M836_EXPECTED_RELEASE_SHA256="${M836_EXPECTED_RELEASE_SHA256}" \
    "${m836_python}" "${m836_driver}" \
    --validate-consumed-attempt \
    --candidate "${m836_candidate}" \
    --release "${m836_release}" \
    --attempt "${m836_attempt}" \
    >>"${m836_stdout_log}" 2>>"${m836_stderr_log}"
m836_driver_rc=$?
set -e
[[ "${m836_driver_rc}" -eq 0 ]] || exit "${m836_driver_rc}"

m836_phase="RUN_PRODUCTION_DRIVER"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M836_EXPECTED_RELEASE_SHA256="${M836_EXPECTED_RELEASE_SHA256}" \
    "${m836_python}" "${m836_driver}" \
    --run-production \
    --candidate "${m836_candidate}" \
    --release "${m836_release}" \
    --attempt "${m836_attempt}" \
    --output "${m836_stage}" \
    >>"${m836_stdout_log}" 2>>"${m836_stderr_log}"
m836_driver_rc=$?
set -e
[[ "${m836_driver_rc}" -eq 0 ]] || exit "${m836_driver_rc}"

m836_phase="VERIFY_AND_SEAL_STAGE"
[[ -d "${m836_stage}" && ! -L "${m836_stage}" && \
   -f "${m836_stage}/result.json" && ! -L "${m836_stage}/result.json" && \
   -f "${m836_stage}/detailed_rows.json" && \
   ! -L "${m836_stage}/detailed_rows.json" ]] || exit 50
(cd "${m836_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m836_phase="PUBLISH_CANONICAL_NOREPLACE"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m836_python}" "${m836_driver}" \
    --publish-no-replace \
    --candidate "${m836_candidate}" \
    --output "${m836_stage}" \
    --publish-to "${m836_result}" >/dev/null
m836_published=1
m836_phase="VERIFY_CANONICAL_PUBLICATION"
[[ -d "${m836_result}" && ! -L "${m836_result}" && \
   ! -e "${m836_stage}" && ! -L "${m836_stage}" && \
   -f "${m836_result}/result.json" && \
   -f "${m836_result}/detailed_rows.json" && \
   -f "${m836_result}/SHA256SUMS" && \
   -f "${m836_result}/SHA256SUMS.seal.sha256" ]] || exit 51
m836_success=1
trap - EXIT
rm -f -- "${m836_stdout_log}" "${m836_stderr_log}"
echo "PASS_M836_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
