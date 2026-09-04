#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M809 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m809_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m809_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m809_repo_root="$(cd "${m809_hw_root}/.." && pwd)"
m809_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m809_driver="${m809_hw_root}/system_simulator/scripts/execute_m809_m785_decoder_physical_residency_production.py"
m809_candidate="${m809_hw_root}/contracts/m809_m785_decoder_physical_residency_production_recovery_candidate_r1_20260829.json"
m809_release="${m809_hw_root}/contracts/m809_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
m809_result="${m809_hw_root}/results/m809_m785_h67_decoder_physical_residency_cycles_r1_20260829"
m809_attempt="${m809_hw_root}/results/.m809_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m809_attempt_stage="${m809_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m809_stage="${m809_result}.stage.$$.${RANDOM}.${RANDOM}"
m809_quarantine="${m809_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m809_partial_artifact="${m809_quarantine}.partial_artifact"
m809_stdout_log="${m809_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m809_stderr_log="${m809_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m809_started=0
m809_published=0
m809_success=0
m809_phase="PRE_ATTEMPT"

m809_fail_closed() {
    local m809_rc=$?
    trap - EXIT
    if [[ "${m809_started}" -eq 1 && "${m809_success}" -ne 1 ]]; then
        [[ "${m809_rc}" -ne 0 ]] || m809_rc=98
        local m809_partial=""
        if [[ "${m809_published}" -eq 1 && \
              ( -e "${m809_result}" || -L "${m809_result}" ) ]]; then
            mv -T --no-clobber -- "${m809_result}" \
                "${m809_partial_artifact}" || exit 99
            m809_partial="${m809_partial_artifact}"
        elif [[ -e "${m809_stage}" || -L "${m809_stage}" ]]; then
            mv -T --no-clobber -- "${m809_stage}" \
                "${m809_partial_artifact}" || exit 99
            m809_partial="${m809_partial_artifact}"
        fi
        [[ -f "${m809_stdout_log}" && ! -L "${m809_stdout_log}" ]] || \
            : >"${m809_stdout_log}"
        [[ -f "${m809_stderr_log}" && ! -L "${m809_stderr_log}" ]] || \
            : >"${m809_stderr_log}"
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m809_python}" "${m809_driver}" \
            --write-failure-receipt \
            --candidate "${m809_candidate}" \
            --release "${m809_release}" \
            --attempt "${m809_attempt}" \
            --runner "${m809_runner}" \
            --stdout-log "${m809_stdout_log}" \
            --stderr-log "${m809_stderr_log}" \
            --output "${m809_quarantine}" \
            --expected-runner-sha256 "${M809_EXPECTED_RUNNER_SHA256}" \
            --expected-release-sha256 "${M809_EXPECTED_RELEASE_SHA256}" \
            --return-code "${m809_rc}" \
            --phase "${m809_phase}" \
            --partial-artifact "${m809_partial}" >/dev/null || exit 99
        [[ -d "${m809_quarantine}" && ! -L "${m809_quarantine}" && \
           -f "${m809_quarantine}/failure.json" && \
           -f "${m809_quarantine}/driver.log" && \
           -f "${m809_quarantine}/SHA256SUMS" && \
           -f "${m809_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m809_stdout_log}" "${m809_stderr_log}"
    fi
    exit "${m809_rc}"
}
trap m809_fail_closed EXIT

[[ "${m809_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m809_runner}" == \
   "${m809_hw_root}/system_simulator/scripts/run_m809_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M809 canonical path drift" >&2
    exit 3
}
[[ -n "${M809_EXPECTED_RUNNER_SHA256:-}" && \
   "${M809_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m809_runner}" | awk '{print $1}')" == \
   "${M809_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M809 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M809_EXPECTED_RELEASE_SHA256:-}" && \
   "${M809_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m809_release}" && ! -L "${m809_release}" && \
   "$(sha256sum "${m809_release}" | awk '{print $1}')" == \
   "${M809_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M809 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m809_python}" && ! -L "${m809_python}" && \
   "$(sha256sum "${m809_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M809 Python identity drift" >&2
    exit 4
}
[[ -f "${m809_driver}" && ! -L "${m809_driver}" && \
   -f "${m809_candidate}" && ! -L "${m809_candidate}" ]] || {
    echo "M809 source files are absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m809_result}" && ! -L "${m809_result}" && \
   ! -e "${m809_attempt}" && ! -L "${m809_attempt}" && \
   ! -e "${m809_attempt_stage}" && ! -L "${m809_attempt_stage}" && \
   ! -e "${m809_stage}" && ! -L "${m809_stage}" && \
   ! -e "${m809_quarantine}" && ! -L "${m809_quarantine}" && \
   ! -e "${m809_partial_artifact}" && ! -L "${m809_partial_artifact}" && \
   ! -e "${m809_stdout_log}" && ! -L "${m809_stdout_log}" && \
   ! -e "${m809_stderr_log}" && ! -L "${m809_stderr_log}" ]] || {
    echo "M809 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

# This exact release/candidate/source-hammer validation is before one-shot
# consumption and creates no result.
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M809_EXPECTED_RELEASE_SHA256="${M809_EXPECTED_RELEASE_SHA256}" \
    "${m809_python}" "${m809_driver}" \
    --validate-release-preflight \
    --candidate "${m809_candidate}" \
    --release "${m809_release}" >/dev/null

m809_free_kib=$(df -Pk "$(dirname "${m809_result}")" | awk 'NR==2 {print $4}')
m809_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m809_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m809_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m809_commit_headroom=$((m809_commit_limit - m809_committed))
[[ "${m809_free_kib}" -ge 1048576 && \
   "${m809_mem_available}" -ge 67108864 && \
   "${m809_commit_headroom}" -ge 67108864 ]] || {
    echo "M809 resource gate failed without consuming one-shot" >&2
    exit 40
}

mkdir "${m809_attempt_stage}"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m809_python}" - "${m809_attempt_stage}/attempt.json" \
    "${m809_runner}" "${m809_driver}" "${m809_candidate}" \
    "${m809_release}" "${m809_result}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out, runner, driver, candidate, release, result = map(Path, sys.argv[1:])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

payload = {
    "schema": "m809_m785_decoder_production_attempt_v1",
    "status": "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY",
    "runner_sha256": digest(runner),
    "driver_sha256": digest(driver),
    "candidate_sha256": digest(candidate),
    "release_sha256": digest(release),
    "canonical_result": str(result),
    "max_attempts": 1,
    "claim_boundary": {
        "cycles_before_result_hammer": False,
        "speedup_before_result_hammer": False,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
               encoding="utf-8")
PY
(cd "${m809_attempt_stage}" && \
    sha256sum attempt.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv -T --no-clobber -- "${m809_attempt_stage}" "${m809_attempt}"
[[ -d "${m809_attempt}" && ! -L "${m809_attempt}" && \
   ! -e "${m809_attempt_stage}" && ! -L "${m809_attempt_stage}" && \
   -f "${m809_attempt}/attempt.json" && \
   -f "${m809_attempt}/SHA256SUMS" && \
   -f "${m809_attempt}/SHA256SUMS.seal.sha256" ]] || exit 41

m809_started=1
m809_phase="ATTEMPT_REQUIRED_PREFLIGHT"
: >"${m809_stdout_log}"
: >"${m809_stderr_log}"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M809_EXPECTED_RELEASE_SHA256="${M809_EXPECTED_RELEASE_SHA256}" \
    "${m809_python}" "${m809_driver}" \
    --validate-consumed-attempt \
    --candidate "${m809_candidate}" \
    --release "${m809_release}" \
    --attempt "${m809_attempt}" \
    >>"${m809_stdout_log}" 2>>"${m809_stderr_log}"
m809_driver_rc=$?
set -e
[[ "${m809_driver_rc}" -eq 0 ]] || exit "${m809_driver_rc}"

m809_phase="RUN_PRODUCTION_DRIVER"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M809_EXPECTED_RELEASE_SHA256="${M809_EXPECTED_RELEASE_SHA256}" \
    "${m809_python}" "${m809_driver}" \
    --run-production \
    --candidate "${m809_candidate}" \
    --release "${m809_release}" \
    --attempt "${m809_attempt}" \
    --output "${m809_stage}" \
    >>"${m809_stdout_log}" 2>>"${m809_stderr_log}"
m809_driver_rc=$?
set -e
[[ "${m809_driver_rc}" -eq 0 ]] || exit "${m809_driver_rc}"

m809_phase="VERIFY_AND_SEAL_STAGE"
[[ -d "${m809_stage}" && ! -L "${m809_stage}" && \
   -f "${m809_stage}/result.json" && ! -L "${m809_stage}/result.json" && \
   -f "${m809_stage}/detailed_rows.json" && \
   ! -L "${m809_stage}/detailed_rows.json" ]] || {
    echo "M809 production did not publish complete staging artifacts" >&2
    exit 50
}
(cd "${m809_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

# The driver uses renameat2(RENAME_NOREPLACE).  A destination created after
# preflight is an EEXIST failure; stage cannot be nested below canonical result.
m809_phase="PUBLISH_CANONICAL_NOREPLACE"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m809_python}" "${m809_driver}" \
    --publish-no-replace \
    --candidate "${m809_candidate}" \
    --output "${m809_stage}" \
    --publish-to "${m809_result}" >/dev/null
m809_published=1
m809_phase="VERIFY_CANONICAL_PUBLICATION"
[[ -d "${m809_result}" && ! -L "${m809_result}" && \
   ! -e "${m809_stage}" && ! -L "${m809_stage}" && \
   -f "${m809_result}/result.json" && \
   ! -L "${m809_result}/result.json" && \
   -f "${m809_result}/detailed_rows.json" && \
   ! -L "${m809_result}/detailed_rows.json" && \
   -f "${m809_result}/SHA256SUMS" && \
   ! -L "${m809_result}/SHA256SUMS" && \
   -f "${m809_result}/SHA256SUMS.seal.sha256" && \
   ! -L "${m809_result}/SHA256SUMS.seal.sha256" ]] || {
    echo "M809 canonical root four-member publication failed" >&2
    exit 51
}
m809_success=1
trap - EXIT
rm -f -- "${m809_stdout_log}" "${m809_stderr_log}"
echo "PASS_M809_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
