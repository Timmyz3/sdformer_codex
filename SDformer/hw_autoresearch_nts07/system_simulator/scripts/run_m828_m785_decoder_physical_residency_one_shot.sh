#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M828 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m828_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m828_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m828_repo_root="$(cd "${m828_hw_root}/.." && pwd)"
m828_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m828_driver="${m828_hw_root}/system_simulator/scripts/execute_m828_m819_decoder_failure_prefix_guard.py"
m828_candidate="${m828_hw_root}/contracts/m828_m785_decoder_failure_prefix_guard_candidate_r1_20260829.json"
m828_release="${m828_hw_root}/contracts/m828_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
m828_result="${m828_hw_root}/results/m828_m785_h67_decoder_physical_residency_cycles_r1_20260829"
m828_attempt="${m828_hw_root}/results/.m828_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m828_attempt_stage="${m828_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m828_stage="${m828_result}.stage.$$.${RANDOM}.${RANDOM}"
m828_quarantine="${m828_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m828_partial_artifact="${m828_quarantine}.partial_artifact"
m828_stdout_log="${m828_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m828_stderr_log="${m828_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m828_started=0
m828_published=0
m828_success=0
m828_phase="PRE_ATTEMPT"

m828_ensure_empty_regular_log() {
    local m828_log="$1"
    if [[ ! -e "${m828_log}" && ! -L "${m828_log}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m828_python}" - "${m828_log}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${m828_log}" && ! -L "${m828_log}" ]]
}

m828_fail_closed() {
    local m828_rc=$?
    trap - EXIT
    if [[ "${m828_started}" -eq 1 && "${m828_success}" -ne 1 ]]; then
        [[ "${m828_rc}" -ne 0 ]] || m828_rc=98
        local m828_partial=""
        if [[ "${m828_published}" -eq 1 && \
              ( -e "${m828_result}" || -L "${m828_result}" ) ]]; then
            mv -T --no-clobber -- "${m828_result}" \
                "${m828_partial_artifact}" || exit 99
            m828_partial="${m828_partial_artifact}"
        elif [[ -e "${m828_stage}" || -L "${m828_stage}" ]]; then
            mv -T --no-clobber -- "${m828_stage}" \
                "${m828_partial_artifact}" || exit 99
            m828_partial="${m828_partial_artifact}"
        fi
        m828_ensure_empty_regular_log "${m828_stdout_log}" || exit 99
        m828_ensure_empty_regular_log "${m828_stderr_log}" || exit 99
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            M828_EXPECTED_RELEASE_SHA256="${M828_EXPECTED_RELEASE_SHA256}" \
            "${m828_python}" "${m828_driver}" \
            --write-failure-receipt \
            --candidate "${m828_candidate}" \
            --release "${m828_release}" \
            --attempt "${m828_attempt}" \
            --runner "${m828_runner}" \
            --stdout-log "${m828_stdout_log}" \
            --stderr-log "${m828_stderr_log}" \
            --output "${m828_quarantine}" \
            --expected-runner-sha256 "${M828_EXPECTED_RUNNER_SHA256}" \
            --expected-release-sha256 "${M828_EXPECTED_RELEASE_SHA256}" \
            --return-code "${m828_rc}" \
            --phase "${m828_phase}" \
            --partial-artifact "${m828_partial}" >/dev/null || exit 99
        [[ -d "${m828_quarantine}" && ! -L "${m828_quarantine}" && \
           -f "${m828_quarantine}/failure.json" && \
           -f "${m828_quarantine}/driver.log" && \
           -f "${m828_quarantine}/SHA256SUMS" && \
           -f "${m828_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m828_stdout_log}" "${m828_stderr_log}"
    fi
    exit "${m828_rc}"
}
trap m828_fail_closed EXIT

[[ "${m828_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m828_runner}" == \
   "${m828_hw_root}/system_simulator/scripts/run_m828_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M828 canonical path drift" >&2
    exit 3
}
[[ -n "${M828_EXPECTED_RUNNER_SHA256:-}" && \
   "${M828_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m828_runner}" | awk '{print $1}')" == \
   "${M828_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M828 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M828_EXPECTED_RELEASE_SHA256:-}" && \
   "${M828_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m828_release}" && ! -L "${m828_release}" && \
   "$(sha256sum "${m828_release}" | awk '{print $1}')" == \
   "${M828_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M828 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m828_python}" && ! -L "${m828_python}" && \
   "$(sha256sum "${m828_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M828 Python identity drift" >&2
    exit 4
}
[[ -f "${m828_driver}" && ! -L "${m828_driver}" && \
   -f "${m828_candidate}" && ! -L "${m828_candidate}" ]] || {
    echo "M828 source files absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m828_result}" && ! -L "${m828_result}" && \
   ! -e "${m828_attempt}" && ! -L "${m828_attempt}" && \
   ! -e "${m828_attempt_stage}" && ! -L "${m828_attempt_stage}" && \
   ! -e "${m828_stage}" && ! -L "${m828_stage}" && \
   ! -e "${m828_quarantine}" && ! -L "${m828_quarantine}" && \
   ! -e "${m828_partial_artifact}" && ! -L "${m828_partial_artifact}" && \
   ! -e "${m828_stdout_log}" && ! -L "${m828_stdout_log}" && \
   ! -e "${m828_stderr_log}" && ! -L "${m828_stderr_log}" ]] || {
    echo "M828 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M828_EXPECTED_RELEASE_SHA256="${M828_EXPECTED_RELEASE_SHA256}" \
    "${m828_python}" "${m828_driver}" \
    --validate-release-preflight \
    --candidate "${m828_candidate}" \
    --release "${m828_release}" >/dev/null

m828_free_kib=$(df -Pk "$(dirname "${m828_result}")" | awk 'NR==2 {print $4}')
m828_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m828_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m828_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m828_commit_headroom=$((m828_commit_limit - m828_committed))
[[ "${m828_free_kib}" -ge 1048576 && \
   "${m828_mem_available}" -ge 67108864 && \
   "${m828_commit_headroom}" -ge 67108864 ]] || {
    echo "M828 resource gate failed without consuming one-shot" >&2
    exit 40
}

# M825 repair: one pinned-directory-FD, two-sample prefix guard is the final
# operation before formal attempt-stage creation. It creates no artifact.
m828_phase="PRE_ATTEMPT_FAILURE_PREFIX_GUARD"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m828_python}" "${m828_driver}" \
    --guard-failure-prefix-absence \
    --candidate "${m828_candidate}" >/dev/null

mkdir "${m828_attempt_stage}"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m828_python}" - "${m828_attempt_stage}/attempt.json" \
    "${m828_runner}" "${m828_driver}" "${m828_candidate}" \
    "${m828_release}" "${m828_result}" <<'PY'
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
    "schema": "m828_m785_decoder_production_attempt_v1",
    "status": "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY",
    "outer_boundary": "M828_FAILURE_PREFIX_GUARD",
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
(cd "${m828_attempt_stage}" && \
    sha256sum attempt.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv -T --no-clobber -- "${m828_attempt_stage}" "${m828_attempt}"
m828_started=1
m828_phase="ATTEMPT_PUBLISHED_POSTCHECK"
[[ -d "${m828_attempt}" && ! -L "${m828_attempt}" && \
   ! -e "${m828_attempt_stage}" && ! -L "${m828_attempt_stage}" && \
   -f "${m828_attempt}/attempt.json" && \
   -f "${m828_attempt}/SHA256SUMS" && \
   -f "${m828_attempt}/SHA256SUMS.seal.sha256" ]] || exit 41

m828_phase="ATTEMPT_REQUIRED_PREFLIGHT"
m828_ensure_empty_regular_log "${m828_stdout_log}"
m828_ensure_empty_regular_log "${m828_stderr_log}"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M828_EXPECTED_RELEASE_SHA256="${M828_EXPECTED_RELEASE_SHA256}" \
    "${m828_python}" "${m828_driver}" \
    --validate-consumed-attempt \
    --candidate "${m828_candidate}" \
    --release "${m828_release}" \
    --attempt "${m828_attempt}" \
    >>"${m828_stdout_log}" 2>>"${m828_stderr_log}"
m828_driver_rc=$?
set -e
[[ "${m828_driver_rc}" -eq 0 ]] || exit "${m828_driver_rc}"

m828_phase="RUN_PRODUCTION_DRIVER"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M828_EXPECTED_RELEASE_SHA256="${M828_EXPECTED_RELEASE_SHA256}" \
    "${m828_python}" "${m828_driver}" \
    --run-production \
    --candidate "${m828_candidate}" \
    --release "${m828_release}" \
    --attempt "${m828_attempt}" \
    --output "${m828_stage}" \
    >>"${m828_stdout_log}" 2>>"${m828_stderr_log}"
m828_driver_rc=$?
set -e
[[ "${m828_driver_rc}" -eq 0 ]] || exit "${m828_driver_rc}"

m828_phase="VERIFY_AND_SEAL_STAGE"
[[ -d "${m828_stage}" && ! -L "${m828_stage}" && \
   -f "${m828_stage}/result.json" && ! -L "${m828_stage}/result.json" && \
   -f "${m828_stage}/detailed_rows.json" && \
   ! -L "${m828_stage}/detailed_rows.json" ]] || exit 50
(cd "${m828_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m828_phase="PUBLISH_CANONICAL_NOREPLACE"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m828_python}" "${m828_driver}" \
    --publish-no-replace \
    --candidate "${m828_candidate}" \
    --output "${m828_stage}" \
    --publish-to "${m828_result}" >/dev/null
m828_published=1
m828_phase="VERIFY_CANONICAL_PUBLICATION"
[[ -d "${m828_result}" && ! -L "${m828_result}" && \
   ! -e "${m828_stage}" && ! -L "${m828_stage}" && \
   -f "${m828_result}/result.json" && \
   -f "${m828_result}/detailed_rows.json" && \
   -f "${m828_result}/SHA256SUMS" && \
   -f "${m828_result}/SHA256SUMS.seal.sha256" ]] || exit 51
m828_success=1
trap - EXIT
rm -f -- "${m828_stdout_log}" "${m828_stderr_log}"
echo "PASS_M828_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
