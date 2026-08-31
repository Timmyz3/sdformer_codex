#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M819 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m819_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m819_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m819_repo_root="$(cd "${m819_hw_root}/.." && pwd)"
m819_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m819_driver="${m819_hw_root}/system_simulator/scripts/execute_m819_m809_decoder_production_delegation_compat.py"
m819_candidate="${m819_hw_root}/contracts/m819_m785_decoder_production_delegation_compat_candidate_r1_20260829.json"
m819_release="${m819_hw_root}/contracts/m819_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
m819_result="${m819_hw_root}/results/m819_m785_h67_decoder_physical_residency_cycles_r1_20260829"
m819_attempt="${m819_hw_root}/results/.m819_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m819_attempt_stage="${m819_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m819_stage="${m819_result}.stage.$$.${RANDOM}.${RANDOM}"
m819_quarantine="${m819_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m819_partial_artifact="${m819_quarantine}.partial_artifact"
m819_stdout_log="${m819_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m819_stderr_log="${m819_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m819_started=0
m819_published=0
m819_success=0
m819_phase="PRE_ATTEMPT"

m819_ensure_empty_regular_log() {
    local m819_log="$1"
    if [[ ! -e "${m819_log}" && ! -L "${m819_log}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m819_python}" - "${m819_log}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${m819_log}" && ! -L "${m819_log}" ]]
}

m819_fail_closed() {
    local m819_rc=$?
    trap - EXIT
    if [[ "${m819_started}" -eq 1 && "${m819_success}" -ne 1 ]]; then
        [[ "${m819_rc}" -ne 0 ]] || m819_rc=98
        local m819_partial=""
        if [[ "${m819_published}" -eq 1 && \
              ( -e "${m819_result}" || -L "${m819_result}" ) ]]; then
            mv -T --no-clobber -- "${m819_result}" \
                "${m819_partial_artifact}" || exit 99
            m819_partial="${m819_partial_artifact}"
        elif [[ -e "${m819_stage}" || -L "${m819_stage}" ]]; then
            mv -T --no-clobber -- "${m819_stage}" \
                "${m819_partial_artifact}" || exit 99
            m819_partial="${m819_partial_artifact}"
        fi
        m819_ensure_empty_regular_log "${m819_stdout_log}" || exit 99
        m819_ensure_empty_regular_log "${m819_stderr_log}" || exit 99
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m819_python}" "${m819_driver}" \
            --write-failure-receipt \
            --candidate "${m819_candidate}" \
            --release "${m819_release}" \
            --attempt "${m819_attempt}" \
            --runner "${m819_runner}" \
            --stdout-log "${m819_stdout_log}" \
            --stderr-log "${m819_stderr_log}" \
            --output "${m819_quarantine}" \
            --expected-runner-sha256 "${M819_EXPECTED_RUNNER_SHA256}" \
            --expected-release-sha256 "${M819_EXPECTED_RELEASE_SHA256}" \
            --return-code "${m819_rc}" \
            --phase "${m819_phase}" \
            --partial-artifact "${m819_partial}" >/dev/null || exit 99
        [[ -d "${m819_quarantine}" && ! -L "${m819_quarantine}" && \
           -f "${m819_quarantine}/failure.json" && \
           -f "${m819_quarantine}/driver.log" && \
           -f "${m819_quarantine}/SHA256SUMS" && \
           -f "${m819_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m819_stdout_log}" "${m819_stderr_log}"
    fi
    exit "${m819_rc}"
}
trap m819_fail_closed EXIT

[[ "${m819_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m819_runner}" == \
   "${m819_hw_root}/system_simulator/scripts/run_m819_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M819 canonical path drift" >&2
    exit 3
}
[[ -n "${M819_EXPECTED_RUNNER_SHA256:-}" && \
   "${M819_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m819_runner}" | awk '{print $1}')" == \
   "${M819_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M819 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M819_EXPECTED_RELEASE_SHA256:-}" && \
   "${M819_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m819_release}" && ! -L "${m819_release}" && \
   "$(sha256sum "${m819_release}" | awk '{print $1}')" == \
   "${M819_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M819 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m819_python}" && ! -L "${m819_python}" && \
   "$(sha256sum "${m819_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M819 Python identity drift" >&2
    exit 4
}
[[ -f "${m819_driver}" && ! -L "${m819_driver}" && \
   -f "${m819_candidate}" && ! -L "${m819_candidate}" ]] || {
    echo "M819 source files absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m819_result}" && ! -L "${m819_result}" && \
   ! -e "${m819_attempt}" && ! -L "${m819_attempt}" && \
   ! -e "${m819_attempt_stage}" && ! -L "${m819_attempt_stage}" && \
   ! -e "${m819_stage}" && ! -L "${m819_stage}" && \
   ! -e "${m819_quarantine}" && ! -L "${m819_quarantine}" && \
   ! -e "${m819_partial_artifact}" && ! -L "${m819_partial_artifact}" && \
   ! -e "${m819_stdout_log}" && ! -L "${m819_stdout_log}" && \
   ! -e "${m819_stderr_log}" && ! -L "${m819_stderr_log}" ]] || {
    echo "M819 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M819_EXPECTED_RELEASE_SHA256="${M819_EXPECTED_RELEASE_SHA256}" \
    "${m819_python}" "${m819_driver}" \
    --validate-release-preflight \
    --candidate "${m819_candidate}" \
    --release "${m819_release}" >/dev/null

m819_free_kib=$(df -Pk "$(dirname "${m819_result}")" | awk 'NR==2 {print $4}')
m819_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m819_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m819_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m819_commit_headroom=$((m819_commit_limit - m819_committed))
[[ "${m819_free_kib}" -ge 1048576 && \
   "${m819_mem_available}" -ge 67108864 && \
   "${m819_commit_headroom}" -ge 67108864 ]] || {
    echo "M819 resource gate failed without consuming one-shot" >&2
    exit 40
}

mkdir "${m819_attempt_stage}"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m819_python}" - "${m819_attempt_stage}/attempt.json" \
    "${m819_runner}" "${m819_driver}" "${m819_candidate}" \
    "${m819_release}" "${m819_result}" <<'PY'
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
    "schema": "m819_m785_decoder_production_attempt_v1",
    "status": "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY",
    "outer_boundary": "M819_DELEGATION_COMPAT",
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
(cd "${m819_attempt_stage}" && \
    sha256sum attempt.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv -T --no-clobber -- "${m819_attempt_stage}" "${m819_attempt}"
m819_started=1
m819_phase="ATTEMPT_PUBLISHED_POSTCHECK"
[[ -d "${m819_attempt}" && ! -L "${m819_attempt}" && \
   ! -e "${m819_attempt_stage}" && ! -L "${m819_attempt_stage}" && \
   -f "${m819_attempt}/attempt.json" && \
   -f "${m819_attempt}/SHA256SUMS" && \
   -f "${m819_attempt}/SHA256SUMS.seal.sha256" ]] || exit 41

m819_phase="ATTEMPT_REQUIRED_PREFLIGHT"
m819_ensure_empty_regular_log "${m819_stdout_log}"
m819_ensure_empty_regular_log "${m819_stderr_log}"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M819_EXPECTED_RELEASE_SHA256="${M819_EXPECTED_RELEASE_SHA256}" \
    "${m819_python}" "${m819_driver}" \
    --validate-consumed-attempt \
    --candidate "${m819_candidate}" \
    --release "${m819_release}" \
    --attempt "${m819_attempt}" \
    >>"${m819_stdout_log}" 2>>"${m819_stderr_log}"
m819_driver_rc=$?
set -e
[[ "${m819_driver_rc}" -eq 0 ]] || exit "${m819_driver_rc}"

m819_phase="RUN_PRODUCTION_DRIVER"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M819_EXPECTED_RELEASE_SHA256="${M819_EXPECTED_RELEASE_SHA256}" \
    "${m819_python}" "${m819_driver}" \
    --run-production \
    --candidate "${m819_candidate}" \
    --release "${m819_release}" \
    --attempt "${m819_attempt}" \
    --output "${m819_stage}" \
    >>"${m819_stdout_log}" 2>>"${m819_stderr_log}"
m819_driver_rc=$?
set -e
[[ "${m819_driver_rc}" -eq 0 ]] || exit "${m819_driver_rc}"

m819_phase="VERIFY_AND_SEAL_STAGE"
[[ -d "${m819_stage}" && ! -L "${m819_stage}" && \
   -f "${m819_stage}/result.json" && ! -L "${m819_stage}/result.json" && \
   -f "${m819_stage}/detailed_rows.json" && \
   ! -L "${m819_stage}/detailed_rows.json" ]] || exit 50
(cd "${m819_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m819_phase="PUBLISH_CANONICAL_NOREPLACE"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m819_python}" "${m819_driver}" \
    --publish-no-replace \
    --candidate "${m819_candidate}" \
    --output "${m819_stage}" \
    --publish-to "${m819_result}" >/dev/null
m819_published=1
m819_phase="VERIFY_CANONICAL_PUBLICATION"
[[ -d "${m819_result}" && ! -L "${m819_result}" && \
   ! -e "${m819_stage}" && ! -L "${m819_stage}" && \
   -f "${m819_result}/result.json" && \
   -f "${m819_result}/detailed_rows.json" && \
   -f "${m819_result}/SHA256SUMS" && \
   -f "${m819_result}/SHA256SUMS.seal.sha256" ]] || exit 51
m819_success=1
trap - EXIT
rm -f -- "${m819_stdout_log}" "${m819_stderr_log}"
echo "PASS_M819_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
