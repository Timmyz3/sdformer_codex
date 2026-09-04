#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M815 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m815_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m815_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m815_repo_root="$(cd "${m815_hw_root}/.." && pwd)"
m815_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m815_driver="${m815_hw_root}/system_simulator/scripts/execute_m815_m809_decoder_production_runner_recovery.py"
m815_candidate="${m815_hw_root}/contracts/m815_m785_decoder_production_runner_recovery_candidate_r1_20260829.json"
m815_release="${m815_hw_root}/contracts/m815_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
m815_result="${m815_hw_root}/results/m815_m785_h67_decoder_physical_residency_cycles_r1_20260829"
m815_attempt="${m815_hw_root}/results/.m815_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m815_attempt_stage="${m815_attempt}.stage.$$.${RANDOM}.${RANDOM}"
m815_stage="${m815_result}.stage.$$.${RANDOM}.${RANDOM}"
m815_quarantine="${m815_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m815_partial_artifact="${m815_quarantine}.partial_artifact"
m815_stdout_log="${m815_result}.driver_stdout.$$.${RANDOM}.${RANDOM}.log"
m815_stderr_log="${m815_result}.driver_stderr.$$.${RANDOM}.${RANDOM}.log"
m815_started=0
m815_published=0
m815_success=0
m815_phase="PRE_ATTEMPT"

m815_ensure_empty_regular_log() {
    local m815_log="$1"
    if [[ ! -e "${m815_log}" && ! -L "${m815_log}" ]]; then
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m815_python}" - "${m815_log}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).open("x").close()
PY
    fi
    [[ -f "${m815_log}" && ! -L "${m815_log}" ]]
}

m815_fail_closed() {
    local m815_rc=$?
    trap - EXIT
    if [[ "${m815_started}" -eq 1 && "${m815_success}" -ne 1 ]]; then
        [[ "${m815_rc}" -ne 0 ]] || m815_rc=98
        local m815_partial=""
        if [[ "${m815_published}" -eq 1 && \
              ( -e "${m815_result}" || -L "${m815_result}" ) ]]; then
            mv -T --no-clobber -- "${m815_result}" \
                "${m815_partial_artifact}" || exit 99
            m815_partial="${m815_partial_artifact}"
        elif [[ -e "${m815_stage}" || -L "${m815_stage}" ]]; then
            mv -T --no-clobber -- "${m815_stage}" \
                "${m815_partial_artifact}" || exit 99
            m815_partial="${m815_partial_artifact}"
        fi
        m815_ensure_empty_regular_log "${m815_stdout_log}" || exit 99
        m815_ensure_empty_regular_log "${m815_stderr_log}" || exit 99
        /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
            "${m815_python}" "${m815_driver}" \
            --write-failure-receipt \
            --candidate "${m815_candidate}" \
            --release "${m815_release}" \
            --attempt "${m815_attempt}" \
            --runner "${m815_runner}" \
            --stdout-log "${m815_stdout_log}" \
            --stderr-log "${m815_stderr_log}" \
            --output "${m815_quarantine}" \
            --expected-runner-sha256 "${M815_EXPECTED_RUNNER_SHA256}" \
            --expected-release-sha256 "${M815_EXPECTED_RELEASE_SHA256}" \
            --return-code "${m815_rc}" \
            --phase "${m815_phase}" \
            --partial-artifact "${m815_partial}" >/dev/null || exit 99
        [[ -d "${m815_quarantine}" && ! -L "${m815_quarantine}" && \
           -f "${m815_quarantine}/failure.json" && \
           -f "${m815_quarantine}/driver.log" && \
           -f "${m815_quarantine}/SHA256SUMS" && \
           -f "${m815_quarantine}/SHA256SUMS.seal.sha256" ]] || exit 99
        rm -f -- "${m815_stdout_log}" "${m815_stderr_log}"
    fi
    exit "${m815_rc}"
}
trap m815_fail_closed EXIT

[[ "${m815_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m815_runner}" == \
   "${m815_hw_root}/system_simulator/scripts/run_m815_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M815 canonical path drift" >&2
    exit 3
}
[[ -n "${M815_EXPECTED_RUNNER_SHA256:-}" && \
   "${M815_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m815_runner}" | awk '{print $1}')" == \
   "${M815_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M815 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M815_EXPECTED_RELEASE_SHA256:-}" && \
   "${M815_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m815_release}" && ! -L "${m815_release}" && \
   "$(sha256sum "${m815_release}" | awk '{print $1}')" == \
   "${M815_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M815 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m815_python}" && ! -L "${m815_python}" && \
   "$(sha256sum "${m815_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M815 Python identity drift" >&2
    exit 4
}
[[ -f "${m815_driver}" && ! -L "${m815_driver}" && \
   -f "${m815_candidate}" && ! -L "${m815_candidate}" ]] || {
    echo "M815 source files are absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m815_result}" && ! -L "${m815_result}" && \
   ! -e "${m815_attempt}" && ! -L "${m815_attempt}" && \
   ! -e "${m815_attempt_stage}" && ! -L "${m815_attempt_stage}" && \
   ! -e "${m815_stage}" && ! -L "${m815_stage}" && \
   ! -e "${m815_quarantine}" && ! -L "${m815_quarantine}" && \
   ! -e "${m815_partial_artifact}" && ! -L "${m815_partial_artifact}" && \
   ! -e "${m815_stdout_log}" && ! -L "${m815_stdout_log}" && \
   ! -e "${m815_stderr_log}" && ! -L "${m815_stderr_log}" ]] || {
    echo "M815 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M815_EXPECTED_RELEASE_SHA256="${M815_EXPECTED_RELEASE_SHA256}" \
    "${m815_python}" "${m815_driver}" \
    --validate-release-preflight \
    --candidate "${m815_candidate}" \
    --release "${m815_release}" >/dev/null

m815_free_kib=$(df -Pk "$(dirname "${m815_result}")" | awk 'NR==2 {print $4}')
m815_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m815_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m815_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m815_commit_headroom=$((m815_commit_limit - m815_committed))
[[ "${m815_free_kib}" -ge 1048576 && \
   "${m815_mem_available}" -ge 67108864 && \
   "${m815_commit_headroom}" -ge 67108864 ]] || {
    echo "M815 resource gate failed without consuming one-shot" >&2
    exit 40
}

mkdir "${m815_attempt_stage}"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m815_python}" - "${m815_attempt_stage}/attempt.json" \
    "${m815_runner}" "${m815_driver}" "${m815_candidate}" \
    "${m815_release}" "${m815_result}" <<'PY'
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
    "schema": "m815_m785_decoder_production_attempt_v1",
    "status": "CONSUMED_IMMEDIATELY_BEFORE_M815_PRODUCTION_REPLAY",
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
(cd "${m815_attempt_stage}" && \
    sha256sum attempt.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv -T --no-clobber -- "${m815_attempt_stage}" "${m815_attempt}"
m815_started=1
m815_phase="ATTEMPT_PUBLISHED_POSTCHECK"
[[ -d "${m815_attempt}" && ! -L "${m815_attempt}" && \
   ! -e "${m815_attempt_stage}" && ! -L "${m815_attempt_stage}" && \
   -f "${m815_attempt}/attempt.json" && \
   -f "${m815_attempt}/SHA256SUMS" && \
   -f "${m815_attempt}/SHA256SUMS.seal.sha256" ]] || exit 41

m815_phase="ATTEMPT_REQUIRED_PREFLIGHT"
m815_ensure_empty_regular_log "${m815_stdout_log}"
m815_ensure_empty_regular_log "${m815_stderr_log}"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M815_EXPECTED_RELEASE_SHA256="${M815_EXPECTED_RELEASE_SHA256}" \
    "${m815_python}" "${m815_driver}" \
    --validate-consumed-attempt \
    --candidate "${m815_candidate}" \
    --release "${m815_release}" \
    --attempt "${m815_attempt}" \
    >>"${m815_stdout_log}" 2>>"${m815_stderr_log}"
m815_driver_rc=$?
set -e
[[ "${m815_driver_rc}" -eq 0 ]] || exit "${m815_driver_rc}"

m815_phase="RUN_PRODUCTION_DRIVER"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M815_EXPECTED_RELEASE_SHA256="${M815_EXPECTED_RELEASE_SHA256}" \
    "${m815_python}" "${m815_driver}" \
    --run-production \
    --candidate "${m815_candidate}" \
    --release "${m815_release}" \
    --attempt "${m815_attempt}" \
    --output "${m815_stage}" \
    >>"${m815_stdout_log}" 2>>"${m815_stderr_log}"
m815_driver_rc=$?
set -e
[[ "${m815_driver_rc}" -eq 0 ]] || exit "${m815_driver_rc}"

m815_phase="VERIFY_AND_SEAL_STAGE"
[[ -d "${m815_stage}" && ! -L "${m815_stage}" && \
   -f "${m815_stage}/result.json" && ! -L "${m815_stage}/result.json" && \
   -f "${m815_stage}/detailed_rows.json" && \
   ! -L "${m815_stage}/detailed_rows.json" ]] || exit 50
(cd "${m815_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m815_phase="PUBLISH_CANONICAL_NOREPLACE"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m815_python}" "${m815_driver}" \
    --publish-no-replace \
    --candidate "${m815_candidate}" \
    --output "${m815_stage}" \
    --publish-to "${m815_result}" >/dev/null
m815_published=1
m815_phase="VERIFY_CANONICAL_PUBLICATION"
[[ -d "${m815_result}" && ! -L "${m815_result}" && \
   ! -e "${m815_stage}" && ! -L "${m815_stage}" && \
   -f "${m815_result}/result.json" && \
   -f "${m815_result}/detailed_rows.json" && \
   -f "${m815_result}/SHA256SUMS" && \
   -f "${m815_result}/SHA256SUMS.seal.sha256" ]] || exit 51
m815_success=1
trap - EXIT
rm -f -- "${m815_stdout_log}" "${m815_stderr_log}"
echo "PASS_M815_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
