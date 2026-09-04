#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M793 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m793_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m793_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m793_repo_root="$(cd "${m793_hw_root}/.." && pwd)"
m793_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m793_driver="${m793_hw_root}/system_simulator/scripts/execute_m793_m785_decoder_physical_residency_production.py"
m793_candidate="${m793_hw_root}/contracts/m793_m785_decoder_physical_residency_production_release_candidate_r1_20260828.json"
m793_release="${m793_hw_root}/contracts/m793_m785_decoder_physical_residency_production_true_release_r1_20260828.json"
m793_result="${m793_hw_root}/results/m793_m785_h67_decoder_physical_residency_cycles_r1_20260828"
m793_attempt="${m793_hw_root}/results/.m793_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m793_stage="${m793_result}.stage.$$.${RANDOM}.${RANDOM}"
m793_quarantine="${m793_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m793_started=0
m793_success=0

m793_fail_closed() {
    local m793_rc=$?
    if [[ "${m793_started}" -eq 1 && "${m793_success}" -ne 1 && \
          ( -e "${m793_stage}" || -L "${m793_stage}" ) ]]; then
        mv -- "${m793_stage}" "${m793_quarantine}" || exit 99
    fi
    exit "${m793_rc}"
}
trap m793_fail_closed EXIT

[[ "${m793_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m793_runner}" == \
   "${m793_hw_root}/system_simulator/scripts/run_m793_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M793 canonical path drift" >&2
    exit 3
}
[[ -n "${M793_EXPECTED_RUNNER_SHA256:-}" && \
   "${M793_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m793_runner}" | awk '{print $1}')" == \
   "${M793_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M793 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M793_EXPECTED_RELEASE_SHA256:-}" && \
   "${M793_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m793_release}" && ! -L "${m793_release}" && \
   "$(sha256sum "${m793_release}" | awk '{print $1}')" == \
   "${M793_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M793 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m793_python}" && ! -L "${m793_python}" && \
   "$(sha256sum "${m793_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M793 Python identity drift" >&2
    exit 4
}
[[ -f "${m793_driver}" && ! -L "${m793_driver}" && \
   -f "${m793_candidate}" && ! -L "${m793_candidate}" ]] || {
    echo "M793 source files are absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m793_result}" && ! -L "${m793_result}" && \
   ! -e "${m793_attempt}" && ! -L "${m793_attempt}" && \
   ! -e "${m793_stage}" && ! -L "${m793_stage}" && \
   ! -e "${m793_quarantine}" && ! -L "${m793_quarantine}" ]] || {
    echo "M793 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

# True-release preflight is before one-shot consumption and creates no result.
# It checks the frozen candidate/source chain while requiring the attempt and
# result to remain absent.
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M793_EXPECTED_RELEASE_SHA256="${M793_EXPECTED_RELEASE_SHA256}" \
    "${m793_python}" "${m793_driver}" \
    --validate-release-preflight \
    --candidate "${m793_candidate}" \
    --release "${m793_release}" >/dev/null

m793_free_kib=$(df -Pk "$(dirname "${m793_result}")" | awk 'NR==2 {print $4}')
m793_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m793_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m793_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m793_commit_headroom=$((m793_commit_limit - m793_committed))
[[ "${m793_free_kib}" -ge 1048576 && \
   "${m793_mem_available}" -ge 67108864 && \
   "${m793_commit_headroom}" -ge 67108864 ]] || {
    echo "M793 resource gate failed without consuming one-shot" >&2
    exit 40
}

mkdir "${m793_attempt}"
mkdir "${m793_attempt}/initial"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M793_EXPECTED_RELEASE_SHA256="${M793_EXPECTED_RELEASE_SHA256}" \
    "${m793_python}" - "${m793_attempt}/initial/attempt.json" \
    "${m793_runner}" "${m793_driver}" "${m793_candidate}" \
    "${m793_release}" "${m793_result}" <<'PY'
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
    "schema": "m793_m785_decoder_production_attempt_v1",
    "status": "CONSUMED_IMMEDIATELY_BEFORE_M793_PRODUCTION_REPLAY",
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
(cd "${m793_attempt}/initial" && \
    sha256sum attempt.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd "${m793_attempt}" && \
    sha256sum initial/SHA256SUMS.seal.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m793_started=1
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M793_EXPECTED_RELEASE_SHA256="${M793_EXPECTED_RELEASE_SHA256}" \
    "${m793_python}" "${m793_driver}" \
    --run-production \
    --candidate "${m793_candidate}" \
    --release "${m793_release}" \
    --attempt "${m793_attempt}" \
    --output "${m793_stage}"

[[ -d "${m793_stage}" && ! -L "${m793_stage}" && \
   -f "${m793_stage}/result.json" && ! -L "${m793_stage}/result.json" && \
   -f "${m793_stage}/detailed_rows.json" && \
   ! -L "${m793_stage}/detailed_rows.json" ]] || {
    echo "M793 production did not publish complete staging artifacts" >&2
    exit 50
}
(cd "${m793_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv -- "${m793_stage}" "${m793_result}"
[[ -d "${m793_result}" && ! -L "${m793_result}" && \
   ! -e "${m793_stage}" && ! -L "${m793_stage}" ]] || {
    echo "M793 atomic result publication failed" >&2
    exit 51
}
m793_success=1
trap - EXIT
echo "PASS_M793_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
