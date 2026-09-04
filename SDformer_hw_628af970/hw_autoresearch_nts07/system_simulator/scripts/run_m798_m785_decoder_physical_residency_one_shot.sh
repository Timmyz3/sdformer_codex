#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M798 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m798_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m798_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m798_repo_root="$(cd "${m798_hw_root}/.." && pwd)"
m798_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m798_driver="${m798_hw_root}/system_simulator/scripts/execute_m798_m785_decoder_physical_residency_production.py"
m798_candidate="${m798_hw_root}/contracts/m798_m785_decoder_physical_residency_production_release_candidate_r1_20260828.json"
m798_release="${m798_hw_root}/contracts/m798_m785_decoder_physical_residency_production_true_release_r1_20260828.json"
m798_result="${m798_hw_root}/results/m798_m785_h67_decoder_physical_residency_cycles_r1_20260828"
m798_attempt="${m798_hw_root}/results/.m798_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
m798_stage="${m798_result}.stage.$$.${RANDOM}.${RANDOM}"
m798_quarantine="${m798_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"
m798_started=0
m798_published=0
m798_success=0

m798_fail_closed() {
    local m798_rc=$?
    if [[ "${m798_started}" -eq 1 && "${m798_success}" -ne 1 ]]; then
        if [[ "${m798_published}" -eq 1 && \
              ( -e "${m798_result}" || -L "${m798_result}" ) ]]; then
            mv -T --no-clobber -- "${m798_result}" "${m798_quarantine}" || exit 99
            [[ ! -e "${m798_result}" && ! -L "${m798_result}" && \
               -d "${m798_quarantine}" && ! -L "${m798_quarantine}" ]] || exit 99
        elif [[ -e "${m798_stage}" || -L "${m798_stage}" ]]; then
            mv -T --no-clobber -- "${m798_stage}" "${m798_quarantine}" || exit 99
            [[ ! -e "${m798_stage}" && ! -L "${m798_stage}" && \
               -d "${m798_quarantine}" && ! -L "${m798_quarantine}" ]] || exit 99
        fi
    fi
    exit "${m798_rc}"
}
trap m798_fail_closed EXIT

[[ "${m798_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m798_runner}" == \
   "${m798_hw_root}/system_simulator/scripts/run_m798_m785_decoder_physical_residency_one_shot.sh" ]] || {
    echo "M798 canonical path drift" >&2
    exit 3
}
[[ -n "${M798_EXPECTED_RUNNER_SHA256:-}" && \
   "${M798_EXPECTED_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m798_runner}" | awk '{print $1}')" == \
   "${M798_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M798 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M798_EXPECTED_RELEASE_SHA256:-}" && \
   "${M798_EXPECTED_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ && \
   -f "${m798_release}" && ! -L "${m798_release}" && \
   "$(sha256sum "${m798_release}" | awk '{print $1}')" == \
   "${M798_EXPECTED_RELEASE_SHA256}" ]] || {
    echo "M798 caller must supply the independently reviewed true-release SHA" >&2
    exit 3
}
[[ -x "${m798_python}" && ! -L "${m798_python}" && \
   "$(sha256sum "${m798_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M798 Python identity drift" >&2
    exit 4
}
[[ -f "${m798_driver}" && ! -L "${m798_driver}" && \
   -f "${m798_candidate}" && ! -L "${m798_candidate}" ]] || {
    echo "M798 source files are absent or nonregular" >&2
    exit 4
}
[[ ! -e "${m798_result}" && ! -L "${m798_result}" && \
   ! -e "${m798_attempt}" && ! -L "${m798_attempt}" && \
   ! -e "${m798_stage}" && ! -L "${m798_stage}" && \
   ! -e "${m798_quarantine}" && ! -L "${m798_quarantine}" ]] || {
    echo "M798 one-shot/result/stage/quarantine path already exists" >&2
    exit 5
}

# This exact release/candidate/source-hammer validation is before one-shot
# consumption and creates no result.
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M798_EXPECTED_RELEASE_SHA256="${M798_EXPECTED_RELEASE_SHA256}" \
    "${m798_python}" "${m798_driver}" \
    --validate-release-preflight \
    --candidate "${m798_candidate}" \
    --release "${m798_release}" >/dev/null

m798_free_kib=$(df -Pk "$(dirname "${m798_result}")" | awk 'NR==2 {print $4}')
m798_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m798_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m798_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m798_commit_headroom=$((m798_commit_limit - m798_committed))
[[ "${m798_free_kib}" -ge 1048576 && \
   "${m798_mem_available}" -ge 67108864 && \
   "${m798_commit_headroom}" -ge 67108864 ]] || {
    echo "M798 resource gate failed without consuming one-shot" >&2
    exit 40
}

mkdir "${m798_attempt}"
mkdir "${m798_attempt}/initial"
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m798_python}" - "${m798_attempt}/initial/attempt.json" \
    "${m798_runner}" "${m798_driver}" "${m798_candidate}" \
    "${m798_release}" "${m798_result}" <<'PY'
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
    "schema": "m798_m785_decoder_production_attempt_v1",
    "status": "CONSUMED_IMMEDIATELY_BEFORE_M798_PRODUCTION_REPLAY",
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
(cd "${m798_attempt}/initial" && \
    sha256sum attempt.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd "${m798_attempt}" && \
    sha256sum initial/SHA256SUMS.seal.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m798_started=1
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M798_EXPECTED_RELEASE_SHA256="${M798_EXPECTED_RELEASE_SHA256}" \
    "${m798_python}" "${m798_driver}" \
    --run-production \
    --candidate "${m798_candidate}" \
    --release "${m798_release}" \
    --attempt "${m798_attempt}" \
    --output "${m798_stage}"

[[ -d "${m798_stage}" && ! -L "${m798_stage}" && \
   -f "${m798_stage}/result.json" && ! -L "${m798_stage}/result.json" && \
   -f "${m798_stage}/detailed_rows.json" && \
   ! -L "${m798_stage}/detailed_rows.json" ]] || {
    echo "M798 production did not publish complete staging artifacts" >&2
    exit 50
}
(cd "${m798_stage}" && \
    sha256sum result.json detailed_rows.json >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

# The driver uses renameat2(RENAME_NOREPLACE).  A destination created after
# preflight is an EEXIST failure; stage cannot be nested below canonical result.
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "${m798_python}" "${m798_driver}" \
    --publish-no-replace \
    --candidate "${m798_candidate}" \
    --output "${m798_stage}" \
    --publish-to "${m798_result}" >/dev/null
m798_published=1
[[ -d "${m798_result}" && ! -L "${m798_result}" && \
   ! -e "${m798_stage}" && ! -L "${m798_stage}" && \
   -f "${m798_result}/result.json" && \
   ! -L "${m798_result}/result.json" && \
   -f "${m798_result}/detailed_rows.json" && \
   ! -L "${m798_result}/detailed_rows.json" && \
   -f "${m798_result}/SHA256SUMS" && \
   ! -L "${m798_result}/SHA256SUMS" && \
   -f "${m798_result}/SHA256SUMS.seal.sha256" && \
   ! -L "${m798_result}/SHA256SUMS.seal.sha256" ]] || {
    echo "M798 canonical root four-member publication failed" >&2
    exit 51
}
m798_success=1
trap - EXIT
echo "PASS_M798_PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED"
