#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m1062_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m1062_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1062_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m1062_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m1062_driver="${m1062_hw}/system_simulator/scripts/execute_m1060_decoder_identity_binding_repair.py"
m1062_driver_sha=440d6a12e19ac5561627ae9181d9b6f8ae1be23b1e988c139816a5261c760eb1
m1062_contract="${m1062_hw}/contracts/m1060_decoder_identity_binding_repair_contract_r1_20260830.json"
m1062_result="${m1062_hw}/results/m1062_m1060_decoder_identity_binding_pilot_r1_20260830"
m1062_attempt="${m1062_hw}/results/.m1062_m1060_decoder_identity_binding_pilot_attempt_consumed"
m1062_work="${m1062_hw}/results/.m1062_m1060_decoder_identity_binding_pilot_r1_20260830.work.$$.$RANDOM.$RANDOM"
m1062_quarantine="${m1062_hw}/results/m1062_m1060_decoder_identity_binding_pilot_r1_20260830.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m1062_timeout=/usr/bin/timeout
m1062_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m1062_flock=/usr/bin/flock
m1062_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
m1062_started=0
m1062_published=0

: "${M1062_EXPECTED_CONTRACT_SHA:?M1062 inert until exact M1060 contract pin}"
: "${M1062_EXPECTED_M1061_REVIEW_SHA:?M1062 requires independent M1061 review pin}"
: "${M1062_EXPECTED_M1061_MANIFEST_SHA:?M1062 requires independent M1061 manifest pin}"
: "${M1062_EXPECTED_M1061_OUTER_SHA:?M1062 requires independent M1061 outer pin}"

m1062_sha(){ /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m1062_python}")" == "${m1062_python}" &&
   "$(m1062_sha "${m1062_python}")" == "${m1062_python_sha}" &&
   "$(m1062_sha "${m1062_driver}")" == "${m1062_driver_sha}" &&
   "$(m1062_sha "${m1062_timeout}")" == "${m1062_timeout_sha}" &&
   "$(m1062_sha "${m1062_flock}")" == "${m1062_flock_sha}" ]] || exit 4

m1062_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "${m1062_python}" "${m1062_driver}" "$@"
}
m1062_auth(){
  m1062_py "$@" \
    --expected-review-sha "${M1062_EXPECTED_M1061_REVIEW_SHA}" \
    --expected-manifest-sha "${M1062_EXPECTED_M1061_MANIFEST_SHA}" \
    --expected-outer-sha "${M1062_EXPECTED_M1061_OUTER_SHA}"
}

m1062_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m1062_started}" -eq 1 && "${m1062_published}" -eq 0 &&
        -d "${m1062_work}" ]]; then
    if ! m1062_py --quarantine --work "${m1062_work}" \
         --quarantine-path "${m1062_quarantine}" \
         --return-code "${rc}" >/dev/null; then
      /usr/bin/printf 'M1062_WORK_RETAINED_NOT_MOVED=%s\n' "${m1062_work}" >&2
    fi
  fi
  if [[ -d "${m1062_attempt}" ]]; then
    /usr/bin/printf 'M1062_CANONICAL_ATTEMPT_PRESERVED=%s\n' "${m1062_attempt}" >&2
  fi
  exit "${rc}"
}
trap m1062_cleanup EXIT HUP INT TERM

[[ ! -e "${m1062_result}" && ! -L "${m1062_result}" &&
   ! -e "${m1062_attempt}" && ! -L "${m1062_attempt}" &&
   ! -e "${m1062_work}" && ! -L "${m1062_work}" &&
   ! -e "${m1062_quarantine}" && ! -L "${m1062_quarantine}" ]] || exit 5

# Root-seal metadata only. No manifest.json or calls/* stat/open/hash here.
m1062_py --validate-pre-attempt-source --contract "${m1062_contract}" \
  --runner "${m1062_runner}" \
  --expected-contract-sha "${M1062_EXPECTED_CONTRACT_SHA}" >/dev/null
m1062_auth --validate-authority >/dev/null

exec 9>"/tmp/m1062_decoder_identity_binding_pilot.lock"
"${m1062_flock}" -n 9 || exit 6
for m1062_process in dc_shell vcs simv fm_shell pt_shell; do
  if /usr/bin/pgrep -u "$(/usr/bin/id -u)" -x "${m1062_process}" >/dev/null; then
    exit 7
  fi
done
read -r m1062_mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r m1062_limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r m1062_used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${m1062_mem}" -ge 16777216 &&
   $((m1062_limit-m1062_used)) -ge 16777216 ]] || exit 8

# First state change: permanently consume the canonical attempt.
m1062_auth --consume-attempt --attempt "${m1062_attempt}" \
  --runner "${m1062_runner}" \
  --expected-contract-sha "${M1062_EXPECTED_CONTRACT_SHA}" >/dev/null
m1062_started=1
/usr/bin/mkdir -m 700 "${m1062_work}"

# Only after attempt consumption: full manifest/member verification and a
# canonical context cross-bound to attempt.json, runner and contract.
m1062_auth --validate-payload-after-attempt --attempt "${m1062_attempt}" \
  --work "${m1062_work}" --runner "${m1062_runner}" \
  --expected-contract-sha "${M1062_EXPECTED_CONTRACT_SHA}" >/dev/null

"${m1062_timeout}" --foreground --signal=TERM --kill-after=60s 14400s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  "${m1062_python}" "${m1062_driver}" --run-pilot \
  --attempt "${m1062_attempt}" --work "${m1062_work}" \
  --runner "${m1062_runner}" \
  --expected-contract-sha "${M1062_EXPECTED_CONTRACT_SHA}" \
  --expected-review-sha "${M1062_EXPECTED_M1061_REVIEW_SHA}" \
  --expected-manifest-sha "${M1062_EXPECTED_M1061_MANIFEST_SHA}" \
  --expected-outer-sha "${M1062_EXPECTED_M1061_OUTER_SHA}" >/dev/null

# Assemble re-derives the canonical context and re-hashes the selected members.
m1062_auth --assemble --attempt "${m1062_attempt}" --work "${m1062_work}" \
  --runner "${m1062_runner}" \
  --expected-contract-sha "${M1062_EXPECTED_CONTRACT_SHA}" >/dev/null
m1062_auth --publish --attempt "${m1062_attempt}" --work "${m1062_work}" \
  --result "${m1062_result}" --runner "${m1062_runner}" \
  --expected-contract-sha "${M1062_EXPECTED_CONTRACT_SHA}" >/dev/null
m1062_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' \
  'PASS M1062 identity-bound diagnostic pilot published; independent result hammer required'
