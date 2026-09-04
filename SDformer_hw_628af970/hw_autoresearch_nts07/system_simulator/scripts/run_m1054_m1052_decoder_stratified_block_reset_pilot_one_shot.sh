#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m1054_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m1054_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1054_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m1054_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m1054_driver="${m1054_hw}/system_simulator/scripts/execute_m1052_decoder_stratified_block_reset_pilot_repair.py"
m1054_driver_sha=756bf90d52505a68f089dd42296244b94b9c9a50cf013efc0dbc02cd6bb25cec
m1054_contract="${m1054_hw}/contracts/m1052_decoder_stratified_block_reset_pilot_repair_contract_r1_20260829.json"
m1054_result="${m1054_hw}/results/m1054_m1052_decoder_stratified_block_reset_pilot_r1_20260829"
m1054_attempt="${m1054_hw}/results/.m1054_m1052_decoder_stratified_block_reset_pilot_attempt_consumed"
m1054_work="${m1054_hw}/results/.m1054_m1052_decoder_stratified_block_reset_pilot_r1_20260829.work.$$.$RANDOM.$RANDOM"
m1054_quarantine="${m1054_hw}/results/m1054_m1052_decoder_stratified_block_reset_pilot_r1_20260829.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m1054_timeout=/usr/bin/timeout
m1054_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m1054_flock=/usr/bin/flock
m1054_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
m1054_started=0
m1054_published=0

: "${M1054_EXPECTED_CONTRACT_SHA:?M1054 inert until exact M1052 contract pin}"
: "${M1054_EXPECTED_M1053_REVIEW_SHA:?M1054 requires independent M1053 review pin}"
: "${M1054_EXPECTED_M1053_MANIFEST_SHA:?M1054 requires independent M1053 manifest pin}"
: "${M1054_EXPECTED_M1053_OUTER_SHA:?M1054 requires independent M1053 outer pin}"

m1054_sha(){ /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m1054_python}")" == "${m1054_python}" &&
   "$(m1054_sha "${m1054_python}")" == "${m1054_python_sha}" &&
   "$(m1054_sha "${m1054_driver}")" == "${m1054_driver_sha}" &&
   "$(m1054_sha "${m1054_timeout}")" == "${m1054_timeout_sha}" &&
   "$(m1054_sha "${m1054_flock}")" == "${m1054_flock_sha}" ]] || exit 4

m1054_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "${m1054_python}" "${m1054_driver}" "$@"
}
m1054_auth(){
  m1054_py "$@" \
    --expected-review-sha "${M1054_EXPECTED_M1053_REVIEW_SHA}" \
    --expected-manifest-sha "${M1054_EXPECTED_M1053_MANIFEST_SHA}" \
    --expected-outer-sha "${M1054_EXPECTED_M1053_OUTER_SHA}"
}

m1054_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m1054_started}" -eq 1 && "${m1054_published}" -eq 0 &&
        -d "${m1054_work}" ]]; then
    if ! m1054_py --quarantine --work "${m1054_work}" \
         --quarantine-path "${m1054_quarantine}" \
         --return-code "${rc}" >/dev/null; then
      /usr/bin/printf 'M1054_WORK_RETAINED_NOT_MOVED=%s\n' "${m1054_work}" >&2
    fi
  fi
  if [[ -d "${m1054_attempt}" ]]; then
    /usr/bin/printf 'M1054_CANONICAL_ATTEMPT_PRESERVED=%s\n' "${m1054_attempt}" >&2
  fi
  exit "${rc}"
}
trap m1054_cleanup EXIT HUP INT TERM

[[ ! -e "${m1054_result}" && ! -L "${m1054_result}" &&
   ! -e "${m1054_attempt}" && ! -L "${m1054_attempt}" &&
   ! -e "${m1054_work}" && ! -L "${m1054_work}" &&
   ! -e "${m1054_quarantine}" && ! -L "${m1054_quarantine}" ]] || exit 5

# This pre-attempt call may read contracts, code, review seals and the M699
# root seal metadata only.  The driver forbids payload-member stat/open/hash.
m1054_py --validate-pre-attempt-source --contract "${m1054_contract}" \
  --runner "${m1054_runner}" \
  --expected-contract-sha "${M1054_EXPECTED_CONTRACT_SHA}" >/dev/null
m1054_auth --validate-authority >/dev/null

exec 9>"/tmp/m1054_decoder_stratified_block_reset_pilot.lock"
"${m1054_flock}" -n 9 || exit 6
for m1054_process in dc_shell vcs simv fm_shell pt_shell; do
  if /usr/bin/pgrep -u "$(/usr/bin/id -u)" -x "${m1054_process}" >/dev/null; then
    exit 7
  fi
done
read -r m1054_mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r m1054_limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r m1054_used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${m1054_mem}" -ge 16777216 &&
   $((m1054_limit-m1054_used)) -ge 16777216 ]] || exit 8

# Permanent attempt consumption is the first state change and occurs before
# work creation and before any full payload-member validation.
m1054_auth --consume-attempt --attempt "${m1054_attempt}" \
  --runner "${m1054_runner}" \
  --expected-contract-sha "${M1054_EXPECTED_CONTRACT_SHA}" >/dev/null
m1054_started=1
/usr/bin/mkdir -m 700 "${m1054_work}"

# Full M699/M705/M785 member hashing begins only here, inside the consumed
# attempt. Any failure is moved to the unique quarantine by the EXIT trap.
m1054_auth --validate-payload-after-attempt --attempt "${m1054_attempt}" \
  --work "${m1054_work}" >/dev/null

"${m1054_timeout}" --foreground --signal=TERM --kill-after=60s 14400s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  "${m1054_python}" "${m1054_driver}" --run-pilot \
  --attempt "${m1054_attempt}" --work "${m1054_work}" \
  --expected-review-sha "${M1054_EXPECTED_M1053_REVIEW_SHA}" \
  --expected-manifest-sha "${M1054_EXPECTED_M1053_MANIFEST_SHA}" \
  --expected-outer-sha "${M1054_EXPECTED_M1053_OUTER_SHA}" >/dev/null
m1054_auth --assemble --work "${m1054_work}" >/dev/null
m1054_auth --publish --work "${m1054_work}" --result "${m1054_result}" >/dev/null
m1054_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' \
  'PASS M1054 strict raw-cycle diagnostic pilot published; independent result hammer required'
