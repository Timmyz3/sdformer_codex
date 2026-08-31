#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m998_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m998_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m998_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m998_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m998_driver="${m998_hw}/system_simulator/scripts/execute_m994_m982_decoder_canonical_attempt_source_r1.py"
m998_contract="${m998_hw}/contracts/m994_m982_decoder_canonical_attempt_source_contract_r1_20260829.json"
m998_result="${m998_hw}/results/m998_m994_decoder_d2d3_10k_canonical_attempt_r1_20260829"
m998_attempt="${m998_hw}/results/.m998_m994_decoder_d2d3_10k_canonical_attempt_consumed"
m998_work="${m998_result}.work.$$.$RANDOM.$RANDOM"
m998_quarantine="${m998_result}.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m998_timeout=/usr/bin/timeout
m998_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m998_started=0
m998_published=0

: "${M998_EXPECTED_M996_RELEASE_SHA:?M998 inert until M996 release}"
: "${M998_EXPECTED_M995_REVIEW_SHA:?M998 requires M995 review SHA}"
: "${M998_EXPECTED_M995_MANIFEST_SHA:?M998 requires M995 manifest SHA}"
: "${M998_EXPECTED_M995_OUTER_SHA:?M998 requires M995 outer SHA}"
: "${M998_EXPECTED_M997_REVIEW_SHA:?M998 requires M997 review SHA}"
: "${M998_EXPECTED_M997_MANIFEST_SHA:?M998 requires M997 manifest SHA}"
: "${M998_EXPECTED_M997_OUTER_SHA:?M998 requires M997 outer SHA}"

m998_sha(){ /usr/bin/sha256sum "$1"|/usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m998_python}")" == "${m998_python}" &&
   "$(m998_sha "${m998_python}")" == "${m998_python_sha}" &&
   "$(m998_sha "${m998_timeout}")" == "${m998_timeout_sha}" ]] || exit 4

m998_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 "${m998_python}" "${m998_driver}" "$@"
}
m998_auth(){
  m998_py "$@" --runner "${m998_runner}" \
    --expected-release-sha "${M998_EXPECTED_M996_RELEASE_SHA}" \
    --expected-source-review-sha "${M998_EXPECTED_M995_REVIEW_SHA}" \
    --expected-source-manifest-sha "${M998_EXPECTED_M995_MANIFEST_SHA}" \
    --expected-source-outer-sha "${M998_EXPECTED_M995_OUTER_SHA}" \
    --expected-release-review-sha "${M998_EXPECTED_M997_REVIEW_SHA}" \
    --expected-release-manifest-sha "${M998_EXPECTED_M997_MANIFEST_SHA}" \
    --expected-release-outer-sha "${M998_EXPECTED_M997_OUTER_SHA}"
}

m998_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m998_started}" -eq 1 && "${m998_published}" -eq 0 &&
        -d "${m998_work}" ]];then
    if ! m998_py --quarantine-work --work "${m998_work}" \
         --quarantine "${m998_quarantine}" --return-code "${rc}" >/dev/null;then
      /usr/bin/printf 'M998_WORK_RETAINED_NOT_MOVED=%s\n' "${m998_work}" >&2
    fi
  fi
  # The canonical attempt is never repaired, removed, renamed, or quarantined.
  # Even an empty directory left by interruption is permanent consumption.
  if [[ -d "${m998_attempt}" ]];then
    /usr/bin/printf 'M998_CANONICAL_ATTEMPT_PRESERVED=%s\n' "${m998_attempt}" >&2
  fi
  exit "${rc}"
}
trap m998_cleanup EXIT HUP INT TERM

[[ ! -e "${m998_result}" && ! -L "${m998_result}" &&
   ! -e "${m998_attempt}" && ! -L "${m998_attempt}" &&
   ! -e "${m998_work}" && ! -L "${m998_work}" ]] || exit 5

m998_py --validate-source --contract "${m998_contract}" \
  --runner "${m998_runner}" >/dev/null
m998_auth --validate-authority >/dev/null

read -r m998_mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r m998_limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r m998_used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${m998_mem}" -ge 16777216 && $((m998_limit-m998_used)) -ge 16777216 ]] || exit 6

# M994 creates the canonical attempt directory directly. There is no random
# attempt stage and no later rename that could reopen the one-shot window.
m998_auth --consume-attempt >/dev/null
m998_started=1
/usr/bin/mkdir -m 700 "${m998_work}"
/usr/bin/printf '%s\n' M998_WORK_CREATED_AFTER_CANONICAL_ATTEMPT >"${m998_work}/WORK_STARTED.txt"

for m998_layer in D2 D3;do
  "${m998_timeout}" --foreground --signal=TERM --kill-after=30s 1800s \
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 "${m998_python}" "${m998_driver}" \
    --run-row "${m998_layer}" --row-stage "${m998_work}/${m998_layer}" \
    --runner "${m998_runner}" \
    --expected-release-sha "${M998_EXPECTED_M996_RELEASE_SHA}" \
    --expected-source-review-sha "${M998_EXPECTED_M995_REVIEW_SHA}" \
    --expected-source-manifest-sha "${M998_EXPECTED_M995_MANIFEST_SHA}" \
    --expected-source-outer-sha "${M998_EXPECTED_M995_OUTER_SHA}" \
    --expected-release-review-sha "${M998_EXPECTED_M997_REVIEW_SHA}" \
    --expected-release-manifest-sha "${M998_EXPECTED_M997_MANIFEST_SHA}" \
    --expected-release-outer-sha "${M998_EXPECTED_M997_OUTER_SHA}" >/dev/null
done

m998_auth --assemble --work "${m998_work}" >/dev/null
m998_auth --publish --work "${m998_work}" >/dev/null
m998_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' 'PASS M998 one D2-then-D3 10K run published; result hammer required'
