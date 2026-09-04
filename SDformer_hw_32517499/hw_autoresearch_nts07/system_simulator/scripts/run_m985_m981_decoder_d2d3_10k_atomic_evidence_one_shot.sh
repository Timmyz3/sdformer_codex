#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m985_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m985_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m985_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m985_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m985_driver="${m985_hw}/system_simulator/scripts/execute_m981_m977_decoder_d2d3_10k_atomic_evidence_r1.py"
m985_contract="${m985_hw}/contracts/m981_m977_decoder_d2d3_10k_atomic_evidence_source_contract_r1_20260829.json"
m985_result="${m985_hw}/results/m985_m981_decoder_d2d3_10k_atomic_evidence_r1_20260829"
m985_attempt="${m985_hw}/results/.m985_m981_decoder_d2d3_10k_atomic_evidence_attempt_consumed"
m985_attempt_stage="${m985_attempt}.stage.$$.$RANDOM.$RANDOM"
m985_work="${m985_result}.work.$$.$RANDOM.$RANDOM"
m985_quarantine="${m985_result}.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m985_timeout=/usr/bin/timeout
m985_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m985_started=0
m985_published=0

: "${M985_EXPECTED_M983_RELEASE_SHA:?M985 inert until M983 release}"
: "${M985_EXPECTED_M982_REVIEW_SHA:?M985 requires M982 review SHA}"
: "${M985_EXPECTED_M982_MANIFEST_SHA:?M985 requires M982 manifest SHA}"
: "${M985_EXPECTED_M982_OUTER_SHA:?M985 requires M982 outer SHA}"
: "${M985_EXPECTED_M984_REVIEW_SHA:?M985 requires M984 review SHA}"
: "${M985_EXPECTED_M984_MANIFEST_SHA:?M985 requires M984 manifest SHA}"
: "${M985_EXPECTED_M984_OUTER_SHA:?M985 requires M984 outer SHA}"

m985_sha(){ /usr/bin/sha256sum "$1"|/usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m985_python}")" == "${m985_python}" &&
   "$(m985_sha "${m985_python}")" == "${m985_python_sha}" &&
   "$(m985_sha "${m985_timeout}")" == "${m985_timeout_sha}" ]] || exit 4

m985_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 "${m985_python}" "${m985_driver}" "$@"
}
m985_auth(){
  m985_py "$@" --runner "${m985_runner}" \
    --expected-release-sha "${M985_EXPECTED_M983_RELEASE_SHA}" \
    --expected-source-review-sha "${M985_EXPECTED_M982_REVIEW_SHA}" \
    --expected-source-manifest-sha "${M985_EXPECTED_M982_MANIFEST_SHA}" \
    --expected-source-outer-sha "${M985_EXPECTED_M982_OUTER_SHA}" \
    --expected-release-review-sha "${M985_EXPECTED_M984_REVIEW_SHA}" \
    --expected-release-manifest-sha "${M985_EXPECTED_M984_MANIFEST_SHA}" \
    --expected-release-outer-sha "${M985_EXPECTED_M984_OUTER_SHA}"
}

m985_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m985_started}" -eq 1 && "${m985_published}" -eq 0 &&
        -d "${m985_work}" ]];then
    if ! m985_py --quarantine-work --work "${m985_work}" \
         --quarantine "${m985_quarantine}" --return-code "${rc}" >/dev/null;then
      /usr/bin/printf 'M985_WORK_RETAINED_NOT_MOVED=%s\n' "${m985_work}" >&2
    fi
  fi
  if [[ -d "${m985_attempt_stage}" ]];then
    /usr/bin/printf 'M985_ATTEMPT_STAGE_RETAINED_NOT_MOVED=%s\n' \
      "${m985_attempt_stage}" >&2
  fi
  exit "${rc}"
}
trap m985_cleanup EXIT HUP INT TERM

[[ ! -e "${m985_result}" && ! -L "${m985_result}" &&
   ! -e "${m985_attempt}" && ! -L "${m985_attempt}" &&
   ! -e "${m985_work}" && ! -L "${m985_work}" ]] || exit 5

m985_py --validate-source --contract "${m985_contract}" \
  --runner "${m985_runner}" >/dev/null
m985_auth --validate-authority >/dev/null

read -r m985_mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r m985_limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r m985_used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${m985_mem}" -ge 16777216 && $((m985_limit-m985_used)) -ge 16777216 ]] || exit 6

m985_auth --consume-attempt --attempt-stage "${m985_attempt_stage}" >/dev/null
m985_started=1
/usr/bin/mkdir -m 700 "${m985_work}"
/usr/bin/printf '%s\n' M985_WORK_CREATED_BEFORE_D2 >"${m985_work}/WORK_STARTED.txt"

# D2 is fully atomically sealed before D3 is created.
"${m985_timeout}" --foreground --signal=TERM --kill-after=30s 1800s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 "${m985_python}" "${m985_driver}" \
  --run-row D2 --row-stage "${m985_work}/D2" --runner "${m985_runner}" \
  --expected-release-sha "${M985_EXPECTED_M983_RELEASE_SHA}" \
  --expected-source-review-sha "${M985_EXPECTED_M982_REVIEW_SHA}" \
  --expected-source-manifest-sha "${M985_EXPECTED_M982_MANIFEST_SHA}" \
  --expected-source-outer-sha "${M985_EXPECTED_M982_OUTER_SHA}" \
  --expected-release-review-sha "${M985_EXPECTED_M984_REVIEW_SHA}" \
  --expected-release-manifest-sha "${M985_EXPECTED_M984_MANIFEST_SHA}" \
  --expected-release-outer-sha "${M985_EXPECTED_M984_OUTER_SHA}" >/dev/null

"${m985_timeout}" --foreground --signal=TERM --kill-after=30s 1800s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 "${m985_python}" "${m985_driver}" \
  --run-row D3 --row-stage "${m985_work}/D3" --runner "${m985_runner}" \
  --expected-release-sha "${M985_EXPECTED_M983_RELEASE_SHA}" \
  --expected-source-review-sha "${M985_EXPECTED_M982_REVIEW_SHA}" \
  --expected-source-manifest-sha "${M985_EXPECTED_M982_MANIFEST_SHA}" \
  --expected-source-outer-sha "${M985_EXPECTED_M982_OUTER_SHA}" \
  --expected-release-review-sha "${M985_EXPECTED_M984_REVIEW_SHA}" \
  --expected-release-manifest-sha "${M985_EXPECTED_M984_MANIFEST_SHA}" \
  --expected-release-outer-sha "${M985_EXPECTED_M984_OUTER_SHA}" >/dev/null

m985_auth --assemble --work "${m985_work}" >/dev/null
m985_auth --publish --work "${m985_work}" >/dev/null
m985_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' 'PASS M985 one D2-then-D3 10K run published; result hammer required'
