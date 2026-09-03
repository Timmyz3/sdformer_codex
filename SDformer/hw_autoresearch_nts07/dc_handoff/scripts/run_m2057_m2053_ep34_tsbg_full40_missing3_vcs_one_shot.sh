#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
PARSER="${HW_ROOT}/system_simulator/scripts/parse_m2057_m2053_ep34_tsbg_full40_missing3_successor.py"
OLD_RAW="${HW_ROOT}/dc_handoff/runs/m2053_m2051_ep34_tsbg_full40_vcs_raw.77266"
OLD_SIMV="${OLD_RAW}/simv"
OLD_COMPILE="${OLD_RAW}/vcs_compile.log"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
STEM=m2057_m2053_ep34_tsbg_full40_missing3_vcs
RESULT="${HW_ROOT}/results/${STEM}_r1_20260903"
ATTEMPT="${HW_ROOT}/results/.${STEM}_attempt_consumed"
RAW="${HW_ROOT}/dc_handoff/runs/${STEM}_raw.$$"
PUBLISH="${HW_ROOT}/results/.${STEM}_publishing.$$"
LOCK="${HW_ROOT}/results/.${STEM}_launch_lock"
SIMV_SHA=80887d96cd4bf3c037eb53f383474f29ab7f35a7406f4c4a175a4ed7f8099789
SLOTS=(86 893 1755)
VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: identity mismatch: ${path}" >&2; exit 3;
  }
}
seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
on_exit() {
  local rc=$?
  set +e
  if [[ ${rc} -ne 0 ]]; then
    if [[ -d "${RAW}" && ! -L "${RAW}" ]]; then
      printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" \
        >"${RAW}/RUN_FAILED_OR_INCOMPLETE.txt"
      seal_dir "${RAW}" || true
    fi
    if [[ -d "${PUBLISH}" && ! -L "${PUBLISH}" ]]; then
      mv -T -- "${PUBLISH}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
    fi
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

# Filled only after the independently reviewed sources are frozen.
sha_exact a2f6dd0f9481fc4aebc02411d4718b68eb53fa196ca0dc6e745776ff0bd0abc6 "${PARSER}"
sha_exact 80887d96cd4bf3c037eb53f383474f29ab7f35a7406f4c4a175a4ed7f8099789 "${OLD_SIMV}"
sha_exact fb774d9d15276c56e02423b3fed31dd767a3124334c21920ac863fbae936a86e "${OLD_COMPILE}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${RAW}" && \
   ! -e "${PUBLISH}" && ! -e "${LOCK}" ]] || exit 4
blocked=' vcs vcs1 vlogan simv dc_shell dc_shell-t pt_shell fm_shell icc2_shell '
for proc in /proc/[0-9]*; do
  [[ -r "${proc}/status" && -r "${proc}/comm" ]] || continue
  real_uid=''
  while IFS=$'\t' read -r key value rest; do
    [[ "${key}" == 'Uid:' ]] && { real_uid="${value}"; break; }
  done <"${proc}/status"
  [[ "${real_uid}" == "${EUID}" ]] || continue
  comm=''; IFS= read -r comm <"${proc}/comm" || continue
  [[ " ${blocked} " != *" ${comm} "* ]] || exit 4
done
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 1048576 && $((commit_limit-committed)) -ge 1048576 ]] || exit 4

cd -- "${REPO_ROOT}"
"/opt/anaconda3/bin/python" -I "${PARSER}" --preflight-old

mkdir -- "${LOCK}" "${ATTEMPT}" "${RAW}" "${PUBLISH}"
printf '%s\n' \
  'status=M2057_MISSING3_ATTEMPT_CONSUMED' \
  'parent_attempt=M2053' \
  'parent_status=FAILED_OR_INCOMPLETE_DO_NOT_CITE' \
  'compiled_simv_reused=true' \
  'license_queries=0' \
  'vcs_compiles=0' \
  'simv_runs=3' \
  'simv_parallelism=1' \
  'slots=86,893,1755' \
  'runtime_switch=-no_save' \
  'retry=false' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"

: >"${RAW}/M2057_RUN_COMMANDS.txt"
for slot in "${SLOTS[@]}"; do
  printf 'slot=%s simv_sha256=%s argv=-no_save +WORKLOAD_SLOT=%s -assert global_finish_maxfail=1\n' \
    "${slot}" "${SIMV_SHA}" "${slot}" >>"${RAW}/M2057_RUN_COMMANDS.txt"
  env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
    SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
    /usr/bin/timeout --signal=TERM --kill-after=10s 180s "${OLD_SIMV}" \
    -no_save +WORKLOAD_SLOT="${slot}" -assert global_finish_maxfail=1 \
    >"${RAW}/sim_slot${slot}.log" 2>&1
done

install -m 664 -- "${OLD_COMPILE}" "${PUBLISH}/vcs_compile.log"
install -m 664 -- "${RAW}/M2057_RUN_COMMANDS.txt" "${PUBLISH}/M2057_RUN_COMMANDS.txt"
for slot in $(/usr/bin/seq 0 1919); do
  source_log="${OLD_RAW}/sim_slot${slot}.log"
  case " ${SLOTS[*]} " in
    *" ${slot} "*) source_log="${RAW}/sim_slot${slot}.log" ;;
  esac
  install -m 664 -- "${source_log}" "${PUBLISH}/sim_slot${slot}.log"
done

"/opt/anaconda3/bin/python" -I "${PARSER}" \
  --new-sim-dir "${RAW}" --merged-sim-dir "${PUBLISH}" \
  --output "${PUBLISH}/result.json"
printf '%s  %s\n' "$(sha_file "${RUNNER}")" "$(basename -- "${RUNNER}")" \
  >"${PUBLISH}/RUNNER_SHA256.txt"
printf '%s\n' \
  'RAW_PASS_M2057_M2053_MISSING3_SUCCESSOR_PENDING_INDEPENDENT_REVIEW' \
  'parent_m2053_status=FAILED_OR_INCOMPLETE_DO_NOT_CITE' \
  'cross_attempt_merge=1917_parent_plus_3_successor' \
  >"${PUBLISH}/RUN_COMPLETE.txt"
seal_dir "${PUBLISH}"
mv -T -- "${PUBLISH}" "${RESULT}"
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2057_M2053_MISSING3_SUCCESSOR_PENDING_INDEPENDENT_REVIEW\n'
