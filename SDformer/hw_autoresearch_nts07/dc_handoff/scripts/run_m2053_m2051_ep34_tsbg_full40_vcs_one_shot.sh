#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
FILELIST="${HW_ROOT}/dc_handoff/filelists/iscas_m2051_ep34_tsbg_full40_cycle_vcs.f"
RTL="${HW_ROOT}/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER="${HW_ROOT}/rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
M803="${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA="${HW_ROOT}/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB="${HW_ROOT}/tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv"
FIXTURE="${HW_ROOT}/tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
STATS="${HW_ROOT}/tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh"
FIXTURE_JSON="${HW_ROOT}/tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
PARSER="${HW_ROOT}/system_simulator/scripts/parse_m2053_ep34_tsbg_full40_vcs.py"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
TOP=tb_m2051_ep34_tsbg_full40_cycle
VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
VCS=${VCS_HOME}/bin/vcs
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
STEM=m2053_m2051_ep34_tsbg_full40_vcs
RESULT="${HW_ROOT}/results/${STEM}_r1_20260903"
ATTEMPT="${HW_ROOT}/results/.${STEM}_attempt_consumed"
RAW="${HW_ROOT}/dc_handoff/runs/${STEM}_raw.$$"
PUBLISH="${HW_ROOT}/results/.${STEM}_publishing.$$"
LOCK="${HW_ROOT}/results/.${STEM}_launch_lock"

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
    fi
    if [[ -d "${PUBLISH}" && ! -L "${PUBLISH}" ]]; then
      mv -T -- "${PUBLISH}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
    fi
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21 "${RTL}"
sha_exact dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c "${ADAPTER}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${M803}"
sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"
sha_exact 64805bdedb7c80d5c6141bc36e59ef61234507b40942e69ccbf4a30ac2383436 "${TB}"
sha_exact 487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0 "${FIXTURE}"
sha_exact 70810fdf3ac4ba2d281d750995810f08561addb50871550aa83343a2a04a6dca "${STATS}"
sha_exact 3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5 "${FIXTURE_JSON}"
sha_exact eb45e899efa03c8ccc17bdc688678e111d1c0e3495848a5a6f095a0a367bfc06 "${FILELIST}"
sha_exact 2dfa31aaad1e1e3b2a4184eca95e4cdd99170a5c5232e4f2c47596ea15f138fd "${PARSER}"
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
[[ "${mem_available}" -ge 8388608 && $((commit_limit-committed)) -ge 8388608 ]] || exit 4

mkdir -- "${LOCK}" "${ATTEMPT}" "${RAW}" "${PUBLISH}"
printf 'status=M2053_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1920\nsimv_parallelism=4\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"

cd -- "${REPO_ROOT}"
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}" >"${RAW}/license_preflight.log" 2>&1
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${VCS}" -full64 -sverilog -assert svaext -top "${TOP}" \
  -f "hw_autoresearch_nts07/dc_handoff/filelists/iscas_m2051_ep34_tsbg_full40_cycle_vcs.f" \
  -o "${RAW}/simv" -Mdir="${RAW}/csrc" >"${RAW}/vcs_compile.log" 2>&1

export RAW VCS_HOME LICENSE_SERVER LICENSE_FILE
run_slot() {
  local slot="$1"
  env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
    SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
    /usr/bin/timeout --signal=TERM --kill-after=10s 180s "${RAW}/simv" \
    +WORKLOAD_SLOT="${slot}" -assert global_finish_maxfail=1 \
    >"${RAW}/sim_slot${slot}.log" 2>&1
}
export -f run_slot
/usr/bin/seq 0 1919 | /usr/bin/xargs -P 4 -n 1 /usr/bin/bash -c 'run_slot "$1"' _

install -m 664 -- "${RAW}/license_preflight.log" "${PUBLISH}/license_preflight.log"
install -m 664 -- "${RAW}/vcs_compile.log" "${PUBLISH}/vcs_compile.log"
for slot in $(/usr/bin/seq 0 1919); do
  install -m 664 -- "${RAW}/sim_slot${slot}.log" "${PUBLISH}/sim_slot${slot}.log"
done
"/opt/anaconda3/bin/python" -I "${PARSER}" \
  --compile-log "${PUBLISH}/vcs_compile.log" --sim-dir "${PUBLISH}" \
  --output "${PUBLISH}/result.json"
printf '%s  %s\n' "$(sha_file "${RUNNER}")" "$(basename -- "${RUNNER}")" \
  >"${PUBLISH}/RUNNER_SHA256.txt"
printf 'RAW_PASS_M2053_EP34_TSBG_FULL40_VCS_PENDING_INDEPENDENT_REVIEW\n' \
  >"${PUBLISH}/RUN_COMPLETE.txt"
seal_dir "${PUBLISH}"
mv -T -- "${PUBLISH}" "${RESULT}"
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2053_EP34_TSBG_FULL40_VCS_PENDING_INDEPENDENT_REVIEW\n'
