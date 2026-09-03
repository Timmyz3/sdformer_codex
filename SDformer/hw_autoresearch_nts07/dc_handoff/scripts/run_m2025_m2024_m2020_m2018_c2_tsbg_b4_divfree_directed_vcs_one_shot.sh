#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
FILELIST="${HW_ROOT}/dc_handoff/filelists/iscas_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs.f"
M803="${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
RTL="${HW_ROOT}/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER="${HW_ROOT}/rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
SVA="${HW_ROOT}/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB="${HW_ROOT}/tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv"
M2019_DIR="${HW_ROOT}/reviews/m2019_m2018_c2_tsbg_b4_divfree_fair_scheduler_source_hammer_r1_20260902"
M2019_REVIEW="${M2019_DIR}/review.json"
M2024_DIR="${HW_ROOT}/reviews/m2024_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_source_hammer_r1_20260902"
M2024_REVIEW="${M2024_DIR}/review.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
VCS=${VCS_HOME}/bin/vcs
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
TOP=tb_m1880_c2_tsbg_b4_real_channel_signed_frontend
RESULT="${HW_ROOT}/results/m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_r1_20260902"
ATTEMPT="${HW_ROOT}/results/.m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_attempt_consumed"
WORK="${HW_ROOT}/results/.m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_work.$$"
LOCK="${HW_ROOT}/results/.m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: identity ${path}" >&2; exit 3;
  }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || return 1
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
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
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" \
      >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21 "${RTL}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${M803}"
sha_exact dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c "${ADAPTER}"
sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"
sha_exact d46a47dada89e16cdc3f2593020a89e3513060a8a1a03ae3a1963d0483b96081 "${TB}"
sha_exact 759a18d9c975ed912b8c75eeeb92b527afb46185c8f8e64f50a8e83f76d86996 "${FILELIST}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact bf1cfc2d1090f5932419e19921a3cd1966adbbec5585ad446e43f9bcb266477d "${M2019_REVIEW}"
verify_dir_seal "${M2019_DIR}"
verify_dir_seal "${M2024_DIR}"
[[ -n "${M2025_EXPECTED_RUNNER_SHA256:-}" &&
   "$(sha_file "${RUNNER}")" == "${M2025_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2025_EXPECTED_M2024_REVIEW_SHA256:-}" &&
   "$(sha_file "${M2024_REVIEW}")" == "${M2025_EXPECTED_M2024_REVIEW_SHA256}" ]] || exit 3
/usr/libexec/platform-python3.6 -I - \
  "${M2019_REVIEW}" "${M2024_REVIEW}" "${RUNNER}" "${FILELIST}" \
  "${RTL}" "${ADAPTER}" "${M803}" "${SVA}" "${TB}" "${DOCS359}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path

(m2019_path, m2024_path, runner, filelist, rtl, adapter, m803, sva, tb,
 docs359) = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
m2019 = json.loads(m2019_path.read_text())
m2024 = json.loads(m2024_path.read_text())
assert m2019['status'] == 'PASS_M2019_M2018_C2_TSBG_B4_DIVFREE_FAIR_SCHEDULER_SOURCE_HAMMER'
assert m2019['score_over_100'] >= 95
assert m2019['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
assert m2019['identity']['m2018_rtl_sha256'] == sha(rtl)
assert m2019['authorization']['next_vcs_source_authoring'] is True
assert m2019['authorization']['eda_execution'] is False
assert m2024['status'] == 'PASS_M2024_M2020_M2018_C2_TSBG_B4_DIVFREE_DIRECTED_VCS_SOURCE_HAMMER'
assert m2024['score_over_100'] >= 95
assert m2024['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
identity = m2024['identity']
for key, path in (
        ('runner_sha256', runner), ('filelist_sha256', filelist),
        ('m2018_rtl_sha256', rtl), ('m2020_adapter_sha256', adapter),
        ('m803_adapter_sha256', m803), ('sva_sha256', sva),
        ('testbench_sha256', tb), ('docs359_sha256', docs359)):
    assert identity[key] == sha(path), (key, identity.get(key), sha(path))
assert m2024['authorization'] == {
    'license_queries': 1, 'vcs_compiles': 1, 'simv_runs': 1,
    'all_other_eda_runs': 0, 'automatic_retry': False}
assert m2024['claim_boundary']['source_only'] is True
assert m2024['claim_boundary']['vcs_result_pending'] is True
assert m2024['claim_boundary']['same_area'] is False
assert m2024['claim_boundary']['exact_cycle_speedup'] is False
assert m2024['claim_boundary']['system_speedup'] is False
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
blocked=' vcs vcs1 vlogan simv dc_shell dc_shell-t pt_shell fm_shell icc2_shell common_shell_ex common_shell_exec common_shell_exe '
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
[[ "${mem_available}" -ge 16777216 && $((commit_limit-committed)) -ge 16777216 ]] || exit 4

cd -- "${REPO_ROOT}"
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2025_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}" >"${WORK}/license_preflight.log" 2>&1
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${VCS}" -full64 -sverilog -assert svaext -top "${TOP}" -f "${FILELIST}" \
  -o "${WORK}/simv" -Mdir="${WORK}/csrc" >"${WORK}/vcs_compile.log" 2>&1
! grep -Eiq 'Warning-\[SVAA-RNF\]|Ignoring.*global_finish_maxfail|global_finish_maxfail.*(ignored|unknown)|Unknown.*global_finish_maxfail|Error-' \
  "${WORK}/vcs_compile.log"
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  /usr/bin/timeout --signal=TERM --kill-after=10s 180s "${WORK}/simv" \
  -assert global_finish_maxfail=1 >"${WORK}/simv.log" 2>&1

EXPECTED_PASS='PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED rows=48 issues=576 products=9216 commits=24 bundles_base=576 bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 retired_replay=1 replay_accept=0 reset=2 recovery=1'
[[ "$(grep -Fc 'PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED' "${WORK}/simv.log")" -eq 1 ]]
[[ "$(grep -Fxc "${EXPECTED_PASS}" "${WORK}/simv.log")" -eq 1 ]]
for phase in reset full_load full_execute retired_replay replay_reset_recovery stale_attack stale_reset_recovery recovery_load recovery_execute final_checks; do
  [[ "$(grep -Fc "M1970_PHASE ${phase}_begin" "${WORK}/simv.log")" -eq 1 ]]
  [[ "$(grep -Fc "M1970_PHASE ${phase}_complete" "${WORK}/simv.log")" -eq 1 ]]
done
[[ "$(grep -Fc 'M1970_LOAD_BEGIN' "${WORK}/simv.log")" -eq 52 ]]
[[ "$(grep -Fc 'M1970_LOAD_COMPLETE' "${WORK}/simv.log")" -eq 52 ]]
[[ "$(grep -Fc 'M1970_LOAD_TIMEOUT' "${WORK}/simv.log")" -eq 0 ]]
! grep -Eiq 'Warning-\[SVAA-RNF\]|Ignoring.*global_finish_maxfail|global_finish_maxfail.*(ignored|unknown)|Unknown.*global_finish_maxfail|: started at .* failed at|Assertion[^[:cntrl:]]*failed|Error-\[SVA|\$(error|fatal)|Fatal:|whole-test watchdog expired|directed timeout|post-reset legal-service timeout' \
  "${WORK}/simv.log"

rm -rf -- "${WORK}/simv" "${WORK}/csrc" "${WORK}/simv.daidir"
cat >"${WORK}/receipt.json" <<'JSON'
{
  "schema": "m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_receipt_r1_v1",
  "status": "RAW_PASS_M2025_M2018_DIVFREE_DIRECTED_VCS_PENDING_INDEPENDENT_RESULT_REVIEW",
  "milestone": "M2025",
  "license_queries": 1,
  "vcs_compiles": 1,
  "simv_runs": 1,
  "automatic_retry": false,
  "directed_geometry": {"bundle": 4, "source_groups": 12, "sources_per_group": 16, "lanes": 16},
  "scope": "G12 directed functional/protocol/recovery regression only",
  "production_g48_dynamic": false,
  "m1970_phase_pairs": 10,
  "m1970_load_begin": 52,
  "m1970_load_complete": 52,
  "m1970_load_timeout": 0,
  "claim_boundary": {
    "behavioral_rtl_directed_only": true,
    "same_area": false,
    "exact_cycle_speedup": false,
    "system_speedup": false,
    "paper_admitted": false,
    "headline": false
  }
}
JSON
printf 'RAW_PASS_M2025_M2018_DIVFREE_DIRECTED_VCS_PENDING_INDEPENDENT_RESULT_REVIEW\n' \
  >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2025_M2018_DIVFREE_DIRECTED_VCS_PENDING_INDEPENDENT_RESULT_REVIEW\n'
