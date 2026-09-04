#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: M2141 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${SCRIPT_DIR}/run_icc2_m2141_library_import_preflight.tcl"
MONITOR="${SCRIPT_DIR}/monitor_m2141_icc2_process_tree.py"
CHECKER="${HW_ROOT}/system_simulator/scripts/check_m2141_icc2_library_import_preflight.py"
CONTRACT="${HW_ROOT}/contracts/m2141_m2136_icc2_library_import_preflight_source_contract_r1_20260904.json"
MASTER_LIST="${HW_ROOT}/dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
MW_MANIFEST="${HW_ROOT}/dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M2136="${HW_ROOT}/reviews/m2136_m2135_m2133_m2029_m2018_matched_macrofree_icc2_pnr_failure_hammer_r1_20260904"
M2146="${HW_ROOT}/reviews/m2146_m2141_m2136_icc2_library_import_preflight_source_hammer_r1_20260904"
M2146_REVIEW="${M2146}/review.json"

M2029="${HW_ROOT}/dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
ORDINARY_V="${M2029}/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v"
TSBG_V="${M2029}/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v"
TECH_BASE=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital
DB_BASE="${TECH_BASE}/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a"
TT_DB="${DB_BASE}/tcbn28hpcplusbwp35p140tt0p9v25c.db"
SS_DB="${DB_BASE}/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
FF_DB="${DB_BASE}/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
MW_REF="${TECH_BASE}/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140"
NXTGRD=/opt/tech/tsmc28/RC_Extraction/starRC/typical/crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical.nxtgrd
LAYER_MAP=/opt/tech/tsmc28/RC_Extraction/starRC/typical/Reference/MAP/star.map_icc_crn28hpc+_1p9m_6x1z1u_ut-alrdl

ICC2=/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_SERVER=27030@ic.ismd-nemo
LICENSE_FILE=/opt/synopsys/Synopsys.dat
RESULT="${HW_ROOT}/dc_handoff/runs/m2147_m2141_icc2_library_import_preflight_raw_r1_20260904"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m2147_m2141_icc2_library_import_preflight_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m2147_m2141_icc2_library_import_preflight_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m2147_m2141_icc2_library_import_preflight_launch_lock"
PRIOR_COLLATERAL="${REPO_ROOT}/icc2_output.txt"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: M2141 identity mismatch: ${path}" >&2
    exit 3
  }
}
sha_executable_exact() {
  sha_exact "$1" "$2"
  [[ -x "$2" ]] || { echo "ERROR: M2141 non-executable source/tool: $2" >&2; exit 3; }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || return 1
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
seal_dir() {
  local dir="$1"
  [[ -z "$(find -P "${dir}" -type l -print -quit)" ]] || return 1
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
snapshot_repo_root() {
  local output="$1"
  (cd -- "${REPO_ROOT}" &&
    find -P . -maxdepth 1 -type f -printf '%P\0' | LC_ALL=C sort -z |
      xargs -0 -r sha256sum --) >"${output}"
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

# Immutable source, technology, failure-diagnosis, and protected-document identities.
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact 0207977db8f1d1ef5d10ca4af97c87b7eb074bb2000d0b7ca11c1a11da7ef552 "${M2136}/review.json"
verify_dir_seal "${M2136}"
sha_exact f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0 "${ORDINARY_V}"
sha_exact 739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af "${TSBG_V}"
sha_exact e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b "${MASTER_LIST}"
sha_exact 7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3 "${MW_MANIFEST}"
sha_exact d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070 "${TT_DB}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${SS_DB}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${FF_DB}"
sha_exact 424477b89c352173da2c3adc1d723764e8ff68425289ef688793be364646fd02 "${NXTGRD}"
sha_exact da6e70dae3b50cc8e7520e3576477f2f80c3ac55dbe2b61baad73eb36fe44ed3 "${LAYER_MAP}"
sha_executable_exact 825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4 "${ICC2}"
sha_executable_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
sha_executable_exact 32399fde5cea3487439feee9b919322197bbbec3730d6d0298bce83a9456c268 "${MONITOR}"
sha_executable_exact 5324e0ac3a8cf53ff00ab9070f340bce24551ca6763081127bfc670b28ac40a0 "${CHECKER}"
(cd -- "${MW_REF}" && sha256sum -c "${MW_MANIFEST}" >/dev/null)
[[ "$(find -P "${MW_REF}" -type f | wc -l)" -eq 1051 ]]
[[ "$(find -P "${MW_REF}/FRAM" -type f | wc -l)" -eq 1044 ]]
[[ "$(find -P "${MW_REF}/CEL" -type f | wc -l)" -eq 2 ]]

# Source hammer binds this exact source package before any license query.
[[ -n "${M2141_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M2141_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2141_EXPECTED_SOURCE_REVIEW_SHA256:-}" && "$(sha_file "${M2146_REVIEW}")" == "${M2141_EXPECTED_SOURCE_REVIEW_SHA256}" ]] || exit 3
verify_dir_seal "${M2146}"
/usr/libexec/platform-python3.6 -I - "${M2146_REVIEW}" "${RUNNER}" "${TCL}" "${MONITOR}" "${CHECKER}" "${CONTRACT}" "${MASTER_LIST}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path
review, runner, tcl, monitor, checker, contract, masters = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads(review.read_text())
assert r['status'].startswith('PASS_M2146')
assert r['score_over_100'] >= 95
assert r['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
for key, path in [('runner_sha256', runner), ('tcl_sha256', tcl),
                  ('monitor_sha256', monitor), ('checker_sha256', checker),
                  ('contract_sha256', contract), ('master_list_sha256', masters)]:
    assert r['identity'][key] == sha(path), (key, r['identity'].get(key), sha(path))
assert r['authorization'] == {'license_queries': 1, 'top_level_icc2_shell_runs': 1,
                              'pnr_runs': 0, 'automatic_retry': False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
sha_exact 0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6 "${PRIOR_COLLATERAL}"
[[ "$(stat -c %s -- "${PRIOR_COLLATERAL}")" -eq 25324 ]]

# No same-UID EDA overlap; this is the unique one-shot top-level ICC2 session.
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked = {'vcs', 'vcs1', 'vlogan', 'simv', 'dc_shell', 'dc_shell-t', 'pt_shell',
           'fm_shell', 'icc2_shell', 'icc2_lm_shell', 'common_shell_exec',
           'common_shell_exe', 'lmutil', 'lmstat'}
hits = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid != os.getuid():
            continue
        comm = (p / 'comm').read_text().strip()
        argv = {Path(x.decode(errors='replace')).name
                for x in (p / 'cmdline').read_bytes().split(b'\0') if x}
    except Exception:
        continue
    if comm in blocked or blocked & argv:
        hits.append((p.name, comm))
if hits:
    raise SystemExit('M2141 same-UID EDA collision: %r' % hits)
PY
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 33554432 && $((commit_limit-committed)) -ge 25165824 ]] || exit 4

mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2147_ATTEMPT_CONSUMED\nlicense_queries=1\ntop_level_icc2_shell_runs=1\npnr_runs=0\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
snapshot_repo_root "${WORK}/repo_root_initial_with_m2135_collateral.sha256"
mkdir -- "${WORK}/prior_m2135_collateral" "${WORK}/isolated_cwd"
mv -- "${PRIOR_COLLATERAL}" "${WORK}/prior_m2135_collateral/icc2_output.txt"
sha_exact 0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6 "${WORK}/prior_m2135_collateral/icc2_output.txt"
snapshot_repo_root "${WORK}/repo_root_before.sha256"

printf '%s\n' \
  'scope=library_import_only' \
  'license_queries=1' \
  'top_level_icc2_shell_runs=1' \
  'pnr_runs=0' \
  'automatic_retry=false' \
  "icc2_path=${ICC2}" \
  'icc2_sha256=825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4' \
  "prior_m2135_collateral_original_path=${PRIOR_COLLATERAL}" \
  'prior_m2135_collateral_sha256=0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6' \
  >"${WORK}/execution_contract.txt"

"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1
grep -Fq 'Users of ICCompilerII:' "${WORK}/license_preflight.log"

ISOLATED="${WORK}/isolated_cwd"
mkdir -- "${ISOLATED}/tmp"
set +e
(
  cd -- "${ISOLATED}"
  exec /usr/bin/timeout --signal=TERM --kill-after=300s 14400s \
    env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR="${ISOLATED}/tmp" \
      SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
      M2141_ISOLATED_CWD="${ISOLATED}" \
      M2141_LIBRARY_CACHE="${ISOLATED}/library_cache" \
      M2141_FRAME_DIR="${ISOLATED}/frame_output" \
      M2141_FRAME_LOG_DIR="${ISOLATED}/frame_logs" \
      M2141_DESIGN_LIB="${ISOLATED}/m2141_disposable_design.nlib" \
      M2141_MW_REF="${MW_REF}" M2141_TT_DB="${TT_DB}" M2141_SS_DB="${SS_DB}" M2141_FF_DB="${FF_DB}" \
      M2141_MASTER_LIST="${MASTER_LIST}" M2141_NXTGRD="${NXTGRD}" M2141_LAYER_MAP="${LAYER_MAP}" \
      M2141_REPORT_DIR="${ISOLATED}/reports" \
      M2141_EXPECTED_RC_TECH_NAME=crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical \
      "${ICC2}" -f "${TCL}"
) >"${WORK}/icc2_preflight.log" 2>&1 &
icc2_pid=$!
"${MONITOR}" --root-pid "${icc2_pid}" --stop-file "${WORK}/process_monitor.stop" \
  --output "${WORK}/process_tree.json" >"${WORK}/process_monitor.log" 2>&1 &
monitor_pid=$!
wait "${icc2_pid}"
icc2_rc=$?
printf '%s\n' "${icc2_rc}" >"${WORK}/icc2_preflight.rc"
: >"${WORK}/process_monitor.stop"
wait "${monitor_pid}"
monitor_rc=$?
set -e
[[ "${icc2_rc}" -eq 0 && "${monitor_rc}" -eq 0 ]]

snapshot_repo_root "${WORK}/repo_root_after.sha256"
cmp -s -- "${WORK}/repo_root_before.sha256" "${WORK}/repo_root_after.sha256"
[[ ! -e "${REPO_ROOT}/icc2_output.txt" ]]
"${CHECKER}" --work "${WORK}" --output "${WORK}/receipt.json" >"${WORK}/checker.log"
grep -Fxq 'PASS_RAW_M2147_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER' "${WORK}/checker.log"
printf 'RAW_PASS_M2147_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER\n' \
  >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2147_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER\n'
