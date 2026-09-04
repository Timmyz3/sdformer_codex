#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: M2168 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${SCRIPT_DIR}/run_icc2_m2153_library_import_preflight.tcl"
MONITOR="${SCRIPT_DIR}/monitor_m2153_icc2_process_tree.py"
INVENTORY="${SCRIPT_DIR}/inventory_m2153_repo_root.py"
CHECKER="${HW_ROOT}/system_simulator/scripts/check_m2164_icc2_library_import_preflight.py"
CONTRACT="${HW_ROOT}/contracts/m2168_m2167_icc2_library_import_preflight_source_contract_r1_20260904.json"
MASTER_LIST="${HW_ROOT}/dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
MW_MANIFEST="${HW_ROOT}/dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M2167="${HW_ROOT}/reviews/m2167_m2166_m2164_icc2_library_preflight_startup_failure_hammer_r1_20260904"
M2167_REVIEW="${M2167}/review.json"
M2169="${HW_ROOT}/reviews/m2169_m2168_m2167_icc2_library_import_preflight_source_hammer_r1_20260904"
M2169_REVIEW="${M2169}/review.json"

TECH_BASE=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital
DB_BASE="${TECH_BASE}/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a"
TT_DB="${DB_BASE}/tcbn28hpcplusbwp35p140tt0p9v25c.db"
SS_DB="${DB_BASE}/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
FF_DB="${DB_BASE}/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
MW_REF="${TECH_BASE}/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140"
NXTGRD=/opt/tech/tsmc28/RC_Extraction/starRC/typical/crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical.nxtgrd
LAYER_MAP=/opt/tech/tsmc28/RC_Extraction/starRC/typical/Reference/MAP/star.map_icc_crn28hpc+_1p9m_6x1z1u_ut-alrdl

ICC2=/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell
ICC2_REAL=/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec
KNOWN_NDM=/opt/synopsys/icc2/V-2023.12-SP3/libraries/syn/gtech.nlib/reflib.ndm
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_SERVER=27030@ic.ismd-nemo
LICENSE_FILE=/opt/synopsys/Synopsys.dat
RESULT="${HW_ROOT}/dc_handoff/runs/m2170_m2168_icc2_library_import_preflight_raw_r1_20260904"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_launch_lock"
PRIOR_COLLATERAL="${REPO_ROOT}/icc2_output.txt"
WORK_ACTIVE=0
icc2_pid=0
monitor_pid=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: M2168 identity mismatch: ${path}" >&2
    exit 3
  }
}
sha_executable_exact() {
  sha_exact "$1" "$2"
  [[ -x "$2" ]] || { echo "ERROR: M2168 non-executable source/tool: $2" >&2; exit 3; }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || return 1
  [[ -z "$(find -P "${dir}" -type l -print -quit)" ]] || return 1
  (cd -- "${dir}" &&
    diff -u <(find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -printf '%P\n' | LC_ALL=C sort) \
      <(awk '{sub(/^\*/, "", $2); print $2}' SHA256SUMS | LC_ALL=C sort) >/dev/null &&
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
seal_dir() {
  local dir="$1"
  [[ -z "$(find -P "${dir}" -type l -print -quit)" ]] || return 1
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
on_exit() {
  local rc=$?
  set +e
  [[ ${monitor_pid} -gt 0 ]] && kill "${monitor_pid}" 2>/dev/null
  [[ ${icc2_pid} -gt 0 ]] && kill "${icc2_pid}" 2>/dev/null
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

# Frozen predecessor, technology, tool and protected-document identities.
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact ee09260d23e4e8b140e4a943c2de58d2c9ad6694ad203b78542f2a6e1fbd3d1a "${M2167_REVIEW}"
verify_dir_seal "${M2167}"
sha_exact e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b "${MASTER_LIST}"
sha_exact 7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3 "${MW_MANIFEST}"
sha_exact d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070 "${TT_DB}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${SS_DB}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${FF_DB}"
sha_exact 424477b89c352173da2c3adc1d723764e8ff68425289ef688793be364646fd02 "${NXTGRD}"
sha_exact da6e70dae3b50cc8e7520e3576477f2f80c3ac55dbe2b61baad73eb36fe44ed3 "${LAYER_MAP}"
sha_executable_exact 825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4 "${ICC2}"
sha_executable_exact 4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c "${ICC2_REAL}"
sha_exact 56f9a2c14fc9ce7d3d7691146bbc89db35c58a4fe40543833a924a23e8ada829 "${KNOWN_NDM}"
sha_executable_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
(cd -- "${MW_REF}" && sha256sum -c "${MW_MANIFEST}" >/dev/null)
[[ "$(find -P "${MW_REF}" -type f | wc -l)" -eq 1051 ]]
[[ -z "$(find -P "${MW_REF}" -type l -print -quit)" ]]

# The independent M2169 source hammer must bind every executable source file.
[[ -n "${M2168_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M2168_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2168_EXPECTED_SOURCE_REVIEW_SHA256:-}" && "$(sha_file "${M2169_REVIEW}")" == "${M2168_EXPECTED_SOURCE_REVIEW_SHA256}" ]] || exit 3
verify_dir_seal "${M2169}"
/usr/libexec/platform-python3.6 -I - "${M2169_REVIEW}" "${RUNNER}" "${TCL}" "${MONITOR}" "${INVENTORY}" "${CHECKER}" "${CONTRACT}" "${MASTER_LIST}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path
review, runner, tcl, monitor, inventory, checker, contract, masters = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
data = json.loads(review.read_text())
assert data['status'] == 'PASS_M2169_M2168_SOURCE_HAMMER__M2170_ONE_SHOT_AUTHORIZED'
assert data['score_over_100'] >= 95
assert data['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
for key, path in [('runner_sha256', runner), ('tcl_sha256', tcl), ('monitor_sha256', monitor),
                  ('inventory_sha256', inventory), ('checker_sha256', checker),
                  ('contract_sha256', contract), ('master_list_sha256', masters)]:
    assert data['identity'][key] == sha(path), (key, data['identity'].get(key), sha(path))
assert data['authorization'] == {'m2170': True, 'license_queries': 1,
                                  'top_level_icc2_shell_runs': 1, 'pnr_runs': 0,
                                  'automatic_retry': False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
sha_exact 0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6 "${PRIOR_COLLATERAL}"
[[ "$(stat -c %s -- "${PRIOR_COLLATERAL}")" -eq 25324 ]]

# One top-level ICC2 session only, and never overlap another same-UID EDA job.
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked = {'vcs', 'vcs1', 'vlogan', 'simv', 'dc_shell', 'dc_shell-t', 'pt_shell',
           'fm_shell', 'icc2_shell', 'icc2_exec', 'dgcom_exec', 'icc2_lm_shell',
           'lm_shell_exec', 'common_shell_exec', 'common_shell_exe', 'lmutil', 'lmstat'}
hits = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        if path.stat().st_uid != os.getuid():
            continue
        comm = (path / 'comm').read_text().strip()
        argv = {Path(item.decode(errors='replace')).name
                for item in (path / 'cmdline').read_bytes().split(b'\0') if item}
        exe = Path(os.readlink(path / 'exe')).name
    except Exception:
        continue
    if comm in blocked or exe in blocked or blocked & argv:
        hits.append((path.name, comm, exe))
if hits:
    raise SystemExit('M2168 same-UID EDA collision: %r' % hits)
PY
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 33554432 && $((commit_limit-committed)) -ge 25165824 ]] || exit 4

mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2170_ATTEMPT_CONSUMED\nlicense_queries=1\ntop_level_icc2_shell_runs=1\npnr_runs=0\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_before.json" >"${WORK}/repo_root_before.log"
mkdir -- "${WORK}/prior_m2135_collateral"
cp --reflink=never --preserve=mode,timestamps -- "${PRIOR_COLLATERAL}" "${WORK}/prior_m2135_collateral/icc2_output.txt"
sha_exact 0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6 "${WORK}/prior_m2135_collateral/icc2_output.txt"

ISOLATED="${WORK}/isolated_cwd"
DESIGN_LIB="${ISOLATED}/m2153_disposable_design.nlib"
FRAME_NDM="${ISOLATED}/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
# One and only one nested-parent creation operation for all seven isolation paths.
mkdir -p -- \
  "${ISOLATED}/home" "${ISOLATED}/tmp" \
  "${ISOLATED}/cache/xdg" "${ISOLATED}/cache/library" \
  "${ISOLATED}/frame_output" "${ISOLATED}/frame_logs" "${ISOLATED}/reports"
/usr/libexec/platform-python3.6 -I - "${ISOLATED}" \
  "${ISOLATED}/home" "${ISOLATED}/tmp" \
  "${ISOLATED}/cache/xdg" "${ISOLATED}/cache/library" \
  "${ISOLATED}/frame_output" "${ISOLATED}/frame_logs" "${ISOLATED}/reports" <<'PY'
from __future__ import print_function
import os, stat, sys
from pathlib import Path
isolated_raw = Path(sys.argv[1])
listed_raw = [Path(value) for value in sys.argv[2:]]
assert len(listed_raw) == 7
assert isolated_raw.is_dir() and not isolated_raw.is_symlink()
isolated = isolated_raw.resolve(strict=True)
assert isolated == isolated_raw.absolute()
for path in listed_raw:
    assert path.is_dir() and not path.is_symlink()
    resolved = path.resolve(strict=True)
    assert resolved != isolated and isolated in resolved.parents
    relative = path.absolute().relative_to(isolated)
    cursor = isolated
    for part in relative.parts:
        cursor = cursor / part
        assert stat.S_ISDIR(os.lstat(str(cursor)).st_mode)
        assert not stat.S_ISLNK(os.lstat(str(cursor)).st_mode)
print('M2168_LAYOUT_GATE_PASS paths=7 strictly_below=true symlinks=0')
PY
[[ ! -e "${DESIGN_LIB}" && ! -L "${DESIGN_LIB}" ]]
[[ ! -e "${FRAME_NDM}" && ! -L "${FRAME_NDM}" ]]
printf 'M2168_OUTPUT_ABSENCE_GATE_PASS design_nlib=absent frame_ndm=absent\n'

/usr/libexec/platform-python3.6 -I - "${WORK}/execution_contract.json" "${ICC2}" "${TCL}" "${ICC2_REAL}" "${ISOLATED}" <<'PY'
from __future__ import print_function
import json, sys
from pathlib import Path
out, wrapper, tcl, real, isolated = map(Path, sys.argv[1:])
payload = {
    'schema': 'm2166_m2164_execution_contract_r1_v1',
    'scope': 'library_import_only',
    'license_queries': 1,
    'top_level_icc2_shell_runs': 1,
    'pnr_runs': 0,
    'automatic_retry': False,
    'icc2_invocation': [str(wrapper), '-no_init', '-f', str(tcl)],
    'icc2_wrapper_sha256': '825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4',
    'icc2_real_exec_path': str(real),
    'icc2_real_exec_sha256': '4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c',
    'isolated_home': str(isolated / 'home'),
    'isolated_tmpdir': str(isolated / 'tmp'),
    'isolated_xdg_cache': str(isolated / 'cache/xdg'),
    'isolated_library_cache': str(isolated / 'cache/library'),
    'prior_m2135_collateral_action': 'copied_byte_exact_original_preserved',
    'prior_m2135_collateral_sha256': '0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6',
}
assert not out.exists() and not out.is_symlink()
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print('M2168_EXECUTION_CONTRACT_WRITE_PASS')
PY
/usr/libexec/platform-python3.6 -I - "${WORK}/execution_contract.json" "${ICC2}" "${TCL}" "${ICC2_REAL}" "${ISOLATED}" <<'PY'
from __future__ import print_function
import json, sys
from pathlib import Path
path, wrapper, tcl, real, isolated = map(Path, sys.argv[1:])
expected = {
    'schema': 'm2166_m2164_execution_contract_r1_v1',
    'scope': 'library_import_only',
    'license_queries': 1,
    'top_level_icc2_shell_runs': 1,
    'pnr_runs': 0,
    'automatic_retry': False,
    'icc2_invocation': [str(wrapper), '-no_init', '-f', str(tcl)],
    'icc2_wrapper_sha256': '825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4',
    'icc2_real_exec_path': str(real),
    'icc2_real_exec_sha256': '4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c',
    'isolated_home': str(isolated / 'home'),
    'isolated_tmpdir': str(isolated / 'tmp'),
    'isolated_xdg_cache': str(isolated / 'cache/xdg'),
    'isolated_library_cache': str(isolated / 'cache/library'),
    'prior_m2135_collateral_action': 'copied_byte_exact_original_preserved',
    'prior_m2135_collateral_sha256': '0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6',
}
assert path.is_file() and not path.is_symlink()
assert json.loads(path.read_text()) == expected
print('M2168_EXECUTION_CONTRACT_REREAD_PASS')
PY
[[ ! -e "${DESIGN_LIB}" && ! -L "${DESIGN_LIB}" ]]
[[ ! -e "${FRAME_NDM}" && ! -L "${FRAME_NDM}" ]]

# No license or Synopsys process is invoked before the layout, absence, and
# exact execution-contract write/read gates above have all succeeded.
"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1
grep -Fq 'Users of ICCompilerII:' "${WORK}/license_preflight.log"

LAUNCH_GATE="${WORK}/launch.gate"
MONITOR_READY="${WORK}/process_monitor.ready"
(
  cd -- "${ISOLATED}"
  while [[ ! -e "${LAUNCH_GATE}" ]]; do /usr/bin/sleep 0.01; done
  exec /usr/bin/timeout --signal=TERM --kill-after=300s 14400s \
    env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
      HOME="${ISOLATED}/home" TMPDIR="${ISOLATED}/tmp" XDG_CACHE_HOME="${ISOLATED}/cache/xdg" \
      SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
      M2153_ISOLATED_CWD="${ISOLATED}" M2153_LIBRARY_CACHE="${ISOLATED}/cache/library" \
      M2153_FRAME_DIR="${ISOLATED}/frame_output" M2153_FRAME_LOG_DIR="${ISOLATED}/frame_logs" \
      M2153_DESIGN_LIB="${DESIGN_LIB}" \
      M2153_MW_REF="${MW_REF}" M2153_TT_DB="${TT_DB}" M2153_SS_DB="${SS_DB}" M2153_FF_DB="${FF_DB}" \
      M2153_MASTER_LIST="${MASTER_LIST}" M2153_NXTGRD="${NXTGRD}" M2153_LAYER_MAP="${LAYER_MAP}" \
      M2153_REPORT_DIR="${ISOLATED}/reports" \
      M2153_EXPECTED_RC_TECH_NAME=crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical \
      "${ICC2}" -no_init -f "${TCL}"
) >"${WORK}/icc2_preflight.log" 2>&1 &
icc2_pid=$!
"${MONITOR}" --root-pid "${icc2_pid}" --stop-file "${WORK}/process_monitor.stop" \
  --ready-file "${MONITOR_READY}" --wrapper-path "${ICC2}" --actual-exec-path "${ICC2_REAL}" \
  --output "${WORK}/process_tree.json" >"${WORK}/process_monitor.log" 2>&1 &
monitor_pid=$!
for _ in $(seq 1 1000); do
  [[ -e "${MONITOR_READY}" ]] && break
  kill -0 "${monitor_pid}" 2>/dev/null || exit 5
  /usr/bin/sleep 0.01
done
[[ -e "${MONITOR_READY}" ]] || exit 5
: >"${LAUNCH_GATE}"

set +e
wait "${icc2_pid}"
icc2_rc=$?
icc2_pid=0
printf '%s\n' "${icc2_rc}" >"${WORK}/icc2_preflight.rc"
: >"${WORK}/process_monitor.stop"
wait "${monitor_pid}"
monitor_rc=$?
monitor_pid=0
set -e
[[ "${icc2_rc}" -eq 0 && "${monitor_rc}" -eq 0 ]]

"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_after.json" >"${WORK}/repo_root_after.log"
cmp -s -- "${WORK}/repo_root_before.json" "${WORK}/repo_root_after.json"
sha_exact 0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6 "${PRIOR_COLLATERAL}"
"${CHECKER}" --work "${WORK}" --output "${WORK}/receipt.json" >"${WORK}/checker.log"
grep -Fxq 'PASS_RAW_M2166_M2164_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2167_INDEPENDENT_RESULT_HAMMER' "${WORK}/checker.log"
printf 'RAW_PASS_M2170_M2168_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2171_INDEPENDENT_RESULT_HAMMER\n' \
  >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2170_M2168_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2171_INDEPENDENT_RESULT_HAMMER\n'
