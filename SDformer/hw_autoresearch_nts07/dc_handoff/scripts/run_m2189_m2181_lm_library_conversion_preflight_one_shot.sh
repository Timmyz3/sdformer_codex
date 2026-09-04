#!/usr/bin/env bash
set -euo pipefail
umask 002
[[ $# -eq 0 ]] || { echo "ERROR: M2189 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${SCRIPT_DIR}/run_lm_m2189_library_conversion_preflight.tcl"
MONITOR="${SCRIPT_DIR}/monitor_m2189_lm_conversion_process_tree.py"
INVENTORY="${SCRIPT_DIR}/inventory_m2153_repo_root.py"
CHECKER="${HW_ROOT}/system_simulator/scripts/check_m2189_lm_library_conversion_preflight.py"
CONTRACT="${HW_ROOT}/contracts/m2189_m2181_lm_library_conversion_preflight_source_contract_r1_20260904.json"
TEST="${HW_ROOT}/tests/test_m2189_lm_library_conversion_preflight_source.py"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M2171="${HW_ROOT}/reviews/m2171_m2170_m2168_icc2_library_import_preflight_failure_hammer_r1_20260904"
M2181="${HW_ROOT}/reviews/m2181_m2180_m2171_lm_library_conversion_preflight_source_hammer_r1_20260904"
M2190="${HW_ROOT}/reviews/m2190_m2189_m2181_lm_library_conversion_preflight_source_hammer_r1_20260904"
MW_MANIFEST="${HW_ROOT}/dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
TECH_BASE=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital
MW_REF="${TECH_BASE}/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140"
LM_SHELL=/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell
LM_EXEC=/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec
MILKYWAY=/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_SERVER=27030@ic.ismd-nemo
LICENSE_FILE=/opt/synopsys/Synopsys.dat
RESULT="${HW_ROOT}/dc_handoff/runs/m2191_m2189_lm_library_conversion_preflight_raw_r1_20260904"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m2191_m2189_lm_library_conversion_preflight_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m2191_m2189_lm_library_conversion_preflight_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m2191_m2189_lm_library_conversion_preflight_launch_lock"
M2182_PERMANENTLY_UNAUTHORIZED=1
WORK_ACTIVE=0
lm_pid=0
monitor_pid=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: M2189 identity mismatch: ${path}" >&2; exit 3; }
}
sha_exec() { sha_exact "$1" "$2"; [[ -x "$2" ]] || exit 3; }
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -z "$(find -P "${dir}" -type l -print -quit)" ]] || return 1
  (cd -- "${dir}" &&
    diff -u <(find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -printf '%P\n' | LC_ALL=C sort) \
      <(awk '{sub(/^\*/, "", $2); print $2}' SHA256SUMS | LC_ALL=C sort) >/dev/null &&
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
seal_dir() {
  local dir="$1"
  [[ -z "$(find -P "${dir}" -type l -print -quit)" ]] || return 1
  (cd -- "${dir}" && find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
    -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
on_exit() {
  local rc=$?
  set +e
  [[ ${monitor_pid} -gt 0 ]] && kill "${monitor_pid}" 2>/dev/null
  [[ ${lm_pid} -gt 0 ]] && kill "${lm_pid}" 2>/dev/null
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

[[ "${M2182_PERMANENTLY_UNAUTHORIZED}" -eq 1 ]]
[[ -z "$(find -P "${HW_ROOT}/dc_handoff/runs" -maxdepth 1 \
  \( -name 'm2182_m2180_lm_library_conversion_preflight_raw_r1_20260904*' \
     -o -name '.m2182_m2180_lm_library_conversion_preflight*' \) -print -quit)" ]]
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact c42ffe2cea367f6a0bb43c73279ec1c340fd20f37fac990a5876c8193b52ccb9 "${M2171}/review.json"
verify_dir_seal "${M2171}"
sha_exact 2364f4323cd7c2beea3468201d5c6e3e0eaca7b8c25516c6a5a30ec530bc575c "${M2181}/review.json"
verify_dir_seal "${M2181}"
sha_exact 7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3 "${MW_MANIFEST}"
sha_exec 1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942 "${LM_SHELL}"
sha_exec 3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab "${LM_EXEC}"
sha_exec 09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec "${MILKYWAY}"
sha_exec e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
(cd -- "${MW_REF}" && sha256sum -c "${MW_MANIFEST}" >/dev/null)
[[ "$(find -P "${MW_REF}" -type f | wc -l)" -eq 1051 ]]
[[ -z "$(find -P "${MW_REF}" -type l -print -quit)" ]]

[[ -n "${M2189_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M2189_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2189_EXPECTED_SOURCE_REVIEW_SHA256:-}" && "$(sha_file "${M2190}/review.json")" == "${M2189_EXPECTED_SOURCE_REVIEW_SHA256}" ]] || exit 3
verify_dir_seal "${M2190}"
/usr/libexec/platform-python3.6 -I - "${M2190}/review.json" "${RUNNER}" "${TCL}" "${MONITOR}" "${CHECKER}" "${TEST}" "${CONTRACT}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path
review, runner, tcl, monitor, checker, test, contract = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d = json.loads(review.read_text())
assert d['status'] == 'PASS_M2190_M2189_SOURCE_HAMMER__M2191_ONE_SHOT_AUTHORIZED'
assert d['score_over_100'] >= 95 and d['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
for key, path in [('runner_sha256', runner), ('tcl_sha256', tcl),
                  ('monitor_sha256', monitor), ('checker_sha256', checker),
                  ('test_sha256', test), ('contract_sha256', contract)]:
    assert d['identity'][key] == sha(path)
assert d['authorization'] == {'m2191': True, 'license_queries': 1,
                               'top_level_lm_shell_runs': 1, 'pnr_runs': 0,
                               'automatic_retry': False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked = {'vcs','simv','dc_shell','pt_shell','fm_shell','icc2_shell','icc2_exec',
           'dgcom_exec','lm_shell','lm_shell_exec','Milkyway','lmutil','lmstat'}
hits=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        if p.stat().st_uid != os.getuid(): continue
        comm=(p/'comm').read_text().strip(); exe=Path(os.readlink(p/'exe')).name
        argv={Path(x.decode(errors='replace')).name for x in (p/'cmdline').read_bytes().split(b'\0') if x}
    except Exception: continue
    if comm in blocked or exe in blocked or blocked & argv: hits.append((p.name,comm,exe))
if hits: raise SystemExit('M2189 same-UID EDA collision: %r' % hits)
PY

mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2191_ATTEMPT_CONSUMED\nlicense_queries=1\ntop_level_lm_shell_runs=1\npnr_runs=0\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_before.json" >"${WORK}/repo_root_before.log"
ISOLATED="${WORK}/isolated_cwd"
FRAME="${ISOLATED}/frame_output/m2189_tcbn28hpcplusbwp35p140_frame.ndm"
mkdir -p -- "${ISOLATED}/home" "${ISOLATED}/tmp" "${ISOLATED}/cache/xdg" \
  "${ISOLATED}/cache/library" "${ISOLATED}/frame_output" "${ISOLATED}/frame_logs" "${ISOLATED}/reports"
/usr/libexec/platform-python3.6 -I - "${ISOLATED}" "${FRAME}" <<'PY'
from __future__ import print_function
import os, stat, sys
from pathlib import Path
root=Path(sys.argv[1]); frame=Path(sys.argv[2]); resolved=root.resolve(strict=True)
assert root.is_dir() and not root.is_symlink() and resolved == root.absolute()
for rel in ('home','tmp','cache/xdg','cache/library','frame_output','frame_logs','reports'):
    p=root/rel; assert p.is_dir() and not p.is_symlink()
    assert resolved in p.resolve(strict=True).parents
    cursor=root
    for part in Path(rel).parts:
        cursor=cursor/part; mode=os.lstat(str(cursor)).st_mode
        assert stat.S_ISDIR(mode) and not stat.S_ISLNK(mode)
assert not frame.exists() and not frame.is_symlink()
assert not list(root.rglob('*.nlib')) and not list(root.rglob('*.ndm'))
print('M2189_ISOLATION_AND_OUTPUT_ABSENCE_PASS paths=7')
PY

/usr/libexec/platform-python3.6 -I - "${WORK}/execution_contract.json" "${LM_SHELL}" "${TCL}" "${LM_EXEC}" "${MILKYWAY}" "${ISOLATED}" "${RUNNER}" <<'PY'
from __future__ import print_function
import json, sys
from pathlib import Path
out,lm,tcl,actual,mw,isolated,runner=map(Path,sys.argv[1:])
d={'schema':'m2191_m2189_lm_execution_contract_r1_v1','scope':'lm_library_conversion_only',
   'license_queries':1,'top_level_lm_shell_runs':1,'milkyway_children':1,
   'pnr_runs':0,'automatic_retry':False,
   'lm_invocation':[str(lm),'-no_init','-f',str(tcl)],
   'lm_shell_sha256':'1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942',
   'lm_shell_exec_path':str(actual),'lm_shell_exec_sha256':'3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab',
   'milkyway_exec_path':str(mw),'milkyway_exec_sha256':'09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec',
   'isolated_root':str(isolated),'runner_path':str(runner)}
assert not out.exists() and not out.is_symlink()
out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
assert json.loads(out.read_text()) == d
print('M2189_EXECUTION_CONTRACT_WRITE_REREAD_PASS')
PY

"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1
grep -Fq 'Users of ICCompilerII:' "${WORK}/license_preflight.log"
LAUNCH_GATE="${WORK}/launch.gate"
MONITOR_READY="${WORK}/process_monitor.ready"
(
  cd -- "${ISOLATED}"
  while [[ ! -e "${LAUNCH_GATE}" ]]; do /usr/bin/sleep 0.01; done
  exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C HOME="${ISOLATED}/home" \
    TMPDIR="${ISOLATED}/tmp" XDG_CACHE_HOME="${ISOLATED}/cache/xdg" \
    SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
    M2189_ISOLATED_CWD="${ISOLATED}" M2189_LIBRARY_CACHE="${ISOLATED}/cache/library" \
    M2189_FRAME_DIR="${ISOLATED}/frame_output" M2189_FRAME_LOG_DIR="${ISOLATED}/frame_logs" \
    M2189_REPORT_DIR="${ISOLATED}/reports" M2189_MW_REF="${MW_REF}" \
    M2189_MILKYWAY_EXEC="${MILKYWAY}" "${LM_SHELL}" -no_init -f "${TCL}"
) >"${WORK}/lm_preflight.log" 2>&1 &
lm_pid=$!
"${MONITOR}" --root-pid "${lm_pid}" --stop-file "${WORK}/process_monitor.stop" \
  --ready-file "${MONITOR_READY}" --wrapper-path "${LM_SHELL}" --actual-exec-path "${LM_EXEC}" \
  --milkyway-path "${MILKYWAY}" --runner-path "${RUNNER}" \
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
wait "${lm_pid}"; lm_rc=$?; lm_pid=0
printf '%s\n' "${lm_rc}" >"${WORK}/lm_preflight.rc"
: >"${WORK}/process_monitor.stop"
wait "${monitor_pid}"; monitor_rc=$?; monitor_pid=0
set -e
[[ "${lm_rc}" -eq 0 && "${monitor_rc}" -eq 0 ]]

"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_after.json" >"${WORK}/repo_root_after.log"
cmp -s -- "${WORK}/repo_root_before.json" "${WORK}/repo_root_after.json"
"${CHECKER}" --work "${WORK}" --output "${WORK}/receipt.json" >"${WORK}/checker.log"
grep -Fxq 'RAW_PASS_M2191_M2189_LM_LIBRARY_CONVERSION_PENDING_M2192_INDEPENDENT_RESULT_HAMMER' "${WORK}/checker.log"
printf 'RAW_PASS_M2191_M2189_LM_LIBRARY_CONVERSION_PENDING_M2192_INDEPENDENT_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2191_M2189_LM_LIBRARY_CONVERSION_PENDING_M2192_INDEPENDENT_RESULT_HAMMER\n'
