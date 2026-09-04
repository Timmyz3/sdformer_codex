#!/usr/bin/env bash
set -euo pipefail
umask 002
[[ $# -eq 0 ]] || { echo "ERROR: M2223 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TCL="${SCRIPT_DIR}/run_lm_m2221_command_option_discovery.tcl"
CENSUS="${SCRIPT_DIR}/census_m2205_same_uid_tools.py"
INVENTORY="${SCRIPT_DIR}/inventory_m2153_repo_root.py"
CHECKER="${HW_ROOT}/system_simulator/scripts/check_m2223_lm_command_option_discovery.py"
CONTRACT="${HW_ROOT}/contracts/m2221_m2208_lm_command_option_discovery_source_contract_r1_20260904.json"
TEST="${HW_ROOT}/tests/test_m2221_lm_command_option_discovery_source.py"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M2208="${HW_ROOT}/reviews/m2208_m2207_m2205_lm_library_conversion_preflight_failure_hammer_r1_20260904"
M2222="${HW_ROOT}/reviews/m2222_m2221_m2208_lm_command_option_discovery_source_hammer_r1_20260904"
MW_MANIFEST="${HW_ROOT}/dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
TECH_BASE=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital
MW_REF="${TECH_BASE}/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140"
LM_SHELL=/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell
LM_EXEC=/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec
MILKYWAY=/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_SERVER=27030@ic.ismd-nemo
LICENSE_FILE=/opt/synopsys/Synopsys.dat
DOC_GENERATE=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/generate_frame_from_mw.2
DOC_SET=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/set_app_options.2
DOC_GET=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/get_app_option_value.2
DOC_REPORT=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/report_app_options.2
DOC_ENUM=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/get_app_options.2
DOC_MW=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat3/lib.setting.milkyway_exec.3
DOC_LOCAL=/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat3/lib.configuration.local_output_dir.3
RESULT="${HW_ROOT}/dc_handoff/runs/m2223_m2221_lm_command_option_discovery_raw_r1_20260904"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m2223_m2221_lm_command_option_discovery_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m2223_m2221_lm_command_option_discovery_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m2223_m2221_lm_command_option_discovery_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: M2223 identity mismatch: ${path}" >&2; exit 3; }
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
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact 8faf4234f4577cbfa751c5efc5c2ea01baa1f1a59fd4261b3dbf0513ccffa6ed "${M2208}/review.json"
verify_dir_seal "${M2208}"
sha_exact 7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3 "${MW_MANIFEST}"
sha_exec 1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942 "${LM_SHELL}"
sha_exec 3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab "${LM_EXEC}"
sha_exec 09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec "${MILKYWAY}"
sha_exec e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
sha_exact f9424346c44d9d48cbae5a3839f26cadad46b4d85e405deb19354356cd232952 "${DOC_GENERATE}"
sha_exact ae28a2f50dc5ed7457adad00428a0c0e7fa57cc4555866015d4ab4563e4ec0da "${DOC_SET}"
sha_exact f0d7b2b4334d00f90432c7fcdb319fe80668578633dfbda0bcdc644302e4e47a "${DOC_GET}"
sha_exact 6be35b3549beaa7ac73886f88cdaf80d40bfd985fc0dd4c96efd3587df89c3ba "${DOC_REPORT}"
sha_exact a9ef5c15a2022c38b0da1140638c1ff23d1806caf939f8e9f0d94ef1eb8b8135 "${DOC_ENUM}"
sha_exact b497b940eaf9c1f044362d701ec2eea5710391f4c5995370cee74d511916a1e9 "${DOC_MW}"
sha_exact 5354ec5b5964e454395a8f8d8cfecd489470d5c6555ec78242213d5925c6d9ea "${DOC_LOCAL}"
(cd -- "${MW_REF}" && sha256sum -c "${MW_MANIFEST}" >/dev/null)
[[ "$(find -P "${MW_REF}" -type f | wc -l)" -eq 1051 ]]
[[ -z "$(find -P "${MW_REF}" -type l -print -quit)" ]]

[[ -n "${M2221_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M2221_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2221_EXPECTED_SOURCE_REVIEW_SHA256:-}" && "$(sha_file "${M2222}/review.json")" == "${M2221_EXPECTED_SOURCE_REVIEW_SHA256}" ]] || exit 3
verify_dir_seal "${M2222}"
/usr/libexec/platform-python3.6 -I - "${M2222}/review.json" "${RUNNER}" "${TCL}" "${CENSUS}" "${INVENTORY}" "${CHECKER}" "${TEST}" "${CONTRACT}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
review,runner,tcl,census,inventory,checker,test,contract=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(review.read_text())
assert d['status']=='PASS_M2222_M2221_SOURCE_HAMMER__M2223_ONE_SHOT_AUTHORIZED'
assert d['score_over_100']>=95 and d['severity_counts']=={'p0':0,'p1':0,'p2':0}
for key,path in [('runner_sha256',runner),('tcl_sha256',tcl),('census_sha256',census),
                 ('inventory_sha256',inventory),('checker_sha256',checker),
                 ('test_sha256',test),('contract_sha256',contract)]:
    assert d['identity'][key]==sha(path)
assert d['authorization']=={'m2223':True,'license_queries':1,
    'top_level_lm_shell_runs':1,'generate_frame_commands':0,'create_lib_commands':0,
    'milkyway_process_runs':0,'pnr_runs':0,'automatic_retry':False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2223_ATTEMPT_CONSUMED\nlicense_queries=1\ntop_level_lm_shell_runs=1\ngenerate_frame_commands=0\ncreate_lib_commands=0\nmilkyway_process_runs=0\npnr_runs=0\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_before.json" >"${WORK}/repo_root_before.log"
"${CENSUS}" --phase before --output "${WORK}/same_uid_census_before.json" >"${WORK}/same_uid_census_before.log"
ISOLATED="${WORK}/isolated_cwd"
mkdir -p -- "${ISOLATED}/home" "${ISOLATED}/tmp" "${ISOLATED}/cache/xdg" \
  "${ISOLATED}/cache/library" "${ISOLATED}/frame_output" "${ISOLATED}/frame_logs" "${ISOLATED}/reports"
/usr/libexec/platform-python3.6 -I - "${ISOLATED}" <<'PY'
from __future__ import print_function
import os,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); resolved=root.resolve(strict=True)
assert root.is_dir() and not root.is_symlink() and resolved==root.absolute()
for rel in ('home','tmp','cache/xdg','cache/library','frame_output','frame_logs','reports'):
    p=root/rel; assert p.is_dir() and not p.is_symlink() and resolved in p.resolve(strict=True).parents
    cursor=root
    for part in Path(rel).parts:
        cursor=cursor/part; mode=os.lstat(str(cursor)).st_mode
        assert stat.S_ISDIR(mode) and not stat.S_ISLNK(mode)
for base in (root,root/'home'):
    assert not list(base.glob('.synopsys*')) and not list(base.glob('*.setup'))
    assert not list(base.glob('.tclshrc'))
assert not list((root/'frame_output').iterdir())
assert not list(root.rglob('*.nlib')) and not list(root.rglob('*.ndm'))
print('M2221_ISOLATION_AND_OUTPUT_ABSENCE_PASS paths=7 setup_files=0')
PY

/usr/libexec/platform-python3.6 -I - "${WORK}/execution_contract.json" "${LM_SHELL}" "${TCL}" "${LM_EXEC}" "${MILKYWAY}" "${ISOLATED}" "${RUNNER}" <<'PY'
from __future__ import print_function
import json,sys
from pathlib import Path
out,lm,tcl,actual,mw,isolated,runner=map(Path,sys.argv[1:])
d={'schema':'m2223_m2221_lm_command_option_discovery_execution_contract_r1_v1',
   'scope':'lm_command_option_discovery_only','startup_mode':'no_init',
   'license_queries':1,'top_level_lm_shell_runs':1,'generate_frame_commands':0,
   'create_lib_commands':0,'milkyway_process_runs':0,'pnr_runs':0,'automatic_retry':False,
   'lm_invocation':[str(lm),'-no_init','-f',str(tcl)],
   'lm_shell_sha256':'1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942',
   'lm_shell_exec_path':str(actual),'lm_shell_exec_sha256':'3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab',
   'milkyway_exec_path':str(mw),'milkyway_exec_sha256':'09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec',
   'isolated_root':str(isolated),'runner_path':str(runner)}
assert not out.exists() and not out.is_symlink(); out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
assert json.loads(out.read_text())==d
print('M2221_EXECUTION_CONTRACT_WRITE_REREAD_PASS')
PY

"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1
grep -Fq 'Users of ICCompilerII:' "${WORK}/license_preflight.log"
set +e
(
  cd -- "${ISOLATED}"
  exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C HOME="${ISOLATED}/home" \
    TMPDIR="${ISOLATED}/tmp" XDG_CACHE_HOME="${ISOLATED}/cache/xdg" \
    SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
    M2221_ISOLATED_CWD="${ISOLATED}" M2221_LIBRARY_CACHE="${ISOLATED}/cache/library" \
    M2221_FRAME_DIR="${ISOLATED}/frame_output" M2221_MILKYWAY_EXEC="${MILKYWAY}" \
    "${LM_SHELL}" -no_init -f "${TCL}"
) >"${WORK}/lm_discovery.log" 2>&1
lm_rc=$?
set -e
printf '%s\n' "${lm_rc}" >"${WORK}/lm_discovery.rc"

"${CENSUS}" --phase after --output "${WORK}/same_uid_census_after.json" >"${WORK}/same_uid_census_after.log"
"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_after.json" >"${WORK}/repo_root_after.log"
cmp -s -- "${WORK}/repo_root_before.json" "${WORK}/repo_root_after.json"
[[ "${lm_rc}" -eq 0 ]]
/usr/libexec/platform-python3.6 -I - "${WORK}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
work=Path(sys.argv[1]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'schema':'m2223_m2221_lm_command_option_discovery_output_manifest_r1_v1',
   'lm_return_code':0,'license_queries':1,'top_level_lm_shell_runs':1,
   'generate_frame_commands':0,'create_lib_commands':0,'milkyway_process_runs':0,
   'pnr_runs':0,'automatic_retry':False,
   'execution_contract_sha256':sha(work/'execution_contract.json'),
   'lm_log_sha256':sha(work/'lm_discovery.log'),
   'same_uid_census_before_sha256':sha(work/'same_uid_census_before.json'),
   'same_uid_census_after_sha256':sha(work/'same_uid_census_after.json'),
   'repo_root_before_sha256':sha(work/'repo_root_before.json'),
   'repo_root_after_sha256':sha(work/'repo_root_after.json')}
out=work/'execution_output_manifest.json'; assert not out.exists(); out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print('M2221_EXECUTION_OUTPUT_MANIFEST_PASS')
PY

"${CHECKER}" --work "${WORK}" --output "${WORK}/receipt.json" >"${WORK}/checker.log"
grep -Fxq 'RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER' "${WORK}/checker.log"
printf 'RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER\n'
