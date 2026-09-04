#!/usr/bin/env bash
set -euo pipefail
umask 002
[[ $# -eq 0 ]] || { echo "ERROR: M2211 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
FILELIST="${HW_ROOT}/dc_handoff/filelists/tcasii_m2197_c2_tsbg_selective_bank_fill_directed_vcs.f"
M803="${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M2018="${HW_ROOT}/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
RTL="${HW_ROOT}/rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
SVA="${HW_ROOT}/verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv"
TB="${HW_ROOT}/tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv"
PARSER="${HW_ROOT}/system_simulator/scripts/parse_m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs.py"
CONTRACT="${HW_ROOT}/contracts/m2209_m2200_selective_bank_fill_vcs_runner_repair_source_contract_r1_20260904.json"
M2210="${HW_ROOT}/reviews/m2210_m2209_m2200_selective_bank_fill_vcs_runner_repair_source_hammer_r1_20260904"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
PYTHON=/opt/anaconda3/bin/python3.12
VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
VCS=${VCS_HOME}/bin/vcs
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
TOP=tb_m2197_c2_tsbg_selective_bank_fill_directed
RESULT="${HW_ROOT}/results/m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904"
ATTEMPT="${HW_ROOT}/results/.m2211_m2209_selective_bank_fill_vcs_attempt_consumed"
WORK="${HW_ROOT}/results/.m2211_m2209_selective_bank_fill_vcs_work.$$"
LOCK="${HW_ROOT}/results/.m2211_m2209_selective_bank_fill_vcs_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: M2211 identity mismatch ${path}" >&2; exit 3; }
}
sha_mode_exact() {
  local expected_sha="$1" expected_mode="$2" executable="$3" path="$4"
  sha_exact "${expected_sha}" "${path}"
  [[ "$(stat -c '%a' -- "${path}")" == "${expected_mode}" ]] || exit 3
  if [[ "${executable}" == "yes" ]]; then [[ -x "${path}" ]] || exit 3
  else [[ ! -x "${path}" ]] || exit 3
  fi
}
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

sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${M803}"
sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21 "${M2018}"
sha_exact f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e "${RTL}"
sha_exact 8003115edb919e9c5c6c9c36ce4ba75dfb37d9ec9f23e7c4cf59e2aed3b461b4 "${SVA}"
sha_exact a8a954826324aa20443e7b2acbbc6a0b1b2a92f83ebdd84bfdbb0879920526e3 "${TB}"
sha_exact 5beddf477b6938b599cfab962eba60f6d79dceeb825380f2e5cdc6f22b49dc13 "${FILELIST}"
sha_mode_exact fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571 664 no "${PARSER}"
sha_mode_exact 873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161 755 yes "${PYTHON}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS}"
sha_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
[[ -n "${M2211_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M2211_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2211_EXPECTED_M2210_REVIEW_SHA256:-}" && "$(sha_file "${M2210}/review.json")" == "${M2211_EXPECTED_M2210_REVIEW_SHA256}" ]] || exit 3
verify_dir_seal "${M2210}"
/usr/libexec/platform-python3.6 -I - "${M2210}/review.json" "${RUNNER}" "${FILELIST}" "${RTL}" "${M803}" "${SVA}" "${TB}" "${PARSER}" "${CONTRACT}" "${PYTHON}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
review,runner,filelist,rtl,m803,sva,tb,parser,contract,python=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(review.read_text())
assert d['status']=='PASS_M2210_M2209_SOURCE_HAMMER__M2211_ONE_SHOT_VCS_AUTHORIZED'
assert d['score_over_100']>=95 and d['severity_counts']=={'p0':0,'p1':0,'p2':0}
for key,path in [('runner_sha256',runner),('filelist_sha256',filelist),('rtl_sha256',rtl),
                 ('m803_sha256',m803),('sva_sha256',sva),('tb_sha256',tb),
                 ('parser_sha256',parser),('contract_sha256',contract),
                 ('python_sha256',python)]:
    assert d['identity'][key]==sha(path),(key,d['identity'].get(key),sha(path))
assert d['authorization']=={'m2211':True,'license_queries':1,'vcs_compiles':1,
                             'simv_runs':1,'parser_runs':1,'all_other_eda_runs':0,
                             'automatic_retry':False,'reuse_old_artifacts':False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked={'vcs','vcs1','vlogan','simv','dc_shell','pt_shell','fm_shell','icc2_shell','icc2_exec','lm_shell','lm_shell_exec'}
hits=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        if p.stat().st_uid!=os.getuid(): continue
        names={(p/'comm').read_text().strip(),Path(os.readlink(p/'exe')).name}
    except Exception: continue
    if names & blocked: hits.append((p.name,sorted(names)))
if hits: raise SystemExit('M2211 same-UID EDA collision: %r'%hits)
PY
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 16777216 && $((commit_limit-committed)) -ge 16777216 ]] || exit 4

cd -- "${REPO_ROOT}"
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2211_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1\nparser_runs=1\nretry=false\nreuse_old_artifacts=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f VCSCompiler_Net >"${WORK}/license_preflight.log" 2>&1
grep -Fq 'Users of VCSCompiler_Net:' "${WORK}/license_preflight.log"
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  "${VCS}" -full64 -sverilog -assert svaext -timescale=1ns/1ps -top "${TOP}" \
  -f "${FILELIST}" -o "${WORK}/simv" -Mdir="${WORK}/csrc" >"${WORK}/vcs_compile.log" 2>&1
set +e
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME="${VCS_HOME}" \
  SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
  /usr/bin/timeout --signal=TERM --kill-after=10s 300s "${WORK}/simv" \
  -assert global_finish_maxfail=1 >"${WORK}/simv.log" 2>&1
sim_rc=$?
set -e
printf '%s\n' "${sim_rc}" >"${WORK}/simv.rc"
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON}" -B "${PARSER}" --sim-log "${WORK}/simv.log" \
  --compile-log "${WORK}/vcs_compile.log" --sim-rc "${WORK}/simv.rc" \
  --output "${WORK}/receipt.json" >"${WORK}/parser.log"
grep -Fxq 'RAW_PASS_M2199_M2197_DIRECTED_VCS_PENDING_M2200_RESULT_HAMMER' "${WORK}/parser.log"
rm -f -- "${WORK}/simv" "${WORK}/vc_hdrs.h"
rm -rf -- "${WORK}/csrc" "${WORK}/simv.daidir" "${WORK}/simv.vdb"
for build_only in simv vc_hdrs.h csrc simv.daidir simv.vdb; do
  [[ ! -e "${WORK}/${build_only}" ]] || exit 5
done
[[ -z "$(find -P "${WORK}" -type l -print -quit)" ]] || exit 5
printf 'RAW_PASS_M2211_M2209_DIRECTED_VCS_PENDING_M2212_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2211_M2209_DIRECTED_VCS_PENDING_M2212_RESULT_HAMMER\n'
