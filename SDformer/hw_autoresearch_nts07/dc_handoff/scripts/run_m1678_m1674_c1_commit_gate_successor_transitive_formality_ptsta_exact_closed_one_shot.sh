#!/usr/bin/env bash
set -euo pipefail
umask 002

# M1678 is a resource-gate-only successor to immutable M1674/M1675/M1676 and
# is source-only until a different author seals M1679 and a separate M1680
# release is caller-pinned.  A released attempt invokes exactly two
# fm_shell processes followed by one independent pt_shell process.  It never
# invokes DC, VCS, PTPX, GPU or remote work and never mutates M993/M1665.
[[ $# -eq 0 ]] || { echo "ERROR: M1678 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
CONTRACT="${HW_ROOT}/contracts/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_contract_r1_20260901.json"
AUTHOR_RECEIPT_DIR="${HW_ROOT}/reviews/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_author_receipt_r1_20260901"
AUTHOR_RECEIPT="${AUTHOR_RECEIPT_DIR}/author_receipt.json"
HAMMER_DIR="${HW_ROOT}/reviews/m1679_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_hammer_r1_20260901"
HAMMER_REVIEW="${HAMMER_DIR}/review.json"
RELEASE="${HW_ROOT}/contracts/m1680_m1679_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_launch_release_r1_20260901.json"
M1674_RUNNER="${HW_ROOT}/dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_exact_closed_one_shot.sh"
M1674_CONTRACT="${HW_ROOT}/contracts/m1674_m1665_c1_transitive_formality_ptsta_source_contract_r1_20260901.json"
M1675_DIR="${HW_ROOT}/reviews/m1675_m1674_m1665_c1_transitive_formality_ptsta_source_hammer_r1_20260901"
M1676_RELEASE="${HW_ROOT}/contracts/m1676_m1675_m1674_m1665_c1_transitive_formality_ptsta_launch_release_r1_20260901.json"

M1665_DIR="${HW_ROOT}/dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
M1665_ORIGINAL="${M1665_DIR}/original_quarantine"
M1667_DIR="${HW_ROOT}/reviews/m1667_m1665_c1_canonical_recovery_result_hammer_r1_20260901"
M993_DIR="${HW_ROOT}/dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
M993_ORIGINAL="${M993_DIR}/original_quarantine"
TOP=m935_m912_three_stage_exact_parent_match_product_capture_island
RTL_FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m962_m935_three_stage_match_macro_aware_dc.f"
RTL="${HW_ROOT}/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M993_NETLIST="${M993_ORIGINAL}/netlist/${TOP}_mapped.v"
M993_SVF="${M993_ORIGINAL}/netlist/${TOP}.svf"
M1665_NETLIST="${M1665_ORIGINAL}/netlist/${TOP}_m1630_residual_hold_closed_mapped.v"
M1665_DDC="${M1665_ORIGINAL}/netlist/${TOP}_m1630_residual_hold_closed.ddc"
M1665_SVF="${M1665_ORIGINAL}/netlist/${TOP}_m1630_residual_hold_closed.svf"
M1665_SDC="${M1665_ORIGINAL}/netlist/${TOP}_m1630_residual_hold_closed_mapped.sdc"
RTL_TO_M993_TCL="${HW_ROOT}/dc_handoff/scripts/run_formality_m1674_c1_rtl_to_m993_transitive.tcl"
GATE_TO_GATE_TCL="${HW_ROOT}/dc_handoff/scripts/run_formality_m1674_c1_m993_to_m1665_gate_to_gate.tcl"
PT_TCL="${HW_ROOT}/dc_handoff/scripts/run_ptsta_m1674_c1_m1665_slowmax_fastmin.tcl"
DOC359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

FM_SHELL=/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell
PT_SHELL=/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
STD_SLOW=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
STD_FAST=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
MACRO_ROOT=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821
MACRO_SLOW="${MACRO_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
MACRO_FAST="${MACRO_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"
MACRO_MANIFEST="${MACRO_ROOT}/SHA256SUMS"

RESULT="${HW_ROOT}/dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_r1_20260901"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_launch_lock"
WORK_ACTIVE=0
COMPLETE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
fail() { echo "ERROR: M1678 $*" >&2; exit 3; }
sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/nonregular ${path}"
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || fail "SHA mismatch ${path}: ${got}"
}
verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}.sha256" && ! -L "${payload}.sha256" && \
     -f "${payload}.sha256.seal.sha256" && ! -L "${payload}.sha256.seal.sha256" ]] || \
    fail "file seal absent ${payload}"
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || \
    fail "file seal invalid ${payload}"
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" && \
     -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "directory seal absent ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || \
    fail "directory seal invalid ${dir}"
}
seal_dir() {
  local dir="$1"
  (cd -- "${dir}" && \
    find -P . -type f ! -path './rtl_to_m993/work/*' \
      ! -path './m993_to_m1665/work/*' ! -path './ptsta/work/*' \
      ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- >SHA256SUMS && \
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
same_uid_eda() {
  /usr/bin/python3 - "$(id -u)" <<'PY'
import os,sys
uid=int(sys.argv[1]); names={'dc_shell','dc_shell-t','fm_shell','fm_shell_exec',
'pt_shell','vcs','vcs1','vlogan','simv','common_shell_exec','common_shell_exe'}
hits=[]
for ent in os.listdir('/proc'):
    if not ent.isdigit(): continue
    try:
        pid=int(ent); lines=open('/proc/%d/status'%pid).read().splitlines()
        puid=int(next(x for x in lines if x.startswith('Uid:')).split()[1])
        stat=open('/proc/%d/stat'%pid).read(); tail=stat[stat.rfind(')')+2:].split()
        state,start=tail[0],int(tail[19])
        comm=open('/proc/%d/comm'%pid).read().strip()
        exe=os.path.basename(os.path.realpath('/proc/%d/exe'%pid))
    except (OSError,StopIteration,ValueError):
        continue
    if puid==uid and state!='Z' and (comm in names or exe in names):
        hits.append('%d:%d:%s:%s'%(pid,start,comm,exe))
print(','.join(sorted(hits)))
PY
}
on_exit() {
  local rc=$?
  set +e
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' \
      "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM

# The caller must pin both immutable source and the future independently
# reviewed release.  No namespace is touched before the whole authority and
# resource preflight has passed.
[[ -n "${M1678_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha_file "${RUNNER}")" == "${M1678_EXPECTED_RUNNER_SHA256}" ]] || \
  fail "caller did not pin exact runner"
[[ -n "${M1678_EXPECTED_RELEASE_SHA256:-}" ]] || fail "missing release pin"
verify_file_seal "${CONTRACT}"
verify_dir_seal "${AUTHOR_RECEIPT_DIR}"
verify_dir_seal "${HAMMER_DIR}"
verify_file_seal "${RELEASE}"
sha_exact "${M1678_EXPECTED_RELEASE_SHA256}" "${RELEASE}"

/usr/bin/python3 - "${RELEASE}" "$(sha_file "${RUNNER}")" \
  "$(sha_file "${CONTRACT}")" "$(sha_file "${AUTHOR_RECEIPT}")" \
  "$(sha_file "${HAMMER_REVIEW}")" <<'PY'
import json,re,sys
p,runner_sha,contract_sha,author_sha,hammer_sha=sys.argv[1:]
d=json.load(open(p))
def exact(o,keys,name):
    if type(o) is not dict or set(o)!=set(keys): raise SystemExit(name+' keyset')
def boolean(o,k,v):
    if type(o[k]) is not bool or o[k] is not v: raise SystemExit('bool '+k)
exact(d,('schema','date','milestone','status','identity','authorization',
         'execution_order','result_policy','claim_boundary'),'top')
if d['schema']!='m1680_m1679_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_launch_release_r1_v1': raise SystemExit('schema')
if d['status']!='AUTHORIZE_ONE_M1678_C1_COMMIT_GATE_SUCCESSOR_FORMALITY_PTSTA_ATTEMPT': raise SystemExit('status')
exact(d['identity'],('runner_path','runner_sha256','source_contract_path',
      'source_contract_sha256','author_receipt_path','author_receipt_sha256',
      'source_hammer_path','source_hammer_sha256','m1674_runner_sha256',
      'm1674_contract_sha256','m1675_review_sha256','m1676_release_sha256',
      'm1665_manifest_sha256','m1667_review_sha256','future_result',
      'future_attempt'),'identity')
expected={'runner_sha256':runner_sha,'source_contract_sha256':contract_sha,
          'author_receipt_sha256':author_sha,'source_hammer_sha256':hammer_sha,
          'm1674_runner_sha256':'55409e053c7392de2e5962d7d8a9430cfc6429483ea3d774cd7ff4906305b944',
          'm1674_contract_sha256':'16424c8442febfccc22d3e0f920c96b4a8f6df7ae3b53dcbca072de9fc5e6bc9',
          'm1675_review_sha256':'644fba82b931b4bcc84287731ce6144a6fae94127fe8b8cf466e2512bf8b88e7',
          'm1676_release_sha256':'121e0843c69dccbb2039d9127e3732754d2d299bf5a818c1c3038b1d940be5a6',
          'm1665_manifest_sha256':'a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72',
          'm1667_review_sha256':'bcec72d13d08ddd38252eda93472a48ee1b9406563780b273544bf863f7b1db0'}
for k,v in expected.items():
    if d['identity'][k]!=v: raise SystemExit('identity '+k)
paths={
'runner_path':'dc_handoff/scripts/run_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_exact_closed_one_shot.sh',
'source_contract_path':'contracts/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_contract_r1_20260901.json',
'author_receipt_path':'reviews/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_author_receipt_r1_20260901/author_receipt.json',
'source_hammer_path':'reviews/m1679_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_hammer_r1_20260901/review.json',
'future_result':'dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_r1_20260901',
'future_attempt':'dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_attempt_consumed'}
for k,v in paths.items():
    if d['identity'][k]!=v: raise SystemExit('path '+k)
exact(d['authorization'],('launch_now','max_attempts','formality_runs','pt_runs',
      'dc_runs','vcs_runs','ptpx_runs','gpu_runs','remote_runs','retry'),'authorization')
expected_auth={'launch_now':True,'max_attempts':1,'formality_runs':2,'pt_runs':1,
 'dc_runs':0,'vcs_runs':0,'ptpx_runs':0,'gpu_runs':0,'remote_runs':0,'retry':False}
if d['authorization']!=expected_auth: raise SystemExit('authorization values')
if d['execution_order']!=['RTL-to-M993 Formality','different-process M993-to-M1665 gate-to-gate Formality','independent PrimeTime slow-max/fast-min','different-author result hammer before any claim']:
    raise SystemExit('execution order')
exact(d['result_policy'],('all_three_processes_must_exit_zero','both_formality_proofs_must_succeed',
      'prime_time_setup_hold_must_be_nonnegative','macro_count_exact','fresh_result_hammer_required'),'result')
if d['result_policy']!={'all_three_processes_must_exit_zero':True,
 'both_formality_proofs_must_succeed':True,
 'prime_time_setup_hold_must_be_nonnegative':True,
 'macro_count_exact':9,'fresh_result_hammer_required':True}: raise SystemExit('result policy')
exact(d['claim_boundary'],('launch_release','formality','prime_time','power','energy',
      'cycle_speedup','system_speedup','paper_ppa_ready','paper_citable','headline'),'claims')
for k in d['claim_boundary']: boolean(d['claim_boundary'],k,k=='launch_release')
PY
/usr/bin/python3 - "${HAMMER_REVIEW}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('status')!='PASS_M1679_M1678_C1_COMMIT_GATE_SUCCESSOR_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT':
    raise SystemExit('hammer status')
a=d.get('authorization')
if type(a) is not dict or a.get('future_m1678_attempts')!=1 or \
   a.get('source_release_authoring') is not True or \
   a.get('all_eda_now') is not False:
    raise SystemExit('hammer authorization')
PY

# Immutable evidence, source and tool identities.
verify_file_seal "${M1674_CONTRACT}"
verify_dir_seal "${M1675_DIR}"
verify_file_seal "${M1676_RELEASE}"
sha_exact 55409e053c7392de2e5962d7d8a9430cfc6429483ea3d774cd7ff4906305b944 "${M1674_RUNNER}"
sha_exact 16424c8442febfccc22d3e0f920c96b4a8f6df7ae3b53dcbca072de9fc5e6bc9 "${M1674_CONTRACT}"
sha_exact 644fba82b931b4bcc84287731ce6144a6fae94127fe8b8cf466e2512bf8b88e7 "${M1675_DIR}/review.json"
sha_exact 73a01b08f7f21781512f0f0c2da38189d2a96875568f1848f17bc6a87cd0e07b "${M1675_DIR}/SHA256SUMS.seal.sha256"
sha_exact 121e0843c69dccbb2039d9127e3732754d2d299bf5a818c1c3038b1d940be5a6 "${M1676_RELEASE}"
sha_exact 5cc03cd4c50de76c5c801e59b9f8513115855beffa1b066d3b188bfa68b9be50 "${M1676_RELEASE}.sha256.seal.sha256"
verify_dir_seal "${M1665_DIR}"
verify_dir_seal "${M1665_ORIGINAL}"
verify_dir_seal "${M1667_DIR}"
verify_dir_seal "${M993_DIR}"
verify_dir_seal "${M993_ORIGINAL}"
sha_exact a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72 "${M1665_DIR}/SHA256SUMS"
sha_exact 12d87acb439b0cc171d3f42cd4f169fa6a531946c9c3c120cc9babc9c36fbc08 "${M1665_DIR}/SHA256SUMS.seal.sha256"
sha_exact 07601960b22b5f1d23226d5a60ce25c92b9652bc9700d058d6a4aea38e08b4e6 "${M1665_DIR}/m1665_recovered_c1_dc_receipt.json"
sha_exact bcec72d13d08ddd38252eda93472a48ee1b9406563780b273544bf863f7b1db0 "${M1667_DIR}/review.json"
sha_exact c942b7b7461fdd4317a398f822d21b3f31be87f7e8f73f17c04bf11e965db5d9 "${M1667_DIR}/SHA256SUMS.seal.sha256"
sha_exact 8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093 "${M993_DIR}/SHA256SUMS"
sha_exact 0cc3b953342d6f149183e5fdf55b97174f69f97701574b0a79f05a5068ff6689 "${M993_DIR}/SHA256SUMS.seal.sha256"
sha_exact e6d9d1ead574e7c4cc446981888aa404d2d92ecd321a6855a43ea498c501e75c "${RTL_FILELIST}"
sha_exact e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact d3a72876d9b40f73c47834da123388fa40263cf017c61586f2113b352a7bc3de "${RTL_TO_M993_TCL}"
sha_exact 6df82c2435ab312263fd133a8e52371ea3de1004bc493d9553879eafaf3d1e12 "${GATE_TO_GATE_TCL}"
sha_exact e289faa0abb9f8e7136305158ef086e20bd7e77d2f960e436f51138a431241a1 "${PT_TCL}"
sha_exact 9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf "${M993_NETLIST}"
sha_exact 8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7 "${M993_SVF}"
sha_exact 842d100f6a3fc26684e13a8065191028af7840685aaf4b7cfa77a4fe998c46ee "${M1665_NETLIST}"
sha_exact 2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0 "${M1665_DDC}"
sha_exact 7c15c1a30827df74c0da35f24f7e88723484c2a211edd3d6c049f52e21dec274 "${M1665_SVF}"
sha_exact 5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198 "${M1665_SDC}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
sha_exact aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b "${FM_SHELL}"
sha_exact afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef "${PT_SHELL}"
sha_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
sha_exact fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490 "${LICENSE_FILE}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${STD_SLOW}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${STD_FAST}"
sha_exact cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${MACRO_SLOW}"
sha_exact 8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f "${MACRO_FAST}"
sha_exact c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${MACRO_MANIFEST}"
(cd -- "${MACRO_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)

# The exact M1665 SDC is the only constraint input.  M1678 is forbidden to
# add a false path, multicycle, min/max delay, disabled arc or case analysis.
[[ "$(grep -Ec '^[[:space:]]*create_clock .* -period 3([.]0+)?([[:space:]]|$)' "${M1665_SDC}")" -eq 1 ]] || fail "3 ns clock identity"
[[ "$(grep -Ec '^[[:space:]]*set_clock_uncertainty -setup 0[.]2([[:space:]]|$)' "${M1665_SDC}")" -eq 1 ]] || fail "setup uncertainty identity"
[[ "$(grep -Ec '^[[:space:]]*set_clock_uncertainty -hold 0[.]05([[:space:]]|$)' "${M1665_SDC}")" -eq 1 ]] || fail "hold uncertainty identity"
! grep -Eq '^[[:space:]]*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis)([[:space:]]|$)' "${M1665_SDC}" || fail "forbidden timing exception"

# No live tool, namespace, resource or license state is consumed until all
# immutable authority checks above pass.  The three EDA calls are sequential.
[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 5
[[ -z "$(same_uid_eda)" ]] || exit 4
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed_as="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
commit_headroom=$((commit_limit-committed_as))
disk_available="$(df -Pk "${HW_ROOT}" | awk 'NR==2 {print $4}')"
[[ "${mem_available}" -ge 16777216 && "${commit_headroom}" -ge 25165824 && \
   "${disk_available}" -ge 4194304 ]] || exit 6

license_tmp="$(mktemp -d /tmp/m1678_c1_license.XXXXXX)"
chmod 0700 "${license_tmp}"
for feature in Formality PrimeTime; do
  "${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f "${feature}" \
    >"${license_tmp}/${feature}.txt" 2>&1 || { rm -rf -- "${license_tmp}"; exit 7; }
  /usr/bin/python3 - "${license_tmp}/${feature}.txt" "${feature}" <<'PY'
import re,sys
t=open(sys.argv[1],errors='replace').read(); f=sys.argv[2]
m=re.search(r'Total of\s+(\d+)\s+licenses? issued;\s+Total of\s+(\d+)\s+licenses? in use',t,re.S)
if not m or int(m.group(1)) <= int(m.group(2)): raise SystemExit('no '+f+' license')
PY
done
rm -rf -- "${license_tmp}"
[[ -z "$(same_uid_eda)" ]] || exit 4
[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 5
mkdir "${LOCK}"
mkdir "${ATTEMPT}"
printf 'status=ATTEMPT_CONSUMED_BEFORE_FIRST_EDA_M1678_RESOURCE_SUCCESSOR\nformality_runs_authorized=2\npt_runs_authorized=1\ncommit_headroom_min_kib=25165824\npredecessor_commit_headroom_min_kib=50331648\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -p "${WORK}/rtl_to_m993/reports" "${WORK}/rtl_to_m993/work" \
  "${WORK}/m993_to_m1665/reports" "${WORK}/m993_to_m1665/work" \
  "${WORK}/ptsta/reports" "${WORK}/ptsta/work"
WORK_ACTIVE=1
cp -- "${CONTRACT}" "${WORK}/source_contract.json"
cp -- "${RELEASE}" "${WORK}/launch_release.json"
cp -- "${HAMMER_REVIEW}" "${WORK}/source_hammer_review.json"
sha256sum -- "${RUNNER}" "${RTL_TO_M993_TCL}" "${GATE_TO_GATE_TCL}" \
  "${PT_TCL}" "${RTL_FILELIST}" "${RTL}" "${MACRO_RTL}" \
  "${M1674_RUNNER}" "${M1674_CONTRACT}" "${M1675_DIR}/review.json" \
  "${M1676_RELEASE}" \
  "${M993_NETLIST}" "${M993_SVF}" "${M1665_NETLIST}" "${M1665_DDC}" \
  "${M1665_SVF}" "${M1665_SDC}" "${STD_SLOW}" "${STD_FAST}" \
  "${MACRO_SLOW}" "${MACRO_FAST}" "${CONTRACT}" "${HAMMER_REVIEW}" \
  "${RELEASE}" "${DOC359}" >"${WORK}/input_sha256.txt"

export M1674_SNAPSHOT_ROOT="${HW_ROOT}"
export M1674_RTL_FILELIST="${RTL_FILELIST}"
export M1674_STD_SLOW_DB="${STD_SLOW}"
export M1674_STD_FAST_DB="${STD_FAST}"
export M1674_MACRO_SLOW_DB="${MACRO_SLOW}"
export M1674_MACRO_FAST_DB="${MACRO_FAST}"
export M1674_M993_MAPPED_NETLIST="${M993_NETLIST}"
export M1674_M993_SVF="${M993_SVF}"
export M1674_M1665_MAPPED_NETLIST="${M1665_NETLIST}"
export M1674_M1665_MAPPED_SDC="${M1665_SDC}"

export M1674_FM_OUTPUT_DIR="${WORK}/rtl_to_m993"
set +e
(cd -- "${WORK}/rtl_to_m993/work" && "${FM_SHELL}" -f "${RTL_TO_M993_TCL}") \
  >"${WORK}/rtl_to_m993/formality.raw.log" 2>&1
rtl_fm_rc=$?
set -e
echo "${rtl_fm_rc}" >"${WORK}/rtl_to_m993/formality.rc"
[[ "${rtl_fm_rc}" -eq 0 ]]
[[ "$(grep -xc 'M1674_C1_RTL_TO_M993_FORMALITY_INTERNAL_COMPLETE=PASS' "${WORK}/rtl_to_m993/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
grep -q 'Verification SUCCEEDED' "${WORK}/rtl_to_m993/reports/formality_status.rpt"
grep -Eq '[1-9][0-9]* Passing compare points' "${WORK}/rtl_to_m993/reports/formality_status.rpt"
grep -q 'No unmatched points' "${WORK}/rtl_to_m993/reports/formality_unmatched.rpt"
grep -q 'No failing compare points' "${WORK}/rtl_to_m993/reports/formality_failing.rpt"
grep -q 'No aborted compare points' "${WORK}/rtl_to_m993/reports/formality_aborted.rpt"
grep -q 'No unverified compare points' "${WORK}/rtl_to_m993/reports/formality_unverified.rpt"
! grep -Eq '^(Error|Fatal):' "${WORK}/rtl_to_m993/formality.raw.log"

export M1674_FM_OUTPUT_DIR="${WORK}/m993_to_m1665"
set +e
(cd -- "${WORK}/m993_to_m1665/work" && "${FM_SHELL}" -f "${GATE_TO_GATE_TCL}") \
  >"${WORK}/m993_to_m1665/formality.raw.log" 2>&1
gate_fm_rc=$?
set -e
echo "${gate_fm_rc}" >"${WORK}/m993_to_m1665/formality.rc"
[[ "${gate_fm_rc}" -eq 0 ]]
[[ "$(grep -xc 'M1674_C1_M993_TO_M1665_GATE_FORMALITY_INTERNAL_COMPLETE=PASS' "${WORK}/m993_to_m1665/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
grep -q 'Verification SUCCEEDED' "${WORK}/m993_to_m1665/reports/formality_status.rpt"
grep -Eq '[1-9][0-9]* Passing compare points' "${WORK}/m993_to_m1665/reports/formality_status.rpt"
grep -q 'No unmatched points' "${WORK}/m993_to_m1665/reports/formality_unmatched.rpt"
grep -q 'No failing compare points' "${WORK}/m993_to_m1665/reports/formality_failing.rpt"
grep -q 'No aborted compare points' "${WORK}/m993_to_m1665/reports/formality_aborted.rpt"
grep -q 'No unverified compare points' "${WORK}/m993_to_m1665/reports/formality_unverified.rpt"
! grep -Eq '^(Error|Fatal):' "${WORK}/m993_to_m1665/formality.raw.log"

export M1674_PT_OUTPUT_DIR="${WORK}/ptsta"
set +e
(cd -- "${WORK}/ptsta/work" && "${PT_SHELL}" -f "${PT_TCL}") \
  >"${WORK}/ptsta/pt.raw.log" 2>&1
pt_rc=$?
set -e
echo "${pt_rc}" >"${WORK}/ptsta/pt.rc"
[[ "${pt_rc}" -eq 0 ]]
[[ "$(grep -xc 'M1674_C1_M1665_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS' "${WORK}/ptsta/PTSTA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
! grep -Eq '^(Error|Fatal):' "${WORK}/ptsta/pt.raw.log"
for report in check_timing.rpt analysis_coverage.rpt global_timing.rpt \
  timing_setup_slow.rpt timing_hold_fast.rpt constraint_violators.rpt \
  clock.rpt exceptions.rpt design.rpt wire_load.rpt libraries.rpt \
  runtime_scope.rpt timing_summary_machine.txt; do
  [[ -s "${WORK}/ptsta/reports/${report}" ]] || exit 30
done
! grep -q 'slack (VIOLATED)' "${WORK}/ptsta/reports/timing_setup_slow.rpt"
! grep -q 'slack (VIOLATED)' "${WORK}/ptsta/reports/timing_hold_fast.rpt"
! grep -q 'slack (VIOLATED)' "${WORK}/ptsta/reports/constraint_violators.rpt"

/usr/bin/python3 - "${WORK}" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1])
def points(rel):
    t=(root/rel).read_text(errors='replace')
    m=re.search(r'(\d+) Passing compare points',t)
    if not m or int(m.group(1))<=0: raise SystemExit('compare points '+rel)
    return int(m.group(1))
machine={}
for line in (root/'ptsta/reports/timing_summary_machine.txt').read_text().splitlines():
    k,v=line.split('=',1); machine[k]=v
setup=float(machine['setup_wns_ns']); hold=float(machine['hold_wns_ns'])
setup_tns=float(machine['setup_tns_ns']); hold_tns=float(machine['hold_tns_ns'])
setup_v=int(machine['setup_violating_paths']); hold_v=int(machine['hold_violating_paths'])
if setup<0 or hold<0 or setup_tns!=0 or hold_tns!=0 or setup_v!=0 or hold_v!=0 or int(machine['macro_count'])!=9:
    raise SystemExit('PT gate')
receipt={
 'schema':'m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_receipt_r1_v1',
 'status':'PASS_M1678_C1_COMMIT_GATE_SUCCESSOR_TRANSITIVE_FORMALITY_AND_INDEPENDENT_PTSTA_PENDING_RESULT_HAMMER',
 'tool_runs':{'fm_shell':2,'pt_shell':1,'dc_shell':0,'vcs':0,'ptpx':0},
 'scheduler_successor':{'only_runtime_gate_changed':'commit_headroom_min_kib',
   'm1674_commit_headroom_min_kib':50331648,
   'm1678_commit_headroom_min_kib':25165824,
   'mem_available_min_kib':16777216,'retry':False},
 'formality':{
   'rtl_to_m993_passing_compare_points':points('rtl_to_m993/reports/formality_status.rpt'),
   'm993_to_m1665_passing_compare_points':points('m993_to_m1665/reports/formality_status.rpt'),
   'transitive_rtl_to_m1665_equivalence_candidate':True,
   'failing_compare_points':0,'aborted_compare_points':0,
   'unverified_compare_points':0,'unmatched_points':0},
 'prime_time':{'clock_period_ns':3.0,'setup_uncertainty_ns':0.2,
   'hold_uncertainty_ns':0.05,'setup_wns_ns':setup,'setup_tns_ns':setup_tns,
   'setup_violating_paths':setup_v,'hold_wns_ns':hold,'hold_tns_ns':hold_tns,
   'hold_violating_paths':hold_v,
   'macro_count':9,'macro_cell':'TS1N28HPCPHVTB128X128M4S',
   'ideal_clock':True,'wireload':'ZeroWireload','spef':False,'pt_eco':False},
 'claim_boundary':{'formality_candidate':True,'prime_time_candidate':True,
   'result_hammered':False,'power':False,'energy':False,'cycle_speedup':False,
   'system_speedup':False,'paper_ppa_ready':False,'paper_citable':False,
   'headline':False}}
(root/'m1678_c1_commit_gate_successor_transitive_formality_ptsta_receipt_r1.json').write_text(
    json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY

printf '%s\n' \
  'status=PASS_M1678_C1_COMMIT_GATE_SUCCESSOR_FORMALITY_PTSTA_PENDING_RESULT_HAMMER' \
  'formality_runs=2' 'pt_runs=1' 'power=false' 'energy=false' \
  'cycle_speedup=false' 'system_speedup=false' 'paper_ppa_ready=false' \
  'paper_citable=false' 'headline=false' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
COMPLETE=1
rmdir -- "${LOCK}"
trap - EXIT INT TERM
echo "PASS_M1678_C1_COMMIT_GATE_SUCCESSOR_FORMALITY_PTSTA_PENDING_RESULT_HAMMER result=${RESULT}"
