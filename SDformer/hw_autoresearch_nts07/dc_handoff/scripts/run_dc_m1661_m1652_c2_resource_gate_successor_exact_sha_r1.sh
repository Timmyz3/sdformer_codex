#!/usr/bin/env bash
set -euo pipefail
umask 002

# M1661 is source-only.  A different-author M1662 hammer and a separately
# sealed M1663 release must exist and be caller-pinned before this runner may
# consume its single all-three-axis DC attempt.

[[ $# -eq 0 ]] || { echo "ERROR: M1661 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
CONTRACT="${HW_ROOT}/contracts/m1661_m1652_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
FILELIST="${HW_ROOT}/dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_axis_logic_only_dc.f"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC="${HW_ROOT}/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M1627="${HW_ROOT}/reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901"
M903="${HW_ROOT}/reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M872_RESULT="${HW_ROOT}/dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
M1634_RUNNER="${HW_ROOT}/dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_three_axis_logic_only_exact_sha_r1.sh"
M1634_CONTRACT="${HW_ROOT}/contracts/m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_source_contract_r1_20260901.json"
M1635_DIR="${HW_ROOT}/reviews/m1635_m1634_m1609_c2_three_axis_dc_source_hammer_r1_20260901"
M1636_RELEASE="${HW_ROOT}/contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_r1_20260901.json"
M1641_DIR="${HW_ROOT}/reviews/m1641_m1636_m1634_m1609_c2_three_axis_dc_release_hammer_r1_20260901"
M1652_RUNNER="${HW_ROOT}/dc_handoff/scripts/run_dc_m1652_m1634_c2_resource_gate_successor_exact_sha_r1.sh"
M1652_CONTRACT="${HW_ROOT}/contracts/m1652_m1634_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
M1653_FAIL_DIR="${HW_ROOT}/reviews/m1653_m1652_m1634_c2_resource_gate_successor_dc_source_hammer_r1_20260901"
HAMMER_DIR="${HW_ROOT}/reviews/m1662_m1661_m1652_c2_resource_gate_successor_dc_source_hammer_r1_20260901"
HAMMER_REVIEW="${HAMMER_DIR}/review.json"
RELEASE="${HW_ROOT}/contracts/m1663_m1662_m1661_m1652_c2_resource_gate_successor_dc_launch_release_r1_20260901.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
DC_SHELL=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
DC_ACTUAL=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
SLOW_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
FAST_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
DESIGN=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
RESULT="${HW_ROOT}/dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "ERROR: missing/nonregular ${path}" >&2; exit 3; }
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || {
    echo "ERROR: SHA mismatch ${path}: ${got}" >&2; exit 3; }
}
sha_tool() {
  local expected="$1" path="$2" got
  [[ -f "${path}" ]] || { echo "ERROR: missing tool ${path}" >&2; exit 3; }
  got="$(sha_file "${path}")"
  [[ "${got}" == "${expected}" ]] || {
    echo "ERROR: tool SHA mismatch ${path}: ${got}" >&2; exit 3; }
}
verify_file_seal() {
  local payload="$1" dir base
  dir="$(dirname -- "${payload}")"; base="$(basename -- "${payload}")"
  [[ -f "${payload}" && ! -L "${payload}" &&
      -f "${payload}.sha256" && ! -L "${payload}.sha256" &&
      -f "${payload}.sha256.seal.sha256" &&
      ! -L "${payload}.sha256.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c "${base}.sha256" >/dev/null &&
    sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" &&
      -f "${dir}/SHA256SUMS" && ! -L "${dir}/SHA256SUMS" &&
      -f "${dir}/SHA256SUMS.seal.sha256" &&
      ! -L "${dir}/SHA256SUMS.seal.sha256" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
  /usr/libexec/platform-python3.6 -I - "${dir}" <<'PY'
from __future__ import print_function
import hashlib, os, stat, sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=root/'SHA256SUMS'; outer=root/'SHA256SUMS.seal.sha256'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert outer.read_text()==sha(manifest)+'  SHA256SUMS\n'
listed={}
for row in manifest.read_text().splitlines():
    digest,name=row.split('  ',1); name=name.lstrip('./')
    assert name not in listed and '..' not in Path(name).parts and not Path(name).is_absolute()
    listed[name]=digest
actual=set()
for base,dirs,files in os.walk(str(root),followlinks=False):
    bp=Path(base)
    for name in list(dirs)+list(files):
        p=bp/name; rel=p.relative_to(root).as_posix(); mode=p.lstat().st_mode
        assert not stat.S_ISLNK(mode), rel
        if stat.S_ISDIR(mode): assert any(p.iterdir()), 'empty unsealed directory '+rel
        if stat.S_ISREG(mode) and rel not in ('SHA256SUMS','SHA256SUMS.seal.sha256'):
            actual.add(rel)
assert actual==set(listed),(set(listed)-actual,actual-set(listed))
for name,digest in listed.items(): assert sha(root/name)==digest,name
PY
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
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' \
      "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

# Frozen tool, technology, evidence and source identities.
sha_tool 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2 "${DC_SHELL}"
sha_tool bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391 "${DC_ACTUAL}"
sha_exact e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL}"
sha_exact fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490 "${LICENSE_FILE}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${SLOW_DB}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${FAST_DB}"
sha_exact c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe "${TCL}"
sha_exact 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5 "${SDC}"
sha_exact 03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f "${FILELIST}"
sha_exact 7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931 "${HW_ROOT}/rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
sha_exact 8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0 "${HW_ROOT}/rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"
sha_exact 529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267 "${HW_ROOT}/rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"
sha_exact f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1 "${HW_ROOT}/rtl_m218/m218_fc2_tagged_slice_service_island.sv"
sha_exact 44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e "${HW_ROOT}/rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv"
sha_exact 3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871 "${HW_ROOT}/rtl_m519/m519_fc2_k1_registered_release_service_island.sv"
sha_exact 010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b "${HW_ROOT}/rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv"
sha_exact 6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815 "${HW_ROOT}/rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv"
sha_exact 11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff "${HW_ROOT}/rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
sha_exact 2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e "${HW_ROOT}/rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"
sha_exact 3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b "${HW_ROOT}/rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
verify_dir_seal "${M1627}"
verify_dir_seal "${M903}"
verify_dir_seal "${M872_RESULT}"
sha_exact ab4f2187667301a37fbd5f523687a8971282e642163d42886edcdc138edc43d4 "${M1627}/review.json"
sha_exact 89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a "${M903}/review.json"

mapfile -t source_rows < <(sed '/^[[:space:]]*$/d' "${FILELIST}")
[[ ${#source_rows[@]} -eq 12 &&
    "${source_rows[0]}" == rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv ]] || {
  echo "ERROR: M1609 must be the unique first compactor definition" >&2; exit 3; }
! grep -Fqx 'rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv' "${FILELIST}" || {
  echo "ERROR: frozen M214 predecessor is forbidden" >&2; exit 3; }
[[ "$(grep -Rho '^module m214_fc2_raw4_to_descriptor4_terminal_hint_compactor\b' \
      "${HW_ROOT}/${source_rows[0]}" | wc -l)" -eq 1 ]] || exit 3
grep -Fq 'm214_fc2_raw4_to_descriptor4_terminal_hint_compactor #(' \
  "${HW_ROOT}/rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv" || exit 3
grep -Fq 'm216_fc2_raw4_to_source_cap_frontend #(' \
  "${HW_ROOT}/rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv" || exit 3
grep -Fq 'm519_fc2_registered_release_standalone_raw4_acc24 #(' \
  "${HW_ROOT}/rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv" || exit 3

verify_file_seal "${CONTRACT}"
/usr/libexec/platform-python3.6 -I - "${CONTRACT}" "${RUNNER}" "${M1627}/review.json" \
  "${M903}/review.json" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
contract,runner,m1627,m903=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads(contract.read_text()); v=json.loads(m1627.read_text()); d=json.loads(m903.read_text())
assert c['status']=='SOURCE_ONLY_M1661_M1652_C2_RESOURCE_GATE_SUCCESSOR__NO_EDA_AUTHORIZED'
assert c['identity']['runner_sha256']==sha(runner)
assert c['identity']['m1627_review_sha256']==sha(m1627)
assert c['identity']['m903_review_sha256']==sha(m903)
auth=c['authorization']
assert auth['dc_runs_now']==0
assert auth['future_dc_shell_runs_max']==3
assert auth['all_other_eda_runs']==0
assert auth['vcs_runs']==0
assert auth['pt_runs']==0
assert auth['formality_runs']==0
assert auth['ptpx_runs']==0
assert auth['gpu_runs']==0
assert auth['remote_runs']==0
assert auth['attempts_created_now']==0
assert auth['retry'] is False
assert v['status']=='PASS_M1627_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_RESULT_HAMMER'
assert v['score']>=95 and v['p0_count']==0 and v['p1_count']==0
assert d['status']=='PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED'
PY

# The old comparison is provenance only.  Every architecture is synthesized
# fresh from the M1609-selected filelist; no M872 DDC/netlist may be copied.
for old_artifact in \
  "${M872_RESULT}/k1/netlist/${DESIGN}.ddc" \
  "${M872_RESULT}/k8/netlist/${DESIGN}.ddc" \
  "${M872_RESULT}/k1x8/netlist/${DESIGN}.ddc"; do
  [[ -f "${old_artifact}" && ! -L "${old_artifact}" ]] || exit 3
done

# Immutable predecessor source/release authority is re-verified.  M1661
# changes no source, physical-flow or result predicate from M1634.
sha_exact da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7 "${M1634_RUNNER}"
verify_file_seal "${M1634_CONTRACT}"
sha_exact 9f5e5b1cb40da5cd403270ba48ceac9b5a7d6aecd79b7ad98cf3d644d0f8f030 "${M1634_CONTRACT}"
verify_dir_seal "${M1635_DIR}"
sha_exact 215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620 "${M1635_DIR}/review.json"
verify_file_seal "${M1636_RELEASE}"
sha_exact 0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088 "${M1636_RELEASE}"
verify_dir_seal "${M1641_DIR}"
sha_exact 278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6 "${M1641_DIR}/review.json"
/usr/libexec/platform-python3.6 -I - "${M1635_DIR}/review.json" "${M1636_RELEASE}"   "${M1641_DIR}/review.json" "${M1634_RUNNER}" "${M1634_CONTRACT}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
m1635,m1636,m1641,runner,contract=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads(m1635.read_text()); r=json.loads(m1636.read_text()); q=json.loads(m1641.read_text())
assert h['status']=='PASS_M1635_M1634_M1609_C2_THREE_AXIS_DC_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT'
assert r['status']=='AUTHORIZE_ONE_M1634_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT'
assert q['status']=='PASS_M1641_M1636_C2_THREE_AXIS_DC_RELEASE_HAMMER__ONE_LAUNCH_ADMITTED'
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert q['identity']['release_sha256']==sha(m1636)
assert q['identity']['runner_sha256']==sha(runner)
assert q['p0_count']==0 and q['p1_count']==0
PY

# The sealed M1652 predecessor and its failed M1653 review are immutable
# negative motivation.  M1653 must remain NO-GO; this fresh identity repairs
# only the executable embedded authorization preflight.
sha_exact 57f9b90642641215c801b0f61302636ddecb81e6b37523763f6523f2862dfdb3 "${M1652_RUNNER}"
verify_file_seal "${M1652_CONTRACT}"
sha_exact 01ee8cff796705c71a0b3c5875046ca32d08935936026315375da797d02d863c "${M1652_CONTRACT}"
verify_dir_seal "${M1653_FAIL_DIR}"
sha_exact 5e3e6c9974e26a28be3e6bae7efc93e661afafaf0ba8b5b9ebf35e5ad0855d6d "${M1653_FAIL_DIR}/review.json"
/usr/libexec/platform-python3.6 -I - "${M1653_FAIL_DIR}/review.json" <<'PY'
from __future__ import print_function
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['status']=='FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE'
assert r['p0_count']==0 and r['p1_count']==1
assert r['authorization']['m1654_release_authoring'] is False
assert r['authorization']['future_dc_attempts']==0
PY

# Independent M1662/M1663 authorization is intentionally absent at M1661
# source authoring.
verify_dir_seal "${HAMMER_DIR}"
verify_file_seal "${RELEASE}"
/usr/libexec/platform-python3.6 -I - "${HAMMER_REVIEW}" "${RELEASE}" \
  "${RUNNER}" "${CONTRACT}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
hammer,release,runner,contract=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads(hammer.read_text()); r=json.loads(release.read_text())
assert h['status']=='PASS_M1662_M1661_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT'
assert h['score']>=95 and h['p0_count']==0 and h['p1_count']==0
assert r['status']=='AUTHORIZE_ONE_M1661_C2_RESOURCE_GATE_SUCCESSOR_DC_ATTEMPT'
assert r['authorization']=={'dc_shell_runs':3,'all_other_eda_runs':0}
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['source_contract_sha256']==sha(contract)
assert r['identity']['hammer_review_sha256']==sha(hammer)
PY
[[ -n "${M1661_EXPECTED_DC_RUNNER_SHA256:-}" &&
   "$(sha_file "${RUNNER}")" == "${M1661_EXPECTED_DC_RUNNER_SHA256}" ]] || {
  echo "ERROR: caller must pin reviewed M1661 runner SHA" >&2; exit 3; }
[[ -n "${M1661_EXPECTED_DC_RELEASE_SHA256:-}" &&
   "$(sha_file "${RELEASE}")" == "${M1661_EXPECTED_DC_RELEASE_SHA256}" ]] || {
  echo "ERROR: caller must pin reviewed M1663 release SHA" >&2; exit 3; }

[[ ! -e "${RESULT}" && ! -L "${RESULT}" &&
   ! -e "${ATTEMPT}" && ! -L "${ATTEMPT}" &&
   ! -e "${WORK}" && ! -L "${WORK}" &&
   ! -e "${LOCK}" && ! -L "${LOCK}" ]] || {
  echo "ERROR: M1661 result/attempt/work/lock identity is not fresh" >&2; exit 4; }
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}
ancestry=set(); pid=os.getpid()
while pid>1 and pid not in ancestry:
    ancestry.add(pid)
    try: pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
    except Exception: break
hits=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) in ancestry: continue
    try:
        if p.stat().st_uid != os.getuid(): continue
        comm=(p/'comm').read_text().strip()
        argv={Path(x.decode(errors='replace')).name for x in
              (p/'cmdline').read_bytes().split(b'\0') if x}
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if comm in blocked or blocked & argv: hits.append((p.name,comm,sorted(argv)))
if hits: raise SystemExit('same-UID DC collision: %r' % hits)
PY
mkdir -- "${LOCK}" || exit 4
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
headroom=$((commit_limit-committed))
[[ "${mem_available}" -ge 100663296 && "${swap_free}" -ge 16777216 &&
    "${headroom}" -ge 50331648 ]] || {
  echo "ERROR: M1661 memory/commit gate not met" >&2; exit 4; }
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null

mkdir -- "${ATTEMPT}"
printf 'status=M1661_ATTEMPT_CONSUMED\ndc_shell_runs=3\naxes=k1,k8,k1x8\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
mkdir -- "${WORK}"
WORK_ACTIVE=1
cp -- "${FILELIST}" "${WORK}/input_filelist.f"
printf 'status=M1661_THREE_AXIS_DC_ATTEMPT_ADMITTED\nclock_period_ns=3.000\naxes=k1,k8,k1x8\nfresh_all_axes=true\nold_netlist_reuse=false\nhold_diagnostic_only=true\ncommit_headroom_gate_kib=50331648\nmem_available_gate_kib=100663296\nswap_free_gate_kib=16777216\nretry=false\n' \
  >"${WORK}/admission.txt"

axis_names=(k1 k8 k1x8)
axis_modes=(0 1 2)
for index in 0 1 2; do
  axis="${axis_names[$index]}"; mode="${axis_modes[$index]}"
  axis_dir="${WORK}/${axis}"
  mkdir -- "${axis_dir}"
  printf 'axis=%s\narch_mode=%s\nsource_filelist_sha256=%s\nm1609_sha256=%s\n' \
    "${axis}" "${mode}" "$(sha_file "${FILELIST}")" \
    "$(sha_file "${HW_ROOT}/rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv")" \
    >"${axis_dir}/input_identity.txt"
  set +e
  env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
    SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
    DESIGN_NAME="${DESIGN}" HW_ROOT="${HW_ROOT}" RTL_FILELIST="${FILELIST}" \
    LIB_DB="${SLOW_DB}" MIN_LIB_DB="${FAST_DB}" SDC_FILE="${SDC}" \
    OUTPUT_DIR="${axis_dir}" OPERATING_CONDITION=ssg0p9v125c \
    CLOCK_PERIOD_NS=3.000 ELAB_PARAMETERS="ARCH_MODE=${mode}" \
    "${DC_SHELL}" -f "${TCL}" >"${axis_dir}/dc.log" 2>&1
  dc_rc=$?
  set -e
  printf '%s\n' "${dc_rc}" >"${axis_dir}/dc.rc"
  [[ "${dc_rc}" -eq 0 ]] || exit "${dc_rc}"

  # HOME is deliberately not manufactured; admit only the known fixed GUI
  # bootstrap diagnostic and reject every other Error/Fatal/link/loop event.
  mapfile -t error_lines < <(rg -n 'Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+' \
      "${axis_dir}/dc.log" || true)
  [[ ${#error_lines[@]} -eq 1 && "${error_lines[0]}" == *'Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl' ]] || {
    echo "ERROR: unexpected DC error/fatal population on ${axis}" >&2; exit 9; }

  required=(
    input_identity.txt dc.log dc.rc TCL_PASS_TERMINAL.txt
    reports/flow_contract.rpt reports/precompile_loop_gate.rpt
    reports/check_design_precompile.rpt reports/check_timing_precompile.rpt
    reports/resources_precompile.rpt reports/references_precompile.rpt
    reports/compile_receipt.rpt reports/hierarchy_postcompile.rpt
    reports/resources_postcompile.rpt reports/references_postcompile.rpt
    reports/qor.rpt reports/area.rpt reports/clocks.rpt reports/ports.rpt
    reports/port_count.txt reports/timing_setup.rpt
    reports/timing_hold_diagnostic.rpt reports/constraint_setup.rpt
    reports/constraint_hold_diagnostic.rpt
    reports/constraint_max_capacitance.rpt
    reports/constraint_max_transition.rpt reports/constraint_max_fanout.rpt
    reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt
    "netlist/${DESIGN}_mapped.v" "netlist/${DESIGN}_mapped.sdc"
    "netlist/${DESIGN}.ddc" "netlist/${DESIGN}.svf"
  )
  for artifact in "${required[@]}"; do
    [[ -s "${axis_dir}/${artifact}" && ! -L "${axis_dir}/${artifact}" ]] || {
      echo "ERROR: missing/nonregular ${axis}/${artifact}" >&2; exit 6; }
  done
  grep -Fxq 'TIM-209=0' "${axis_dir}/reports/precompile_loop_gate.rpt" || exit 9
  grep -Fxq 'OPT-150=0' "${axis_dir}/reports/precompile_loop_gate.rpt" || exit 9
  grep -Fxq 'compile_ultra_count=1' "${axis_dir}/reports/compile_receipt.rpt" || exit 9
  grep -Fxq 'incremental_compile_count=0' "${axis_dir}/reports/compile_receipt.rpt" || exit 9
  grep -Fxq 'hold_optimization_count=0' "${axis_dir}/reports/compile_receipt.rpt" || exit 9
  grep -Fq 'slack (MET)' "${axis_dir}/reports/timing_setup.rpt" || exit 9
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_setup.rpt" || exit 9
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_max_capacitance.rpt" || exit 9
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_max_transition.rpt" || exit 9
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_max_fanout.rpt" || exit 9
done

/usr/libexec/platform-python3.6 -I - "${WORK}" "${RUNNER}" "${CONTRACT}" \
  "${RELEASE}" "${FILELIST}" "${M1627}/review.json" "${M903}/review.json" <<'PY'
from __future__ import print_function
import hashlib,json,math,re,sys
from pathlib import Path
root,runner,contract,release,filelist,m1627,m903=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
design='m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24'
axes={}
for name,mode in [('k1',0),('k8',1),('k1x8',2)]:
    d=root/name
    area_text=(d/'reports/area.rpt').read_text(errors='replace')
    match=re.search(r'Total cell area:\s*([0-9.]+)',area_text)
    if not match: raise SystemExit('missing area '+name)
    area=float(match.group(1))
    if not math.isfinite(area) or area<=0: raise SystemExit('invalid area '+name)
    setup=(d/'reports/timing_setup.rpt').read_text(errors='replace')
    slacks=[float(x) for x in re.findall(r'\bslack \(MET\)\s+([0-9.+-]+)',setup)]
    if not slacks or min(slacks)<0: raise SystemExit('setup not met '+name)
    qor=(d/'reports/qor.rpt').read_text(errors='replace')
    drc=re.search(r'Nets With Violations:\s*([0-9.]+)',qor)
    if not drc or int(float(drc.group(1)))!=0: raise SystemExit('DRC violation '+name)
    net=d/'netlist'
    artifacts={
      'mapped_verilog_sha256':sha(net/(design+'_mapped.v')),
      'mapped_sdc_sha256':sha(net/(design+'_mapped.sdc')),
      'ddc_sha256':sha(net/(design+'.ddc')),
      'svf_sha256':sha(net/(design+'.svf')),
    }
    axes[name]={'arch_mode':mode,'area_um2':area,'minimum_reported_setup_slack_ns':min(slacks),
                'setup_met':True,'hold_closed':False,'hold_report_present':True,
                'design_rule_violating_nets':0,'fresh_mapped_artifacts':artifacts}
receipt={
 'schema':'m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_receipt_r1_v1',
 'status':'PASS_RAW_M1661_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RESULT_HAMMER',
 'axes':axes,'axis_order':['k1','k8','k1x8'],'fresh_all_axes':True,
 'old_m872_netlist_reuse':False,'clock_period_ns':3.0,'setup_uncertainty_ns':0.2,
 'hold_uncertainty_ns':0.05,'ideal_clock':True,'wireload':'ZeroWireload',
 'logic_only_pre_macro':True,'macro_count':0,'hold_diagnostic_only':True,
 'identity':{'runner_sha256':sha(runner),'contract_sha256':sha(contract),
             'release_sha256':sha(release),'filelist_sha256':sha(filelist),
             'm1627_review_sha256':sha(m1627),'m903_review_sha256':sha(m903)},
 'execution':{'dc_shell_runs':3,'compile_ultra_per_axis':1,'automatic_retry':False,
              'vcs_runs':0,'pt_runs':0,'ptpx_runs':0,'formality_runs':0},
 'resource_gate':{'commit_headroom_min_kib':50331648,
                  'mem_available_min_kib':100663296,
                  'swap_free_min_kib':16777216,
                  'same_uid_dc_collision_tolerance':0},
 'claim_boundary':{'fresh_m1609_three_axis_setup_area':True,'hold_closed':False,
                   'power':False,'energy':False,'formality':False,'paper_ppa_ready':False,
                   'system_speedup':False,'paper_headline':False}}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
PY
printf 'PASS_M1661_M1609_C2_REGISTERED_FAULT_THREE_AXIS_LOGIC_ONLY_DC\n' \
  >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
[[ ! -e "${RESULT}" && ! -L "${RESULT}" ]] || exit 8
mv -T -n -- "${WORK}" "${RESULT}"
[[ -d "${RESULT}" && ! -L "${RESULT}" && ! -e "${WORK}" ]] || exit 8
WORK_ACTIVE=0
trap - EXIT INT TERM HUP
rmdir -- "${LOCK}"
printf 'M1661 executable-preflight resource-gate successor fresh M1609 three-axis DC result published; independent result hammer required\n'
