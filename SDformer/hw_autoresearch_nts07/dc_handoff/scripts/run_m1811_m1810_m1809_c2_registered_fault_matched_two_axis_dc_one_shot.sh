#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
DESIGN=m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24
FILELIST="${HW_ROOT}/dc_handoff/filelists/iscas_m1809_c2_registered_fault_matched_k8_k1x8_logic_only_dc.f"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC="${HW_ROOT}/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
REVIEW_DIR="${HW_ROOT}/reviews/m1810_m1809_c2_registered_fault_matched_two_axis_dc_source_hammer_r1_20260902"
REVIEW="${REVIEW_DIR}/review.json"
RUNNER_REVIEW_DIR="${HW_ROOT}/reviews/m1816_m1811_c2_registered_fault_matched_two_axis_dc_runner_hammer_r1_20260902"
RUNNER_REVIEW="${RUNNER_REVIEW_DIR}/review.json"
M1802="${HW_ROOT}/reviews/m1802_m1801_c2_registered_public_fault_evidence_successor_source_hammer_r1_20260902/review.json"
M1804="${HW_ROOT}/reviews/m1804_m1803_c2_registered_fault_two_vcs_result_hammer_r1_20260902/review.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
DC_SHELL=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
SLOW_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
FAST_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
RESULT="${HW_ROOT}/dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 3; }
  [[ "$(sha_file "${path}")" == "${expected}" ]] || { echo "ERROR: SHA ${path}" >&2; exit 3; }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
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
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

sha_exact 1dc9703bafb12ed35dda1dc9b7248881145d600c06129b00b34b7308eaeaf661 "${FILELIST}"
sha_exact c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe "${TCL}"
sha_exact 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5 "${SDC}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact 405ac73f401440245e3edaea7c6e23a222883c44e8ed77e732983df721664c66 "${HW_ROOT}/rtl_m1809/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24.sv"
sha_exact fcd002804f1086d90237ddb36ed2178213ef5992adde18d148f6c14ff11db18d "${HW_ROOT}/rtl_m1801/m1801_c2_registered_public_fault_export.sv"
sha_exact f77ac9f343961ea37a277c106ebe099191cf7005c35dcbd8eb98e01b1eccb59c "${HW_ROOT}/rtl_m1801/m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24.sv"

declare -A SOURCE_SHA=(
  [rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv]=7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931
  [rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv]=8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0
  [rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv]=529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267
  [rtl_m218/m218_fc2_tagged_slice_service_island.sv]=f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1
  [rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv]=44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e
  [rtl_m519/m519_fc2_k1_registered_release_service_island.sv]=3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871
  [rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv]=010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b
  [rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv]=6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815
  [rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv]=11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff
  [rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv]=cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156
  [rtl_m1801/m1801_c2_registered_public_fault_export.sv]=fcd002804f1086d90237ddb36ed2178213ef5992adde18d148f6c14ff11db18d
  [rtl_m1801/m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24.sv]=f77ac9f343961ea37a277c106ebe099191cf7005c35dcbd8eb98e01b1eccb59c
  [rtl_m1809/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24.sv]=405ac73f401440245e3edaea7c6e23a222883c44e8ed77e732983df721664c66
)
while IFS= read -r source; do
  [[ -n "${SOURCE_SHA[$source]:-}" ]] || { echo "ERROR: unexpected source ${source}" >&2; exit 3; }
  sha_exact "${SOURCE_SHA[$source]}" "${HW_ROOT}/${source}"
done <"${FILELIST}"
[[ "$(wc -l <"${FILELIST}")" -eq 13 ]] || exit 3

verify_dir_seal "${REVIEW_DIR}"
verify_dir_seal "${RUNNER_REVIEW_DIR}"
/usr/libexec/platform-python3.6 -I - "${REVIEW}" "${M1802}" "${M1804}" "${FILELIST}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
review,m1802,m1804,filelist=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads(review.read_text()); s=json.loads(m1802.read_text()); v=json.loads(m1804.read_text())
assert r['status'].startswith('PASS_M1810')
assert r['score']>=95 and r['p0_count']==0 and r['p1_count']==0 and r['p2_count']==0
assert r['authorization']['future_dc_shell_runs_max']==2
assert r['authorization']['all_other_eda_runs']==0
assert r['identity']['filelist_sha256']==sha(filelist)
assert s['status'].startswith('PASS_M1802') and s['severity_counts']['p0']==0 and s['severity_counts']['p1']==0
assert v['status'].startswith('PASS_M1804') and v['severity_counts']['p0']==0 and v['severity_counts']['p1']==0
PY
/usr/libexec/platform-python3.6 -I - "${RUNNER_REVIEW}" "${RUNNER}" "${REVIEW}" <<'PY'
from __future__ import print_function
import hashlib,json,sys
from pathlib import Path
rr,runner,source_review=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads(rr.read_text())
assert r['status'].startswith('PASS_M1816')
assert r['score']>=95 and r['p0_count']==0 and r['p1_count']==0 and r['p2_count']==0
assert r['identity']['runner_sha256']==sha(runner)
assert r['identity']['m1810_review_sha256']==sha(source_review)
assert r['authorization']=={'execution_allowed':True,'dc_shell_runs':2,'all_other_eda_runs':0}
PY
[[ -n "${M1811_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M1811_EXPECTED_RUNNER_SHA256}" ]] || { echo "ERROR: caller must pin runner SHA" >&2; exit 3; }
[[ -n "${M1811_EXPECTED_REVIEW_SHA256:-}" && "$(sha_file "${REVIEW}")" == "${M1811_EXPECTED_REVIEW_SHA256}" ]] || { echo "ERROR: caller must pin review SHA" >&2; exit 3; }

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || { echo "ERROR: identity not fresh" >&2; exit 4; }
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}
hits=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        if p.stat().st_uid!=os.getuid(): continue
        comm=(p/'comm').read_text().strip()
        argv={Path(x.decode(errors='replace')).name for x in (p/'cmdline').read_bytes().split(b'\0') if x}
    except Exception: continue
    if comm in blocked or blocked & argv: hits.append((p.name,comm))
if hits: raise SystemExit('same-UID DC collision: %r' % hits)
PY
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 67108864 && $((commit_limit-committed)) -ge 33554432 ]] || { echo "ERROR: memory gate" >&2; exit 4; }
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M1811_ATTEMPT_CONSUMED\ndc_shell_runs=2\naxes=k8,k1x8\nretry=false\n' >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
cp -- "${FILELIST}" "${WORK}/input_filelist.f"

axis_names=(k8 k1x8)
axis_modes=(0 1)
for index in 0 1; do
  axis="${axis_names[$index]}"; mode="${axis_modes[$index]}"; axis_dir="${WORK}/${axis}"
  mkdir -- "${axis_dir}"
  env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
    SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
    DESIGN_NAME="${DESIGN}" HW_ROOT="${HW_ROOT}" RTL_FILELIST="${FILELIST}" \
    LIB_DB="${SLOW_DB}" MIN_LIB_DB="${FAST_DB}" SDC_FILE="${SDC}" \
    OUTPUT_DIR="${axis_dir}" OPERATING_CONDITION=ssg0p9v125c \
    CLOCK_PERIOD_NS=3.000 ELAB_PARAMETERS="ARCH_MODE=${mode}" \
    "${DC_SHELL}" -f "${TCL}" >"${axis_dir}/dc.log" 2>&1
  printf '0\n' >"${axis_dir}/dc.rc"
  for artifact in TCL_PASS_TERMINAL.txt reports/area.rpt reports/qor.rpt reports/timing_setup.rpt reports/constraint_setup.rpt reports/precompile_loop_gate.rpt "netlist/${DESIGN}_mapped.v" "netlist/${DESIGN}_mapped.sdc" "netlist/${DESIGN}.ddc" "netlist/${DESIGN}.svf"; do
    [[ -s "${axis_dir}/${artifact}" && ! -L "${axis_dir}/${artifact}" ]] || { echo "ERROR: missing ${axis}/${artifact}" >&2; exit 6; }
  done
  grep -Fxq 'TIM-209=0' "${axis_dir}/reports/precompile_loop_gate.rpt"
  grep -Fxq 'OPT-150=0' "${axis_dir}/reports/precompile_loop_gate.rpt"
  grep -Fq 'slack (MET)' "${axis_dir}/reports/timing_setup.rpt"
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_setup.rpt"
done

/usr/libexec/platform-python3.6 -I - "${WORK}" "${RUNNER}" "${REVIEW}" "${FILELIST}" <<'PY'
from __future__ import print_function
import hashlib,json,math,re,sys
from pathlib import Path
root,runner,review,filelist=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
axes={}
design='m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24'
for name,mode in [('k8',0),('k1x8',1)]:
    d=root/name
    a=re.search(r'Total cell area:\s*([0-9.]+)',(d/'reports/area.rpt').read_text(errors='replace'))
    if not a: raise SystemExit('missing area '+name)
    area=float(a.group(1))
    slacks=[float(x) for x in re.findall(r'\bslack \(MET\)\s+([0-9.+-]+)',(d/'reports/timing_setup.rpt').read_text(errors='replace'))]
    if not math.isfinite(area) or area<=0 or not slacks or min(slacks)<0: raise SystemExit('invalid axis '+name)
    axes[name]={'arch_mode':mode,'area_um2':area,'minimum_reported_setup_slack_ns':min(slacks),'setup_met':True,'hold_closed':False,'mapped_verilog_sha256':sha(d/'netlist'/(design+'_mapped.v')),'ddc_sha256':sha(d/'netlist'/(design+'.ddc'))}
ratio=axes['k1x8']['area_um2']/axes['k8']['area_um2']
receipt={'schema':'m1811_c2_registered_fault_matched_two_axis_dc_receipt_r1_v1','status':'PASS_RAW_M1811_C2_MATCHED_TWO_AXIS_DC_PENDING_INDEPENDENT_RESULT_REVIEW','axes':axes,'clock_period_ns':3.0,'area_efficiency_k8_vs_k1x8':ratio,'k8_area_reduction_fraction_vs_k1x8':1.0-1.0/ratio,'identity':{'runner_sha256':sha(runner),'review_sha256':sha(review),'filelist_sha256':sha(filelist)},'execution':{'dc_shell_runs':2,'retry':False,'vcs_runs':0,'pt_runs':0,'ptpx_runs':0},'claim_boundary':{'logic_only_pre_macro':True,'ideal_clock':True,'wireload':'ZeroWireload','hold_closed':False,'power':False,'system_speedup':False,'paper_ppa_ready':False}}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
PY
printf 'PASS_M1811_C2_REGISTERED_FAULT_MATCHED_TWO_AXIS_DC\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -n -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
trap - EXIT INT TERM HUP
rmdir -- "${LOCK}"
echo "M1811 raw two-axis DC published; independent result review required"
