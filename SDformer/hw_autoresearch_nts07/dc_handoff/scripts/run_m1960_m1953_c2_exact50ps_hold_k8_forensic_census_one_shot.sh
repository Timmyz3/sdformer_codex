#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash
# M1960 additive C2 exact-50-ps K8 pilot.  It persists per-invocation forensic
# census and independently reports absolute and relative post-hold area.
set -euo pipefail
umask 002
[[ $# -eq 4 ]] || { echo 'ERROR: expected runner_sha review_sha release_sha audit_sha' >&2; exit 2; }
RUNNER_SHA=$1; REVIEW_SHA=$2; RELEASE_SHA=$3; AUDIT_SHA=$4
for digest in "$@"; do [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || exit 2; done

HW=/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
RUNNER=${HW}/dc_handoff/scripts/run_m1960_m1953_c2_exact50ps_hold_k8_forensic_census_one_shot.sh
TCL=${HW}/dc_handoff/scripts/run_dc_m1939_m1918_c2_exact50ps_hold_repair_candidate.tcl
DOC359=${HW}/docs/359_DATE终局冻结_20260813.md
M1811=${HW}/dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902
M1830=${HW}/reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902
M1938=${HW}/reviews/m1938_m1918_c2_fastmin_hold_failure_readonly_review_r1_20260902
M1940=${HW}/reviews/m1940_m1939_c2_exact50ps_hold_source_hammer_r1_20260902
M1944FAIL=${HW}/reviews/m1944_m1943_c2_exact50ps_hold_k8_pilot_runner_hammer_r1_20260902
M1953FAIL=${HW}/reviews/m1953_m1952_c2_exact50ps_hold_k8_area_checked_runner_hammer_r1_20260902
M1961=${HW}/reviews/m1961_m1960_c2_exact50ps_hold_k8_forensic_census_runner_hammer_r1_20260902
M1962=${HW}/contracts/m1962_m1961_m1960_c2_exact50ps_hold_k8_forensic_census_launch_release_r1_20260902.json
M1963=${HW}/reviews/m1963_m1962_c2_exact50ps_hold_k8_forensic_census_launch_release_audit_r1_20260902
DESIGN=m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24
K8_DDC=${M1811}/k8/netlist/${DESIGN}.ddc
K8_SDC=${M1811}/k8/netlist/${DESIGN}_mapped.sdc
DC=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
LM=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICFILE=/opt/synopsys/Synopsys.dat
LIC=27030@ic.ismd-nemo
SLOW=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
FAST=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
SHA=/usr/bin/sha256sum; AWK=/usr/bin/awk; FIND=/usr/bin/find
SORT=/usr/bin/sort; XARGS=/usr/bin/xargs; GREP=/usr/bin/grep
MKDIR=/usr/bin/mkdir; MV=/usr/bin/mv; RMDIR=/usr/bin/rmdir
ENV=/usr/bin/env; PY=/usr/bin/python3
ATTEMPT=${HW}/dc_handoff/runs/.m1960_c2_exact50ps_hold_k8_forensic_census_attempt_consumed
RESULT=${HW}/dc_handoff/runs/m1960_c2_exact50ps_hold_k8_forensic_census_r1_20260902
FAILURE=${HW}/dc_handoff/runs/m1960_c2_exact50ps_hold_k8_forensic_census_r1_20260902.failed_or_incomplete.quarantine
WORK=${HW}/dc_handoff/runs/.m1960_c2_exact50ps_hold_k8_forensic_census_work.$$
LOCK=${HW}/dc_handoff/runs/.m1960_c2_exact50ps_hold_k8_forensic_census_launch_lock
ACTIVE=0
LOCK_HELD=0
LICENSE_OBSERVED=0
DC_OBSERVED=0

shaf(){ "${SHA}" -- "$1" | "${AWK}" '{print $1}'; }
exact(){ [[ -f $2 && ! -L $2 && "$(shaf "$2")" == "$1" ]] || exit 3; }
vseal(){ [[ -d $1 && ! -L $1 ]] && (cd "$1" && "${SHA}" -c SHA256SUMS >/dev/null && "${SHA}" -c SHA256SUMS.seal.sha256 >/dev/null); }
seal(){ (cd "$1" && "${FIND}" -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -printf '%P\0' | LC_ALL=C "${SORT}" -z | "${XARGS}" -0 -r "${SHA}" -- >SHA256SUMS && "${SHA}" SHA256SUMS >SHA256SUMS.seal.sha256 && "${SHA}" -c SHA256SUMS >/dev/null && "${SHA}" -c SHA256SUMS.seal.sha256 >/dev/null); }
publish(){ [[ -d $1 && ! -e $2 ]] && "${MV}" -T -n -- "$1" "$2" && [[ ! -e $1 && -d $2 && ! -L $2 ]] && vseal "$2"; }
census(){
  local tmp=${WORK}/.tool_census.$$
  printf 'license_queries_authorized=1\ndc_shell_runs_authorized=1\nlicense_queries_observed=%s\ndc_shell_runs_observed=%s\n' "${LICENSE_OBSERVED}" "${DC_OBSERVED}" >"${tmp}"
  "${MV}" -f -- "${tmp}" "${WORK}/tool_census.txt"
}
finish(){
  rc=$1; trap - EXIT INT TERM HUP; set +e
  if [[ ${rc} -ne 0 && ${ACTIVE} -eq 1 && -d ${WORK} && ! -L ${WORK} ]]; then
    census
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nlicense_queries_observed=%s\ndc_shell_runs_observed=%s\nretry=false\n' "${rc}" "${LICENSE_OBSERVED}" "${DC_OBSERVED}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    set -e; seal "${WORK}"; publish "${WORK}" "${FAILURE}"; set +e
  fi
  if [[ ${LOCK_HELD} -eq 1 ]]; then "${RMDIR}" "${LOCK}" 2>/dev/null; fi
  exit "${rc}"
}
trap 'finish $?' EXIT
trap 'finish 130' INT
trap 'finish 143' TERM
trap 'finish 129' HUP

exact "${RUNNER_SHA}" "${RUNNER}"
exact 0257e1e42f7e37588d9339ffb5b017252a7ccf399428b656b223d6706087b56e "${TCL}"
exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
exact c2f2b7b538cccb39efb76dc3f524efd1777327a6732a7bd498d58cd208e43ad7 "${K8_DDC}"
exact af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018 "${K8_SDC}"
exact 695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066 "${M1811}/SHA256SUMS"; vseal "${M1811}"
exact 79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b "${M1830}/review.json"; vseal "${M1830}"
exact 25d4fc6e68b66e024bb2bb6321f044aecaf80720b37c69d99e548737417281c5 "${M1938}/review.json"; vseal "${M1938}"
exact 9b54201789bc04cad06b3250f5aa31b57db37dbad72373c298deb7feb10ea117 "${M1940}/review.json"; vseal "${M1940}"
exact c36c158ae3f832778a714d55c7867c9a76f8ea788a5dbb859ed01379963dd3d2 "${M1944FAIL}/review.json"; vseal "${M1944FAIL}"
exact 2c618cd7915d4364077eb426da3c3e2392b81098ab694ed338a2450a2f8e8fe0 "${M1953FAIL}/review.json"; vseal "${M1953FAIL}"
exact "${REVIEW_SHA}" "${M1961}/review.json"; vseal "${M1961}"
exact "${RELEASE_SHA}" "${M1962}"
exact "${AUDIT_SHA}" "${M1963}/review.json"; vseal "${M1963}"

"${PY}" -I - "${M1961}/review.json" "${M1962}" "${M1963}/review.json" "${RUNNER_SHA}" "${REVIEW_SHA}" "${RELEASE_SHA}" <<'PY'
import json,sys
from pathlib import Path
r,l,a=(json.loads(Path(x).read_text()) for x in sys.argv[1:4])
rs,vs,ls=sys.argv[4:7]
assert r['schema']=='m1961_m1960_c2_exact50ps_hold_k8_forensic_census_runner_hammer_review_r1_v1'
assert r['milestone']=='M1961'
assert r['reviewer_identity']=='/root/m1961_c2_exact50_census_runner_review'
assert r['status']=='PASS_M1961_M1960_C2_EXACT50PS_HOLD_K8_FORENSIC_CENSUS_RUNNER_HAMMER__AUTHORIZE_RELEASE_ONLY'
assert [r['p0_count'],r['p1_count'],r['p2_count']]==[0,0,0]
assert r['identity']['runner_sha256']==rs
assert l['schema']=='m1962_m1961_m1960_c2_exact50ps_hold_k8_forensic_census_launch_release_r1_v1'
assert l['status']=='AUTHORIZE_ONE_M1960_C2_EXACT50PS_HOLD_K8_FORENSIC_CENSUS_ATTEMPT'
assert l['identity']=={'runner_sha256':rs,'runner_review_sha256':vs,'m1939_tcl_sha256':'0257e1e42f7e37588d9339ffb5b017252a7ccf399428b656b223d6706087b56e','m1938_failure_review_sha256':'25d4fc6e68b66e024bb2bb6321f044aecaf80720b37c69d99e548737417281c5','m1940_source_review_sha256':'9b54201789bc04cad06b3250f5aa31b57db37dbad72373c298deb7feb10ea117','m1944_failure_review_sha256':'c36c158ae3f832778a714d55c7867c9a76f8ea788a5dbb859ed01379963dd3d2','m1953_failure_review_sha256':'2c618cd7915d4364077eb426da3c3e2392b81098ab694ed338a2450a2f8e8fe0'}
assert l['budget']=={'license_queries':1,'dc_shell_runs':1,'automatic_retry':False}
assert l['axes']==['k8']
assert l['gates']=={'dc_setup_met':True,'dc_hold_met':True,'drc_violations_max':0,'area_ceiling_percent':5.0,'formality_required_after_dc':True,'pt_required_after_dc':True}
assert a['schema']=='m1963_m1962_c2_exact50ps_hold_k8_forensic_census_launch_release_audit_review_r1_v1'
assert a['milestone']=='M1963'
assert a['reviewer_identity']=='/root/m1963_c2_exact50_census_release_audit'
assert a['status']=='PASS_M1963_M1962_C2_EXACT50PS_HOLD_K8_FORENSIC_CENSUS_LAUNCH_RELEASE_AUDIT__AUTHORIZE_ONE_ATTEMPT'
assert [a['p0_count'],a['p1_count'],a['p2_count']]==[0,0,0]
assert a['identity']=={'runner_sha256':rs,'runner_review_sha256':vs,'release_sha256':ls}
PY

[[ ! -e ${ATTEMPT} && ! -e ${RESULT} && ! -e ${FAILURE} && ! -e ${WORK} && ! -e ${LOCK} ]] || exit 4
blocked=' dc_shell dc_shell-t pt_shell fm_shell icc2_shell common_shell_ex common_shell_exec common_shell_exe '
for proc in /proc/[0-9]*; do
  [[ -r ${proc}/status && -r ${proc}/comm ]] || continue
  uid=''; while IFS=$'\t' read -r k v z; do [[ ${k} == Uid: ]] && { uid=${v}; break; }; done <"${proc}/status"
  [[ ${uid} == "${EUID}" ]] || continue
  comm=''; IFS= read -r comm <"${proc}/comm" || continue
  [[ " ${blocked} " != *" ${comm} "* ]] || exit 4
done
ma=0; cl=0; ca=0
while IFS=' :' read -r k v u; do case ${k} in MemAvailable) ma=${v};; CommitLimit) cl=${v};; Committed_AS) ca=${v};; esac; done </proc/meminfo
[[ ${ma} -ge 67108864 && $((cl-ca)) -ge 33554432 ]] || exit 4

"${MKDIR}" "${LOCK}"; LOCK_HELD=1; ACTIVE=1; "${MKDIR}" "${WORK}"
census
"${MKDIR}" "${WORK}/attempt_stage"
printf 'status=M1960_ATTEMPT_CONSUMED\nlicense_queries_authorized=1\ndc_shell_runs_authorized=1\naxes=k8\nretry=false\n' >"${WORK}/attempt_stage/ATTEMPT_CONSUMED.txt"
seal "${WORK}/attempt_stage"; publish "${WORK}/attempt_stage" "${ATTEMPT}"
LICENSE_OBSERVED=1; census
"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SNPSLMD_LICENSE_FILE="${LIC}" LM_LICENSE_FILE="${LICFILE}" "${LM}" lmstat -c "${LIC}" -f Design-Compiler >"${WORK}/license_preflight.log" 2>&1

n=k8; design=${DESIGN}_ARCH_MODE0; d=${WORK}/${n}; "${MKDIR}" "${d}"
DC_OBSERVED=1; census
"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
  SNPSLMD_LICENSE_FILE="${LIC}" LM_LICENSE_FILE="${LICFILE}" \
  M1939_INPUT_DDC="${K8_DDC}" M1939_INPUT_SDC="${K8_SDC}" \
  M1939_STD_SLOW_DB="${SLOW}" M1939_STD_FAST_DB="${FAST}" \
  M1939_OUTPUT_DIR="${d}" M1939_EXPECTED_DESIGN="${design}" M1939_AXIS="${n}" \
  M1939_AREA_BASELINE_UM2=130822.775176 M1939_AREA_CEILING_UM2=137363.9139348 \
  "${DC}" -f "${TCL}" >"${d}/dc.log" 2>&1
printf '0\n' >"${d}/dc.rc"
for f in TCL_INTERNAL_COMPLETE.txt reports/setup_posthold_summary_machine.txt reports/hold_posthold_summary_machine.txt reports/constraint_design_rules_posthold.rpt reports/area_posthold.rpt "netlist/${design}_m1939_fastmin_hold_repaired_mapped.v" "netlist/${design}_m1939_fastmin_hold_repaired_mapped.sdc" "netlist/${design}_m1939_fastmin_hold_repaired.ddc" "netlist/${design}_m1939_fastmin_hold_repaired.svf"; do [[ -s ${d}/${f} && ! -L ${d}/${f} ]] || exit 6; done
"${GREP}" -Fxq status=M1939_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED "${d}/TCL_INTERNAL_COMPLETE.txt"
"${GREP}" -Fxq status=MET "${d}/reports/setup_posthold_summary_machine.txt"; "${GREP}" -Fxq violating_paths=0 "${d}/reports/setup_posthold_summary_machine.txt"
"${GREP}" -Fxq status=MET "${d}/reports/hold_posthold_summary_machine.txt"; "${GREP}" -Fxq violating_paths=0 "${d}/reports/hold_posthold_summary_machine.txt"
met=$("${GREP}" -Fc 'This design has no violated constraints.' "${d}/reports/constraint_design_rules_posthold.rpt")
vio=$("${GREP}" -Ec '\(VIOLATED\)' "${d}/reports/constraint_design_rules_posthold.rpt" || true)
[[ ${met} -eq 5 && ${vio} -eq 0 ]] || exit 6
printf 'status=MET\nno_violated_constraint_sections=%s\nviolated_rows=%s\n' "${met}" "${vio}" >"${d}/reports/drc_posthold_summary_machine.txt"
area_rows=$("${GREP}" -Ec '^[[:space:]]*Total cell area:[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$' "${d}/reports/area_posthold.rpt" || true)
[[ ${area_rows} -eq 1 ]] || exit 6
post_area=$("${AWK}" '$1=="Total" && $2=="cell" && $3=="area:" {print $4}' "${d}/reports/area_posthold.rpt")
read -r area_ratio area_growth_percent < <("${PY}" -I - "${post_area}" <<'PY'
import math,sys
value=float(sys.argv[1])
baseline=130822.775176
assert math.isfinite(value) and value > 0.0 and value <= 137363.9139348
print('{:.12f} {:.9f}'.format(value/baseline,(value/baseline-1.0)*100.0))
PY
)
printf 'status=MET\nreport_area_unique_rows=%s\npre_total_cell_area_um2=130822.775176\npost_total_cell_area_um2=%s\narea_ratio=%s\narea_growth_percent=%s\narea_ceiling_um2=137363.9139348\n' "${area_rows}" "${post_area}" "${area_ratio}" "${area_growth_percent}" >"${d}/reports/area_posthold_summary_machine.txt"
printf '%s\n' schema=m1960_c2_exact50ps_hold_k8_forensic_census_receipt_r1_v1 status=RAW_PASS_AWAIT_RESULT_HAMMER_TRANSITIVE_FORMALITY_PT axes=k8 license_queries_authorized=1 "license_queries_observed=${LICENSE_OBSERVED}" dc_shell_runs_authorized=1 "dc_shell_runs_observed=${DC_OBSERVED}" retry=false clock_period_ns=3.000 setup_uncertainty_ns=0.200 reported_hold_uncertainty_ns=0.050 optimization_hold_uncertainty_ns=0.050 pre_total_cell_area_um2=130822.775176 "post_total_cell_area_um2=${post_area}" "area_ratio=${area_ratio}" "area_growth_percent=${area_growth_percent}" area_ceiling_percent=5.0 post_area_independently_parsed=true functional_rtl_modified=false logic_only=true formality=false prime_time=false power=false paper_ppa_ready=false system_speedup=false >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1960_C2_EXACT50PS_HOLD_K8_FORENSIC_CENSUS__AWAIT_RESULT_HAMMER_FORMALITY_PT\n' >"${WORK}/RUN_COMPLETE.txt"
seal "${WORK}"; publish "${WORK}" "${RESULT}"; ACTIVE=0
"${RMDIR}" "${LOCK}"; LOCK_HELD=0; trap - EXIT INT TERM HUP
echo 'M1960 raw C2 exact50ps K8 forensic-census pilot published; result hammer and Formality/PT required'
