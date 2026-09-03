#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash
# M1900 additive C2 hold-repair runner. It requires a separately reviewed
# source, launch release, and release audit before one two-axis DC campaign.
set -euo pipefail
umask 002
[[ $# -eq 4 ]] || { echo 'ERROR: expected runner_sha review_sha release_sha audit_sha' >&2; exit 2; }
RUNNER_SHA=$1; REVIEW_SHA=$2; RELEASE_SHA=$3; AUDIT_SHA=$4
for digest in "$@"; do [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || exit 2; done

HW=/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
RUNNER=${HW}/dc_handoff/scripts/run_m1900_m1896_c2_fastmin_hold_cleanenv_release_two_axis_one_shot.sh
TCL=${HW}/dc_handoff/scripts/run_dc_m1892_m1811_c2_fastmin_hold_repair_candidate.tcl
DOC359=${HW}/docs/359_DATE终局冻结_20260813.md
M1811=${HW}/dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902
M1830=${HW}/reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902
M1893R2=${HW}/reviews/m1893r2_m1893_m1892_c2_fastmin_hold_source_identity_correction_r1_20260902
M1897=${HW}/reviews/m1897_m1896_c2_fastmin_hold_two_axis_runner_hammer_r1_20260902
M1901=${HW}/reviews/m1901_m1900_c2_fastmin_hold_cleanenv_runner_hammer_r1_20260902
M1902=${HW}/contracts/m1902_m1901_m1900_c2_fastmin_hold_two_axis_launch_release_r1_20260902.json
M1903=${HW}/reviews/m1903_m1902_c2_fastmin_hold_launch_release_audit_r1_20260902
DESIGN=m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24
K8_DDC=${M1811}/k8/netlist/${DESIGN}.ddc
K8_SDC=${M1811}/k8/netlist/${DESIGN}_mapped.sdc
K1X8_DDC=${M1811}/k1x8/netlist/${DESIGN}.ddc
K1X8_SDC=${M1811}/k1x8/netlist/${DESIGN}_mapped.sdc
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
ATTEMPT=${HW}/dc_handoff/runs/.m1900_c2_fastmin_hold_two_axis_attempt_consumed
RESULT=${HW}/dc_handoff/runs/m1900_c2_fastmin_hold_two_axis_r1_20260902
FAILURE=${HW}/dc_handoff/runs/m1900_c2_fastmin_hold_two_axis_r1_20260902.failed_or_incomplete.quarantine
WORK=${HW}/dc_handoff/runs/.m1900_c2_fastmin_hold_two_axis_work.$$
LOCK=${HW}/dc_handoff/runs/.m1900_c2_fastmin_hold_two_axis_launch_lock
ACTIVE=0

shaf(){ "${SHA}" -- "$1" | "${AWK}" '{print $1}'; }
exact(){ [[ -f $2 && ! -L $2 && "$(shaf "$2")" == "$1" ]] || exit 3; }
vseal(){ [[ -d $1 && ! -L $1 ]] && (cd "$1" && "${SHA}" -c SHA256SUMS >/dev/null && "${SHA}" -c SHA256SUMS.seal.sha256 >/dev/null); }
seal(){ (cd "$1" && "${FIND}" -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -printf '%P\0' | LC_ALL=C "${SORT}" -z | "${XARGS}" -0 -r "${SHA}" -- >SHA256SUMS && "${SHA}" SHA256SUMS >SHA256SUMS.seal.sha256 && "${SHA}" -c SHA256SUMS >/dev/null && "${SHA}" -c SHA256SUMS.seal.sha256 >/dev/null); }
publish(){ [[ -d $1 && ! -e $2 ]] && "${MV}" -T -n -- "$1" "$2" && [[ ! -e $1 && -d $2 && ! -L $2 ]] && vseal "$2"; }
finish(){
  rc=$?; trap - EXIT INT TERM HUP; set +e
  if [[ ${rc} -ne 0 && ${ACTIVE} -eq 1 && -d ${WORK} && ! -L ${WORK} ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    set -e; seal "${WORK}"; publish "${WORK}" "${FAILURE}"; set +e
  fi
  "${RMDIR}" "${LOCK}" 2>/dev/null
  exit "${rc}"
}
trap finish EXIT INT TERM HUP

exact "${RUNNER_SHA}" "${RUNNER}"
exact b01b22661dbd3789984aa78eb86f6b996f41a398e749a8e874e917b070e9885f "${TCL}"
exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"
exact c2f2b7b538cccb39efb76dc3f524efd1777327a6732a7bd498d58cd208e43ad7 "${K8_DDC}"
exact af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018 "${K8_SDC}"
exact 7c73ef9ed0a2c224a006023fc46b136c7c15783b5df6bd085805130d57c2dfda "${K1X8_DDC}"
exact 1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a "${K1X8_SDC}"
exact 695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066 "${M1811}/SHA256SUMS"; vseal "${M1811}"
exact 79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b "${M1830}/review.json"; vseal "${M1830}"
exact d3b71bf7f287adb51c545233c3b85da1ab7be8bfa07c897a8527ac86ab00915c "${M1893R2}/review.json"; vseal "${M1893R2}"
exact d4fb097adcbc361634d94ec72a834981e941c1442e98de168d464feafb01db8b "${M1897}/review.json"; vseal "${M1897}"
exact "${REVIEW_SHA}" "${M1901}/review.json"; vseal "${M1901}"
exact "${RELEASE_SHA}" "${M1902}"
exact "${AUDIT_SHA}" "${M1903}/review.json"; vseal "${M1903}"

"${PY}" -I - "${M1901}/review.json" "${M1902}" "${M1903}/review.json" "${RUNNER_SHA}" "${REVIEW_SHA}" "${RELEASE_SHA}" <<'PY'
import json,sys
from pathlib import Path
r,l,a=(json.loads(Path(x).read_text()) for x in sys.argv[1:4])
rs,vs,ls=sys.argv[4:7]
assert r['status']=='PASS_M1901_M1900_C2_FASTMIN_HOLD_CLEANENV_RUNNER_HAMMER__AUTHORIZE_RELEASE_ONLY'
assert [r['p0_count'],r['p1_count'],r['p2_count']]==[0,0,0]
assert r['identity']['runner_sha256']==rs
assert l['schema']=='m1902_m1901_m1900_c2_fastmin_hold_two_axis_launch_release_r1_v1'
assert l['status']=='AUTHORIZE_ONE_M1900_C2_FASTMIN_HOLD_TWO_AXIS_ATTEMPT'
assert l['identity']=={'runner_sha256':rs,'runner_review_sha256':vs,'m1892_tcl_sha256':'b01b22661dbd3789984aa78eb86f6b996f41a398e749a8e874e917b070e9885f','m1893r2_review_sha256':'d3b71bf7f287adb51c545233c3b85da1ab7be8bfa07c897a8527ac86ab00915c','m1897_failure_review_sha256':'d4fb097adcbc361634d94ec72a834981e941c1442e98de168d464feafb01db8b'}
assert l['budget']=={'license_queries':1,'dc_shell_runs':2,'automatic_retry':False}
assert l['axes']==['k8','k1x8']
assert l['gates']=={'dc_setup_met':True,'dc_hold_met':True,'drc_violations_max':0,'area_ceiling_percent':5.0,'formality_required_after_dc':True,'pt_required_after_dc':True}
assert a['status']=='PASS_M1903_M1902_C2_FASTMIN_HOLD_LAUNCH_RELEASE_AUDIT__AUTHORIZE_ONE_ATTEMPT'
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

"${MKDIR}" "${LOCK}"; "${MKDIR}" "${WORK}"; ACTIVE=1
"${MKDIR}" "${WORK}/attempt_stage"
printf 'status=M1900_ATTEMPT_CONSUMED\nlicense_queries=1\ndc_shell_runs=2\naxes=k8,k1x8\nretry=false\n' >"${WORK}/attempt_stage/ATTEMPT_CONSUMED.txt"
seal "${WORK}/attempt_stage"; publish "${WORK}/attempt_stage" "${ATTEMPT}"
"${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SNPSLMD_LICENSE_FILE="${LIC}" LM_LICENSE_FILE="${LICFILE}" "${LM}" lmstat -c "${LIC}" -f Design-Compiler >"${WORK}/license_preflight.log" 2>&1

names=(k8 k1x8)
designs=(${DESIGN}_ARCH_MODE0 ${DESIGN}_ARCH_MODE1)
ddcs=("${K8_DDC}" "${K1X8_DDC}"); sdcs=("${K8_SDC}" "${K1X8_SDC}")
areas=(130822.775176 585534.971643); ceilings=(137363.9139348 614811.72022515)
for i in 0 1; do
  n=${names[$i]}; d=${WORK}/${n}; "${MKDIR}" "${d}"
  "${ENV}" -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp SNPSLMD_LICENSE_FILE="${LIC}" LM_LICENSE_FILE="${LICFILE}" M1892_INPUT_DDC="${ddcs[$i]}" M1892_INPUT_SDC="${sdcs[$i]}" M1892_STD_SLOW_DB="${SLOW}" M1892_STD_FAST_DB="${FAST}" M1892_OUTPUT_DIR="${d}" M1892_EXPECTED_DESIGN="${designs[$i]}" M1892_AXIS="${n}" M1892_AREA_BASELINE_UM2="${areas[$i]}" M1892_AREA_CEILING_UM2="${ceilings[$i]}" "${DC}" -f "${TCL}" >"${d}/dc.log" 2>&1
  printf '0\n' >"${d}/dc.rc"
  for f in TCL_INTERNAL_COMPLETE.txt reports/setup_posthold_summary_machine.txt reports/hold_posthold_summary_machine.txt reports/constraint_design_rules_posthold.rpt reports/area_posthold.rpt "netlist/${designs[$i]}_m1892_fastmin_hold_repaired_mapped.v" "netlist/${designs[$i]}_m1892_fastmin_hold_repaired_mapped.sdc" "netlist/${designs[$i]}_m1892_fastmin_hold_repaired.ddc" "netlist/${designs[$i]}_m1892_fastmin_hold_repaired.svf"; do [[ -s ${d}/${f} && ! -L ${d}/${f} ]] || exit 6; done
  "${GREP}" -Fxq status=MET "${d}/reports/setup_posthold_summary_machine.txt"; "${GREP}" -Fxq violating_paths=0 "${d}/reports/setup_posthold_summary_machine.txt"
  "${GREP}" -Fxq status=MET "${d}/reports/hold_posthold_summary_machine.txt"; "${GREP}" -Fxq violating_paths=0 "${d}/reports/hold_posthold_summary_machine.txt"
  met=$("${GREP}" -Fc 'This design has no violated constraints.' "${d}/reports/constraint_design_rules_posthold.rpt")
  vio=$("${GREP}" -Ec '\(VIOLATED\)' "${d}/reports/constraint_design_rules_posthold.rpt" || true)
  [[ ${met} -eq 5 && ${vio} -eq 0 ]] || exit 6
  printf 'status=MET\nno_violated_constraint_sections=%s\nviolated_rows=%s\n' "${met}" "${vio}" >"${d}/reports/drc_posthold_summary_machine.txt"
done
printf '%s\n' schema=m1900_c2_fastmin_hold_two_axis_receipt_r1_v1 status=RAW_PASS_AWAIT_RESULT_HAMMER_TRANSITIVE_FORMALITY_PT axes=k8,k1x8 license_queries=1 dc_shell_runs=2 retry=false clock_period_ns=3.000 setup_uncertainty_ns=0.200 reported_hold_uncertainty_ns=0.050 optimization_hold_uncertainty_ns=0.070 functional_rtl_modified=false logic_only=true formality=false prime_time=false power=false paper_ppa_ready=false system_speedup=false >"${WORK}/receipt.txt"
printf 'RAW_PASS_M1900_C2_HOLD_REPAIR__AWAIT_RESULT_HAMMER_FORMALITY_PT\n' >"${WORK}/RUN_COMPLETE.txt"
seal "${WORK}"; publish "${WORK}" "${RESULT}"; ACTIVE=0
trap - EXIT INT TERM HUP; "${RMDIR}" "${LOCK}"
echo 'M1900 raw C2 hold repair published; result hammer and Formality/PT required'

