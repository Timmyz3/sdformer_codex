#!/usr/bin/env bash
set -euo pipefail

m479r2_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m479r2_hw="$(cd "${m479r2_dc_root}/.." && pwd)"
m479r2_run="${M479R2_DC_RUN:-${m479r2_dc_root}/runs/m479r2_lane_local_dc_3p000ns_r1_20260827}"
m479r2_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m479r2_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m479r2_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m479r2_core="rtl_m479_lane_local/m479_lane_local_parent_queue_pipeline.sv"
m479r2_wrapper="rtl_m479_lane_local/m479_lane_local_backpressure_safe_parent_queue_pipeline.sv"
m479r2_filelist="dc_handoff/filelists/date_m479_lane_local_parent_queue_dc.f"
m479r2_sdc="dc_handoff/constraints/date_m479_lane_local_parent_queue_3ns.sdc"
m479r2_tcl="dc_handoff/scripts/run_dc_m479_lane_local_parent_queue_exact_sha.tcl"
m479r2_contract="contracts/m479r2_lane_local_parent_queue_logic_only_dc_contract_r1_20260827.json"
m479r2_hammer="results/m479_independent_hammer_review_r1_20260826/m479_independent_hammer_review_r1.json"
m479r2_failure="dc_handoff/runs/m477_m476r2_backpressure_safe_parent_queue_dc_3p000ns_r1_20260826/m477_dc_failure_receipt_r1.json"

m479r2_sha() { sha256sum "$1" | awk '{print $1}'; }
m479r2_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 3; }
    [[ "$(m479r2_sha "${path}")" == "${expected}" ]] || {
        echo "M479r2 SHA mismatch ${path}" >&2
        exit 3
    }
}

[[ ! -e "${m479r2_run}" ]] || { echo "M479r2 output exists ${m479r2_run}" >&2; exit 5; }
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -f '/common_shell_exec -shell dc_shell ' >/dev/null; then
    echo "M479r2 refuses to collide with another Design Compiler run" >&2
    exit 4
fi

cd "${m479r2_hw}"
m479r2_expect "${m479r2_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m479r2_expect "${m479r2_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m479r2_expect "${m479r2_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m479r2_expect "${m479r2_core}" 382a6f46352d01d08b5f7885801b5b562a618e2df21180aa6444f59543aeda1c
m479r2_expect "${m479r2_wrapper}" 597b1bd08a4434bdde193ca1c818628d44d0ba6dfd4df7ca0ce1c2984ebc29b2
m479r2_expect "${m479r2_filelist}" 3330a9073323367c6baf9b3961ff046af44b480d420e20cb2814b2c75e3aac30
m479r2_expect "${m479r2_sdc}" b768d3d094b63d445fdf576be6ace5b134f861434a953862e30401098f740327
m479r2_expect "${m479r2_tcl}" ad0dedf1d7e7202ebc29a0d0d518ab890255dabfd73b34159662e600eb1129f5
m479r2_expect "${m479r2_contract}" 2fe438f0ed15d6afd0962e927b0b1381e13824251feddc0eb0698ed051d31eb0
m479r2_expect "${m479r2_hammer}" 4d44ed1295e3db2d475dbd5201f3a6ec693d0a843b1ea184e38871f114069050
m479r2_expect "${m479r2_failure}" d27b08fec5a17735b2db2aecf8289b40b2c155f14e10e7a1a7d0527cfe3339ba
m479r2_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
(cd results/m479_independent_hammer_review_r1_20260826 && \
  sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
python3 - <<'PY'
import json
from pathlib import Path
hammer = json.loads(Path('results/m479_independent_hammer_review_r1_20260826/m479_independent_hammer_review_r1.json').read_text())
failure = json.loads(Path('dc_handoff/runs/m477_m476r2_backpressure_safe_parent_queue_dc_3p000ns_r1_20260826/m477_dc_failure_receipt_r1.json').read_text())
assert hammer['status'] == 'PASS_M479_INDEPENDENT_VCS_STATIC_WITH_P1'
assert hammer['verdict'] == 'GO_SAME_CONSTRAINT_3NS_DC_DRC_DIAGNOSTIC_ONLY'
assert hammer['p0_findings'] == []
assert failure['status'] == 'FAIL_M477_M476R2_PREMACRO_DC_DESIGN_RULE_CONSTRAINTS_NOT_CLEAN'
PY

mkdir -p "${m479r2_run}"
m479r2_complete=0
trap 'm479r2_rc=$?; if [[ ${m479r2_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m479r2_rc}" >"${m479r2_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
sha256sum "${m479r2_core}" "${m479r2_wrapper}" "${m479r2_filelist}" \
    "${m479r2_sdc}" "${m479r2_tcl}" "${m479r2_contract}" \
    "${m479r2_hammer}" "${m479r2_failure}" \
    docs/359_DATE终局冻结_20260813.md "${m479r2_dc}" "${m479r2_slow}" \
    "${m479r2_fast}" >"${m479r2_run}/input_sha256.txt"
cp "${m479r2_contract}" "${m479r2_run}/contract.json"

export HW_ROOT="${m479r2_hw}"
export RTL_FILELIST="${m479r2_hw}/${m479r2_filelist}"
export LIB_DB="${m479r2_slow}"
export MIN_LIB_DB="${m479r2_fast}"
export SDC_FILE="${m479r2_hw}/${m479r2_sdc}"
export OUTPUT_DIR="${m479r2_run}"
export CLOCK_PERIOD_NS=3.000
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m479r2_dc}" -f "${m479r2_hw}/${m479r2_tcl}" >"${m479r2_run}/dc.log" 2>&1
m479r2_rc=$?
set -e
echo "${m479r2_rc}" >"${m479r2_run}/dc.rc"
[[ "${m479r2_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m479r2_run}/dc.log"
grep -Fq "Current design is 'm479_lane_local_backpressure_safe_parent_queue_pipeline'." "${m479r2_run}/dc.log"
! grep -Fq "Current design is 'm476r2_backpressure_safe_parent_queue_pipeline'." "${m479r2_run}/dc.log"
grep -Fq 'Thank you...' "${m479r2_run}/dc.log"
for report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt hierarchy_postcompile.rpt \
        resources_postcompile.rpt references_postcompile.rpt; do
    [[ -s "${m479r2_run}/reports/${report}" ]] || exit 30
done
mapped="${m479r2_run}/netlist/m479_lane_local_backpressure_safe_parent_queue_pipeline_mapped.v"
[[ -s "${mapped}" ]] || exit 31
grep -Fq 'module m479_lane_local_backpressure_safe_parent_queue_pipeline' "${mapped}"
! grep -Fq 'module m476r2_backpressure_safe_parent_queue_pipeline' "${mapped}"
grep -Fq 'slack (VIOLATED)' "${m479r2_run}/reports/timing_setup.rpt" \
    "${m479r2_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m479r2_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33

m479r2_area=$(awk '/Total cell area:/ {print $4; exit}' "${m479r2_run}/reports/area.rpt")
m479r2_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m479r2_run}/reports/area.rpt")
m479r2_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m479r2_run}/reports/area.rpt")
m479r2_combo=$(awk '/Number of combinational cells:/ {print $5; exit}' "${m479r2_run}/reports/area.rpt")
m479r2_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m479r2_run}/reports/qor.rpt")
m479r2_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m479r2_run}/reports/timing_setup.rpt")
m479r2_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m479r2_run}/reports/timing_hold.rpt")
for value in "${m479r2_area}" "${m479r2_cells}" "${m479r2_seq}" \
        "${m479r2_combo}" "${m479r2_levels}" "${m479r2_setup}" "${m479r2_hold}"; do
    [[ -n "${value}" ]] || exit 34
done

python3 - "${m479r2_run}" "${m479r2_area}" "${m479r2_cells}" \
    "${m479r2_seq}" "${m479r2_combo}" "${m479r2_levels}" \
    "${m479r2_setup}" "${m479r2_hold}" <<'PY'
import hashlib, json
from pathlib import Path
import sys
run = Path(sys.argv[1])
area, cells, seq, combo, levels, setup, hold = (
    float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
    float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
m477 = json.loads(Path('dc_handoff/runs/m477_m476r2_backpressure_safe_parent_queue_dc_3p000ns_r1_20260826/m477_dc_failure_receipt_r1.json').read_text())
base = m477['measured_but_not_admissible']
receipt = {
    'schema': 'm479r2_lane_local_parent_queue_logic_only_dc_receipt_v1',
    'status': 'PASS_M479R2_LANE_LOCAL_LOGIC_ONLY_DC_3NS_CLEAN',
    'tool': 'Synopsys Design Compiler V-2023.12-SP3',
    'design_identity': 'm479_lane_local_backpressure_safe_parent_queue_pipeline',
    'technology': 'TSMC28 HPC+ standard cells',
    'operating_condition': 'ssg0p9v125c',
    'clock_period_ns': 3.0,
    'measured': {
        'cell_area_um2': area, 'cell_count': cells,
        'sequential_cells': seq, 'combinational_cells': combo,
        'logic_levels': levels, 'setup_worst_slack_ns': setup,
        'hold_worst_slack_ns': hold,
    },
    'm477_failed_diagnostic_comparison': {
        'm477_area_um2_not_admissible': base['cell_area_um2'],
        'area_ratio_vs_m477_diagnostic': area / base['cell_area_um2'],
        'area_delta_percent_vs_m477_diagnostic': (area / base['cell_area_um2'] - 1) * 100,
        'm477_design_rules_clean': False,
        'm479r2_design_rules_clean': True,
    },
    'macro_count': 0,
    'admission': {
        'm479_logic_only_dc_sta': True,
        'same_constraint_m477_drc_comparison': True,
        'formality': False, 'physical_timing': False,
        'scratch_or_psum_macro_ppa': False, 'power': False, 'energy': False,
        'performance_admitted': False, 'paper_ppa_ready': False,
        'full_network': False, 'system_speedup': False, 'date_headline': False,
    },
    'required_next_gate': 'Independent receipt-blind DC hammer, then Formality only if the hammer confirms the exact M479 identity and clean DRC.'
}
(run/'m479r2_lane_local_parent_queue_logic_only_dc_receipt_r1.json').write_text(json.dumps(receipt, indent=2)+'\n')
(run/'RUN_COMPLETE.txt').write_text(
    'PASS_M479R2_LANE_LOCAL_LOGIC_ONLY_DC_3NS_CLEAN\n'
    f'cell_area_um2={area}\nsetup_worst_slack_ns={setup}\n'
    f'hold_worst_slack_ns={hold}\ndesign_identity=m479_lane_local_backpressure_safe_parent_queue_pipeline\n'
    'macro_count=0\nperformance_admitted=false\npaper_ppa_ready=false\n')
files = [p for p in sorted(run.rglob('*')) if p.is_file() and p.name not in {'evidence_manifest.sha256','evidence_manifest.seal.sha256'}]
(run/'evidence_manifest.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n' for p in files))
(run/'evidence_manifest.seal.sha256').write_text(hashlib.sha256((run/'evidence_manifest.sha256').read_bytes()).hexdigest()+'  evidence_manifest.sha256\n')
PY
(cd "${m479r2_run}" && sha256sum -c evidence_manifest.sha256 >/dev/null && sha256sum -c evidence_manifest.seal.sha256 >/dev/null)
m479r2_complete=1
rm -f "${m479r2_run}/RUN_FAILED_OR_INCOMPLETE.txt"
