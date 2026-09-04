#!/usr/bin/env bash
set -euo pipefail

m477_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m477_hw="$(cd "${m477_dc_root}/.." && pwd)"
m477_run="${M477_DC_RUN:-${m477_dc_root}/runs/m477_m476r2_backpressure_safe_parent_queue_dc_3p000ns_r1_20260826}"
m477_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m477_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m477_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m477_filelist="dc_handoff/filelists/date_m476r2_backpressure_safe_parent_queue_dc.f"
m477_sdc="dc_handoff/constraints/date_m476r2_backpressure_safe_parent_queue_3ns.sdc"
m477_tcl="dc_handoff/scripts/run_dc_m476r2_backpressure_safe_parent_queue_exact_sha.tcl"
m477_contract="contracts/m477_m476r2_backpressure_safe_logic_only_dc_contract_r1_20260826.json"
m477_baseline="dc_handoff/runs/m475_m474_fused_parent_dual_update_dc_3p000ns_r1_20260826"

m477_sha() { sha256sum "$1" | awk '{print $1}'; }
m477_expect() {
    local m477_path=$1
    local m477_expected=$2
    [[ -f "${m477_path}" ]] || { echo "missing ${m477_path}" >&2; exit 3; }
    [[ "$(m477_sha "${m477_path}")" == "${m477_expected}" ]] || {
        echo "M477 SHA mismatch ${m477_path}" >&2
        exit 3
    }
}

[[ ! -e "${m477_run}" ]] || { echo "M477 output exists ${m477_run}" >&2; exit 5; }
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    echo "M477 refuses to collide with another Design Compiler run" >&2
    exit 4
fi
cd "${m477_hw}"
m477_expect "${m477_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m477_expect "${m477_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m477_expect "${m477_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m477_expect rtl_m476/m476_dual_slot_parent_queue_pipeline.sv c5aa9d0cceb4e353c2457afb6b554403d333720d84dec6fe1b0a982769893c55
m477_expect rtl_m476r2/m476r2_backpressure_safe_parent_queue_pipeline.sv 4620d4666b44843be17306c984006a4423f43ad97103fd2a419aa8d901ccc37c
m477_expect "${m477_filelist}" 1e9d4c20421742c1e4df4a295305d459577b1a425ffe04a6f0982ee9fc5323be
m477_expect "${m477_sdc}" 8ddd3fd9278443082581c95ec92d07188efd4fb9659cf989616a03d963e643a8
m477_expect "${m477_tcl}" 0d5c761dbe66cb3d820c5689a5d56c2d03ad1830a5f45ffcb9671df3469564d3
m477_expect "${m477_contract}" a320b990d8eea5cd307eb80cfa7afdd3b5105d5857a08ffb70c360f2bdbc0b05
m477_expect results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json 36e99e859e77f0f61e12eb238360a7828d794854bab8934ea12810020c558e5c
m477_expect results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/SHA256SUMS c9b4860c4b2a6b72ffbe0394824ad9b3efedb748a5d74f697d43e1ae31ed1cf6
m477_expect results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/SHA256SUMS.seal.sha256 25b5a356a00fadf620d2d45dade5482e664387d4569aad9ab7ea83d682776fc0
m477_expect results/m476r2_independent_hammer_review_r1_20260826/m476r2_independent_hammer_review_r1.json 1ec68d48bded366763a4bf7a3307ce153332be45ef7b61b1005e38b9a923bda1
m477_expect results/m476r2_independent_hammer_review_r1_20260826/SHA256SUMS 9f8134efd1e7079fe5d94928d60dad55896f1bb7959b5812a9d20490aba69d06
m477_expect results/m476r2_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 b7bc89490a96b103c737889764677c8313c32c07263eef3e1f654158d08896c0
m477_expect "${m477_baseline}/m475_m474_fused_parent_dual_update_logic_only_dc_receipt_r1.json" 9909c34a57efa823cdf9c4a052e27c12acfbb2b0e49b8852db6d059d4fd2ee74
m477_expect "${m477_baseline}/evidence_manifest.sha256" 84ea164aa16ffc1807c92fcc09658b723b4a409f05aa9ba99940b43471099c95
m477_expect "${m477_baseline}/evidence_manifest.seal.sha256" c8da7cc718f52bf403074ca1b40b533cee916d5cb8437b7feb3737e6ef2cc271
m477_expect results/m475_independent_hammer_review_r1_20260826/m475_independent_hammer_review_r1.json 459079052437bb2b260c087c35e64cfc75b1066a8cc156892e77d15fdf2b8dfe
m477_expect dc_handoff/constraints/date_m474_fused_parent_dual_update_3ns.sdc 09846a3645de26a89454893e89bef05b1b4b0d2cd1591ff176da5533ace6fdbe
m477_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(cd results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826 &&
  sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd results/m476r2_independent_hammer_review_r1_20260826 &&
  sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd "${m477_baseline}" &&
  sha256sum -c evidence_manifest.sha256 >/dev/null &&
  sha256sum -c evidence_manifest.seal.sha256 >/dev/null)

python3 - <<'PY'
import json
from pathlib import Path

vcs = json.loads(Path('results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json').read_text())
hammer = json.loads(Path('results/m476r2_independent_hammer_review_r1_20260826/m476r2_independent_hammer_review_r1.json').read_text())
baseline = json.loads(Path('dc_handoff/runs/m475_m474_fused_parent_dual_update_dc_3p000ns_r1_20260826/m475_m474_fused_parent_dual_update_logic_only_dc_receipt_r1.json').read_text())
baseline_review = json.loads(Path('results/m475_independent_hammer_review_r1_20260826/m475_independent_hammer_review_r1.json').read_text())
assert vcs['status'] == 'PASS_M476R2_EXACT_SHA_STALE_RAW_P0_CLOSED_MICRO_ONLY'
assert vcs['claim_boundary']['m473_performance_admitted'] is False
assert hammer['status'] == 'PASS_M476R2_INDEPENDENT_HAMMER_P0_CLOSED_WITH_P1'
assert hammer['verdict'] == 'GO_SAME_CONSTRAINT_3NS_PREMACRO_DC_COMPARE_ONLY'
assert hammer['score_out_of_100'] == 93
assert hammer['p0_findings'] == []
assert hammer['admission']['same_constraint_three_ns_dc_compare_allowed'] is True
assert hammer['admission']['m473_performance_admitted'] is False
assert baseline['status'] == 'PASS_M475_M474_FUSED_PIPELINE_LOGIC_ONLY_DC_3NS'
assert baseline['clock_period_ns'] == 3.0 and baseline['macro_count'] == 0
assert baseline['admission']['m473_performance_admitted'] is False
assert baseline_review['status'] == 'PASS_M475_INDEPENDENT_RECEIPT_BLIND_HAMMER_WITH_P1'
assert baseline_review['p0_findings'] == []

def semantic_sdc(path):
    return '\n'.join(line.rstrip() for line in Path(path).read_text().splitlines()
                     if not line.lstrip().startswith('#')).strip()

assert semantic_sdc('dc_handoff/constraints/date_m474_fused_parent_dual_update_3ns.sdc') == semantic_sdc('dc_handoff/constraints/date_m476r2_backpressure_safe_parent_queue_3ns.sdc')
PY

mkdir -p "${m477_run}"
m477_complete=0
trap 'm477_rc=$?; if [[ ${m477_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m477_rc}" >"${m477_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
sha256sum \
    rtl_m476/m476_dual_slot_parent_queue_pipeline.sv \
    rtl_m476r2/m476r2_backpressure_safe_parent_queue_pipeline.sv \
    "${m477_filelist}" "${m477_sdc}" "${m477_tcl}" "${m477_contract}" \
    dc_handoff/scripts/run_dc_m477_m476r2_backpressure_safe_parent_queue_exact_sha.sh \
    results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json \
    results/m476r2_independent_hammer_review_r1_20260826/m476r2_independent_hammer_review_r1.json \
    "${m477_baseline}/m475_m474_fused_parent_dual_update_logic_only_dc_receipt_r1.json" \
    results/m475_independent_hammer_review_r1_20260826/m475_independent_hammer_review_r1.json \
    dc_handoff/constraints/date_m474_fused_parent_dual_update_3ns.sdc \
    docs/359_DATE终局冻结_20260813.md \
    "${m477_dc}" "${m477_slow}" "${m477_fast}" >"${m477_run}/input_sha256.txt"
cp "${m477_contract}" "${m477_run}/contract.json"

export DESIGN_NAME=m476r2_backpressure_safe_parent_queue_pipeline
export HW_ROOT="${m477_hw}"
export RTL_FILELIST="${m477_hw}/${m477_filelist}"
export LIB_DB="${m477_slow}"
export MIN_LIB_DB="${m477_fast}"
export SDC_FILE="${m477_hw}/${m477_sdc}"
export OUTPUT_DIR="${m477_run}"
export CLOCK_PERIOD_NS=3.000
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m477_dc}" -f "${m477_hw}/${m477_tcl}" >"${m477_run}/dc.log" 2>&1
m477_rc=$?
set -e
echo "${m477_rc}" >"${m477_run}/dc.rc"
[[ "${m477_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m477_run}/dc.log"
grep -Fq 'Thank you...' "${m477_run}/dc.log"
for m477_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt; do
    [[ -s "${m477_run}/reports/${m477_report}" ]] || exit 30
done
[[ -s "${m477_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m477_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m477_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m477_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m477_run}/reports/timing_setup.rpt" \
    "${m477_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m477_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33

m477_area=$(awk '/Total cell area:/ {print $4; exit}' "${m477_run}/reports/area.rpt")
m477_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m477_run}/reports/area.rpt")
m477_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m477_run}/reports/area.rpt")
m477_combo=$(awk '/Number of combinational cells:/ {print $5; exit}' "${m477_run}/reports/area.rpt")
m477_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m477_run}/reports/qor.rpt")
m477_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m477_run}/reports/timing_setup.rpt")
m477_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m477_run}/reports/timing_hold.rpt")
for m477_value in "${m477_area}" "${m477_cells}" "${m477_seq}" \
        "${m477_combo}" "${m477_levels}" "${m477_setup}" "${m477_hold}"; do
    [[ -n "${m477_value}" ]] || exit 34
done
awk -v x="${m477_area}" 'BEGIN {exit !(x > 0)}'
awk -v x="${m477_setup}" 'BEGIN {exit !(x >= 0.0)}'
awk -v x="${m477_hold}" 'BEGIN {exit !(x >= 0.0)}'

python3 - "${m477_run}" "${m477_area}" "${m477_cells}" \
    "${m477_seq}" "${m477_combo}" "${m477_levels}" \
    "${m477_setup}" "${m477_hold}" <<'PY'
import hashlib, json
from pathlib import Path
import sys
run = Path(sys.argv[1])
area, cells, seq, combo, levels, setup, hold = (
    float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
base = {
    'cell_area_um2': 37316.285232, 'cell_count': 35333,
    'sequential_cells': 4509, 'combinational_cells': 30824,
    'logic_levels': 70.0, 'setup_worst_slack_ns': 0.0,
    'hold_worst_slack_ns': 0.01,
}
measured = {
    'cell_area_um2': area, 'cell_count': cells,
    'sequential_cells': seq, 'combinational_cells': combo,
    'logic_levels': levels, 'setup_worst_slack_ns': setup,
    'hold_worst_slack_ns': hold,
}
comparison = {
    'baseline': 'M475/M474 one-slot fused parent dual-update logic-only DC',
    'area_ratio_vs_m475': area / base['cell_area_um2'],
    'area_delta_um2': area - base['cell_area_um2'],
    'area_delta_percent': (area / base['cell_area_um2'] - 1.0) * 100.0,
    'cell_count_delta': cells - base['cell_count'],
    'sequential_cell_delta': seq - base['sequential_cells'],
    'combinational_cell_delta': combo - base['combinational_cells'],
    'logic_level_delta': levels - base['logic_levels'],
    'setup_slack_delta_ns': setup - base['setup_worst_slack_ns'],
    'hold_slack_delta_ns': hold - base['hold_worst_slack_ns'],
}
receipt = {
    'schema': 'm477_m476r2_backpressure_safe_logic_only_dc_receipt_v1',
    'status': 'PASS_M477_M476R2_BACKPRESSURE_SAFE_LOGIC_ONLY_DC_3NS',
    'tool': 'Synopsys Design Compiler V-2023.12-SP3',
    'technology': 'TSMC28 HPC+ standard cells',
    'operating_condition': 'ssg0p9v125c',
    'clock_period_ns': 3.0,
    'measured': measured,
    'm475_same_constraint_baseline': base,
    'comparison': comparison,
    'macro_count': 0,
    'external_memory_cuts': {
        'parent_scratch': '64x1152b synchronous 1R1W = 9 KiB, excluded',
        'resident_psum_if_64_rows': '64x1824b = 14.25 KiB, excluded',
        'internal_parent_response_slots': '2x1152b = 288 B, synthesized'
    },
    'admission': {
        'm476r2_logic_only_dc_sta': True,
        'same_constraint_m475_cost_comparison': True,
        'three_ns_premacro_timing_met': setup >= 0 and hold >= 0,
        'r2_full_wrapper_regression': False,
        'r2_formality': False,
        'physical_timing': False,
        'scratch_macro_area_timing_power': False,
        'resident_psum_macro_area_timing_power': False,
        'm473_performance_admitted': False,
        'power': False, 'energy': False, 'paper_ppa_ready': False,
        'full_network': False, 'system_speedup': False,
        'date_headline': False
    },
    'required_next_gate': (
        'Independent receipt-blind DC hammer. Before Formality or controller '
        'admission, run the full M476 suite through the r2 wrapper or prove '
        'hazard-false transparency; macro banking and PrimeTime remain mandatory.'
    )
}
(run/'m477_m476r2_backpressure_safe_logic_only_dc_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
(run/'RUN_COMPLETE.txt').write_text(
    'PASS_M477_M476R2_BACKPRESSURE_SAFE_LOGIC_ONLY_DC_3NS\n'
    f'cell_area_um2={area}\nsetup_worst_slack_ns={setup}\n'
    f'hold_worst_slack_ns={hold}\nmacro_count=0\n'
    f'area_ratio_vs_m475={comparison["area_ratio_vs_m475"]}\n'
    'm473_performance_admitted=false\npaper_ppa_ready=false\n')
files = [p for p in sorted(run.rglob('*')) if p.is_file() and
         p.name not in {'evidence_manifest.sha256','evidence_manifest.seal.sha256'}]
(run/'evidence_manifest.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n'
    for p in files))
(run/'evidence_manifest.seal.sha256').write_text(
    f"{hashlib.sha256((run/'evidence_manifest.sha256').read_bytes()).hexdigest()}  evidence_manifest.sha256\n")
PY

m477_complete=1
echo "PASS_M477_M476R2_BACKPRESSURE_SAFE_LOGIC_ONLY_DC run=${m477_run}"
