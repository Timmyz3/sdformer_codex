#!/usr/bin/env bash
set -euo pipefail

m475_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m475_hw="$(cd "${m475_dc_root}/.." && pwd)"
m475_run="${M475_DC_RUN:-${m475_dc_root}/runs/m475_m474_fused_parent_dual_update_dc_3p000ns_r1_20260826}"
m475_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m475_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m475_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m475_filelist="dc_handoff/filelists/date_m474_fused_parent_dual_update_dc.f"
m475_sdc="dc_handoff/constraints/date_m474_fused_parent_dual_update_3ns.sdc"
m475_tcl="dc_handoff/scripts/run_dc_m474_fused_parent_dual_update_exact_sha.tcl"
m475_contract="contracts/m475_m474_fused_parent_dual_update_logic_only_dc_contract_r1_20260826.json"

m475_sha() { sha256sum "$1" | awk '{print $1}'; }
m475_expect() {
    local m475_path=$1
    local m475_expected=$2
    [[ -f "${m475_path}" ]] || { echo "missing ${m475_path}" >&2; exit 3; }
    [[ "$(m475_sha "${m475_path}")" == "${m475_expected}" ]] || {
        echo "M475 SHA mismatch ${m475_path}" >&2
        exit 3
    }
}

[[ ! -e "${m475_run}" ]] || { echo "M475 output exists ${m475_run}" >&2; exit 5; }
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    echo "M475 refuses to collide with another Design Compiler run" >&2
    exit 4
fi
cd "${m475_hw}"
m475_expect "${m475_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m475_expect "${m475_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m475_expect "${m475_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m475_expect rtl_m474/m474_fused_parent_dual_update_pipeline.sv 30fdf778e5baea959c793c7b2f9d9e332364b84717f9ffd2f8ad74d85280d57c
m475_expect "${m475_filelist}" 15dff6a2ec48021842615a13b830b89f70328f125feeaf51c190deb749c367c7
m475_expect "${m475_sdc}" 09846a3645de26a89454893e89bef05b1b4b0d2cd1591ff176da5533ace6fdbe
m475_expect "${m475_tcl}" 1b7b848c99abd9a52ba686ffe1dbaa2aa0278fd5675ba0b772f13ed53a91b703
m475_expect "${m475_contract}" ec48ec25b121787f44b9ec99e9b078631573167902e497ac4eb96cef776b8213
m475_expect results/m474_fused_parent_dual_update_vcs_r1_20260826/m474_fused_parent_dual_update_vcs_receipt_r1.json 4c7a77e4c9f26476c27a2e64194cc3d42343b4303d2d9ecea7c4c4a17b681f44
m475_expect results/m474_fused_parent_dual_update_vcs_r1_20260826/SHA256SUMS 67f31a260b424b6802cca43800df9d4f4cc15ef996912f180b1f72757c2b07b4
m475_expect results/m474_fused_parent_dual_update_vcs_r1_20260826/SHA256SUMS.seal.sha256 190771692c00697126b09b5e2db5340fbfcb618114b09aafb62e384e69327298
m475_expect results/m474_independent_hammer_review_r1_20260826/m474_independent_hammer_review_r1.json a684444122429752a1f2d49b21db59a5db3547df4fdf9e19d7eb3f56ccdbb7ce
m475_expect results/m474_independent_hammer_review_r1_20260826/SHA256SUMS c7a5a1d401c7f0e21cd54259155a0ea84a753c2e200287b69d429d9ae4d7fbb5
m475_expect results/m474_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 b6c997b68f5246de0f78e42d0e3ded4872af3c93ffbf0ff365bee7657ab3599a
m475_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(cd results/m474_fused_parent_dual_update_vcs_r1_20260826 &&
  sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)
(cd results/m474_independent_hammer_review_r1_20260826 &&
  sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)

python3 - <<'PY'
import json
from pathlib import Path
vcs = json.loads(Path('results/m474_fused_parent_dual_update_vcs_r1_20260826/m474_fused_parent_dual_update_vcs_receipt_r1.json').read_text())
hammer = json.loads(Path('results/m474_independent_hammer_review_r1_20260826/m474_independent_hammer_review_r1.json').read_text())
assert vcs['status'] == 'PASS_EXACT_SHA_SYNOPSYS_VCS_MICRO_FUNCTIONAL_ONLY'
assert vcs['admission']['m473_performance_admitted'] is False
assert hammer['status'] == 'PASS_M474_INDEPENDENT_POST_RUN_HAMMER'
assert hammer['verdict'] == 'GO_TO_3P0NS_PREMACRO_DC_STA_MICRO_LOGIC_ONLY'
assert hammer['score_out_of_100'] == 96
PY

mkdir -p "${m475_run}"
m475_complete=0
trap 'm475_rc=$?; if [[ ${m475_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m475_rc}" >"${m475_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
sha256sum \
    rtl_m474/m474_fused_parent_dual_update_pipeline.sv \
    "${m475_filelist}" "${m475_sdc}" "${m475_tcl}" "${m475_contract}" \
    results/m474_fused_parent_dual_update_vcs_r1_20260826/m474_fused_parent_dual_update_vcs_receipt_r1.json \
    results/m474_independent_hammer_review_r1_20260826/m474_independent_hammer_review_r1.json \
    "${m475_slow}" "${m475_fast}" >"${m475_run}/input_sha256.txt"
cp "${m475_contract}" "${m475_run}/contract.json"

export DESIGN_NAME=m474_fused_parent_dual_update_pipeline
export HW_ROOT="${m475_hw}"
export RTL_FILELIST="${m475_hw}/${m475_filelist}"
export LIB_DB="${m475_slow}"
export MIN_LIB_DB="${m475_fast}"
export SDC_FILE="${m475_hw}/${m475_sdc}"
export OUTPUT_DIR="${m475_run}"
export CLOCK_PERIOD_NS=3.000
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m475_dc}" -f "${m475_hw}/${m475_tcl}" >"${m475_run}/dc.log" 2>&1
m475_rc=$?
set -e
echo "${m475_rc}" >"${m475_run}/dc.rc"
[[ "${m475_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m475_run}/dc.log"
grep -Fq 'Thank you...' "${m475_run}/dc.log"
for m475_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt; do
    [[ -s "${m475_run}/reports/${m475_report}" ]] || exit 30
done
[[ -s "${m475_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m475_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m475_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m475_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m475_run}/reports/timing_setup.rpt" \
    "${m475_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m475_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33

m475_area=$(awk '/Total cell area:/ {print $4; exit}' "${m475_run}/reports/area.rpt")
m475_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m475_run}/reports/area.rpt")
m475_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m475_run}/reports/area.rpt")
m475_combo=$(awk '/Number of combinational cells:/ {print $5; exit}' "${m475_run}/reports/area.rpt")
m475_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m475_run}/reports/qor.rpt")
m475_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m475_run}/reports/timing_setup.rpt")
m475_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m475_run}/reports/timing_hold.rpt")
for m475_value in "${m475_area}" "${m475_cells}" "${m475_seq}" \
        "${m475_combo}" "${m475_levels}" "${m475_setup}" "${m475_hold}"; do
    [[ -n "${m475_value}" ]] || exit 34
done
awk -v x="${m475_area}" 'BEGIN {exit !(x > 0)}'
awk -v x="${m475_setup}" 'BEGIN {exit !(x >= 0.0)}'
awk -v x="${m475_hold}" 'BEGIN {exit !(x >= 0.0)}'

python3 - "${m475_run}" "${m475_area}" "${m475_cells}" \
    "${m475_seq}" "${m475_combo}" "${m475_levels}" \
    "${m475_setup}" "${m475_hold}" <<'PY'
import hashlib, json
from pathlib import Path
import sys
run = Path(sys.argv[1])
area, cells, seq, combo, levels, setup, hold = (
    float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
receipt = {
    'schema': 'm475_m474_fused_parent_dual_update_logic_only_dc_receipt_v1',
    'status': 'PASS_M475_M474_FUSED_PIPELINE_LOGIC_ONLY_DC_3NS',
    'tool': 'Synopsys Design Compiler V-2023.12-SP3',
    'technology': 'TSMC28 HPC+ standard cells',
    'operating_condition': 'ssg0p9v125c',
    'clock_period_ns': 3.0,
    'cell_area_um2': area,
    'cell_count': cells,
    'sequential_cells': seq,
    'combinational_cells': combo,
    'logic_levels': levels,
    'setup_worst_slack_ns': setup,
    'hold_worst_slack_ns': hold,
    'macro_count': 0,
    'external_memory_cuts': {
        'parent_scratch': '64 rows x 96 lanes x signed12, 1R1W, excluded',
        'resident_psum': '96 lanes x signed19 interface, excluded'
    },
    'admission': {
        'm474_logic_only_dc_sta': True,
        'three_ns_premacro_timing_met': setup >= 0 and hold >= 0,
        'physical_timing': False,
        'scratch_macro_area_timing_power': False,
        'resident_psum_macro_area_timing_power': False,
        'm473_full_controller_rtl': False,
        'm473_performance_admitted': False,
        'power': False, 'energy': False, 'paper_ppa_ready': False,
        'full_network': False, 'system_speedup': False,
        'date_headline': False
    },
    'required_next_gate': (
        'Independent DC hammer; then target 144-byte 1R1W scratch macro '
        'timing/energy or a conservative macro envelope before M473 performance admission.'
    )
}
(run/'m475_m474_fused_parent_dual_update_logic_only_dc_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
(run/'RUN_COMPLETE.txt').write_text(
    'PASS_M475_M474_FUSED_PIPELINE_LOGIC_ONLY_DC_3NS\n'
    f'cell_area_um2={area}\nsetup_worst_slack_ns={setup}\n'
    f'hold_worst_slack_ns={hold}\nmacro_count=0\n'
    'm473_performance_admitted=false\npaper_ppa_ready=false\n')
files = [p for p in sorted(run.rglob('*')) if p.is_file() and
         p.name not in {'evidence_manifest.sha256','evidence_manifest.seal.sha256'}]
(run/'evidence_manifest.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n'
    for p in files))
(run/'evidence_manifest.seal.sha256').write_text(
    f"{hashlib.sha256((run/'evidence_manifest.sha256').read_bytes()).hexdigest()}  evidence_manifest.sha256\n")
PY

m475_complete=1
echo "PASS_M475_M474_FUSED_PIPELINE_LOGIC_ONLY_DC run=${m475_run}"
