#!/usr/bin/env bash
set -euo pipefail

m485_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m485_hw="$(cd "${m485_dc_root}/.." && pwd)"
m485_runner="$(realpath "${BASH_SOURCE[0]}")"
m485_run="${M493_DC_RUN:-${m485_dc_root}/runs/m493_fc2_matched_cutthrough_logic_only_dc_3p000ns_r1_20260827}"
m485_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m485_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m485_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m485_f493=dc_handoff/filelists/date_m493_fc2_matched_8bank_logic_only_dc.f
m485_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
m485_tcl=dc_handoff/scripts/run_dc_m485_fc2_matched_top_logic_only.tcl
m485_contract=contracts/m493_fc2_matched_cutthrough_logic_only_dc_contract_r1_20260827.json
m485_vcs=results/m492_fc2_cutthrough_8bank_equal_bandwidth_vcs_r1_exact_20260827/m492_fc2_cutthrough_8bank_equal_bandwidth_vcs_receipt_r1.json

m485_sha() { sha256sum "$1" | awk '{print $1}'; }
m485_expect() {
    local m485_path=$1 m485_expected=$2
    [[ -f "${m485_path}" ]] || { echo "missing ${m485_path}" >&2; exit 3; }
    [[ "$(m485_sha "${m485_path}")" == "${m485_expected}" ]] || {
        echo "M493 SHA mismatch ${m485_path}" >&2
        exit 3
    }
}

[[ ! -e "${m485_run}" ]] || {
    echo "M493 refuses to overwrite ${m485_run}" >&2
    exit 5
}
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -f '/common_shell_exec -shell dc_shell ' >/dev/null; then
    echo "M493 refuses to collide with another Design Compiler run" >&2
    exit 4
fi

cd "${m485_hw}"
m485_expect "${m485_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m485_expect "${m485_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m485_expect "${m485_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m485_expect rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5
m485_expect rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv 8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0
m485_expect rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv 529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267
m485_expect rtl_m218/m218_fc2_tagged_slice_service_island.sv f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1
m485_expect rtl_m219/m219_fc2_k1_cropped_tagged_slice_service_island.sv 75c4690ec04653084fb59fd75c5ba7ac329807975d76c9ffc43b6304bd4e1d47
m485_expect rtl_m342/m342_fc2_standalone_raw4_acc24.sv 309759bfa6eeb303143e707bd3df269eddcd31e34e79ed662d507c363ba4d904
m485_expect rtl_m349/m349_fc2_k1x8_raw4_acc24.sv ddcf6c051a43813f84fe94a789f209160d522e8a8be79a3fc7b572133393b2c9
m485_expect rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv 597e4d9e9a606afa58111d01be8e8304e4fb5d4656cabdd4da9fca4b8393f43b
m485_expect rtl_m491/m491_fc2_k8_cutthrough_8bank_raw4_acc24.sv efc8ed5ca96cd0b38ae4a1d9fb1c00bb4b5f1100396519d3749c512f9cc9f596
m485_expect rtl_m493/m493_fc2_matched_8bank_raw4_acc24.sv d416b5cfc99c6adfe57e898fb7b7ef0f4c8a13286d12cf90e5de64bd94c36471
m485_expect "${m485_f493}" 40be33503e80e04852a0a689ac3d4363c45d81b434992340db82a7d1e8b09bef
m485_expect "${m485_sdc}" 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5
m485_expect "${m485_tcl}" 6b9491022cd8f219a57fa12eac38506a72973b1d8d2f92bf43330522132c934b
m485_expect "${m485_contract}" eeb78f6324da5eaaf387f92274d5bec29026319f1c5b6d2d5a109e016cb3fc86
m485_expect "${m485_vcs}" beec54d7a3318fdfa22d3ef8c7eb89b3ce6de33fe8ffff2477563a6a99d2aadf
m485_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m485_run}"
m485_complete=0
trap 'm485_rc=$?; if [[ ${m485_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m485_rc}" >"${m485_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
sha256sum rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv \
    rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv \
    rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv \
    rtl_m218/m218_fc2_tagged_slice_service_island.sv \
    rtl_m219/m219_fc2_k1_cropped_tagged_slice_service_island.sv \
    rtl_m342/m342_fc2_standalone_raw4_acc24.sv \
    rtl_m349/m349_fc2_k1x8_raw4_acc24.sv \
    rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv \
    rtl_m491/m491_fc2_k8_cutthrough_8bank_raw4_acc24.sv \
    rtl_m493/m493_fc2_matched_8bank_raw4_acc24.sv \
    "${m485_f493}" "${m485_sdc}" "${m485_tcl}" \
    "${m485_contract}" "${m485_vcs}" \
    docs/359_DATE终局冻结_20260813.md \
    "${m485_dc}" "${m485_slow}" "${m485_fast}" \
    >"${m485_run}/input_sha256.txt"
cp "${m485_contract}" "${m485_run}/contract.json"
sha256sum "${m485_runner}" >"${m485_run}/runner_sha256.txt"

export HW_ROOT="${m485_hw}"
export LIB_DB="${m485_slow}"
export MIN_LIB_DB="${m485_fast}"
export SDC_FILE="${m485_hw}/${m485_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m485_run_point() {
    local m485_id=$1 m485_top=$2 m485_filelist=$3 m485_parameters=$4
    local m485_dir="${m485_run}/${m485_id}"
    mkdir -p "${m485_dir}"
    export DESIGN_NAME="${m485_top}"
    export RTL_FILELIST="${m485_hw}/${m485_filelist}"
    export OUTPUT_DIR="${m485_dir}"
    export ELAB_PARAMETERS="${m485_parameters}"
    set +e
    "${m485_dc}" -f "${m485_hw}/${m485_tcl}" \
        >"${m485_dir}/dc.log" 2>&1
    local m485_rc=$?
    set -e
    echo "${m485_rc}" >"${m485_dir}/dc.rc"
    [[ "${m485_rc}" -eq 0 ]] || return 20
    ! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' \
        "${m485_dir}/dc.log" || return 21
    grep -Fq 'Thank you...' "${m485_dir}/dc.log" || return 22
    for m485_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt hierarchy_postcompile.rpt \
            resources_postcompile.rpt references_postcompile.rpt ports.rpt; do
        [[ -s "${m485_dir}/reports/${m485_report}" ]] || return 30
    done
    [[ -s "${m485_dir}/netlist/${m485_top}_mapped.v" ]] || return 31
    ! grep -Fq 'slack (VIOLATED)' "${m485_dir}/reports/timing_setup.rpt" \
        "${m485_dir}/reports/timing_hold.rpt" || return 32
    [[ "$(grep -Fc 'This design has no violated constraints.' \
        "${m485_dir}/reports/constraint_violators.rpt")" -eq 5 ]] \
        || return 33
    ! grep -Eqi 'unresolved reference|multiply driven|latch inferred' \
        "${m485_dir}/dc.log" "${m485_dir}/reports/check_design_postcompile.rpt" \
        || return 34

    local m485_area m485_cells m485_seq m485_combo m485_levels
    local m485_path m485_setup m485_hold m485_ports
    m485_area=$(awk '/Total cell area:/ {print $4; exit}' "${m485_dir}/reports/area.rpt")
    m485_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m485_dir}/reports/area.rpt")
    m485_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m485_dir}/reports/area.rpt")
    m485_combo=$(awk '/Number of combinational cells:/ {print $5; exit}' "${m485_dir}/reports/area.rpt")
    m485_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m485_dir}/reports/qor.rpt")
    m485_path=$(awk '/Critical Path Length:/ {print $4; exit}' "${m485_dir}/reports/qor.rpt")
    m485_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m485_dir}/reports/timing_setup.rpt")
    m485_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m485_dir}/reports/timing_hold.rpt")
    m485_ports=$(awk '/^Information: There are [0-9]+ ports/ {print $4; exit}' "${m485_dir}/reports/ports.rpt" || true)
    for m485_value in "${m485_area}" "${m485_cells}" "${m485_seq}" \
            "${m485_combo}" "${m485_levels}" "${m485_path}" \
            "${m485_setup}" "${m485_hold}" "${m485_ports}"; do
        [[ -n "${m485_value}" ]] || return 35
    done
    awk -v x="${m485_setup}" 'BEGIN {exit !(x >= 0)}'
    awk -v x="${m485_hold}" 'BEGIN {exit !(x >= 0)}'
    printf '%s\n' \
        "status=PASS_M493_${m485_id^^}_LOGIC_ONLY_DC_3NS_CLEAN" \
        "design=${m485_top}" \
        "elaboration_parameters=${m485_parameters:-none}" \
        "cell_area_um2=${m485_area}" \
        "cell_count=${m485_cells}" \
        "sequential_cells=${m485_seq}" \
        "combinational_cells=${m485_combo}" \
        "logic_levels=${m485_levels}" \
        "critical_path_length_ns=${m485_path}" \
        "setup_worst_slack_ns=${m485_setup}" \
        "hold_worst_slack_ns=${m485_hold}" \
        "reported_port_count=${m485_ports:-unknown}" \
        "macro_count=0" \
        "paper_ppa_ready=false" \
        "system_speedup=false" \
        >"${m485_dir}/RUN_COMPLETE.txt"
    sha256sum "${m485_dir}/dc.log" "${m485_dir}/reports/"*.rpt \
        "${m485_dir}/netlist/"* "${m485_dir}/RUN_COMPLETE.txt" \
        >"${m485_dir}/evidence_manifest.sha256"
}

m485_run_point k8_cutthrough m493_fc2_matched_8bank_raw4_acc24 \
    "${m485_f493}" ARCH_K8=1
m485_run_point k1x8 m493_fc2_matched_8bank_raw4_acc24 \
    "${m485_f493}" ARCH_K8=0

python3 - "${m485_run}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])

def read_point(name):
    values = {}
    for line in (run/name/'RUN_COMPLETE.txt').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key] = value
    return {
        'cell_area_um2': float(values['cell_area_um2']),
        'cell_count': int(values['cell_count']),
        'sequential_cells': int(values['sequential_cells']),
        'combinational_cells': int(values['combinational_cells']),
        'logic_levels': float(values['logic_levels']),
        'critical_path_length_ns': float(values['critical_path_length_ns']),
        'setup_worst_slack_ns': float(values['setup_worst_slack_ns']),
        'hold_worst_slack_ns': float(values['hold_worst_slack_ns']),
        'reported_port_count': values['reported_port_count'],
    }

k8, k1x8 = (read_point(x) for x in ('k8_cutthrough', 'k1x8'))
k8_k1x8_area = k8['cell_area_um2'] / k1x8['cell_area_um2']
k8_k1x8_seq = k8['sequential_cells'] / k1x8['sequential_cells']
ports_identical = k8['reported_port_count'] == k1x8['reported_port_count']
gates = {
    'both_dc_constraint_clean': all(
        p['setup_worst_slack_ns'] >= 0 and p['hold_worst_slack_ns'] >= 0
        for p in (k8, k1x8)),
    'reported_port_count_identical': ports_identical,
    'prior_vcs_equal_bandwidth_throughput_parity': True,
    'k8_over_k1x8_area_lte_0p50': k8_k1x8_area <= 0.5,
    'k8_over_k1x8_sequential_cells_lte_0p50': k8_k1x8_seq <= 0.5,
    'throughput_per_area_improvement_gte_2': (1.0 / k8_k1x8_area) >= 2.0,
}
logic_gate = all(gates.values())
receipt = {
    'schema': 'm493_fc2_matched_cutthrough_logic_only_dc_receipt_v1',
    'status': ('PASS_M493_MATCHED_CUTTHROUGH_LOGIC_PARETO_GATE'
               if logic_gate else 'PASS_M493_DC_BUT_LOGIC_PARETO_NO_GO'),
    'tool': 'Synopsys Design Compiler V-2023.12-SP3',
    'technology': 'TSMC28 HPC+ standard cells',
    'operating_condition': 'ssg0p9v125c',
    'clock_period_ns': 3.0,
    'clock_network': 'ideal',
    'wireload': 'ZeroWireload',
    'macro_count_each': 0,
    'measured': {'k8_cutthrough': k8, 'k1x8': k1x8},
    'measured_area_ratios': {
        'k8_over_k1x8': k8_k1x8_area,
        'k1x8_over_k8': 1.0 / k8_k1x8_area,
    },
    'measured_sequential_cell_ratios': {
        'k8_over_k1x8': k8_k1x8_seq,
        'k1x8_over_k8': 1.0 / k8_k1x8_seq,
    },
    'throughput_per_area_improvement_k8_over_k1x8': 1.0 / k8_k1x8_area,
    'prior_vcs_cycle_ratios_not_remeasured_here': {
        'k8_over_k1x8_equal_peak_bandwidth': 1.0,
    },
    'hard_gate_results': gates,
    'logic_pareto_gate': logic_gate,
    'fairness_limitations': [
        'Both elaborations use the same M493 top and eight scalar bank endpoints.',
        'Debug counters are not observable at M493 and may be optimized away.',
        'Weight SRAM macros and explicit paper context macros are excluded.'
    ],
    'admission': {
        'matched_compile_and_constraints': True,
        'canonical_bank_wrapper': True,
        'formality': False,
        'power': False,
        'energy': False,
        'paper_ppa_ready': False,
        'full_ffn': False,
        'full_network': False,
        'system_speedup': False,
        'date_headline': False,
    },
    'required_next_gate': (
        'Independent receipt-blind hammer, then Formality and matched SAIF/PTPX energy per completed work only if logic_pareto_gate is true.'
    )
}
(run/'m493_fc2_matched_cutthrough_logic_only_dc_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
(run/'RUN_COMPLETE.txt').write_text(
    receipt['status'] + '\n'
    + f"k8_over_k1x8_area_ratio={k8_k1x8_area:.12f}\n"
    + f"k8_over_k1x8_sequential_cell_ratio={k8_k1x8_seq:.12f}\n"
    + f"throughput_per_area_improvement={1.0 / k8_k1x8_area:.12f}\n"
    + 'paper_ppa_ready=false\nsystem_speedup=false\n')
files = [p for p in sorted(run.rglob('*')) if p.is_file()
         and p.name not in {'evidence_manifest.sha256',
                            'evidence_manifest.seal.sha256'}]
(run/'evidence_manifest.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n'
    for p in files))
(run/'evidence_manifest.seal.sha256').write_text(
    hashlib.sha256((run/'evidence_manifest.sha256').read_bytes()).hexdigest()
    + '  evidence_manifest.sha256\n')
PY
(cd "${m485_run}" && sha256sum -c evidence_manifest.sha256 >/dev/null \
    && sha256sum -c evidence_manifest.seal.sha256 >/dev/null)
m485_complete=1
rm -f "${m485_run}/RUN_FAILED_OR_INCOMPLETE.txt"
echo "PASS M493 matched cutthrough logic-only DC sealed at ${m485_run}"
