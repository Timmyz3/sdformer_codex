#!/usr/bin/env bash
set -euo pipefail

m431_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m431_hw="$(cd "${m431_dc_root}/.." && pwd)"
m431_runner="$(realpath "${BASH_SOURCE[0]}")"
m431_run="${M431_DC_RUN:-${m431_dc_root}/runs/m431_m414_saif_tracked_dc_3p000ns_r1_20260826}"
m431_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m431_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m431_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m431_filelist="dc_handoff/filelists/date_m414_balanced_selected_slice_rtl.f"
m431_sdc="dc_handoff/constraints/date_m412_m405_selected_slice_3ns.sdc"
m431_tcl="dc_handoff/scripts/run_dc_m431_m414_saif_tracked_selected_slice.tcl"
m431_contract="contracts/m431_m414_saif_tracked_dc_diagnostic_contract_r1_20260826.json"
m431_top="m405_q32_elastic_selected_slice"
m431_m416="dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826"
m431_m429="results/m429_m425r4_saif_independent_hammer_r1_20260826"

m431_sha() { sha256sum "$1" | awk '{print $1}'; }
m431_expect() {
    local m431_path=$1
    local m431_expected=$2
    [[ -f "${m431_path}" ]] || exit 3
    [[ "$(m431_sha "${m431_path}")" == "${m431_expected}" ]] || exit 3
}

[[ ! -e "${m431_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m431_hw}"
m431_expect "${m431_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m431_expect "${m431_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m431_expect "${m431_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m431_expect "${m431_tcl}" d40d72d22ec7508a0239387ba30bb9710fd04ca7a1accdbe9960b82300e2a33a
m431_expect "${m431_contract}" b86858557d3a66017ad97a73b75e855601c6e506ae934fb1c43fdc00f0b86766
m431_expect "${m431_filelist}" 5e93db53c751df3ca3c4cefec8376434f31cbe6561b15e0848ec2d872adc1f92
m431_expect "${m431_sdc}" 565f486c7537484b0b6c11db7e53e4afc6962f2f73827a30764c3fe70bf3bb29
m431_expect rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
m431_expect rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv 819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744
m431_expect rtl_m405/m405_q32_elastic_selected_slice.sv 91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd
m431_expect "${m431_m416}/evidence_manifest.seal.sha256" 40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79
m431_expect "${m431_m429}/SHA256SUMS.seal.sha256" 06496b718f116ad1e1d1c84bda095f319fc9b10b9bcf3b554e042e585c87fa33
m431_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m431_run}"
m431_complete=0
trap 'm431_rc=$?; if [[ ${m431_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m431_rc}" >"${m431_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
(cd "${m431_m416}" && sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) \
    >"${m431_run}/upstream_seal_checks.log" 2>&1
(cd "${m431_m429}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m431_run}/upstream_seal_checks.log" 2>&1
sha256sum "${m431_tcl}" "${m431_contract}" "${m431_filelist}" \
    "${m431_sdc}" "${m431_slow}" "${m431_fast}" \
    rtl_m414/m414_q32_balanced16_zero_stop_controller.sv \
    rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv \
    rtl_m405/m405_q32_elastic_selected_slice.sv \
    "${m431_m416}/evidence_manifest.seal.sha256" \
    "${m431_m429}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md >"${m431_run}/input_sha256.txt"
cp "${m431_contract}" "${m431_run}/contract.json"

export DESIGN_NAME="${m431_top}"
export HW_ROOT="${m431_hw}"
export RTL_FILELIST="${m431_hw}/${m431_filelist}"
export LIB_DB="${m431_slow}"
export MIN_LIB_DB="${m431_fast}"
export SDC_FILE="${m431_hw}/${m431_sdc}"
export OUTPUT_DIR="${m431_run}"
export OPERATING_CONDITION=ssg0p9v125c
set +e
"${m431_dc}" -f "${m431_hw}/${m431_tcl}" >"${m431_run}/dc.log" 2>&1
m431_rc=$?
set -e
printf '%s\n' "${m431_rc}" >"${m431_run}/dc.rc"
[[ "${m431_rc}" -eq 0 ]] || exit 20
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m431_run}/dc.log"; then
    exit 21
fi
grep -Fq 'Thank you...' "${m431_run}/dc.log" || exit 22
for m431_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt saif_map_report.rpt; do
    [[ -f "${m431_run}/reports/${m431_report}" ]] || exit 23
done
for m431_netlist in "${m431_top}_mapped.v" "${m431_top}_mapped.sdc" \
        "${m431_top}.ddc" "${m431_top}.svf" \
        "${m431_top}.saif_map.bin" "${m431_top}.ptpx_saif_map.tcl"; do
    [[ -s "${m431_run}/netlist/${m431_netlist}" ]] || exit 24
done
if grep -Fq 'slack (VIOLATED)' "${m431_run}/reports/timing_setup.rpt" \
        "${m431_run}/reports/timing_hold.rpt"; then
    exit 25
fi
if grep -Eiq 'unresolved reference|black box|inferred latch|timing loop' \
        "${m431_run}/reports/check_design_postcompile.rpt" \
        "${m431_run}/reports/check_timing_postcompile.rpt"; then
    exit 26
fi
grep -Fq 'set_rtl_to_gate_name' \
    "${m431_run}/netlist/${m431_top}.ptpx_saif_map.tcl" || exit 27

python3 - "${m431_run}" "${m431_m416}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
reference_run = Path(sys.argv[2])
reports = run / "reports"

def first(pattern: str, path: Path, cast=float):
    match = re.search(pattern, path.read_text(errors="replace"), re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {pattern!r} in {path}")
    return cast(match.group(1))

area = first(r"Total cell area:\s+([0-9.]+)", reports / "area.rpt")
cells = first(r"Number of cells:\s+([0-9]+)", reports / "area.rpt", int)
seq = first(r"Number of sequential cells:\s+([0-9]+)", reports / "area.rpt", int)
levels = first(r"Levels of Logic:\s+([0-9.]+)", reports / "qor.rpt")
setup = first(r"slack \(MET\)\s+([-0-9.]+)", reports / "timing_setup.rpt")
hold = first(r"slack \(MET\)\s+([-0-9.]+)", reports / "timing_hold.rpt")
reference = json.loads((reference_run / "m416_m414_balanced_selected_slice_dc_receipt_r1.json").read_text())
ref = reference["m416_balanced"]
area_fraction = area / ref["cell_area_um2"] - 1.0
if seq != ref["sequential_cells"]:
    raise SystemExit(f"sequential population drift {seq} != {ref['sequential_cells']}")
if abs(area_fraction) > 0.01 or setup < 0 or hold < 0:
    raise SystemExit(f"M431 reproduction gate failed area_fraction={area_fraction} setup={setup} hold={hold}")
map_file = run / "netlist/m405_q32_elastic_selected_slice.ptpx_saif_map.tcl"
map_text = map_file.read_text(errors="replace")
map_entries = len(re.findall(r"^\s*set_rtl_to_gate_name\b", map_text, re.MULTILINE))
if map_entries < 100:
    raise SystemExit(f"SAIF map unexpectedly small: {map_entries}")
receipt = {
    "schema": "m431_m414_saif_tracked_dc_diagnostic_receipt_v1",
    "status": "PASS_M431_SAIF_TRACKED_DC_DIAGNOSTIC",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "tracked_netlist": {
        "cell_area_um2": area,
        "cell_count": cells,
        "sequential_cells": seq,
        "logic_levels": levels,
        "setup_worst_slack_ns": setup,
        "hold_worst_slack_ns": hold,
    },
    "m416_reproduction": {
        "reference_cell_area_um2": ref["cell_area_um2"],
        "cell_area_delta_fraction": area_fraction,
        "reference_sequential_cells": ref["sequential_cells"],
        "sequential_cells_equal": True,
    },
    "saif_mapping": {
        "method": "Synopsys saif_map transformation tracking",
        "ptpx_essential_map_entries": map_entries,
        "ptpx_map_file": map_file.name,
        "coverage_measured": False,
        "minimum_coverage_fraction_for_power": 0.95,
    },
    "claim_boundary": {
        "dc_mapping_diagnostic": True,
        "formality": False,
        "mapped_annotation_coverage": False,
        "power": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(run / "m431_m414_saif_tracked_dc_diagnostic_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m431_runner}" >"${m431_run}/runner_sha256.txt"
printf '%s\n' PASS_M431_SAIF_TRACKED_DC_DIAGNOSTIC \
    >"${m431_run}/RUN_COMPLETE.txt"
find "${m431_run}" -type f \
    ! -name evidence_manifest.sha256 ! -name evidence_manifest.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m431_run}/evidence_manifest.sha256"
sha256sum "${m431_run}/evidence_manifest.sha256" \
    >"${m431_run}/evidence_manifest.seal.sha256"
m431_complete=1
echo "PASS M431 SAIF-tracked DC diagnostic sealed at ${m431_run}"
