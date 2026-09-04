#!/usr/bin/env bash
set -euo pipefail

m416_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m416_hw="$(cd "${m416_dc_root}/.." && pwd)"
m416_runner="$(realpath "${BASH_SOURCE[0]}")"
m416_run="${M416_DC_RUN:-${m416_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826}"
m416_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m416_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m416_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m416_filelist="dc_handoff/filelists/date_m414_balanced_selected_slice_rtl.f"
m416_sdc="dc_handoff/constraints/date_m412_m405_selected_slice_3ns.sdc"
m416_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m416_contract="contracts/m416_m414_balanced_selected_slice_dc_contract_r1_20260826.json"
m416_top="m405_q32_elastic_selected_slice"

m416_sha() { sha256sum "$1" | awk '{print $1}'; }
m416_expect() {
    local m416_path=$1
    local m416_expected=$2
    [[ -f "${m416_path}" ]] || exit 3
    [[ "$(m416_sha "${m416_path}")" == "${m416_expected}" ]] || exit 3
}

[[ ! -e "${m416_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m416_hw}"
m416_expect "${m416_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m416_expect "${m416_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m416_expect "${m416_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m416_expect "${m416_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m416_expect "${m416_contract}" 69073f1f51c359cb4373425b283778c9c7ca461d6acb1538aa951ddeba0b468f
m416_expect rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
m416_expect rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv 819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744
m416_expect rtl_m405/m405_q32_elastic_selected_slice.sv 91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd
m416_expect "${m416_filelist}" 5e93db53c751df3ca3c4cefec8376434f31cbe6561b15e0848ec2d872adc1f92
m416_expect "${m416_sdc}" 565f486c7537484b0b6c11db7e53e4afc6962f2f73827a30764c3fe70bf3bb29
m416_expect dc_handoff/runs/m412_dual_standalone_logic_only_dc_3p000ns_r1_20260826/m412_dual_standalone_logic_only_dc_receipt_r1.json f650871eee5fa2cee412bfcdf46fbd1ce2e67069e02f3e7374149fcf4c366442
m416_expect results/m413_m412_dual_dc_independent_hammer_r1_20260826/m413_m412_dual_dc_independent_hammer_review_r1.json de9bf0d1d0da77f13185e91f7f23681255b2966782a7a33a8db191633973728d
m416_expect results/m414_q32_balanced16_vcs_r1_20260826/m414_q32_balanced16_zero_stop_vcs_receipt_r1.json 032bedb7ee15080083cacceee49b7ecb0a2fe92ccb456aceeebc8e86928a183d
m416_expect results/m414_q32_balanced16_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 5baeba2fca6f9d350d3c0dc363c405eeda76fb8deaa4dcf709914885748c772f
m416_expect contracts/m415_m414_balanced_vcs_independent_hammer_contract_r1_20260826.json 5a855fecef04b47aa3f16c5b1eb7c2a64b0ea7db7720c47177d0cfc50a496312
m416_expect results/m415_m414_balanced_vcs_independent_hammer_r1_20260826/m415_m414_balanced_vcs_independent_hammer_review_r1.json 1dcb3f71a9d1c23773e4051059299f49d7719382fc1a0243019377a64665c981
m416_expect results/m415_m414_balanced_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 98db30a32a11957795bc4a1e937864718c486d11eb0b77b25aecfdd852278167
m416_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m416_run}"
m416_complete=0
trap 'm416_rc=$?; if [[ ${m416_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m416_rc}" >"${m416_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
(
    cd results/m415_m414_balanced_vcs_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m416_run}/m415_seal_check.log" 2>&1
sha256sum \
    rtl_m414/m414_q32_balanced16_zero_stop_controller.sv \
    rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv \
    rtl_m405/m405_q32_elastic_selected_slice.sv \
    "${m416_filelist}" "${m416_sdc}" "${m416_tcl}" \
    "${m416_contract}" "${m416_slow}" "${m416_fast}" \
    results/m414_q32_balanced16_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
    results/m415_m414_balanced_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m416_run}/input_sha256.txt"
cp "${m416_contract}" "${m416_run}/contract.json"

export DESIGN_NAME="${m416_top}"
export HW_ROOT="${m416_hw}"
export RTL_FILELIST="${m416_hw}/${m416_filelist}"
export LIB_DB="${m416_slow}"
export MIN_LIB_DB="${m416_fast}"
export SDC_FILE="${m416_hw}/${m416_sdc}"
export OUTPUT_DIR="${m416_run}"
export OPERATING_CONDITION=ssg0p9v125c
set +e
"${m416_dc}" -f "${m416_hw}/${m416_tcl}" >"${m416_run}/dc.log" 2>&1
m416_rc=$?
set -e
echo "${m416_rc}" >"${m416_run}/dc.rc"
[[ "${m416_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m416_run}/dc.log"
grep -Fq 'Thank you...' "${m416_run}/dc.log"
for m416_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt hold_guard_contract.rpt; do
    [[ -s "${m416_run}/reports/${m416_report}" ]] || exit 30
done
[[ -s "${m416_run}/netlist/${m416_top}_mapped.v" &&
   -s "${m416_run}/netlist/${m416_top}_mapped.sdc" &&
   -s "${m416_run}/netlist/${m416_top}.ddc" &&
   -s "${m416_run}/netlist/${m416_top}.svf" ]] || exit 31
if grep -Fq 'slack (VIOLATED)' "${m416_run}/reports/timing_setup.rpt" \
        "${m416_run}/reports/timing_hold.rpt"; then
    exit 32
fi
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m416_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33
grep -qx 'additional_hold_guard_ns=0.025' \
    "${m416_run}/reports/hold_guard_contract.rpt"
grep -Eq 'set_clock_uncertainty( +-hold)? +0\.1(00)? ' \
    "${m416_run}/netlist/${m416_top}_mapped.sdc"
if grep -Eiq 'unresolved reference|black box|inferred latch|timing loop' \
        "${m416_run}/reports/check_design_postcompile.rpt" \
        "${m416_run}/reports/check_timing_postcompile.rpt"; then
    exit 35
fi

python3 - "${m416_run}" <<'PY'
import json
from pathlib import Path
import re
import sys

run = Path(sys.argv[1])
reports = run / "reports"
def first(pattern, path, cast=float):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {pattern!r} in {path}")
    return cast(match.group(1))

area = first(r"Total cell area:\s+([0-9.]+)", reports / "area.rpt")
cells = first(r"Number of cells:\s+([0-9]+)", reports / "area.rpt", int)
seq = first(r"Number of sequential cells:\s+([0-9]+)", reports / "area.rpt", int)
levels = first(r"Levels of Logic:\s+([0-9.]+)", reports / "qor.rpt")
setup = first(r"slack \(MET\)\s+([-0-9.]+)", reports / "timing_setup.rpt")
hold = first(r"slack \(MET\)\s+([-0-9.]+)", reports / "timing_hold.rpt")
if setup < 0.1 or hold < 0.0 or levels > 60:
    raise SystemExit(f"M416 robustness gate failed setup={setup} hold={hold} levels={levels}")

ref = (reports / "references_postcompile.rpt").read_text(errors="replace")
delay_count = 0
delay_area = 0.0
lines = ref.splitlines()
for index, line in enumerate(lines[:-1]):
    if line.startswith("DEL"):
        fields = lines[index + 1].split()
        if len(fields) >= 3:
            delay_count += int(fields[1])
            delay_area += float(fields[2])

old = {"cell_area_um2": 24885.377609, "cell_count": 21582,
       "sequential_cells": 4100, "logic_levels": 111.0,
       "setup_worst_slack_ns": 0.0008, "hold_worst_slack_ns": 0.0250}
new = {"cell_area_um2": area, "cell_count": cells,
       "sequential_cells": seq, "logic_levels": levels,
       "setup_worst_slack_ns": setup, "hold_worst_slack_ns": hold,
       "macro_count": 0, "delay_cell_count": delay_count,
       "delay_cell_area_um2": delay_area,
       "delay_cell_area_fraction": delay_area / area}
receipt = {
    "schema": "m416_m414_balanced_selected_slice_dc_receipt_v1",
    "status": "PASS_M416_BALANCED_SELECTED_SLICE_ROBUST_3NS_DC_SCREEN",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "m412_serial_reference": old,
    "m416_balanced": new,
    "delta": {
        "cell_area_um2": area - old["cell_area_um2"],
        "cell_area_fraction": area / old["cell_area_um2"] - 1.0,
        "cell_count": cells - old["cell_count"],
        "logic_levels": levels - old["logic_levels"],
        "setup_slack_ns": setup - old["setup_worst_slack_ns"],
    },
    "functional_context": {"full_runtime_rows": 51840000,
                           "m401_matcher_cycles": 67912100,
                           "task_ledger_change": 0,
                           "accuracy_change": False},
    "hold_guard_context": {"mapping_guard_ns": 0.025,
                           "delay_cells_causally_attributed_to_guard": False,
                           "guard_ab_required_for_exact_attribution": True},
    "claim_boundary": {"balanced_selected_slice_logic_only_dc": True,
                       "robust_3ns_pre_macro_screen": True,
                       "physical_sram": False, "physical_timing": False,
                       "formality": False, "primetime": False,
                       "saif_or_ptpx": False, "energy": False,
                       "new_cycle_speedup": False,
                       "system_speedup": False,
                       "paper_ppa_ready": False, "date_headline": False},
}
(run / "m416_m414_balanced_selected_slice_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

python3 - "${m416_run}/m416_m414_balanced_selected_slice_dc_receipt_r1.json" \
    "${m416_run}/RUN_COMPLETE.txt" <<'PY'
import json
from pathlib import Path
import sys
r = json.loads(Path(sys.argv[1]).read_text())
n = r["m416_balanced"]
d = r["delta"]
lines = [f"status={r['status']}", "tool=Synopsys_DC_V-2023.12-SP3",
         "clock_period_ns=3.000",
         f"cell_area_um2={n['cell_area_um2']}",
         f"cell_count={n['cell_count']}",
         f"sequential_cells={n['sequential_cells']}",
         f"logic_levels={n['logic_levels']}",
         f"setup_worst_slack_ns={n['setup_worst_slack_ns']}",
         f"hold_worst_slack_ns={n['hold_worst_slack_ns']}",
         f"area_delta_fraction_vs_m412={d['cell_area_fraction']}",
         f"logic_level_delta_vs_m412={d['logic_levels']}",
         "task_ledger_change=0", "macro_count=0", "formality=false",
         "primetime=false", "energy=false", "system_speedup=false",
         "paper_ppa_ready=false", "headline=false"]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY

sha256sum "${m416_runner}" >"${m416_run}/runner_sha256.txt"
find "${m416_run}" -type f \
    ! -name evidence_manifest.sha256 ! -name evidence_manifest.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m416_run}/evidence_manifest.sha256"
sha256sum "${m416_run}/evidence_manifest.sha256" \
    >"${m416_run}/evidence_manifest.seal.sha256"
m416_complete=1
echo "PASS_M416_BALANCED_SELECTED_SLICE_DC run=${m416_run}"
