#!/usr/bin/env bash
set -euo pipefail

m412_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m412_hw="$(cd "${m412_dc_root}/.." && pwd)"
m412_runner="$(realpath "${BASH_SOURCE[0]}")"
m412_run="${M412_DC_RUN:-${m412_dc_root}/runs/m412_dual_standalone_logic_only_dc_3p000ns_r1_20260826}"
m412_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m412_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m412_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m412_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m412_contract="contracts/m412_dual_standalone_logic_only_dc_contract_r1_20260826.json"

m412_sha() { sha256sum "$1" | awk '{print $1}'; }
m412_expect() {
    local m412_path=$1
    local m412_expected=$2
    [[ -f "${m412_path}" ]] || exit 3
    [[ "$(m412_sha "${m412_path}")" == "${m412_expected}" ]] || exit 3
}

[[ ! -e "${m412_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m412_hw}"

m412_expect "${m412_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m412_expect "${m412_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m412_expect "${m412_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m412_expect "${m412_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m412_expect "${m412_contract}" 30064df765a0a58163c4ce9eae22c87e78af557a7647c4ff56839878380acab6
m412_expect rtl_m405/m405_q32_serial16_zero_stop_controller.sv f412ab817eb29ab303da9ec011379a853efc567fa5a6a29a73fced52cf503b42
m412_expect rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv 819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744
m412_expect rtl_m405/m405_q32_elastic_selected_slice.sv 91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd
m412_expect rtl_m384/m384_active_descriptor_streaming_controller.sv 7a93b60b327d7a92fb19028d754e3d5ed444c91c5a8d8a7ddd50ce03bb679512
m412_expect dc_handoff/filelists/date_m412_m405_selected_slice_rtl.f e0ef128a2ae9e351ecd98c45c19e9706e983cb9cef82913600febf48dae0f58e
m412_expect dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_rtl.f c3db231e355357c138247c0c76a0352d80d5574a863988fb9af2746be9c37467
m412_expect dc_handoff/constraints/date_m412_m405_selected_slice_3ns.sdc 565f486c7537484b0b6c11db7e53e4afc6962f2f73827a30764c3fe70bf3bb29
m412_expect dc_handoff/constraints/date_m384_active_descriptor_streaming_controller_3ns.sdc 25939ff975096245a2f696a2a22f0f555eab340f3857d0fcd5aa897dedbbe866
m412_expect results/m405r3_selected_slice_integration_vcs_r1_20260826/m405r3_selected_slice_integration_vcs_receipt_r1.json aabfb0863cb39c5457e4a02e253b467e24f39f3121f030aaf6bd520851c2ac61
m412_expect results/m405r3_selected_slice_integration_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 7a6ded4969a4afb8263129b5282fefcf642b0951142ea9bb2174fdcc34c67be3
m412_expect results/m407_m405r3_integration_independent_hammer_r1_20260826/m407_m405r3_integration_independent_hammer_review_r1.json af279c4d7cc07d8517cbf72fb12ccf4600b66609493af0cda35cb1251b2285e6
m412_expect results/m407_m405r3_integration_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 d2ecf11f0b6fd0a710e961350329b4c728a033816259a32777ef0bb1b0f40fbf
m412_expect results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/m410r2_h67_q32_full_runtime_vcs_receipt_r2.json ef25a15655ad1ce1aec57a35125e973ec7e9fa3ca78df2a327b810ef5ca09a04
m412_expect results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.seal.sha256 bac41c1f8fe14c3250323659e3c5ef02848c55a3e0c28caadc97774e2529f1b6
m412_expect results/m411_m410r2_full_runtime_vcs_independent_hammer_r1_20260826/m411_m410r2_full_runtime_vcs_independent_hammer_review_r1.json 64e43aa0b7424ca55f751f4162cd7ba9da3571e10ab7f69458748de885fc51c7
m412_expect results/m411_m410r2_full_runtime_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 6bbcedb292a0d22ace98eeb969ed903d0cf0bd2f348a815d2a8b1a3bf95a68e2
m412_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m412_run}"
m412_complete=0
trap 'm412_rc=$?; if [[ ${m412_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m412_rc}" >"${m412_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
{
    sha256sum \
        rtl_m405/m405_q32_serial16_zero_stop_controller.sv \
        rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv \
        rtl_m405/m405_q32_elastic_selected_slice.sv \
        rtl_m384/m384_active_descriptor_streaming_controller.sv \
        dc_handoff/filelists/date_m412_m405_selected_slice_rtl.f \
        dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_rtl.f \
        dc_handoff/constraints/date_m412_m405_selected_slice_3ns.sdc \
        dc_handoff/constraints/date_m384_active_descriptor_streaming_controller_3ns.sdc \
        "${m412_tcl}" "${m412_contract}" "${m412_slow}" "${m412_fast}" \
        results/m405r3_selected_slice_integration_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
        results/m407_m405r3_integration_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
        results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.seal.sha256 \
        results/m411_m410r2_full_runtime_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
        docs/359_DATE终局冻结_20260813.md
} >"${m412_run}/input_sha256.txt"
cp "${m412_contract}" "${m412_run}/contract.json"

m412_run_one() {
    local m412_label=$1
    local m412_top=$2
    local m412_filelist=$3
    local m412_sdc=$4
    local m412_out="${m412_run}/${m412_label}"
    mkdir -p "${m412_out}"
    export DESIGN_NAME="${m412_top}"
    export HW_ROOT="${m412_hw}"
    export RTL_FILELIST="${m412_hw}/${m412_filelist}"
    export LIB_DB="${m412_slow}"
    export MIN_LIB_DB="${m412_fast}"
    export SDC_FILE="${m412_hw}/${m412_sdc}"
    export OUTPUT_DIR="${m412_out}"
    export OPERATING_CONDITION=ssg0p9v125c

    set +e
    "${m412_dc}" -f "${m412_hw}/${m412_tcl}" >"${m412_out}/dc.log" 2>&1
    local m412_rc=$?
    set -e
    echo "${m412_rc}" >"${m412_out}/dc.rc"
    [[ "${m412_rc}" -eq 0 ]]
    ! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m412_out}/dc.log"
    grep -Fq 'Thank you...' "${m412_out}/dc.log"
    local m412_report
    for m412_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt resources_postcompile.rpt \
            references_postcompile.rpt hold_guard_contract.rpt; do
        [[ -s "${m412_out}/reports/${m412_report}" ]] || return 30
    done
    [[ -s "${m412_out}/netlist/${m412_top}_mapped.v" &&
       -s "${m412_out}/netlist/${m412_top}_mapped.sdc" &&
       -s "${m412_out}/netlist/${m412_top}.ddc" &&
       -s "${m412_out}/netlist/${m412_top}.svf" ]] || return 31
    if grep -Fq 'slack (VIOLATED)' "${m412_out}/reports/timing_setup.rpt" \
            "${m412_out}/reports/timing_hold.rpt"; then
        return 32
    fi
    [[ "$(grep -Fc 'This design has no violated constraints.' \
        "${m412_out}/reports/constraint_violators.rpt")" -eq 5 ]] || return 33
    grep -qx 'additional_hold_guard_ns=0.025' \
        "${m412_out}/reports/hold_guard_contract.rpt"
    grep -Eq 'set_clock_uncertainty( +-hold)? +0\.1(00)? ' \
        "${m412_out}/netlist/${m412_top}_mapped.sdc"
    if grep -Eiq 'unresolved reference|black box|inferred latch|timing loop' \
            "${m412_out}/reports/check_design_postcompile.rpt" \
            "${m412_out}/reports/check_timing_postcompile.rpt"; then
        return 35
    fi
}

m412_run_one selected_slice m405_q32_elastic_selected_slice \
    dc_handoff/filelists/date_m412_m405_selected_slice_rtl.f \
    dc_handoff/constraints/date_m412_m405_selected_slice_3ns.sdc
m412_run_one descriptor_controller m384_active_descriptor_streaming_controller \
    dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_rtl.f \
    dc_handoff/constraints/date_m384_active_descriptor_streaming_controller_3ns.sdc

python3 - "${m412_run}" <<'PY'
import json
from pathlib import Path
import re
import sys

run = Path(sys.argv[1])

def first(pattern, path, cast=float):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {pattern!r} in {path}")
    return cast(match.group(1))

def collect(label):
    root = run / label / "reports"
    area = first(r"Total cell area:\s+([0-9.]+)", root / "area.rpt")
    cells = first(r"Number of cells:\s+([0-9]+)", root / "area.rpt", int)
    sequential = first(r"Number of sequential cells:\s+([0-9]+)", root / "area.rpt", int)
    levels = first(r"Levels of Logic:\s+([0-9.]+)", root / "qor.rpt")
    setup = first(r"slack \(MET\)\s+([-0-9.]+)", root / "timing_setup.rpt")
    hold = first(r"slack \(MET\)\s+([-0-9.]+)", root / "timing_hold.rpt")
    if not (area > 0 and setup >= 0 and hold >= 0):
        raise SystemExit(f"failed numeric gate for {label}")
    return {
        "cell_area_um2": area,
        "cell_count": cells,
        "sequential_cells": sequential,
        "logic_levels": levels,
        "setup_worst_slack_ns": setup,
        "hold_worst_slack_ns": hold,
        "macro_count": 0,
    }

selected = collect("selected_slice")
controller = collect("descriptor_controller")
receipt = {
    "schema": "m412_dual_standalone_logic_only_dc_receipt_v1",
    "status": "PASS_M412_BOTH_STANDALONE_LOGIC_ONLY_DC_3NS",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "selected_slice": selected,
    "descriptor_controller": controller,
    "arithmetic_area_sum_um2": selected["cell_area_um2"] + controller["cell_area_um2"],
    "area_sum_is_physical_integration": False,
    "macro_count": 0,
    "metric_context": {
        "m401_four_h67_bottleneck_conv_trace_cycle_speedup_vs_strong_baseline": 1.1563713549830412,
        "speedup_upgraded_by_m412": False,
    },
    "claim_boundary": {
        "selected_slice_logic_only_dc": True,
        "q32_controller_logic_only_dc": True,
        "separate_synthesis": True,
        "physical_integration": False,
        "physical_sram": False,
        "physical_timing": False,
        "formality": False,
        "primetime": False,
        "saif_or_ptpx": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(run / "m412_dual_standalone_logic_only_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python3 - "${m412_run}/m412_dual_standalone_logic_only_dc_receipt_r1.json" \
    "${m412_run}/RUN_COMPLETE.txt" <<'PY'
import json
from pathlib import Path
import sys
r = json.loads(Path(sys.argv[1]).read_text())
lines = [
    f"status={r['status']}",
    "tool=Synopsys_DC_V-2023.12-SP3",
    "clock_period_ns=3.000",
]
for label in ("selected_slice", "descriptor_controller"):
    point = r[label]
    for key in ("cell_area_um2", "cell_count", "sequential_cells", "logic_levels", "setup_worst_slack_ns", "hold_worst_slack_ns"):
        lines.append(f"{label}_{key}={point[key]}")
lines += [
    f"arithmetic_area_sum_um2={r['arithmetic_area_sum_um2']}",
    "area_sum_is_physical_integration=false",
    "macro_count=0",
    "physical_sram=false",
    "formality=false",
    "primetime=false",
    "system_speedup=false",
    "paper_ppa_ready=false",
    "headline=false",
]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY

sha256sum "${m412_runner}" >"${m412_run}/runner_sha256.txt"
find "${m412_run}" -type f \
    ! -name evidence_manifest.sha256 ! -name evidence_manifest.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m412_run}/evidence_manifest.sha256"
sha256sum "${m412_run}/evidence_manifest.sha256" \
    >"${m412_run}/evidence_manifest.seal.sha256"
m412_complete=1
echo "PASS_M412_DUAL_STANDALONE_LOGIC_ONLY_DC run=${m412_run}"
