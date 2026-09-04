#!/usr/bin/env bash
set -euo pipefail

m387_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m387_hw="$(cd "${m387_dc_root}/.." && pwd)"
m387_runner="$(realpath "${BASH_SOURCE[0]}")"
m387_run="${M387_DC_RUN:-${m387_dc_root}/runs/m387_m384_active_descriptor_controller_dc_3p000ns_r1b_20260826}"
m387_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m387_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m387_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m387_filelist="dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_rtl.f"
m387_sdc="dc_handoff/constraints/date_m384_active_descriptor_streaming_controller_3ns.sdc"
m387_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m387_contract="contracts/m387_m384_active_descriptor_controller_logic_only_dc_contract_r1_20260826.json"

m387_sha() { sha256sum "$1" | awk '{print $1}'; }
m387_expect() {
    local m387_path=$1
    local m387_expected=$2
    [[ -f "${m387_path}" ]] || exit 3
    [[ "$(m387_sha "${m387_path}")" == "${m387_expected}" ]] || exit 3
}

[[ ! -e "${m387_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then
    exit 4
fi
cd "${m387_hw}"
m387_expect "${m387_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m387_expect "${m387_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m387_expect "${m387_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m387_expect "rtl_m384/m384_active_descriptor_streaming_controller.sv" 15f0e1d8aebfcb66ed58cefed988bde855a8b2a351e32c86beb2381a8c4e6b38
m387_expect "${m387_filelist}" c3db231e355357c138247c0c76a0352d80d5574a863988fb9af2746be9c37467
m387_expect "${m387_sdc}" 25939ff975096245a2f696a2a22f0f555eab340f3857d0fcd5aa897dedbbe866
m387_expect "${m387_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m387_expect "${m387_contract}" c8a8c57dbe847919efa5dcd712d402e73de3bcd9c3ebcbd8e8a66ae72bca79e2
m387_expect "contracts/m384_active_descriptor_streaming_controller_directed_vcs_contract_r1_20260826.json" 7dc11d0ddb090768f89bcef397dd8d2520a23eac38d22804067249251bf1bec9
m387_expect "results/m384_active_descriptor_streaming_controller_vcs_r1b_20260826/m384_active_descriptor_streaming_controller_vcs_receipt_r1b.json" f1c357ddf34ae3ee6c4ea21b527b89b7ae84a709c2da70d98b7f1920a9e09677
m387_expect "results/m384_active_descriptor_streaming_controller_vcs_r1b_20260826/RUN_MANIFEST.seal.sha256" bf639636b808d40b567c6a07614c54eb2303c6f3e828da290db92435335f9d54
m387_expect "results/m385_m384_active_descriptor_streaming_controller_independent_hammer_r1_20260826/m385_m384_independent_hammer_review_r1.json" 70765825cf15cbc74db300b9ea97c317daade5b1dec8940e7aacf4bd38173232
m387_expect "results/m385_m384_active_descriptor_streaming_controller_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256" 376716b4de1e2367f8312efdf36365e387584eaa237c78a0e72476027beff22f
m387_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m387_run}"
m387_complete=0
trap 'm387_rc=$?; if [[ ${m387_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m387_rc}" >"${m387_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
{
    sha256sum \
        rtl_m384/m384_active_descriptor_streaming_controller.sv \
        "${m387_filelist}" "${m387_sdc}" "${m387_tcl}" \
        "${m387_contract}" "${m387_slow}" "${m387_fast}" \
        results/m384_active_descriptor_streaming_controller_vcs_r1b_20260826/RUN_MANIFEST.seal.sha256 \
        results/m385_m384_active_descriptor_streaming_controller_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
        docs/359_DATE终局冻结_20260813.md
} >"${m387_run}/input_sha256.txt"
cp "${m387_contract}" "${m387_run}/contract.json"

export DESIGN_NAME=m384_active_descriptor_streaming_controller
export HW_ROOT="${m387_hw}"
export RTL_FILELIST="${m387_hw}/${m387_filelist}"
export LIB_DB="${m387_slow}"
export MIN_LIB_DB="${m387_fast}"
export SDC_FILE="${m387_hw}/${m387_sdc}"
export OUTPUT_DIR="${m387_run}"
export OPERATING_CONDITION=ssg0p9v125c

set +e
"${m387_dc}" -f "${m387_hw}/${m387_tcl}" >"${m387_run}/dc.log" 2>&1
m387_rc=$?
set -e
echo "${m387_rc}" >"${m387_run}/dc.rc"
[[ "${m387_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m387_run}/dc.log"
grep -Fq 'Thank you...' "${m387_run}/dc.log"
for m387_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt hold_guard_contract.rpt; do
    [[ -s "${m387_run}/reports/${m387_report}" ]] || exit 30
done
[[ -s "${m387_run}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m387_run}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m387_run}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m387_run}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m387_run}/reports/timing_setup.rpt" \
    "${m387_run}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m387_run}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33
grep -qx 'additional_hold_guard_ns=0.025' \
    "${m387_run}/reports/hold_guard_contract.rpt"
grep -Eq 'set_clock_uncertainty( +-hold)? +0\.1(00)? ' \
    "${m387_run}/netlist/${DESIGN_NAME}_mapped.sdc"
grep -Eiq 'unresolved reference|black box|inferred latch|timing loop' \
    "${m387_run}/reports/check_design_postcompile.rpt" \
    "${m387_run}/reports/check_timing_postcompile.rpt" && exit 35 || true

m387_area=$(awk '/Total cell area:/ {print $4; exit}' "${m387_run}/reports/area.rpt")
m387_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m387_run}/reports/area.rpt")
m387_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m387_run}/reports/area.rpt")
m387_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m387_run}/reports/qor.rpt")
m387_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m387_run}/reports/timing_setup.rpt")
m387_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m387_run}/reports/timing_hold.rpt")
for m387_value in "${m387_area}" "${m387_cells}" "${m387_seq}" \
        "${m387_levels}" "${m387_setup}" "${m387_hold}"; do
    [[ -n "${m387_value}" ]] || exit 34
done
awk -v x="${m387_area}" 'BEGIN {exit !(x > 0)}'
awk -v x="${m387_setup}" 'BEGIN {exit !(x >= 0.0)}'
awk -v x="${m387_hold}" 'BEGIN {exit !(x >= 0.0)}'

python3 - "${m387_run}" "${m387_area}" "${m387_cells}" \
    "${m387_seq}" "${m387_levels}" "${m387_setup}" "${m387_hold}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
area = float(sys.argv[2])
cells = int(sys.argv[3])
seq = int(sys.argv[4])
levels = float(sys.argv[5])
setup = float(sys.argv[6])
hold = float(sys.argv[7])
receipt = {
    "schema": "m387_m384_active_descriptor_controller_logic_only_dc_receipt_r1b",
    "status": "PASS_M387_R1B_M384_CONTROLLER_LOGIC_ONLY_DC_3NS",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "cell_area_um2": area,
    "cell_count": cells,
    "sequential_cells": seq,
    "logic_levels": levels,
    "setup_worst_slack_ns": setup,
    "hold_worst_slack_ns": hold,
    "macro_count": 0,
    "architecture_boundary": {
        "phase_center_register_bits": 512,
        "descriptor_response_fifo_entries": 8,
        "descriptor_response_fifo_payload_bits": 48,
        "external_descriptor_sram": True,
        "external_pwp_sram": True,
        "q32_matcher_integrated": False,
        "pwp_compute_integrated": False,
    },
    "performance_context": {
        "m381_four_bottleneck_conv_module_speedup": 1.076382876808849,
        "m381_speedup_upgraded_by_dc": False,
        "rtl_to_frozen_17280_phase_cycle_match": False,
    },
    "claim_boundary": {
        "isolated_controller_logic_only_dc": True,
        "physical_descriptor_sram": False,
        "physical_pwp_sram": False,
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
(run / "m387_m384_active_descriptor_controller_logic_only_dc_receipt_r1b.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
    echo "status=PASS_M387_R1B_M384_CONTROLLER_LOGIC_ONLY_DC_3NS"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "cell_area_um2=${m387_area}"
    echo "cell_count=${m387_cells}"
    echo "sequential_cells=${m387_seq}"
    echo "logic_levels=${m387_levels}"
    echo "setup_worst_slack_ns=${m387_setup}"
    echo "hold_worst_slack_ns=${m387_hold}"
    echo "synthesis_hold_guard_ns=0.025"
    echo "publication_hold_uncertainty_ns=0.100"
    echo "macro_count=0"
    echo "physical_descriptor_sram=false"
    echo "physical_pwp_sram=false"
    echo "physical_timing=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} >"${m387_run}/RUN_COMPLETE.txt"
sha256sum "${m387_run}"/dc.log "${m387_run}"/reports/*.rpt \
    "${m387_run}"/netlist/* \
    "${m387_run}"/m387_m384_active_descriptor_controller_logic_only_dc_receipt_r1b.json \
    "${m387_run}"/RUN_COMPLETE.txt >"${m387_run}/evidence_manifest.sha256"
sha256sum "${m387_run}/evidence_manifest.sha256" \
    >"${m387_run}/evidence_manifest.seal.sha256"
sha256sum "${m387_runner}" >"${m387_run}/runner_sha256.txt"
m387_complete=1
echo "PASS_M387_M384_CONTROLLER_LOGIC_ONLY_DC run=${m387_run}"
