#!/usr/bin/env bash
set -euo pipefail

m437_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m437_hw="$(cd "${m437_dc_root}/.." && pwd)"
m437_runner="$(realpath "${BASH_SOURCE[0]}")"
m437_run="${M437_RUN_DIR:-${m437_dc_root}/runs/m437_m431_full_saif_map_annotation_recovery_r1_20260826}"
m437_source="${m437_dc_root}/runs/m431_m414_saif_tracked_dc_3p000ns_r1_20260826"
m437_saif_run="${m437_hw}/results/m425r4_h67_balanced_selected_slice_direct_saif_r4_20260826"
m437_hammer="${m437_hw}/results/m429_m425r4_saif_independent_hammer_r1_20260826"
m437_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m437_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m437_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m437_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m437_ddc="${m437_source}/netlist/m405_q32_elastic_selected_slice.ddc"
m437_binary_map="${m437_source}/netlist/m405_q32_elastic_selected_slice.saif_map.bin"
m437_netlist="${m437_source}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m437_sdc="${m437_source}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m437_saif="${m437_saif_run}/m405_q32_elastic_selected_slice_rtl.saif"
m437_export_tcl="${m437_hw}/dc_handoff/scripts/export_m437_m431_full_ptpx_saif_map.tcl"
m437_pt_tcl="${m437_hw}/dc_handoff/scripts/run_ptpx_m432_m431_m425_saif_annotation_diagnostic.tcl"
m437_contract="${m437_hw}/contracts/m437_m431_full_saif_map_annotation_recovery_contract_r1_20260826.json"

m437_sha() { sha256sum "$1" | awk '{print $1}'; }
m437_expect() { [[ -f "$1" && "$(m437_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m437_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then
    exit 4
fi
m437_expect "${m437_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m437_expect "${m437_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m437_expect "${m437_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m437_expect "${m437_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m437_expect "${m437_ddc}" fe32aae0f705c410e55c5191a8771bb19d195c03e736d548b104f20457b4502a
m437_expect "${m437_binary_map}" 7d485acb60dc8bfa71d971111176b95631c34e989a976d549d1d9a4e02114f0c
m437_expect "${m437_netlist}" dd42a5b82d2e1782f5511d93405b8f6aad94ab3eaf47fb1a615eeee60a5e6723
m437_expect "${m437_sdc}" f0c30f09360f71ab8d801943f1c0c372a3596ffb43b3506925c860c54b38708f
m437_expect "${m437_source}/evidence_manifest.seal.sha256" 1562ab70877b853ba87d5fe35c1c52d61dcbc4eb3610f9a01c0d6e43e675d05c
m437_expect "${m437_saif}" 32f24bd5d0e663bf23c70aefa232e5cfe94dea5bad3762b92c801b9dff737dcd
m437_expect "${m437_saif_run}/RUN_MANIFEST.seal.sha256" b702148541e783b29fdcb3df632a06eb3fa481e86f58b3b6509d0bcd17e205da
m437_expect "${m437_hammer}/SHA256SUMS.seal.sha256" 06496b718f116ad1e1d1c84bda095f319fc9b10b9bcf3b554e042e585c87fa33
m437_expect "${m437_export_tcl}" 385bbab31abd7cd11b4952ff7f2ccda607cc5f1f6c1bdb01415c6b4e883d518e
m437_expect "${m437_pt_tcl}" 5780bc068cd2082e970cdd323ca55f08fd960dd271221633fa46ac6c37492867
m437_expect "${m437_contract}" 63f980259c1fe59adbed39707dbe7090c396c0f9658f1514cc1ab35fd2d97fa1
m437_expect "${m437_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m437_run}/work" "${m437_run}/reports" "${m437_run}/netlist"
m437_complete=0
trap 'm437_rc=$?; if [[ ${m437_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m437_rc}" >"${m437_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
(cd "${m437_source}" && sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) >"${m437_run}/upstream_seal_checks.log" 2>&1
(cd "${m437_saif_run}" && sha256sum -c RUN_MANIFEST.sha256 && \
    sha256sum -c RUN_MANIFEST.seal.sha256) >>"${m437_run}/upstream_seal_checks.log" 2>&1
(cd "${m437_hammer}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) >>"${m437_run}/upstream_seal_checks.log" 2>&1
sha256sum "${m437_dc}" "${m437_pt}" "${m437_slow}" "${m437_fast}" \
    "${m437_ddc}" "${m437_binary_map}" "${m437_netlist}" "${m437_sdc}" \
    "${m437_saif}" "${m437_export_tcl}" "${m437_pt_tcl}" \
    "${m437_contract}" "${m437_source}/evidence_manifest.seal.sha256" \
    "${m437_saif_run}/RUN_MANIFEST.seal.sha256" \
    "${m437_hammer}/SHA256SUMS.seal.sha256" \
    "${m437_hw}/docs/359_DATE终局冻结_20260813.md" \
    >"${m437_run}/input_sha256.txt"
cp "${m437_contract}" "${m437_run}/contract.json"

export DESIGN_NAME=m405_q32_elastic_selected_slice
export LIB_DB="${m437_slow}" MIN_LIB_DB="${m437_fast}"
export DDC_FILE="${m437_ddc}" BINARY_SAIF_MAP="${m437_binary_map}"
export OUTPUT_DIR="${m437_run}"
set +e
(cd "${m437_run}/work" && "${m437_dc}" -f "${m437_export_tcl}") \
    >"${m437_run}/dc_export.log" 2>&1
m437_dc_rc=$?
set -e
printf '%s\n' "${m437_dc_rc}" >"${m437_run}/dc_export.rc"
[[ "${m437_dc_rc}" -eq 0 ]] || exit 20
if grep -Eq '^Error:|^Fatal:' "${m437_run}/dc_export.log"; then exit 21; fi
m437_full_map="${m437_run}/netlist/m405_q32_elastic_selected_slice.ptpx_saif_map.full.tcl"
m437_essential_map="${m437_run}/netlist/m405_q32_elastic_selected_slice.ptpx_saif_map.essential.tcl"
[[ -s "${m437_full_map}" && -s "${m437_essential_map}" ]] || exit 22
m437_full_count="$(grep -c '^set_rtl_to_gate_name' "${m437_full_map}")"
m437_essential_count="$(grep -c '^set_rtl_to_gate_name' "${m437_essential_map}")"
[[ "${m437_full_count}" -ge "${m437_essential_count}" && \
   "${m437_essential_count}" -eq 7035 ]] || exit 23

export MAPPED_NETLIST="${m437_netlist}" MAPPED_SDC="${m437_sdc}"
export RTL_GATE_MAP_TCL="${m437_full_map}" SAIF_FILE="${m437_saif}"
export OPERATING_CONDITION=ssg0p9v125c
export SAIF_INSTANCE=tb_m425_h67_balanced_selected_slice_direct_saif/dut
set +e
(cd "${m437_run}/work" && "${m437_pt}" -f "${m437_pt_tcl}") \
    >"${m437_run}/pt_annotation.log" 2>&1
m437_pt_rc=$?
set -e
printf '%s\n' "${m437_pt_rc}" >"${m437_run}/pt_annotation.rc"
[[ "${m437_pt_rc}" -eq 0 ]] || exit 30
if grep -Eq '^Error:|^Fatal:' "${m437_run}/pt_annotation.log"; then exit 31; fi
grep -Fqx 'M432_M431_M425_SAIF_ANNOTATION_DIAGNOSTIC_INTERNAL_COMPLETE=PASS' \
    "${m437_run}/PTPX_ANNOTATION_DIAGNOSTIC_INTERNAL_COMPLETE.txt" || exit 32

python3 - "${m437_run}" "${m437_full_count}" "${m437_essential_count}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
full_count = int(sys.argv[2])
essential_count = int(sys.argv[3])
log = (run / "pt_annotation.log").read_text(errors="replace")
coverage_text = (run / "reports/switching_coverage.rpt").read_text(errors="replace")
annotated_match = re.search(
    r"Total number of nets = (\d+).*?Number of annotated nets = (\d+) \(([0-9.]+)%\)",
    log, re.DOTALL)
coverage_match = re.search(
    r"^m405_q32_elastic_selected_slice\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$",
    coverage_text, re.MULTILINE)
if not annotated_match or not coverage_match:
    raise SystemExit("M437 could not parse PrimeTime annotation/coverage")
total_nets = int(annotated_match.group(1))
annotated_nets = int(annotated_match.group(2))
annotated_percent = float(annotated_match.group(3))
coverage_percent = float(coverage_match.group(1))
covered_nets = int(coverage_match.group(2))
coverage_total = int(coverage_match.group(3))
if total_nets != 22800 or coverage_total != total_nets:
    raise SystemExit("M437 mapped net population drift")
passes = coverage_percent >= 95.0 and annotated_percent >= 95.0
status = ("PASS_M437_FULL_MAP_ANNOTATION_AT_LEAST_95" if passes else
          "COMPLETE_M437_FULL_MAP_ANNOTATION_BELOW_95_NO_GO_POWER")
receipt = {
    "schema": "m437_m431_full_saif_map_annotation_recovery_receipt_v1",
    "status": status,
    "mapping": {
        "method": "M431 sealed binary saif_map to PT-PX full map",
        "essential_map_entries": essential_count,
        "full_map_entries": full_count,
    },
    "primetime_annotation": {
        "total_nets": total_nets,
        "annotated_nets": annotated_nets,
        "annotated_percent": annotated_percent,
        "covered_nets_at_least_one_toggle": covered_nets,
        "switching_coverage_percent": coverage_percent,
        "minimum_required_percent": 95.0,
        "passes": passes,
    },
    "decision": ("GO_SEPARATE_PTPX_AFTER_INDEPENDENT_REVIEW" if passes else
                 "NO_GO_POWER_ENERGY_PROCEED_TO_GATE_LEVEL_SAIF"),
    "claim_boundary": {
        "annotation_diagnostic": True,
        "update_power_called": False,
        "report_power_called": False,
        "power": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(run / "m437_m431_full_saif_map_annotation_recovery_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(run / "RUN_COMPLETE.txt").write_text(status + "\n")
PY

sha256sum "${m437_runner}" >"${m437_run}/runner_sha256.txt"
find "${m437_run}" -type f \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m437_run}/RUN_MANIFEST.sha256"
sha256sum "${m437_run}/RUN_MANIFEST.sha256" \
    >"${m437_run}/RUN_MANIFEST.seal.sha256"
m437_complete=1
echo "M437 full-map annotation recovery complete at ${m437_run}"
