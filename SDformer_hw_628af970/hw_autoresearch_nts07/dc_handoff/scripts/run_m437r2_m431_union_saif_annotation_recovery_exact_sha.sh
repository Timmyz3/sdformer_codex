#!/usr/bin/env bash
set -euo pipefail

m437r2_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m437r2_hw="$(cd "${m437r2_dc_root}/.." && pwd)"
m437r2_runner="$(realpath "${BASH_SOURCE[0]}")"
m437r2_run="${M437R2_RUN_DIR:-${m437r2_dc_root}/runs/m437r2_m431_union_saif_annotation_recovery_r1_20260826}"
m437r2_source="${m437r2_dc_root}/runs/m431_m414_saif_tracked_dc_3p000ns_r1_20260826"
m437r2_failed="${m437r2_dc_root}/runs/m437_m431_full_saif_map_annotation_recovery_r1_20260826"
m437r2_saif_run="${m437r2_hw}/results/m425r4_h67_balanced_selected_slice_direct_saif_r4_20260826"
m437r2_hammer="${m437r2_hw}/results/m429_m425r4_saif_independent_hammer_r1_20260826"
m437r2_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m437r2_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m437r2_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m437r2_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m437r2_ddc="${m437r2_source}/netlist/m405_q32_elastic_selected_slice.ddc"
m437r2_binary_map="${m437r2_source}/netlist/m405_q32_elastic_selected_slice.saif_map.bin"
m437r2_netlist="${m437r2_source}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m437r2_sdc="${m437r2_source}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m437r2_saif="${m437r2_saif_run}/m405_q32_elastic_selected_slice_rtl.saif"
m437r2_export_tcl="${m437r2_hw}/dc_handoff/scripts/export_m437_m431_full_ptpx_saif_map.tcl"
m437r2_pt_tcl="${m437r2_hw}/dc_handoff/scripts/run_ptpx_m437r2_union_saif_annotation_diagnostic.tcl"
m437r2_contract="${m437r2_hw}/contracts/m437r2_m431_union_saif_map_annotation_recovery_contract_r1_20260826.json"

m437r2_sha() { sha256sum "$1" | awk '{print $1}'; }
m437r2_expect() { [[ -f "$1" && "$(m437r2_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m437r2_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 4; fi
m437r2_expect "${m437r2_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m437r2_expect "${m437r2_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m437r2_expect "${m437r2_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m437r2_expect "${m437r2_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m437r2_expect "${m437r2_ddc}" fe32aae0f705c410e55c5191a8771bb19d195c03e736d548b104f20457b4502a
m437r2_expect "${m437r2_binary_map}" 7d485acb60dc8bfa71d971111176b95631c34e989a976d549d1d9a4e02114f0c
m437r2_expect "${m437r2_netlist}" dd42a5b82d2e1782f5511d93405b8f6aad94ab3eaf47fb1a615eeee60a5e6723
m437r2_expect "${m437r2_sdc}" f0c30f09360f71ab8d801943f1c0c372a3596ffb43b3506925c860c54b38708f
m437r2_expect "${m437r2_source}/evidence_manifest.seal.sha256" 1562ab70877b853ba87d5fe35c1c52d61dcbc4eb3610f9a01c0d6e43e675d05c
m437r2_expect "${m437r2_saif}" 32f24bd5d0e663bf23c70aefa232e5cfe94dea5bad3762b92c801b9dff737dcd
m437r2_expect "${m437r2_saif_run}/RUN_MANIFEST.seal.sha256" b702148541e783b29fdcb3df632a06eb3fa481e86f58b3b6509d0bcd17e205da
m437r2_expect "${m437r2_hammer}/SHA256SUMS.seal.sha256" 06496b718f116ad1e1d1c84bda095f319fc9b10b9bcf3b554e042e585c87fa33
m437r2_expect "${m437r2_export_tcl}" 385bbab31abd7cd11b4952ff7f2ccda607cc5f1f6c1bdb01415c6b4e883d518e
m437r2_expect "${m437r2_pt_tcl}" dc5d0629030a45f7febebbcd08cb7c5b9c14c505059d607aeede74355471d8e3
m437r2_expect "${m437r2_contract}" f7b9af81c57f5ce33b046c9cb0294e9196503baba2c3be009c4002a73bf8dbfe
m437r2_expect "${m437r2_failed}/RUN_FAILED_OR_INCOMPLETE.txt" 0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466
m437r2_expect "${m437r2_failed}/dc_export.log" 47bafb4dacb48e4f909fc13b7fed0a0101d732df97bbf7985033b90814afa288
m437r2_expect "${m437r2_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m437r2_run}/work" "${m437r2_run}/reports" "${m437r2_run}/netlist"
m437r2_complete=0
trap 'm437r2_rc=$?; if [[ ${m437r2_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m437r2_rc}" >"${m437r2_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
(cd "${m437r2_source}" && sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) >"${m437r2_run}/upstream_seal_checks.log" 2>&1
(cd "${m437r2_saif_run}" && sha256sum -c RUN_MANIFEST.sha256 && \
    sha256sum -c RUN_MANIFEST.seal.sha256) >>"${m437r2_run}/upstream_seal_checks.log" 2>&1
(cd "${m437r2_hammer}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) >>"${m437r2_run}/upstream_seal_checks.log" 2>&1
sha256sum "${m437r2_dc}" "${m437r2_pt}" "${m437r2_slow}" \
    "${m437r2_fast}" "${m437r2_ddc}" "${m437r2_binary_map}" \
    "${m437r2_netlist}" "${m437r2_sdc}" "${m437r2_saif}" \
    "${m437r2_export_tcl}" "${m437r2_pt_tcl}" "${m437r2_contract}" \
    "${m437r2_failed}/RUN_FAILED_OR_INCOMPLETE.txt" \
    "${m437r2_failed}/dc_export.log" \
    "${m437r2_hw}/docs/359_DATE终局冻结_20260813.md" \
    >"${m437r2_run}/input_sha256.txt"
cp "${m437r2_contract}" "${m437r2_run}/contract.json"

export DESIGN_NAME=m405_q32_elastic_selected_slice
export LIB_DB="${m437r2_slow}" MIN_LIB_DB="${m437r2_fast}"
export DDC_FILE="${m437r2_ddc}" BINARY_SAIF_MAP="${m437r2_binary_map}"
export OUTPUT_DIR="${m437r2_run}"
set +e
(cd "${m437r2_run}/work" && "${m437r2_dc}" -f "${m437r2_export_tcl}") \
    >"${m437r2_run}/dc_export.log" 2>&1
m437r2_dc_rc=$?
set -e
printf '%s\n' "${m437r2_dc_rc}" >"${m437r2_run}/dc_export.rc"
[[ "${m437r2_dc_rc}" -eq 0 ]] || exit 20
if grep -Eq '^Error:|^Fatal:' "${m437r2_run}/dc_export.log"; then exit 21; fi
m437r2_reg_map="${m437r2_run}/netlist/m405_q32_elastic_selected_slice.ptpx_saif_map.full.tcl"
m437r2_essential_map="${m437r2_run}/netlist/m405_q32_elastic_selected_slice.ptpx_saif_map.essential.tcl"
[[ -s "${m437r2_reg_map}" && -s "${m437r2_essential_map}" ]] || exit 22
python3 - "${m437r2_reg_map}" "${m437r2_essential_map}" \
    "${m437r2_run}/mapping_class_audit.json" <<'PY'
import json
import sys
from pathlib import Path

def commands(path):
    return {line for line in Path(path).read_text(errors="replace").splitlines()
            if line.startswith("set_rtl_to_gate_name")}
registers = commands(sys.argv[1])
essential = commands(sys.argv[2])
audit = {"default_sequential_entries": len(registers),
         "essential_entries": len(essential),
         "intersection_entries": len(registers & essential),
         "union_entries": len(registers | essential)}
if audit != {"default_sequential_entries": 4100,
             "essential_entries": 7035,
             "intersection_entries": 0,
             "union_entries": 11135}:
    raise SystemExit(f"M437R2 mapping class drift: {audit}")
Path(sys.argv[3]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
PY

export MAPPED_NETLIST="${m437r2_netlist}" MAPPED_SDC="${m437r2_sdc}"
export ESSENTIAL_RTL_GATE_MAP_TCL="${m437r2_essential_map}"
export REGISTER_RTL_GATE_MAP_TCL="${m437r2_reg_map}"
export SAIF_FILE="${m437r2_saif}" OPERATING_CONDITION=ssg0p9v125c
export SAIF_INSTANCE=tb_m425_h67_balanced_selected_slice_direct_saif/dut
set +e
(cd "${m437r2_run}/work" && "${m437r2_pt}" -f "${m437r2_pt_tcl}") \
    >"${m437r2_run}/pt_annotation.log" 2>&1
m437r2_pt_rc=$?
set -e
printf '%s\n' "${m437r2_pt_rc}" >"${m437r2_run}/pt_annotation.rc"
[[ "${m437r2_pt_rc}" -eq 0 ]] || exit 30
if grep -Eq '^Error:|^Fatal:' "${m437r2_run}/pt_annotation.log"; then exit 31; fi
grep -Fqx 'M437R2_UNION_SAIF_ANNOTATION_INTERNAL_COMPLETE=PASS' \
    "${m437r2_run}/PTPX_UNION_ANNOTATION_INTERNAL_COMPLETE.txt" || exit 32

python3 - "${m437r2_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
log = (run / "pt_annotation.log").read_text(errors="replace")
coverage_text = (run / "reports/switching_coverage.rpt").read_text(errors="replace")
annotated_match = re.search(
    r"Total number of nets = (\d+).*?Number of annotated nets = (\d+) \(([0-9.]+)%\)",
    log, re.DOTALL)
coverage_match = re.search(
    r"^m405_q32_elastic_selected_slice\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$",
    coverage_text, re.MULTILINE)
if not annotated_match or not coverage_match:
    raise SystemExit("M437R2 could not parse PrimeTime annotation/coverage")
total = int(annotated_match.group(1))
annotated = int(annotated_match.group(2))
annotated_pct = float(annotated_match.group(3))
coverage_pct = float(coverage_match.group(1))
covered = int(coverage_match.group(2))
coverage_total = int(coverage_match.group(3))
if total != 22800 or coverage_total != total:
    raise SystemExit("M437R2 mapped net population drift")
passes = annotated_pct >= 95.0 and coverage_pct >= 95.0
status = ("PASS_M437R2_UNION_ANNOTATION_AT_LEAST_95" if passes else
          "COMPLETE_M437R2_UNION_ANNOTATION_BELOW_95_NO_GO_POWER")
receipt = {
    "schema": "m437r2_m431_union_saif_annotation_recovery_receipt_v1",
    "status": status,
    "mapping": json.loads((run / "mapping_class_audit.json").read_text()),
    "primetime_annotation": {
        "total_nets": total, "annotated_nets": annotated,
        "annotated_percent": annotated_pct,
        "covered_nets_at_least_one_toggle": covered,
        "switching_coverage_percent": coverage_pct,
        "minimum_required_percent": 95.0, "passes": passes,
    },
    "decision": ("GO_SEPARATE_PTPX_AFTER_INDEPENDENT_REVIEW" if passes else
                 "NO_GO_POWER_ENERGY_PROCEED_TO_GATE_LEVEL_SAIF"),
    "claim_boundary": {
        "annotation_diagnostic": True, "update_power_called": False,
        "report_power_called": False, "power": False, "energy": False,
        "system_speedup": False, "paper_ppa_ready": False,
        "headline": False,
    },
}
(run / "m437r2_m431_union_saif_annotation_recovery_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(run / "RUN_COMPLETE.txt").write_text(status + "\n")
PY

sha256sum "${m437r2_runner}" >"${m437r2_run}/runner_sha256.txt"
find "${m437r2_run}" -type f \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m437r2_run}/RUN_MANIFEST.sha256"
sha256sum "${m437r2_run}/RUN_MANIFEST.sha256" \
    >"${m437r2_run}/RUN_MANIFEST.seal.sha256"
m437r2_complete=1
echo "M437R2 union-map annotation recovery complete at ${m437r2_run}"
