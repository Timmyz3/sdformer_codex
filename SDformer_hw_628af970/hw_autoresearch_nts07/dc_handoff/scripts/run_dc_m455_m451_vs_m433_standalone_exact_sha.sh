#!/usr/bin/env bash
set -euo pipefail

m455_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m455_hw="$(cd "${m455_dc_root}/.." && pwd)"
m455_runner="$(realpath "${BASH_SOURCE[0]}")"
m455_run="${M455_DC_RUN:-${m455_dc_root}/runs/m455_m451_vs_m433_standalone_dc_3p000ns_r1_20260826}"
m455_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m455_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m455_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m455_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m455_fl="dc_handoff/filelists/date_m455_m451_k1_fused_adapter_rtl.f"
m455_sdc="dc_handoff/constraints/date_m439_pwp_adapter_3ns.sdc"
m455_contract="contracts/m455_m451_vs_m433_standalone_dc_contract_r1_20260826.json"

m455_sha() { sha256sum "$1" | awk '{print $1}'; }
m455_expect() { [[ -f "$1" && "$(m455_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m455_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then exit 4; fi
cd "${m455_hw}"
m455_expect "${m455_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m455_expect "${m455_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m455_expect "${m455_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m455_expect "${m455_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m455_expect "${m455_fl}" eb29883d7e9c9301ef753fbb650caa659d6d2ab4051def06a27eafdb23918cdd
m455_expect "${m455_sdc}" 565f486c7537484b0b6c11db7e53e4afc6962f2f73827a30764c3fe70bf3bb29
m455_expect "${m455_contract}" ca97c75c53299a60325be2fb16f3d411b3fe675a588a59eea9f28a48356e9444
m455_expect rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv b09172c5ca5c6fccddad0ccd19f37ffaae032cfe26350297f9ffcb3df65e2307
m455_expect results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 9b6fad46290411d90e9d28e40202981b64d8ccb178f607f23370ce213c6fd3e3
m455_expect results/m452_m451_independent_hammer_r1_20260826/RUN_MANIFEST.seal.sha256 13873fcd25dbe9b74bfd8095f2a13ac10115f037f052aa7b32b9b8d2ae16598e
m455_expect dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 98696f3bd166172aa294d2d24fb5d16f6fa7211a8da939fb99c035506d3eaa1a
m455_expect dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/m439_serial_vs_dualcoread_adapters_dc_receipt_r1.json 9f8d14bec581114e80886c172397b04043965f8c9930acaca742167927072133
m455_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m455_run}/candidate"
m455_complete=0
trap 'm455_rc=$?; if [[ ${m455_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m455_rc}" >"${m455_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

(cd results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826 && \
    sha256sum -c RUN_MANIFEST.sha256 && sha256sum -c RUN_MANIFEST.seal.sha256) \
    >"${m455_run}/upstream_seal_checks.log" 2>&1
(cd results/m452_m451_independent_hammer_r1_20260826 && \
    sha256sum -c RUN_MANIFEST.sha256 && sha256sum -c RUN_MANIFEST.seal.sha256) \
    >>"${m455_run}/upstream_seal_checks.log" 2>&1
(cd dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826 && \
    sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) \
    >>"${m455_run}/upstream_seal_checks.log" 2>&1

sha256sum "${m455_tcl}" "${m455_fl}" "${m455_sdc}" "${m455_contract}" \
    "${m455_slow}" "${m455_fast}" \
    rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv \
    results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
    results/m452_m451_independent_hammer_r1_20260826/RUN_MANIFEST.seal.sha256 \
    dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 \
    dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/m439_serial_vs_dualcoread_adapters_dc_receipt_r1.json \
    docs/359_DATE终局冻结_20260813.md >"${m455_run}/input_sha256.txt"
cp "${m455_contract}" "${m455_run}/contract.json"

export DESIGN_NAME=m451_exact_k1_fused_pwp_correction_adapter
export HW_ROOT="${m455_hw}"
export RTL_FILELIST="${m455_hw}/${m455_fl}"
export LIB_DB="${m455_slow}" MIN_LIB_DB="${m455_fast}"
export SDC_FILE="${m455_hw}/${m455_sdc}"
export OUTPUT_DIR="${m455_run}/candidate"
export OPERATING_CONDITION=ssg0p9v125c
set +e
"${m455_dc}" -f "${m455_hw}/${m455_tcl}" \
    >"${m455_run}/candidate/dc.log" 2>&1
m455_rc=$?
set -e
printf '%s\n' "${m455_rc}" >"${m455_run}/candidate/dc.rc"
[[ "${m455_rc}" -eq 0 ]] || exit 20
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' \
        "${m455_run}/candidate/dc.log"; then exit 21; fi
grep -Fq 'Thank you...' "${m455_run}/candidate/dc.log" || exit 22
for m455_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt hold_guard_contract.rpt; do
    [[ -s "${m455_run}/candidate/reports/${m455_report}" ]] || exit 23
done
[[ -s "${m455_run}/candidate/netlist/${DESIGN_NAME}_mapped.v" && \
   -s "${m455_run}/candidate/netlist/${DESIGN_NAME}_mapped.sdc" && \
   -s "${m455_run}/candidate/netlist/${DESIGN_NAME}.ddc" && \
   -s "${m455_run}/candidate/netlist/${DESIGN_NAME}.svf" ]] || exit 24
if grep -Fq 'slack (VIOLATED)' \
        "${m455_run}/candidate/reports/timing_setup.rpt" \
        "${m455_run}/candidate/reports/timing_hold.rpt"; then exit 25; fi
if grep -Eiq 'unresolved reference|black box|inferred latch|timing loop' \
        "${m455_run}/candidate/reports/check_design_postcompile.rpt" \
        "${m455_run}/candidate/reports/check_timing_postcompile.rpt"; then exit 26; fi

python3 - "${m455_run}" \
    dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/m439_serial_vs_dualcoread_adapters_dc_receipt_r1.json <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
reference = json.loads(Path(sys.argv[2]).read_text())["dual_coread"]
reports = run / "candidate" / "reports"

def first(pattern, path, cast=float):
    match = re.search(pattern, path.read_text(errors="replace"), re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {pattern!r} in {path}")
    return cast(match.group(1))

candidate = {
    "cell_area_um2": first(r"Total cell area:\s+([0-9.]+)", reports/"area.rpt"),
    "cell_count": first(r"Number of cells:\s+([0-9]+)", reports/"area.rpt", int),
    "sequential_cells": first(r"Number of sequential cells:\s+([0-9]+)", reports/"area.rpt", int),
    "logic_levels": first(r"Levels of Logic:\s+([0-9.]+)", reports/"qor.rpt"),
    "setup_worst_slack_ns": first(r"slack \(MET\)\s+([-0-9.]+)", reports/"timing_setup.rpt"),
    "hold_worst_slack_ns": first(r"slack \(MET\)\s+([-0-9.]+)", reports/"timing_hold.rpt"),
    "macro_count": 0,
}
if candidate["setup_worst_slack_ns"] < 0 or candidate["hold_worst_slack_ns"] < 0:
    raise SystemExit("negative candidate slack")
area_ratio = candidate["cell_area_um2"] / reference["cell_area_um2"]
ff_ratio = candidate["sequential_cells"] / reference["sequential_cells"]
opportunity = 517041352 / 430154216
receipt = {
    "schema": "m455_m451_vs_m433_standalone_dc_receipt_v1",
    "status": "PASS_M455_M451_STANDALONE_3NS_DC",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "reference_m433_from_sealed_m439": reference,
    "candidate_m451": candidate,
    "comparison": {
        "m451_to_m433_area_ratio": area_ratio,
        "m451_to_m433_area_delta_fraction": area_ratio - 1.0,
        "m451_to_m433_ff_ratio": ff_ratio,
        "m451_minus_m433_logic_levels": candidate["logic_levels"] - reference["logic_levels"],
        "cycle_opportunity_vs_m430": opportunity,
        "standalone_adapter_opportunity_throughput_per_area_ratio": opportunity / area_ratio,
        "diagnostic_only": True,
    },
    "claim_boundary": {
        "standalone_logic_only_dc": True, "macro_count": 0,
        "memory_port_concurrency_proven": False,
        "address_generation_present": False,
        "integrated_old_psum_proven": False,
        "cycle_speedup_admitted": False,
        "resource_normalized_speedup": False,
        "formality": False, "primetime": False,
        "power": False, "energy": False, "system_speedup": False,
        "paper_ppa_ready": False, "date_headline": False,
    },
}
(run/"m455_m451_vs_m433_standalone_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m455_runner}" >"${m455_run}/runner_sha256.txt"
printf '%s\n' PASS_M455_M451_STANDALONE_3NS_DC >"${m455_run}/RUN_COMPLETE.txt"
(
    cd "${m455_run}"
    find . -type f ! -name evidence_manifest.sha256 \
        ! -name evidence_manifest.seal.sha256 -print0 | sort -z | \
        xargs -0 sha256sum >evidence_manifest.sha256
    [[ "$(wc -l <evidence_manifest.sha256)" -ge 20 ]]
    ! grep -Eq '  -$' evidence_manifest.sha256
    sha256sum -c evidence_manifest.sha256 >/dev/null
    sha256sum evidence_manifest.sha256 >evidence_manifest.seal.sha256
    sha256sum -c evidence_manifest.seal.sha256 >/dev/null
)
m455_complete=1
echo "PASS M455 M451-vs-M433 standalone DC sealed at ${m455_run}"
