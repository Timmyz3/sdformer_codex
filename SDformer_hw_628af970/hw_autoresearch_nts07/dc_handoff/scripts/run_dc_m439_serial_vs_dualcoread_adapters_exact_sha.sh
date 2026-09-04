#!/usr/bin/env bash
set -euo pipefail

m439_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m439_hw="$(cd "${m439_dc_root}/.." && pwd)"
m439_runner="$(realpath "${BASH_SOURCE[0]}")"
m439_run="${M439_DC_RUN:-${m439_dc_root}/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826}"
m439_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m439_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m439_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m439_tcl="dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl"
m439_serial_fl="dc_handoff/filelists/date_m439_m405_serial_adapter_rtl.f"
m439_dual_fl="dc_handoff/filelists/date_m439_m433_dualcoread_adapter_rtl.f"
m439_sdc="dc_handoff/constraints/date_m439_pwp_adapter_3ns.sdc"
m439_contract="contracts/m439_m405_serial_vs_m433_dualcoread_adapter_dc_contract_r1_20260826.json"

m439_sha() { sha256sum "$1" | awk '{print $1}'; }
m439_expect() { [[ -f "$1" && "$(m439_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m439_run}" ]] || exit 5
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null; then exit 4; fi
cd "${m439_hw}"
m439_expect "${m439_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m439_expect "${m439_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m439_expect "${m439_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m439_expect "${m439_tcl}" b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f
m439_expect "${m439_serial_fl}" 1d89d09f1589dbd30815ea8c2fd985d1bb0e29d94066a9337ca50592a066684c
m439_expect "${m439_dual_fl}" 99f23e242a5eff6526e2118c6811b3b09493ce43bbc01d0b91e0ea9be7e19303
m439_expect "${m439_sdc}" 565f486c7537484b0b6c11db7e53e4afc6962f2f73827a30764c3fe70bf3bb29
m439_expect "${m439_contract}" f59a58be539a734b04fbbb8f4de9cdb4f7f33661cf02fd4ed49938ea1782698a
m439_expect rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv 819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744
m439_expect rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
m439_expect results/m433_exact_dualbank_coread_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 d57308dcabd40945f827fa0dfba0f18c7374f5d710722d2121e1084cd5b6d375
m439_expect reviews/m434_m433_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 5b271829327c5ae5632b42b44c22a37515f022426217663fac9a7e9001455aa1
m439_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m439_run}"
m439_complete=0
trap 'm439_rc=$?; if [[ ${m439_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m439_rc}" >"${m439_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
(cd results/m433_exact_dualbank_coread_directed_vcs_r1_20260826 && \
    sha256sum -c RUN_MANIFEST.sha256 && sha256sum -c RUN_MANIFEST.seal.sha256) \
    >"${m439_run}/upstream_seal_checks.log" 2>&1
(cd reviews/m434_m433_independent_hammer_r1_20260826 && \
    sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m439_run}/upstream_seal_checks.log" 2>&1
sha256sum "${m439_tcl}" "${m439_serial_fl}" "${m439_dual_fl}" \
    "${m439_sdc}" "${m439_contract}" "${m439_slow}" "${m439_fast}" \
    rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv \
    rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv \
    results/m433_exact_dualbank_coread_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
    reviews/m434_m433_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m439_run}/input_sha256.txt"
cp "${m439_contract}" "${m439_run}/contract.json"

m439_run_one() {
    local m439_label=$1
    local m439_top=$2
    local m439_filelist=$3
    local m439_out="${m439_run}/${m439_label}"
    mkdir -p "${m439_out}"
    export DESIGN_NAME="${m439_top}" HW_ROOT="${m439_hw}"
    export RTL_FILELIST="${m439_hw}/${m439_filelist}"
    export LIB_DB="${m439_slow}" MIN_LIB_DB="${m439_fast}"
    export SDC_FILE="${m439_hw}/${m439_sdc}" OUTPUT_DIR="${m439_out}"
    export OPERATING_CONDITION=ssg0p9v125c
    set +e
    "${m439_dc}" -f "${m439_hw}/${m439_tcl}" >"${m439_out}/dc.log" 2>&1
    local m439_rc=$?
    set -e
    printf '%s\n' "${m439_rc}" >"${m439_out}/dc.rc"
    [[ "${m439_rc}" -eq 0 ]] || return 20
    if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "${m439_out}/dc.log"; then return 21; fi
    grep -Fq 'Thank you...' "${m439_out}/dc.log" || return 22
    local m439_report
    for m439_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt hold_guard_contract.rpt; do
        [[ -s "${m439_out}/reports/${m439_report}" ]] || return 23
    done
    [[ -s "${m439_out}/netlist/${m439_top}_mapped.v" && \
       -s "${m439_out}/netlist/${m439_top}_mapped.sdc" && \
       -s "${m439_out}/netlist/${m439_top}.ddc" && \
       -s "${m439_out}/netlist/${m439_top}.svf" ]] || return 24
    if grep -Fq 'slack (VIOLATED)' "${m439_out}/reports/timing_setup.rpt" \
            "${m439_out}/reports/timing_hold.rpt"; then return 25; fi
    if grep -Eiq 'unresolved reference|black box|inferred latch|timing loop' \
            "${m439_out}/reports/check_design_postcompile.rpt" \
            "${m439_out}/reports/check_timing_postcompile.rpt"; then return 26; fi
}

m439_run_one serial m405_exact_elastic_pwp_issue_adapter "${m439_serial_fl}"
m439_run_one dual_coread m433_exact_dualbank_coread_pwp_adapter "${m439_dual_fl}"

python3 - "${m439_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
def first(pattern, path, cast=float):
    match = re.search(pattern, path.read_text(errors="replace"), re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {pattern!r} in {path}")
    return cast(match.group(1))
def collect(label):
    reports = run / label / "reports"
    point = {
        "cell_area_um2": first(r"Total cell area:\s+([0-9.]+)", reports/"area.rpt"),
        "cell_count": first(r"Number of cells:\s+([0-9]+)", reports/"area.rpt", int),
        "sequential_cells": first(r"Number of sequential cells:\s+([0-9]+)", reports/"area.rpt", int),
        "logic_levels": first(r"Levels of Logic:\s+([0-9.]+)", reports/"qor.rpt"),
        "setup_worst_slack_ns": first(r"slack \(MET\)\s+([-0-9.]+)", reports/"timing_setup.rpt"),
        "hold_worst_slack_ns": first(r"slack \(MET\)\s+([-0-9.]+)", reports/"timing_hold.rpt"),
        "macro_count": 0,
    }
    if point["setup_worst_slack_ns"] < 0 or point["hold_worst_slack_ns"] < 0:
        raise SystemExit(f"negative slack in {label}")
    return point
serial = collect("serial")
dual = collect("dual_coread")
area_ratio = dual["cell_area_um2"] / serial["cell_area_um2"]
receipt = {
    "schema": "m439_m405_serial_vs_m433_dualcoread_adapter_dc_receipt_v1",
    "status": "PASS_M439_BOTH_STANDALONE_ADAPTERS_3NS_DC",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "technology": "TSMC28 HPC+ standard cells",
    "clock_period_ns": 3.0,
    "serial": serial,
    "dual_coread": dual,
    "comparison": {
        "dual_to_serial_area_ratio": area_ratio,
        "dual_to_serial_area_delta_fraction": area_ratio - 1.0,
        "dual_to_serial_ff_ratio": dual["sequential_cells"] / serial["sequential_cells"],
        "wide_block_raw_throughput_ratio": 2.0,
        "wide_block_throughput_per_area_ratio": 2.0 / area_ratio,
        "logical_peak_input_bandwidth_ratio": 1.5,
        "physical_peak_input_bandwidth_ratio": 160.0 / 96.0,
    },
    "m430_context": {
        "cycles": 517041352,
        "speedup_vs_strong_zero": 1.435375301,
        "scope": "four H67 bottleneck Conv3x3 only",
        "dc_does_not_readmit_or_upgrade_cycle_claim": True,
    },
    "claim_boundary": {
        "standalone_logic_only_dc": True, "macro_count": 0,
        "formality": False, "primetime": False,
        "power": False, "energy": False, "system_speedup": False,
        "paper_ppa_ready": False, "headline": False,
    },
}
(run/"m439_serial_vs_dualcoread_adapters_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m439_runner}" >"${m439_run}/runner_sha256.txt"
printf '%s\n' PASS_M439_BOTH_STANDALONE_ADAPTERS_3NS_DC >"${m439_run}/RUN_COMPLETE.txt"
find "${m439_run}" -type f \
    ! -name evidence_manifest.sha256 ! -name evidence_manifest.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m439_run}/evidence_manifest.sha256"
sha256sum "${m439_run}/evidence_manifest.sha256" >"${m439_run}/evidence_manifest.seal.sha256"
m439_complete=1
echo "PASS M439 serial-vs-dual adapter DC sealed at ${m439_run}"
