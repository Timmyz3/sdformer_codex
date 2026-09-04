#!/usr/bin/env bash
set -euo pipefail

m448r2_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m448r2_hw="$(cd "${m448r2_dc_root}/.." && pwd)"
m448r2_runner="$(realpath "${BASH_SOURCE[0]}")"
m448r2_run="${M448R2_RUN_DIR:-${m448r2_dc_root}/runs/m448r2_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r2_20260826}"
m448r2_m431="${m448r2_dc_root}/runs/m431_m414_saif_tracked_dc_3p000ns_r1_20260826"
m448r2_m438="${m448r2_dc_root}/runs/m438_m431_direct_mapped_gate_saif_r1_20260826"
m448r2_m446="${m448r2_hw}/reviews/m446_m438_gate_saif_independent_hammer_r1_20260826"
m448r2_r1="${m448r2_dc_root}/runs/m448_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r1_20260826"
m448r2_diag_a="${m448r2_dc_root}/runs/m448_tt_power_ramp_diagnostic_a_frozen_sdc_r1_20260826"
m448r2_diag_b="${m448r2_dc_root}/runs/m448_tt_power_ramp_diagnostic_b_input100ps_r1_20260826"
m448r2_diag_c="${m448r2_dc_root}/runs/m448_tt_power_ramp_diagnostic_c_nonclock100ps_r1_20260826"
m448r2_contract="contracts/m448r2_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_contract_r2_20260826.json"
m448r2_tcl="dc_handoff/scripts/run_ptpx_m448r2_m431_m438_prelayout_stdcell_tt0p9v25c.tcl"
m448r2_slew_source="dc_handoff/constraints/date_m54_m66_core_ab_3ns.sdc"
m448r2_netlist="${m448r2_m431}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m448r2_sdc="${m448r2_m431}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m448r2_saif="${m448r2_m438}/m405_q32_elastic_selected_slice_mapped_gate.saif"
m448r2_tt="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db"
m448r2_ss="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m448r2_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"

m448r2_sha() { sha256sum "$1" | awk '{print $1}'; }

[[ ! -e "${m448r2_run}" ]] || exit 2
mkdir -p "${m448r2_run}/work" "${m448r2_run}/reports"
m448r2_complete=0
trap 'm448r2_rc=$?; if [[ ${m448r2_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m448r2_rc}" >"${m448r2_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m448r2_hw}"

declare -A m448r2_expected=(
    ["${m448r2_contract}"]="29f879a39f7887ed2aa1c335a9de26e0734a9649ca048a9365e85e035adc3e64"
    ["${m448r2_tcl}"]="6307be74fe8da77ada183f702c9faf7850f4a5af58f36a7e95d0877a186ef159"
    ["${m448r2_slew_source}"]="1ccc9313f8fd4cb866aadf7ba8a77ae0290822ccff7dfe06a120d5fecccd78f9"
    ["${m448r2_netlist}"]="dd42a5b82d2e1782f5511d93405b8f6aad94ab3eaf47fb1a615eeee60a5e6723"
    ["${m448r2_sdc}"]="f0c30f09360f71ab8d801943f1c0c372a3596ffb43b3506925c860c54b38708f"
    ["${m448r2_m431}/evidence_manifest.seal.sha256"]="1562ab70877b853ba87d5fe35c1c52d61dcbc4eb3610f9a01c0d6e43e675d05c"
    ["${m448r2_saif}"]="3d95d2aa51e1455b5d420330c96dee61f58bbc58d33133a54b6b71cb77814d4b"
    ["${m448r2_m438}/m438_m431_direct_mapped_gate_saif_receipt_r1.json"]="be35985a43fea45c747bb8b9cd514aacee0c92ffc673f82e0a26737e802113b5"
    ["${m448r2_m438}/RUN_MANIFEST.seal.sha256"]="bfc05d234e2cae16e0a6aa26b1016dbf0a1a504380e2c6d59c21ca10a2443fd8"
    ["${m448r2_m446}/m446_m438_gate_saif_independent_review_r1.json"]="a0c1107225e62ef53ea73a66a0d958aa1e577199bd424aeb121d131a9019438e"
    ["${m448r2_m446}/SHA256SUMS.seal.sha256"]="d170d4db5728cbe6f31fad95a302f8cf97634cca65cd5cf1d4af3ec2fe8e395c"
    ["${m448r2_r1}/RUN_FAILED_OR_INCOMPLETE.txt"]="69f54defb365a7f6594d18a5f67d3c277fdaa852358c20dd914fb9d19b3d3a20"
    ["${m448r2_r1}/ptpx.rc"]="9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
    ["${m448r2_diag_a}/check_power_verbose.rpt"]="3dfbd30cb72ceecbff820cdaca76e4e12adb820015da1fc280b55380377208b4"
    ["${m448r2_diag_b}/check_power_verbose.rpt"]="83ec2f52a3edbde192bac408b5e0eafe4af84effe8d68bfa6740749bcd1b84ea"
    ["${m448r2_diag_c}/check_power_verbose.rpt"]="0f076ea70f3e891d9a3f1a53066f7d1954cfc6ee2e1148c36f665cb7e82bb6c9"
    ["${m448r2_diag_c}/u10556_input_net.rpt"]="c5c8ace5bf5cf5bcc8f646d4678084074d98ee3fcdee6928e279e5e64c2b044d"
    ["${m448r2_tt}"]="d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070"
    ["${m448r2_ss}"]="79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
    ["${m448r2_pt}"]="afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: >"${m448r2_run}/preflight_sha_checks.txt"
for m448r2_path in "${!m448r2_expected[@]}"; do
    m448r2_observed="$(m448r2_sha "${m448r2_path}")"
    printf 'path=%s expected=%s observed=%s\n' "${m448r2_path}" \
        "${m448r2_expected[${m448r2_path}]}" "${m448r2_observed}" \
        >>"${m448r2_run}/preflight_sha_checks.txt"
    [[ "${m448r2_observed}" == "${m448r2_expected[${m448r2_path}]}" ]] || exit 10
done

(cd "${m448r2_m431}" && sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) \
    >"${m448r2_run}/upstream_seal_checks.log" 2>&1
(cd "${m448r2_m438}" && sha256sum -c RUN_MANIFEST.sha256 && \
    sha256sum -c RUN_MANIFEST.seal.sha256) \
    >>"${m448r2_run}/upstream_seal_checks.log" 2>&1
(cd "${m448r2_m446}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m448r2_run}/upstream_seal_checks.log" 2>&1

python3 - "${m448r2_contract}" \
    "${m448r2_m438}/m438_m431_direct_mapped_gate_saif_receipt_r1.json" \
    "${m448r2_m446}/m446_m438_gate_saif_independent_review_r1.json" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text())
m438 = json.loads(Path(sys.argv[2]).read_text())
m446 = json.loads(Path(sys.argv[3]).read_text())
if contract["status_before_execution"] != "FROZEN_BEFORE_SYNOPSYS_PRIMETIME_PX_RERUN":
    raise SystemExit("M448R2 contract is not frozen")
if contract["supersession"]["status"] != "FAILED_OR_INCOMPLETE_DO_NOT_CITE":
    raise SystemExit("M448 r1 failure is not preserved")
if contract["input_slew_contract"]["nonclock_primary_inputs"] != 1666:
    raise SystemExit("M448R2 nonclock input population is not frozen")
if m438["decision"] != "GO_SEPARATE_PTPX_AFTER_INDEPENDENT_REVIEW":
    raise SystemExit("M438 does not admit separate PTPX")
if m446["verdict"] != "GO_SEPARATE_PRELAYOUT_PTPX_WITH_SCOPED_CLAIMS":
    raise SystemExit("M446 does not admit scoped PTPX")
if m446["severity_counts"]["P0"] != 0:
    raise SystemExit("M446 has P0 findings")
PY

python3 - "${m448r2_saif}" "${m448r2_run}/saif_preflight_receipt_r2.json" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

saif = Path(sys.argv[1])
out = Path(sys.argv[2])
text = saif.read_text(errors="strict")
if '(TIMESCALE 1 ns)' not in text:
    raise SystemExit("M448R2 SAIF timescale is not 1 ns")
duration_match = re.search(r"\(DURATION\s+([0-9.]+)\)", text)
if not duration_match or float(duration_match.group(1)) != 6288008.5:
    raise SystemExit("M448R2 SAIF duration drift")
required_scope = (
    "(INSTANCE tb_m425_h67_balanced_selected_slice_direct_saif\n"
    "   (INSTANCE dut\n"
    "      (INSTANCE u_gate\n"
    "         (NET")
if required_scope not in text:
    raise SystemExit("M448R2 exact gate-only SAIF scope missing")
entry_re = re.compile(
    r"\(([^()\s]+)\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+"
    r"\(TX\s+(\d+)\)\s+\(TC\s+(\d+)\)", re.MULTILINE)
entries = [(name, int(t0), int(t1), int(tx), int(tc))
           for name, t0, t1, tx, tc in entry_re.findall(text)]
duration = float(duration_match.group(1))
nonzero = sum(tc > 0 for _, _, _, _, tc in entries)
tx_nonzero = sum(tx > 0 for _, _, _, tx, _ in entries)
tx_total = sum(tx for _, _, _, tx, _ in entries)
if len(entries) != 22800 or nonzero != 21827 or tx_nonzero != 0 or tx_total != 0:
    raise SystemExit("M448R2 SAIF population/TX gate failed")
if any(abs((t0 + t1 + tx) - duration) > 0.5 for _, t0, t1, tx, _ in entries):
    raise SystemExit("M448R2 SAIF residence-time inconsistency")
clocks = [e for e in entries if e[0] == "clk_core"]
if clocks != [("clk_core", 3144004, 3144005, 0, 4192005)]:
    raise SystemExit("M448R2 clock activity drift")
receipt = {
    "schema": "date.m448r2_saif_preflight_receipt.v2",
    "status": "PASS_BEFORE_PRIMETIME_POWER",
    "sha256": hashlib.sha256(saif.read_bytes()).hexdigest(),
    "scope": "tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate",
    "timescale": "1 ns", "duration_ns": duration,
    "measurement_cycles": 2096003, "entries": len(entries),
    "nonzero_toggle_entries": nonzero,
    "nonzero_toggle_coverage_percent": nonzero / len(entries) * 100.0,
    "nonzero_tx_entries": tx_nonzero, "total_tx_duration_ns": tx_total,
    "clock": {"name": "clk_core", "t0_ns": clocks[0][1],
              "t1_ns": clocks[0][2], "tx_ns": clocks[0][3],
              "toggle_count": clocks[0][4]},
}
out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${!m448r2_expected[@]}" >"${m448r2_run}/input_sha256.txt"
cp "${m448r2_contract}" "${m448r2_run}/contract.json"
sha256sum "${m448r2_runner}" >"${m448r2_run}/runner_sha256.txt"

if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 20; fi
export DESIGN_NAME=m405_q32_elastic_selected_slice
export TT_LIB_DB="${m448r2_tt}" SDC_LIB_DB="${m448r2_ss}"
export MAPPED_NETLIST="${m448r2_netlist}" MAPPED_SDC="${m448r2_sdc}"
export GATE_SAIF_FILE="${m448r2_saif}" OUTPUT_DIR="${m448r2_run}"
export POWER_OPERATING_CONDITION=tt0p9v25c
export POWER_LIBRARY_NAME=tcbn28hpcplusbwp35p140tt0p9v25c
export SAIF_INSTANCE=tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate
export SAIF_DURATION_NS=6288008.5 MEASUREMENT_CYCLES=2096003
export SAIF_TX_NONZERO_ENTRIES=0
set +e
(cd "${m448r2_run}/work" && "${m448r2_pt}" -f "${m448r2_hw}/${m448r2_tcl}") \
    >"${m448r2_run}/ptpx.log" 2>&1
m448r2_rc=$?
set -e
printf '%s\n' "${m448r2_rc}" >"${m448r2_run}/ptpx.rc"
[[ ${m448r2_rc} -eq 0 ]] || exit 21
if grep -Eq '^Error:|^Fatal:' "${m448r2_run}/ptpx.log"; then exit 22; fi
grep -Fqx 'M448R2_PTPX_POWER_GATE_PASS_PRE_UPDATE=PASS' \
    "${m448r2_run}/PTPX_POWER_GATE_PASS_PRE_UPDATE.txt" || exit 23
grep -Fqx 'M448R2_M431_M438_PRELAYOUT_STDCELL_PTPX_INTERNAL_COMPLETE=PASS' \
    "${m448r2_run}/PTPX_INTERNAL_COMPLETE.txt" || exit 24
[[ "$(grep -Ec '^update_power([[:space:]]|$)' "${m448r2_run}/ptpx.log")" -eq 3 ]] || exit 25
[[ "$(grep -Ec '^report_power([[:space:]]|$)' "${m448r2_run}/ptpx.log")" -eq 6 ]] || exit 26

python3 - "${m448r2_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
saif_gate = (run / "reports/saif_annotation_summary.rpt").read_text(errors="replace")
coverage_text = (run / "reports/switching_coverage.rpt").read_text(errors="replace")
verbose_text = (run / "reports/ptpx_power_primary_100ps_verbose.rpt").read_text(errors="replace")
hier_text = (run / "reports/ptpx_power_primary_100ps_hierarchy.rpt").read_text(errors="replace")
clock_text = (run / "reports/ptpx_clock.rpt").read_text(errors="replace")
ptlog = (run / "ptpx.log").read_text(errors="replace")

ann = re.search(
    r"Total number of nets = (\d+).*?Number of annotated nets = (\d+) \(([0-9.]+)%\).*?"
    r"Total number of leaf cells = (\d+).*?Number of fully annotated leaf cells = (\d+) \(([0-9.]+)%\)",
    saif_gate, re.S)
cov = re.search(
    r"^m405_q32_elastic_selected_slice\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$",
    coverage_text, re.M)
if not ann or not cov:
    raise SystemExit("M448R2 cannot parse annotation gates")
annotation_tuple = (int(ann.group(1)), int(ann.group(2)), float(ann.group(3)),
                    int(ann.group(4)), int(ann.group(5)), float(ann.group(6)),
                    int(cov.group(2)), int(cov.group(3)))
if annotation_tuple != (22800, 22800, 100.0, 20803, 20803, 100.0, 21827, 22800):
    raise SystemExit("M448R2 annotation population drift")
nonzero_percent = float(cov.group(1))
if nonzero_percent < 95.0:
    raise SystemExit("M448R2 nonzero coverage below 95 percent")

for label in ("primary_100ps", "sensitivity_050ps", "sensitivity_200ps"):
    check = (run / f"reports/ptpx_check_power_{label}_pre_update.rpt").read_text(errors="replace")
    if "check_power succeeded." not in check:
        raise SystemExit(f"M448R2 {label} check_power did not succeed")
    if re.search(r"Warning:|out_of_range|out of ramp range", check):
        raise SystemExit(f"M448R2 {label} check_power warning")

if not re.search(r"Dynamic Power Units\s*=\s*1 mW", verbose_text):
    raise SystemExit("M448R2 dynamic power unit is not mW")
if not re.search(r"Leakage Power Units\s*=\s*1 mW", verbose_text):
    raise SystemExit("M448R2 leakage power unit is not mW")
if "Operating Conditions: tt0p9v25c" not in verbose_text:
    raise SystemExit("M448R2 TT operating condition missing")
if "tcbn28hpcplusbwp35p140tt0p9v25c" not in verbose_text:
    raise SystemExit("M448R2 TT library missing")
if not re.search(r"core_clk\s+3\.00\s+\{0 1\.5\}", clock_text):
    raise SystemExit("M448R2 3 ns core clock missing")
if "m405_q32_elastic_selected_slice" not in hier_text:
    raise SystemExit("M448R2 hierarchy report missing top")
if re.search(r"unresolved reference|could not resolve", ptlog, re.I):
    raise SystemExit("M448R2 unresolved reference")

def parse_power(label):
    text = (run / f"reports/ptpx_power_{label}.rpt").read_text(errors="replace")
    def value(name):
        match = re.search(rf"{re.escape(name)}\s*=\s*([0-9.eE+-]+)", text)
        if not match:
            raise SystemExit(f"M448R2 {label} missing {name}")
        return float(match.group(1))
    result = {
        "internal": value("Cell Internal Power"),
        "net_switching": value("Net Switching Power"),
        "leakage": value("Cell Leakage Power"),
        "total": value("Total Power"),
    }
    if min(result.values()) < 0:
        raise SystemExit(f"M448R2 {label} negative power")
    component_sum = result["internal"] + result["net_switching"] + result["leakage"]
    if abs(result["total"] - component_sum) > max(1e-8, result["total"] * 2e-5):
        raise SystemExit(f"M448R2 {label} component sum mismatch")
    return result

p050 = parse_power("sensitivity_050ps")
p100 = parse_power("primary_100ps")
p200 = parse_power("sensitivity_200ps")
duration_ns = 6288008.5
measurement_cycles = 2096003
ns_per_cycle = duration_ns / measurement_cycles
def energy(power_mw):
    return power_mw * ns_per_cycle

clock_group = re.search(
    r"^clock_network\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)",
    (run / "reports/ptpx_power_primary_100ps.rpt").read_text(errors="replace"), re.M)
if not clock_group:
    raise SystemExit("M448R2 clock-network group missing")

receipt = {
    "schema": "date.m448r2_m431_m438_prelayout_stdcell_ptpx_receipt.v2",
    "status": "PASS_M448R2_PRELAYOUT_STDCELL_SELECTED_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER",
    "tool": "Synopsys PrimeTime PX W-2024.09-SP3",
    "analysis": {
        "mode": "averaged", "corner": "tt0p9v25c",
        "library": "tcbn28hpcplusbwp35p140tt0p9v25c",
        "voltage_v": 0.9, "temperature_c": 25,
        "clock_period_ns": 3.0, "clock_frequency_mhz": 333.333333333,
        "clock_network": "ideal_no_cts",
        "wireload": "ZeroWireload", "spef": False, "macro_count": 0,
        "power_units": "mW",
    },
    "input_slew": {
        "nonclock_primary_inputs": 1666, "includes_reset_n": True,
        "primary_ps": 100, "sensitivity_only_ps": [50, 200],
        "all_three_check_power_out_of_range_ramps": 0,
        "all_three_check_power_missing_tables": 0,
        "all_three_check_power_missing_functions": 0,
        "reset_signoff": False,
    },
    "activity": {
        "scope": "tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate",
        "saif_duration_ns": duration_ns, "measurement_cycles": measurement_cycles,
        "effective_ns_per_measured_cycle": ns_per_cycle,
        "total_nets": 22800, "exact_annotated_nets": 22800,
        "exact_annotation_percent": 100.0,
        "total_leaf_cells": 20803, "fully_annotated_leaf_cells": 20803,
        "fully_annotated_leaf_percent": 100.0,
        "nonzero_toggle_nets": 21827,
        "nonzero_toggle_coverage_percent_reported": nonzero_percent,
        "nonzero_toggle_coverage_percent_exact": 21827 / 22800 * 100.0,
        "nonzero_tx_entries": 0,
    },
    "primary_100ps_prelayout_standard_cell_power_mw": p100,
    "primary_100ps_prelayout_standard_cell_energy_per_measured_cycle_pj": {
        **{key: energy(value) for key, value in p100.items()},
        "derivation": "power_mW * saif_duration_ns / measurement_cycles; 1 mW*ns = 1 pJ",
    },
    "input_slew_sensitivity_total_power_mw": {
        "50ps": p050["total"], "100ps_primary": p100["total"],
        "200ps": p200["total"],
        "50ps_vs_primary_ratio": p050["total"] / p100["total"],
        "200ps_vs_primary_ratio": p200["total"] / p100["total"],
        "max_abs_delta_vs_primary_percent": max(
            abs(p050["total"] / p100["total"] - 1),
            abs(p200["total"] / p100["total"] - 1)) * 100.0,
    },
    "primary_100ps_power_group_clock_network_mw": {
        "internal": float(clock_group.group(1)),
        "switching": float(clock_group.group(2)),
        "leakage": float(clock_group.group(3)),
        "total": float(clock_group.group(4)),
        "boundary": "includes register clock-pin internal power; no CTS buffers or extracted clock interconnect are modeled",
    },
    "population": {"phases": 64, "source_rows": 192000,
                   "pwp_rows": 63067, "contributions": 921166,
                   "reconstructed_lanes": 48435456},
    "claim_boundary": {
        "prelayout_standard_cell_m416_selected_slice_power": True,
        "prelayout_standard_cell_m416_selected_slice_energy_per_measured_cycle": True,
        "input_slew_sensitivity": True, "reset_signoff": False,
        "sram_power": False, "macro_power": False,
        "extracted_interconnect_power": False, "full_conv_power": False,
        "full_network_power": False, "system_energy": False,
        "system_speedup": False, "paper_ppa_ready": False,
        "headline": False, "pending_independent_hammer": True,
    },
}
(run / "m448r2_m431_m438_prelayout_stdcell_ptpx_receipt_r2.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")

md = "# M448R2 M431/M438 prelayout standard-cell PTPX\n\n"
md += "Status: **PASS, pending independent hammer.** Scope is the M416 balanced selected slice only.\n\n"
md += "- Corner: TT 0.9 V, 25 C; ideal 3.0 ns clock (333.333333 MHz), ZeroWireload, no SPEF, 0 macro.\n"
md += "- Input slew: 100 ps primary on all 1,666 nonclock inputs including reset_n; 50/200 ps are sensitivity only. All three check_power runs have 0 ramp/missing-table/missing-function findings.\n"
md += f"- Activity: 64 phases / 192,000 rows / {measurement_cycles:,} measured cycles / {duration_ns:,.1f} ns; 22,800/22,800 exact annotation, 21,827/22,800 nonzero ({21827/22800*100:.6f}%), TX=0.\n"
md += f"- Primary 100 ps power: internal {p100['internal']:.8g} mW; net switching {p100['net_switching']:.8g} mW; leakage {p100['leakage']:.8g} mW; total {p100['total']:.8g} mW.\n"
md += f"- Primary energy per measured cycle: {energy(p100['total']):.8g} pJ/cycle.\n"
md += f"- Total-power sensitivity: 50 ps {p050['total']:.8g} mW ({p050['total']/p100['total']:.6f}x primary), 200 ps {p200['total']:.8g} mW ({p200['total']/p100['total']:.6f}x primary).\n\n"
md += "reset_n slew is not recovery/removal/min-pulse/reset signoff. Clock-network group includes register clock-pin internal power but no CTS. SRAM, macros, extracted interconnect, four-Conv and full-network energy are excluded. This is not paper-PPA or speedup evidence.\n"
(run / "m448r2_m431_m438_prelayout_stdcell_ptpx_receipt_r2.md").write_text(md)
PY

grep -Fqx 'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4  docs/359_DATE终局冻结_20260813.md' \
    <(sha256sum docs/359_DATE终局冻结_20260813.md) || exit 30
printf '%s\n' 'PASS_M448R2_PRELAYOUT_STDCELL_SELECTED_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER' \
    >"${m448r2_run}/RUN_COMPLETE.txt"
find "${m448r2_run}" -type f ! -path '*/work/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m448r2_run}/RUN_MANIFEST.sha256"
sha256sum "${m448r2_run}/RUN_MANIFEST.sha256" \
    >"${m448r2_run}/RUN_MANIFEST.seal.sha256"
m448r2_complete=1
echo "M448R2 prelayout standard-cell PTPX complete at ${m448r2_run}"
