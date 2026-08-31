#!/usr/bin/env bash
set -euo pipefail

m448_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m448_hw="$(cd "${m448_dc_root}/.." && pwd)"
m448_runner="$(realpath "${BASH_SOURCE[0]}")"
m448_run="${M448_RUN_DIR:-${m448_dc_root}/runs/m448_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r1_20260826}"
m448_m431="${m448_dc_root}/runs/m431_m414_saif_tracked_dc_3p000ns_r1_20260826"
m448_m438="${m448_dc_root}/runs/m438_m431_direct_mapped_gate_saif_r1_20260826"
m448_m446="${m448_hw}/reviews/m446_m438_gate_saif_independent_hammer_r1_20260826"
m448_contract="contracts/m448_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_contract_r1_20260826.json"
m448_tcl="dc_handoff/scripts/run_ptpx_m448_m431_m438_prelayout_stdcell_tt0p9v25c.tcl"
m448_netlist="${m448_m431}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m448_sdc="${m448_m431}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m448_saif="${m448_m438}/m405_q32_elastic_selected_slice_mapped_gate.saif"
m448_tt="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db"
m448_ss="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m448_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"

m448_sha() { sha256sum "$1" | awk '{print $1}'; }
m448_expect() { [[ -f "$1" && "$(m448_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m448_run}" ]] || exit 2
mkdir -p "${m448_run}/work" "${m448_run}/reports"
m448_complete=0
trap 'm448_rc=$?; if [[ ${m448_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m448_rc}" >"${m448_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m448_hw}"

declare -A m448_expected=(
    ["${m448_contract}"]="d7b70157e729002d9e7de1bc33f2a9a1754e71b893228d4dacf094cf18a14f3a"
    ["${m448_tcl}"]="c8ad9c88ad47b70ad64418fc0d7c5a915f6cbe28a16e3a50698733d82627483e"
    ["${m448_netlist}"]="dd42a5b82d2e1782f5511d93405b8f6aad94ab3eaf47fb1a615eeee60a5e6723"
    ["${m448_sdc}"]="f0c30f09360f71ab8d801943f1c0c372a3596ffb43b3506925c860c54b38708f"
    ["${m448_m431}/evidence_manifest.seal.sha256"]="1562ab70877b853ba87d5fe35c1c52d61dcbc4eb3610f9a01c0d6e43e675d05c"
    ["${m448_saif}"]="3d95d2aa51e1455b5d420330c96dee61f58bbc58d33133a54b6b71cb77814d4b"
    ["${m448_m438}/m438_m431_direct_mapped_gate_saif_receipt_r1.json"]="be35985a43fea45c747bb8b9cd514aacee0c92ffc673f82e0a26737e802113b5"
    ["${m448_m438}/RUN_MANIFEST.seal.sha256"]="bfc05d234e2cae16e0a6aa26b1016dbf0a1a504380e2c6d59c21ca10a2443fd8"
    ["${m448_m446}/m446_m438_gate_saif_independent_review_r1.json"]="a0c1107225e62ef53ea73a66a0d958aa1e577199bd424aeb121d131a9019438e"
    ["${m448_m446}/SHA256SUMS.seal.sha256"]="d170d4db5728cbe6f31fad95a302f8cf97634cca65cd5cf1d4af3ec2fe8e395c"
    ["${m448_tt}"]="d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070"
    ["${m448_ss}"]="79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
    ["${m448_pt}"]="afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: >"${m448_run}/preflight_sha_checks.txt"
for m448_path in "${!m448_expected[@]}"; do
    m448_observed="$(m448_sha "${m448_path}")"
    printf 'path=%s expected=%s observed=%s\n' "${m448_path}" \
        "${m448_expected[${m448_path}]}" "${m448_observed}" \
        >>"${m448_run}/preflight_sha_checks.txt"
    [[ "${m448_observed}" == "${m448_expected[${m448_path}]}" ]] || exit 10
done

(cd "${m448_m431}" && sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) \
    >"${m448_run}/upstream_seal_checks.log" 2>&1
(cd "${m448_m438}" && sha256sum -c RUN_MANIFEST.sha256 && \
    sha256sum -c RUN_MANIFEST.seal.sha256) \
    >>"${m448_run}/upstream_seal_checks.log" 2>&1
(cd "${m448_m446}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m448_run}/upstream_seal_checks.log" 2>&1

python3 - "${m448_contract}" "${m448_m438}/m438_m431_direct_mapped_gate_saif_receipt_r1.json" \
    "${m448_m446}/m446_m438_gate_saif_independent_review_r1.json" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text())
m438 = json.loads(Path(sys.argv[2]).read_text())
m446 = json.loads(Path(sys.argv[3]).read_text())
if contract["status_before_execution"] != "FROZEN_BEFORE_SYNOPSYS_PRIMETIME_PX":
    raise SystemExit("M448 contract is not frozen")
if m438["decision"] != "GO_SEPARATE_PTPX_AFTER_INDEPENDENT_REVIEW":
    raise SystemExit("M438 does not admit separate PTPX")
if m446["verdict"] != "GO_SEPARATE_PRELAYOUT_PTPX_WITH_SCOPED_CLAIMS":
    raise SystemExit("M446 does not admit scoped PTPX")
if m446["severity_counts"]["P0"] != 0:
    raise SystemExit("M446 has P0 findings")
PY

python3 - "${m448_saif}" "${m448_run}/saif_preflight_receipt_r1.json" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

saif = Path(sys.argv[1])
out = Path(sys.argv[2])
text = saif.read_text(errors="strict")
if '(TIMESCALE 1 ns)' not in text:
    raise SystemExit("M448 SAIF timescale is not 1 ns")
duration_match = re.search(r"\(DURATION\s+([0-9.]+)\)", text)
if not duration_match:
    raise SystemExit("M448 SAIF duration missing")
duration = float(duration_match.group(1))
if duration != 6288008.5:
    raise SystemExit(f"M448 SAIF duration drift: {duration}")
required_scope = (
    "(INSTANCE tb_m425_h67_balanced_selected_slice_direct_saif\n"
    "   (INSTANCE dut\n"
    "      (INSTANCE u_gate\n"
    "         (NET")
if required_scope not in text:
    raise SystemExit("M448 exact gate-only SAIF scope missing")
entry_re = re.compile(
    r"\(([^()\s]+)\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+"
    r"\(TX\s+(\d+)\)\s+\(TC\s+(\d+)\)", re.MULTILINE)
entries = [(name, int(t0), int(t1), int(tx), int(tc))
           for name, t0, t1, tx, tc in entry_re.findall(text)]
if len(entries) != 22800:
    raise SystemExit(f"M448 SAIF entry drift: {len(entries)}")
nonzero = sum(tc > 0 for _, _, _, _, tc in entries)
tx_nonzero = sum(tx > 0 for _, _, _, tx, _ in entries)
tx_total = sum(tx for _, _, _, tx, _ in entries)
if nonzero != 21827:
    raise SystemExit(f"M448 nonzero-toggle drift: {nonzero}")
if tx_nonzero != 0 or tx_total != 0:
    raise SystemExit(f"M448 TX gate failed: entries={tx_nonzero} total={tx_total}")
if any(abs((t0 + t1 + tx) - duration) > 0.5 for _, t0, t1, tx, _ in entries):
    raise SystemExit("M448 SAIF residence-time inconsistency")
clocks = [e for e in entries if e[0] == "clk_core"]
if clocks != [("clk_core", 3144004, 3144005, 0, 4192005)]:
    raise SystemExit(f"M448 clock activity drift: {clocks}")
receipt = {
    "schema": "date.m448_saif_preflight_receipt.v1",
    "status": "PASS_BEFORE_PRIMETIME_POWER",
    "sha256": hashlib.sha256(saif.read_bytes()).hexdigest(),
    "scope": "tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate",
    "timescale": "1 ns",
    "duration_ns": duration,
    "measurement_cycles": 2096003,
    "entries": len(entries),
    "nonzero_toggle_entries": nonzero,
    "nonzero_toggle_coverage_percent": nonzero / len(entries) * 100.0,
    "nonzero_tx_entries": tx_nonzero,
    "total_tx_duration_ns": tx_total,
    "clock": {
        "name": "clk_core", "t0_ns": clocks[0][1], "t1_ns": clocks[0][2],
        "tx_ns": clocks[0][3], "toggle_count": clocks[0][4]
    }
}
out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${!m448_expected[@]}" >"${m448_run}/input_sha256.txt"
cp "${m448_contract}" "${m448_run}/contract.json"
sha256sum "${m448_runner}" >"${m448_run}/runner_sha256.txt"

# One PrimeTime-family process at a time.  A busy tool queue is a fail-closed
# environmental stop, not permission to overlap analyses.
if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 20; fi

export DESIGN_NAME=m405_q32_elastic_selected_slice
export TT_LIB_DB="${m448_tt}" SDC_LIB_DB="${m448_ss}"
export MAPPED_NETLIST="${m448_netlist}" MAPPED_SDC="${m448_sdc}"
export GATE_SAIF_FILE="${m448_saif}" OUTPUT_DIR="${m448_run}"
export POWER_OPERATING_CONDITION=tt0p9v25c
export POWER_LIBRARY_NAME=tcbn28hpcplusbwp35p140tt0p9v25c
export SAIF_INSTANCE=tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate
export SAIF_DURATION_NS=6288008.5 MEASUREMENT_CYCLES=2096003
export SAIF_TX_NONZERO_ENTRIES=0
set +e
(cd "${m448_run}/work" && "${m448_pt}" -f "${m448_hw}/${m448_tcl}") \
    >"${m448_run}/ptpx.log" 2>&1
m448_rc=$?
set -e
printf '%s\n' "${m448_rc}" >"${m448_run}/ptpx.rc"
[[ ${m448_rc} -eq 0 ]] || exit 21
if grep -Eq '^Error:|^Fatal:|M448_FAIL_' "${m448_run}/ptpx.log"; then exit 22; fi
grep -Fqx 'M448_PTPX_POWER_GATE_PASS_PRE_UPDATE=PASS' \
    "${m448_run}/PTPX_POWER_GATE_PASS_PRE_UPDATE.txt" || exit 23
grep -Fqx 'M448_M431_M438_PRELAYOUT_STDCELL_PTPX_INTERNAL_COMPLETE=PASS' \
    "${m448_run}/PTPX_INTERNAL_COMPLETE.txt" || exit 24
[[ "$(grep -Ec '^update_power([[:space:]]|$)' "${m448_run}/ptpx.log")" -eq 1 ]] || exit 25
[[ "$(grep -Ec '^report_power([[:space:]]|$)' "${m448_run}/ptpx.log")" -eq 4 ]] || exit 26

python3 - "${m448_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
saif_gate = (run / "reports/saif_annotation_summary.rpt").read_text(errors="replace")
coverage_text = (run / "reports/switching_coverage.rpt").read_text(errors="replace")
power_text = (run / "reports/ptpx_power.rpt").read_text(errors="replace")
verbose_text = (run / "reports/ptpx_power_verbose.rpt").read_text(errors="replace")
hier_text = (run / "reports/ptpx_power_hierarchy.rpt").read_text(errors="replace")
clock_text = (run / "reports/ptpx_clock.rpt").read_text(errors="replace")
check_power = (run / "reports/ptpx_check_power_post_update.rpt").read_text(errors="replace")
ptlog = (run / "ptpx.log").read_text(errors="replace")

ann = re.search(
    r"Total number of nets = (\d+).*?Number of annotated nets = (\d+) \(([0-9.]+)%\).*?"
    r"Total number of leaf cells = (\d+).*?Number of fully annotated leaf cells = (\d+) \(([0-9.]+)%\)",
    saif_gate, re.S)
cov = re.search(
    r"^m405_q32_elastic_selected_slice\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$",
    coverage_text, re.M)
if not ann or not cov:
    raise SystemExit("M448 cannot parse post-run annotation gates")
total_nets, annotated_nets = int(ann.group(1)), int(ann.group(2))
annotated_percent = float(ann.group(3))
total_leaf, annotated_leaf = int(ann.group(4)), int(ann.group(5))
annotated_leaf_percent = float(ann.group(6))
nonzero_percent, nonzero_nets, coverage_total = (
    float(cov.group(1)), int(cov.group(2)), int(cov.group(3)))
if (total_nets, annotated_nets, annotated_percent, total_leaf, annotated_leaf,
        annotated_leaf_percent, nonzero_nets, coverage_total) != (
        22800, 22800, 100.0, 20803, 20803, 100.0, 21827, 22800):
    raise SystemExit("M448 post-run annotation population drift")
if nonzero_percent < 95.0:
    raise SystemExit("M448 post-run nonzero coverage below 95 percent")

if "Power-specific unit information" not in verbose_text:
    raise SystemExit("M448 verbose power units missing")
if not re.search(r"Dynamic Power Units\s*=\s*1 mW", verbose_text):
    raise SystemExit("M448 dynamic power unit is not explicit mW")
if not re.search(r"Leakage Power Units\s*=\s*1 mW", verbose_text):
    raise SystemExit("M448 leakage power unit is not explicit mW")
if "tcbn28hpcplusbwp35p140tt0p9v25c" not in verbose_text:
    raise SystemExit("M448 TT library missing from verbose power report")
if "tt0p9v25c" not in verbose_text:
    raise SystemExit("M448 TT operating condition missing from verbose report")
if not re.search(r"core_clk\s+3\.00\s+\{0 1\.5\}", clock_text):
    raise SystemExit("M448 3 ns core clock missing")
if "m405_q32_elastic_selected_slice" not in hier_text:
    raise SystemExit("M448 hierarchy report does not contain top design")
if re.search(r"unresolved|black box", check_power, re.I) and not re.search(
        r"black box.*0", check_power, re.I):
    raise SystemExit("M448 check_power reports unresolved/black-box risk")
if re.search(r"unresolved reference|could not resolve", ptlog, re.I):
    raise SystemExit("M448 PrimeTime log contains unresolved reference")

def value(label):
    match = re.search(rf"{re.escape(label)}\s*=\s*([0-9.eE+-]+)", power_text)
    if not match:
        raise SystemExit(f"M448 missing {label}")
    return float(match.group(1))

switching_mw = value("Net Switching Power")
internal_mw = value("Cell Internal Power")
leakage_mw = value("Cell Leakage Power")
total_mw = value("Total Power")
if min(switching_mw, internal_mw, leakage_mw, total_mw) < 0:
    raise SystemExit("M448 negative power component")
if abs(total_mw - (switching_mw + internal_mw + leakage_mw)) > max(1e-8, total_mw * 2e-5):
    raise SystemExit("M448 power component sum mismatch")

duration_ns = 6288008.5
measurement_cycles = 2096003
effective_ns_per_cycle = duration_ns / measurement_cycles
def energy_per_cycle_pj(power_mw):
    # 1 mW * 1 ns = 1 pJ.
    return power_mw * effective_ns_per_cycle

receipt = {
    "schema": "date.m448_m431_m438_prelayout_stdcell_ptpx_receipt.v1",
    "status": "PASS_M448_PRELAYOUT_STDCELL_MODULE_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER",
    "tool": "Synopsys PrimeTime PX W-2024.09-SP3",
    "analysis": {
        "mode": "averaged",
        "corner": "tt0p9v25c",
        "library": "tcbn28hpcplusbwp35p140tt0p9v25c",
        "voltage_v": 0.9,
        "temperature_c": 25,
        "clock_period_ns": 3.0,
        "clock_frequency_mhz": 333.333333333,
        "clock_network": "ideal",
        "wireload": "ZeroWireload",
        "spef": False,
        "macro_count": 0,
        "power_units": "mW",
    },
    "activity": {
        "scope": "tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate",
        "saif_duration_ns": duration_ns,
        "measurement_cycles": measurement_cycles,
        "effective_ns_per_measured_cycle": effective_ns_per_cycle,
        "total_nets": total_nets,
        "exact_annotated_nets": annotated_nets,
        "exact_annotation_percent": annotated_percent,
        "total_leaf_cells": total_leaf,
        "fully_annotated_leaf_cells": annotated_leaf,
        "fully_annotated_leaf_percent": annotated_leaf_percent,
        "nonzero_toggle_nets": nonzero_nets,
        "nonzero_toggle_coverage_percent_reported": nonzero_percent,
        "nonzero_toggle_coverage_percent_exact": nonzero_nets / total_nets * 100.0,
        "nonzero_tx_entries": 0,
    },
    "prelayout_standard_cell_power_mw": {
        "internal": internal_mw,
        "net_switching": switching_mw,
        "leakage": leakage_mw,
        "total": total_mw,
    },
    "prelayout_standard_cell_energy_per_measured_cycle_pj": {
        "internal": energy_per_cycle_pj(internal_mw),
        "net_switching": energy_per_cycle_pj(switching_mw),
        "leakage": energy_per_cycle_pj(leakage_mw),
        "total": energy_per_cycle_pj(total_mw),
        "derivation": "power_mW * saif_duration_ns / measurement_cycles; 1 mW*ns = 1 pJ",
    },
    "population": {
        "phases": 64, "source_rows": 192000, "pwp_rows": 63067,
        "contributions": 921166, "reconstructed_lanes": 48435456,
    },
    "claim_boundary": {
        "prelayout_standard_cell_module_slice_power": True,
        "prelayout_standard_cell_module_slice_energy_per_measured_cycle": True,
        "sram_power": False, "macro_power": False,
        "extracted_interconnect_power": False, "full_conv_power": False,
        "full_network_power": False, "system_energy": False,
        "system_speedup": False, "paper_ppa_ready": False,
        "headline": False, "pending_independent_hammer": True,
    },
}
(run / "m448_m431_m438_prelayout_stdcell_ptpx_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")

md = "# M448 M431/M438 prelayout standard-cell PTPX\n\n"
md += "Status: **PASS, pending independent hammer.** This is a module-slice, prelayout standard-cell-only result.\n\n"
md += f"- Corner: TT 0.9 V, 25 C; ideal 3.0 ns clock ({333.333333333:.9f} MHz), ZeroWireload, no SPEF, 0 macro.\n"
md += f"- Activity: 64 phases / 192,000 rows / {measurement_cycles:,} measured cycles / {duration_ns:,.1f} ns.\n"
md += f"- Annotation: {annotated_nets:,}/{total_nets:,} exact (100%); {nonzero_nets:,}/{total_nets:,} nonzero ({nonzero_nets/total_nets*100:.6f}%); TX=0.\n"
md += f"- Power: internal {internal_mw:.8g} mW; net switching {switching_mw:.8g} mW; leakage {leakage_mw:.8g} mW; total {total_mw:.8g} mW.\n"
md += f"- Energy per measured cycle: total {energy_per_cycle_pj(total_mw):.8g} pJ/cycle (from explicit mW and SAIF time).\n\n"
md += "Not included: SRAM, macros, extracted interconnect, full four-Conv or full-network energy. This is not paper-PPA ready and is not a speedup claim.\n"
(run / "m448_m431_m438_prelayout_stdcell_ptpx_receipt_r1.md").write_text(md)
PY

grep -Fqx 'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4  docs/359_DATE终局冻结_20260813.md' \
    <(sha256sum docs/359_DATE终局冻结_20260813.md) || exit 30

printf '%s\n' 'PASS_M448_PRELAYOUT_STDCELL_MODULE_SLICE_PTPX_PENDING_INDEPENDENT_HAMMER' \
    >"${m448_run}/RUN_COMPLETE.txt"
find "${m448_run}" -type f ! -path '*/work/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m448_run}/RUN_MANIFEST.sha256"
sha256sum "${m448_run}/RUN_MANIFEST.sha256" \
    >"${m448_run}/RUN_MANIFEST.seal.sha256"
m448_complete=1
echo "M448 prelayout standard-cell PTPX complete at ${m448_run}"
