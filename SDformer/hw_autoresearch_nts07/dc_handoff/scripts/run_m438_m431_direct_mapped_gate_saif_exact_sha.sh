#!/usr/bin/env bash
set -euo pipefail

m438_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m438_hw="$(cd "${m438_dc_root}/.." && pwd)"
m438_runner="$(realpath "${BASH_SOURCE[0]}")"
m438_run="${M438_RUN_DIR:-${m438_dc_root}/runs/m438_m431_direct_mapped_gate_saif_r1_20260826}"
m438_source="${m438_dc_root}/runs/m431_m414_saif_tracked_dc_3p000ns_r1_20260826"
m438_subset="${m438_hw}/results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826"
m438_m429="${m438_hw}/results/m429_m425r4_saif_independent_hammer_r1_20260826"
m438_m437r2="${m438_dc_root}/runs/m437r2_m431_union_saif_annotation_recovery_r1_20260826"
m438_contract="contracts/m438_m431_direct_mapped_gate_saif_contract_r1_20260826.json"
m438_wrapper="tb_m405/m438_mapped_gate_selected_slice_wrapper.sv"
m438_assertions="verif_m405/m405_q32_elastic_selected_slice_assertions.sv"
m438_tb="tb_m405/tb_m425_h67_balanced_selected_slice_direct_saif.sv"
m438_ucli="dc_handoff/scripts/m438_mapped_gate_selected_slice_saif.ucli.tcl"
m438_pt_tcl="dc_handoff/scripts/run_pt_m438_mapped_gate_saif_annotation_only.tcl"
m438_netlist="${m438_source}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m438_sdc="${m438_source}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m438_gate_netlist="${m438_run}/netlist/m405_q32_elastic_selected_slice_mapped_gate.v"
m438_gate_saif="${m438_run}/m405_q32_elastic_selected_slice_mapped_gate.saif"
m438_vcs="/opt/synopsys/vcs/V-2023.12-SP1"
m438_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m438_cell_v="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v"
m438_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m438_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"

m438_sha() { sha256sum "$1" | awk '{print $1}'; }
m438_expect() { [[ -f "$1" && "$(m438_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m438_run}" ]] || exit 2
mkdir -p "${m438_run}/csrc" "${m438_run}/netlist" \
    "${m438_run}/pt_work" "${m438_run}/reports"
m438_complete=0
trap 'm438_rc=$?; if [[ ${m438_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m438_rc}" >"${m438_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m438_hw}"

declare -A m438_expected=(
    ["${m438_contract}"]="b376ca33cfd745d664c5c74ee9b8bf90629a5ad4980443b36f9570cd5c82a201"
    ["${m438_wrapper}"]="7e7f20cfb637ff05fb4ec5c5cf1e9295982e336192dabe241dd8ab64179a209a"
    ["${m438_assertions}"]="71a190e373ec0016cc09314276d03f3b40d7e7731c108b3734bc29c384abfa4b"
    ["${m438_tb}"]="1ee804adc1ddaca965d9b0d4395d8f084e98009855b4f91ac456a0432871c12d"
    ["${m438_ucli}"]="abebbb19d92e6ac7a1ef0ecac0c869394a3dfe78d05d3c8eec9f5e250324564d"
    ["${m438_pt_tcl}"]="54b603dc77ebf23c59257d0c5cf067bdbaa9acc83ab9b464b7ce84dbbd93c130"
    ["${m438_netlist}"]="dd42a5b82d2e1782f5511d93405b8f6aad94ab3eaf47fb1a615eeee60a5e6723"
    ["${m438_sdc}"]="f0c30f09360f71ab8d801943f1c0c372a3596ffb43b3506925c860c54b38708f"
    ["${m438_source}/evidence_manifest.seal.sha256"]="1562ab70877b853ba87d5fe35c1c52d61dcbc4eb3610f9a01c0d6e43e675d05c"
    ["${m438_subset}/m425_h67_phase_config_768.memh"]="08c03c014290a709bffe461d1cd77dfe42d02f02bb9b7756cce06996f18cb1de"
    ["${m438_subset}/m425_h67_runtime_rows_32.memh"]="666312a60d33b1ee0579b05bb3f9e6f9ddf1c86be8ddf5795af2348772f1780b"
    ["${m438_subset}/m425_h67_static_pwp_1281.memh"]="cd6ef528ba76ed4d470e14bd688d8b2c3b48b38fe15ff7d5d3bb8969ae0506cf"
    ["${m438_subset}/SHA256SUMS.seal.sha256"]="56964e3d7c8d424dc0d720da803e105a63aeb334b819d993c4c4bd36ee65601a"
    ["${m438_m429}/SHA256SUMS.seal.sha256"]="06496b718f116ad1e1d1c84bda095f319fc9b10b9bcf3b554e042e585c87fa33"
    ["${m438_m437r2}/RUN_MANIFEST.seal.sha256"]="fa79a962e8a5bdd4d65abffacb54db5facdc86a845ce029d609d7731fbb84ef0"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
m438_expect "${m438_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m438_expect "${m438_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m438_expect "${m438_cell_v}" 3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a
m438_expect "${m438_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m438_expect "${m438_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
: >"${m438_run}/preflight_sha_checks.txt"
for m438_path in "${!m438_expected[@]}"; do
    m438_observed="$(m438_sha "${m438_path}")"
    printf 'path=%s expected=%s observed=%s\n' "${m438_path}" \
        "${m438_expected[${m438_path}]}" "${m438_observed}" \
        >>"${m438_run}/preflight_sha_checks.txt"
    [[ "${m438_observed}" == "${m438_expected[${m438_path}]}" ]] || exit 10
done

(cd "${m438_source}" && sha256sum -c evidence_manifest.sha256 && \
    sha256sum -c evidence_manifest.seal.sha256) \
    >"${m438_run}/upstream_seal_checks.log" 2>&1
(cd "${m438_subset}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m438_run}/upstream_seal_checks.log" 2>&1
(cd "${m438_m429}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m438_run}/upstream_seal_checks.log" 2>&1
(cd "${m438_m437r2}" && sha256sum -c RUN_MANIFEST.sha256 && \
    sha256sum -c RUN_MANIFEST.seal.sha256) \
    >>"${m438_run}/upstream_seal_checks.log" 2>&1

# Create a simulation-only module rename in the fresh run directory.  The
# sealed M431 source netlist is never edited.
sed '0,/^module m405_q32_elastic_selected_slice /s//module m405_q32_elastic_selected_slice_mapped_gate /' \
    "${m438_netlist}" >"${m438_gate_netlist}"
m438_expect "${m438_gate_netlist}" 7a267625d4e522fd0e51a7019c1c3fcf23616b1250972542f7275e8e5c5f1544
[[ "$(rg -c '^module m405_q32_elastic_selected_slice_mapped_gate ' "${m438_gate_netlist}")" -eq 1 ]] || exit 11
[[ "$(rg -c '^module m405_q32_elastic_selected_slice ' "${m438_netlist}")" -eq 1 ]] || exit 12

sha256sum "${!m438_expected[@]}" "${m438_vcs}/bin/vcs" "${m438_pt}" \
    "${m438_cell_v}" "${m438_slow}" "${m438_fast}" "${m438_gate_netlist}" \
    >"${m438_run}/input_sha256.txt"
cp "${m438_contract}" "${m438_run}/contract.json"

export VCS_HOME="${m438_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
export M438_GATE_SAIF_FILE="${m438_gate_saif}"
set +e
"${m438_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -debug_access+all \
    -cm assert -Mdir="${m438_run}/csrc" "${m438_cell_v}" \
    "${m438_gate_netlist}" "${m438_wrapper}" "${m438_assertions}" \
    "${m438_tb}" -top tb_m425_h67_balanced_selected_slice_direct_saif \
    -o "${m438_run}/simv" >"${m438_run}/compile.log" 2>&1
m438_rc=$?
set -e
printf '%s\n' "${m438_rc}" >"${m438_run}/compile.rc"
[[ ${m438_rc} -eq 0 && -x "${m438_run}/simv" ]] || exit 20
if grep -Eiq 'Error-\[|^Error:|^Fatal:' "${m438_run}/compile.log"; then exit 21; fi

m438_start="$(date +%s)"
set +e
"${m438_run}/simv" -no_save -lca +M425_UCLI_SAIF_STOP \
    +ntb_random_seed=43820260826 \
    "+M425_CONFIG=${m438_subset}/m425_h67_phase_config_768.memh" \
    "+M425_ROWS=${m438_subset}/m425_h67_runtime_rows_32.memh" \
    "+M425_PWP=${m438_subset}/m425_h67_static_pwp_1281.memh" \
    -ucli -do "${m438_hw}/${m438_ucli}" -cm assert \
    -assert "report=${m438_run}/assert.report" \
    >"${m438_run}/sim.log" 2>&1
m438_rc=$?
set -e
m438_end="$(date +%s)"
printf '%s\n' "${m438_rc}" >"${m438_run}/sim.rc"
printf '%s\n' "$((m438_end-m438_start))" >"${m438_run}/sim_wall_seconds.txt"
[[ ${m438_rc} -eq 0 ]] || exit 22
[[ -s "${m438_gate_saif}" ]] || exit 23
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]|protocol_error=1' \
        "${m438_run}/sim.log" "${m438_run}/assert.report"; then exit 24; fi
grep -Eq 'PASS M425 H67 balanced selected-slice direct-SAIF activity phases=64 rows=192000 pass0=192000 pass1=61285 early=11923 zero=93037 pop1=25755 pwp_rows=63067 low=504536 high=416630 narrow=87906 wide=416630 contributions=921166 reconstructed_lanes=48435456 metadata_mismatches=0 matcher_arithmetic_mismatches=0 codec_arithmetic_mismatches=0 reconstruction_mismatches=0 bitmap_mismatches=0 unknown_transactions=0 protocol_error=0.*measurement_cycles=2096003' \
    "${m438_run}/sim.log" || exit 25

python3 - "${m438_run}" "${m438_gate_saif}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
saif = Path(sys.argv[2])
sim = (run / "sim.log").read_text(errors="replace")
pattern = (
    r"PASS M425 H67 balanced selected-slice direct-SAIF activity "
    r"phases=(\d+) rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) "
    r"zero=(\d+) pop1=(\d+) pwp_rows=(\d+) low=(\d+) high=(\d+) "
    r"narrow=(\d+) wide=(\d+) contributions=(\d+) "
    r"reconstructed_lanes=(\d+).*measurement_cycles=(\d+)")
match = re.search(pattern, sim)
if not match:
    raise SystemExit("missing gate replay PASS ledger")
values = [int(value) for value in match.groups()]
expected = [64,192000,192000,61285,11923,93037,25755,63067,
            504536,416630,87906,416630,921166,48435456,2096003]
if values != expected:
    raise SystemExit(f"mapped gate population drift: {values}")

text = saif.read_text(errors="strict")
duration_match = re.search(r"\(DURATION\s+(\d+(?:\.\d+)?)\)", text)
if not duration_match or float(duration_match.group(1)) <= 0:
    raise SystemExit("gate SAIF duration missing/nonpositive")
entry = re.compile(
    r"\(([^()\s]+)\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+"
    r"\(TX\s+(\d+)\)\s+\(TC\s+(\d+)\)", re.MULTILINE)
entries = [(name, int(t0), int(t1), int(tx), int(tc))
           for name,t0,t1,tx,tc in entry.findall(text)]
if len(entries) < 20000:
    raise SystemExit(f"gate SAIF signal population too small: {len(entries)}")
nonzero = sum(item[4] > 0 for item in entries)
if nonzero < 10000:
    raise SystemExit(f"gate SAIF toggle population too small: {nonzero}")
protocol = [item for item in entries if item[0] == "protocol_error"]
if not protocol or any(item[3] != 0 or item[4] != 0 for item in protocol):
    raise SystemExit("gate SAIF protocol_error missing, X, or toggled")

receipt = {
    "schema": "m438_m431_direct_mapped_gate_saif_pre_pt_receipt_v1",
    "status": "PASS_M438_MAPPED_GATE_FUNCTIONAL_REPLAY_AND_DIRECT_SAIF",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "simulation": "zero_delay_no_sdf",
    "population": {
        "phases": values[0], "source_rows": values[1],
        "pass0_tasks": values[2], "pass1_tasks": values[3],
        "early_stops": values[4], "zero_rows": values[5],
        "pop1_rows": values[6], "pwp_rows": values[7],
        "low_accepts": values[8], "high_accepts": values[9],
        "narrow_blocks": values[10], "wide_blocks": values[11],
        "contributions": values[12], "reconstructed_lanes": values[13],
        "measurement_cycles": values[14],
    },
    "mismatches": {
        "metadata": 0, "matcher_arithmetic": 0,
        "codec_arithmetic": 0, "reconstruction": 0,
        "bitmap": 0, "accepted_transaction_unknowns": 0,
        "protocol": 0, "assertion": 0,
    },
    "saif": {
        "scope": "tb_m425_h67_balanced_selected_slice_direct_saif.dut.u_gate",
        "wrapper_included": False, "bytes": saif.stat().st_size,
        "sha256": hashlib.sha256(saif.read_bytes()).hexdigest(),
        "duration_raw": float(duration_match.group(1)),
        "parsed_signal_entries": len(entries),
        "nonzero_toggle_signal_entries": nonzero,
        "nonzero_tx_signal_entries": sum(item[3] > 0 for item in entries),
        "total_tx_duration_raw": sum(item[3] for item in entries),
        "protocol_error_toggle_count": 0,
    },
    "sim_wall_seconds": int((run / "sim_wall_seconds.txt").read_text()),
    "claim_boundary": {
        "mapped_gate_functional_replay": True,
        "mapped_gate_activity": True, "paper_power_eligible": False,
        "power": False, "energy": False, "system_speedup": False,
        "paper_ppa_ready": False, "headline": False,
    },
}
(run / "m438_m431_direct_mapped_gate_saif_pre_pt_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
sha256sum "${m438_gate_saif}" >"${m438_run}/GATE_SAIF.sha256"

if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then exit 30; fi
export DESIGN_NAME=m405_q32_elastic_selected_slice
export LIB_DB="${m438_slow}" MIN_LIB_DB="${m438_fast}"
export MAPPED_NETLIST="${m438_netlist}" MAPPED_SDC="${m438_sdc}"
export GATE_SAIF_FILE="${m438_gate_saif}" OUTPUT_DIR="${m438_run}"
export OPERATING_CONDITION=ssg0p9v125c
export SAIF_INSTANCE=tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate
set +e
(cd "${m438_run}/pt_work" && "${m438_pt}" -f "${m438_hw}/${m438_pt_tcl}") \
    >"${m438_run}/pt_annotation.log" 2>&1
m438_rc=$?
set -e
printf '%s\n' "${m438_rc}" >"${m438_run}/pt_annotation.rc"
[[ ${m438_rc} -eq 0 ]] || exit 31
if grep -Eq '^Error:|^Fatal:' "${m438_run}/pt_annotation.log"; then exit 32; fi
if grep -Eq '^(update_power|report_power)([[:space:]]|$)' \
        "${m438_run}/pt_annotation.log"; then exit 33; fi
grep -Fqx 'M438_GATE_SAIF_ANNOTATION_INTERNAL_COMPLETE=PASS' \
    "${m438_run}/PT_GATE_SAIF_ANNOTATION_INTERNAL_COMPLETE.txt" || exit 34

python3 - "${m438_run}" <<'PY'
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
    raise SystemExit("could not parse PrimeTime gate-SAIF annotation/coverage")
total = int(annotated_match.group(1))
annotated = int(annotated_match.group(2))
annotated_pct = float(annotated_match.group(3))
coverage_pct = float(coverage_match.group(1))
covered = int(coverage_match.group(2))
coverage_total = int(coverage_match.group(3))
if total != 22800 or coverage_total != total:
    raise SystemExit("mapped net population drift")
passes = annotated_pct >= 95.0 and coverage_pct >= 95.0
pre = json.loads((run / "m438_m431_direct_mapped_gate_saif_pre_pt_receipt_r1.json").read_text())
status = ("PASS_M438_DIRECT_GATE_SAIF_ANNOTATION_AT_LEAST_95" if passes else
          "COMPLETE_M438_DIRECT_GATE_SAIF_ANNOTATION_BELOW_95_NO_GO_POWER")
receipt = {
    "schema": "m438_m431_direct_mapped_gate_saif_receipt_v1",
    "status": status,
    "vcs_gate_replay": pre,
    "primetime_annotation": {
        "total_nets": total, "annotated_nets": annotated,
        "annotated_percent": annotated_pct,
        "covered_nets_at_least_one_toggle": covered,
        "switching_coverage_percent": coverage_pct,
        "minimum_required_percent": 95.0, "passes": passes,
    },
    "decision": ("GO_SEPARATE_PTPX_AFTER_INDEPENDENT_REVIEW" if passes else
                 "NO_GO_POWER_ENERGY"),
    "claim_boundary": {
        "mapped_gate_functional_replay": True,
        "mapped_gate_activity": True, "annotation_diagnostic": True,
        "update_power_called": False, "report_power_called": False,
        "power": False, "energy": False, "system_speedup": False,
        "paper_ppa_ready": False, "headline": False,
    },
}
(run / "m438_m431_direct_mapped_gate_saif_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(run / "RUN_COMPLETE.txt").write_text(status + "\n")
PY

sha256sum "${m438_runner}" >"${m438_run}/runner_sha256.txt"
find "${m438_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -path '*/pt_work/*' ! -name RUN_MANIFEST.sha256 \
    ! -name RUN_MANIFEST.seal.sha256 -print0 | sort -z | xargs -0 sha256sum \
    >"${m438_run}/RUN_MANIFEST.sha256"
sha256sum "${m438_run}/RUN_MANIFEST.sha256" \
    >"${m438_run}/RUN_MANIFEST.seal.sha256"
m438_complete=1
echo "M438 direct mapped-gate SAIF and annotation diagnostic complete at ${m438_run}"
