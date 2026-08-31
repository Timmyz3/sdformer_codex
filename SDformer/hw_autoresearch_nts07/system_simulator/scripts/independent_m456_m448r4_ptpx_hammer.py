#!/usr/bin/env python3
"""Independent, read-only hammer of the sealed M448R4 PrimeTime PX run."""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(actual: float, expected: float, atol: float = 1e-12) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol),
            f"numeric mismatch: {actual} != {expected}")


def snapshot(root: Path) -> dict:
    result = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        result[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return result


def parse_power(report: Path) -> dict:
    text = report.read_text(errors="replace")
    require(re.search(r"(?m)^\s*-unit mW\s*$", text) is not None,
            f"{report}: report header does not declare mW")
    fields = {
        "internal": "Cell Internal Power",
        "net_switching": "Net Switching Power",
        "leakage": "Cell Leakage Power",
        "total": "Total Power",
    }
    parsed = {"unit": "mW"}
    for key, label in fields.items():
        matches = re.findall(rf"{re.escape(label)}\s*=\s*([0-9.eE+-]+)", text)
        require(len(matches) == 1, f"{report}: expected unique {label}, got {len(matches)}")
        parsed[key] = float(matches[0])
    component_sum = float(parsed["internal"]) + float(parsed["net_switching"]) + float(parsed["leakage"])
    require(abs(float(parsed["total"]) - component_sum) <= 1e-6,
            f"{report}: rounded components do not sum to total")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    script = Path(__file__).resolve()
    hw_root = script.parents[2]
    repo_root = hw_root.parent
    contract_path = args.contract.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    contract = json.loads(contract_path.read_text())
    require(contract["schema"] == "m456_m448r4_independent_hammer_contract_v1",
            "unexpected M456 contract schema")
    require(contract["status"] == "FROZEN_BEFORE_INDEPENDENT_RECOMPUTATION",
            "M456 contract not frozen")

    for label, item in contract["inputs"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = hw_root / path
        require(path.is_file(), f"missing input {label}: {path}")
        require(sha256(path) == item["sha256"], f"input hash mismatch: {label}")

    runs = hw_root / "dc_handoff" / "runs"
    r1 = runs / "m448_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r1_20260826"
    r2 = runs / "m448r2_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r2_20260826"
    r3 = runs / "m448r3_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r3_20260826"
    r4 = runs / "m448r4_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r4_20260826"
    r4_before = snapshot(r4)

    # Supersession is established from failure markers and the defective R3 manifest,
    # never from the numeric power outputs in those obsolete directories.
    r1_marker = (r1 / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().splitlines()
    r2_marker = (r2 / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().splitlines()
    require(r1_marker == ["status=FAILED_OR_INCOMPLETE_DO_NOT_CITE", "runner_exit_code=22"],
            "R1 failure marker drift")
    require(r2_marker == ["status=FAILED_OR_INCOMPLETE_DO_NOT_CITE", "runner_exit_code=25"],
            "R2 failure marker drift")
    require((r1 / "ptpx.rc").read_text().strip() == "0", "R1 pt rc drift")
    require((r2 / "ptpx.rc").read_text().strip() == "0", "R2 pt rc drift")
    r1_check = (r1 / "reports" / "ptpx_check_power_pre_update.rpt").read_text(errors="replace")
    r1_ramps = re.search(r"Warning: There are (\d+) out_of_range ramps\.", r1_check)
    require(r1_ramps is not None and int(r1_ramps.group(1)) == 4139,
            "R1 4139-ramp supersession evidence missing")
    r2_runner_text = (hw_root / "dc_handoff" / "scripts" /
                      "run_m448r2_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_exact_sha.sh").read_text()
    require("grep -Ec '^update_power([[:space:]]|$)'" in r2_runner_text and "|| exit 25" in r2_runner_text,
            "R2 invalid source-echo audit not found")
    require(len(re.findall(r"(?m)^\s*update_power\s*$", (r2 / "ptpx.log").read_text())) == 1,
            "R2 observed source-echo count drift")

    r3_manifest = (r3 / "RUN_MANIFEST.sha256").read_text().splitlines()
    require(len(r3_manifest) == 1, "R3 invalid manifest no longer has one line")
    require(r3_manifest[0] == hashlib.sha256(b"").hexdigest() + "  -",
            "R3 vacuous stdin manifest signature drift")
    require(sha256(r3 / "RUN_MANIFEST.sha256") ==
            "abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a",
            "R3 invalid manifest hash drift")
    require(sha256(r3 / "RUN_MANIFEST.seal.sha256") ==
            "657898a0d7f7d1b421281f509718e371357863be28c93023a7e6ccba32d11f35",
            "R3 invalid seal hash drift")

    # Independently parse and verify every entry in the corrected R4 relative manifest.
    manifest_path = r4 / "R4_RUN_MANIFEST.sha256"
    manifest_lines = manifest_path.read_text().splitlines()
    targets = []
    for line in manifest_lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"malformed R4 manifest line: {line}")
        expected_hash, target = match.groups()
        require(target.startswith("./"), f"non-relative R4 manifest target: {target}")
        require(target != "-", "R4 manifest contains stdin target")
        require(target != "./work" and not target.startswith("./work/"),
                f"R4 manifest contains work target: {target}")
        target_path = r4 / target[2:]
        require(target_path.is_file(), f"R4 manifest target missing: {target}")
        require(sha256(target_path) == expected_hash, f"R4 manifest hash mismatch: {target}")
        targets.append(target)
    required_targets = set(contract["expected"]["required_r4_manifest_targets"])
    require(len(manifest_lines) == 44, f"R4 manifest entry count {len(manifest_lines)} != 44")
    require(len(set(targets)) == len(targets), "R4 manifest contains duplicate targets")
    require(required_targets <= set(targets), "R4 manifest misses required targets")
    seal_line = (r4 / "R4_RUN_MANIFEST.seal.sha256").read_text().strip()
    require(seal_line == f"{sha256(manifest_path)}  R4_RUN_MANIFEST.sha256",
            "R4 outer manifest seal mismatch")
    seal_process = subprocess.run(
        ["sha256sum", "-c", "R4_RUN_MANIFEST.sha256"], cwd=r4,
        universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    outer_process = subprocess.run(
        ["sha256sum", "-c", "R4_RUN_MANIFEST.seal.sha256"], cwd=r4,
        universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output_dir / "r4_manifest_self_check_from_r4_cwd.log").write_text(
        f"cwd={r4}\nrc={seal_process.returncode}\n{seal_process.stdout}")
    (output_dir / "r4_outer_seal_self_check_from_r4_cwd.log").write_text(
        f"cwd={r4}\nrc={outer_process.returncode}\n{outer_process.stdout}")
    require(seal_process.returncode == 0 and outer_process.returncode == 0,
            "R4 sha256sum self-check failed")

    require((r4 / "ptpx.rc").read_text().strip() == "0", "R4 pt_shell rc is nonzero")
    expected_ledger = contract["expected"]["runtime_ledger"]
    ledger = (r4 / "power_call_ledger.txt").read_text().splitlines()
    require(ledger == expected_ledger and len(ledger) == 9, "R4 runtime ledger drift")

    check_power = {}
    for label in ("primary_100ps", "sensitivity_050ps", "sensitivity_200ps"):
        check_text = (r4 / "reports" / f"ptpx_check_power_{label}_pre_update.rpt").read_text(errors="replace")
        metrics = {
            "succeeded": "check_power succeeded." in check_text,
            "warning_findings": len(re.findall(r"(?mi)^Warning:", check_text)),
            "ramp_findings": len(re.findall(r"(?i)out_of_range ramps|out of ramp range", check_text)),
            "missing_table_findings": len(re.findall(r"(?i)missing table", check_text)),
            "missing_function_findings": len(re.findall(r"(?i)missing function", check_text)),
        }
        require(metrics == {
            "succeeded": True,
            "warning_findings": 0,
            "ramp_findings": 0,
            "missing_table_findings": 0,
            "missing_function_findings": 0,
        }, f"{label} check_power finding: {metrics}")
        check_power[label] = metrics

    power = {
        "50ps": parse_power(r4 / "reports" / "ptpx_power_sensitivity_050ps.rpt"),
        "100ps": parse_power(r4 / "reports" / "ptpx_power_primary_100ps.rpt"),
        "200ps": parse_power(r4 / "reports" / "ptpx_power_sensitivity_200ps.rpt"),
    }
    expected_power = contract["expected"]["power_mw"]
    for point, expected_fields in expected_power.items():
        for field, expected in expected_fields.items():
            close(float(power[point][field]), float(expected), 1e-12)

    verbose = (r4 / "reports" / "ptpx_power_primary_100ps_verbose.rpt").read_text(errors="replace")
    for unit_line in ("Voltage Units = 1 V", "Capacitance Units = 1 pf",
                      "Time Units = 1 ns", "Dynamic Power Units = 1 mW",
                      "Leakage Power Units = 1 mW"):
        require(unit_line in verbose, f"missing power unit declaration: {unit_line}")
    require("Operating Conditions: tt0p9v25c" in verbose, "TT operating condition absent")
    require("Library: tcbn28hpcplusbwp35p140tt0p9v25c" in verbose, "TT library absent")
    require(re.search(r"m405_q32_elastic_selected_slice\s+ZeroWireload", verbose) is not None,
            "ZeroWireload not observed")

    annotation = (r4 / "reports" / "saif_annotation_summary.rpt").read_text(errors="replace")
    coverage = (r4 / "reports" / "switching_coverage.rpt").read_text(errors="replace")
    annotation_match = re.search(
        r"Total number of nets = (\d+).*?Number of annotated nets = (\d+) \(([0-9.]+)%\).*?"
        r"Total number of leaf cells = (\d+).*?Number of fully annotated leaf cells = (\d+) \(([0-9.]+)%\)",
        annotation, re.S)
    coverage_match = re.search(
        r"(?m)^m405_q32_elastic_selected_slice\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$", coverage)
    require(annotation_match is not None and coverage_match is not None,
            "cannot parse R4 SAIF annotation reports")
    report_activity = {
        "total_nets": int(annotation_match.group(1)),
        "annotated_nets": int(annotation_match.group(2)),
        "annotated_percent": float(annotation_match.group(3)),
        "total_leaf_cells": int(annotation_match.group(4)),
        "fully_annotated_leaf_cells": int(annotation_match.group(5)),
        "fully_annotated_leaf_percent": float(annotation_match.group(6)),
        "nonzero_reported_percent": float(coverage_match.group(1)),
        "nonzero_nets": int(coverage_match.group(2)),
        "coverage_total_nets": int(coverage_match.group(3)),
    }
    require(report_activity == {
        "total_nets": 22800,
        "annotated_nets": 22800,
        "annotated_percent": 100.0,
        "total_leaf_cells": 20803,
        "fully_annotated_leaf_cells": 20803,
        "fully_annotated_leaf_percent": 100.0,
        "nonzero_reported_percent": 95.73,
        "nonzero_nets": 21827,
        "coverage_total_nets": 22800,
    }, f"R4 report activity drift: {report_activity}")

    saif_path = hw_root / "dc_handoff" / "runs" / "m438_m431_direct_mapped_gate_saif_r1_20260826" / "m405_q32_elastic_selected_slice_mapped_gate.saif"
    saif_text = saif_path.read_text(errors="strict")
    duration_match = re.search(r"\(DURATION\s+([0-9.]+)\)", saif_text)
    entries = re.findall(
        r"\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+\(TX\s+(\d+)\)\s+\(TC\s+(\d+)\)", saif_text)
    require(duration_match is not None and len(entries) == 22800, "raw SAIF population drift")
    raw_activity = {
        "duration_ns": float(duration_match.group(1)),
        "entries": len(entries),
        "nonzero_toggle_entries": sum(int(item[3]) > 0 for item in entries),
        "nonzero_tx_entries": sum(int(item[2]) > 0 for item in entries),
        "total_tx_duration_ns": sum(int(item[2]) for item in entries),
    }
    require(raw_activity == {
        "duration_ns": 6288008.5,
        "entries": 22800,
        "nonzero_toggle_entries": 21827,
        "nonzero_tx_entries": 0,
        "total_tx_duration_ns": 0,
    }, f"raw SAIF activity drift: {raw_activity}")
    reset_match = re.search(
        r"\(reset_n\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+\(TX\s+(\d+)\)\s+\(TC\s+(\d+)\)",
        saif_text)
    require(reset_match is not None and tuple(map(int, reset_match.groups())) == (0, 6288009, 0, 0),
            "reset_n is not static high in raw SAIF")

    measurement_cycles = 2096003
    duration_ns = raw_activity["duration_ns"]
    ns_per_measured_cycle = float(duration_ns) / measurement_cycles
    energy = {
        field: float(power["100ps"][field]) * ns_per_measured_cycle
        for field in ("internal", "net_switching", "leakage", "total")
    }
    close(energy["total"], 18.76142256815862, 1e-12)
    sensitivity = {
        "50ps_vs_100ps_ratio": float(power["50ps"]["total"]) / float(power["100ps"]["total"]),
        "200ps_vs_100ps_ratio": float(power["200ps"]["total"]) / float(power["100ps"]["total"]),
    }
    sensitivity["max_abs_delta_percent"] = max(
        abs(sensitivity["50ps_vs_100ps_ratio"] - 1.0),
        abs(sensitivity["200ps_vs_100ps_ratio"] - 1.0)) * 100.0
    close(sensitivity["max_abs_delta_percent"], 0.0070833642251688644, 1e-14)

    primary_text = (r4 / "reports" / "ptpx_power_primary_100ps.rpt").read_text()
    clock_match = re.search(
        r"(?m)^clock_network\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+).*?\bi\s*$",
        primary_text)
    require(clock_match is not None, "clock_network group with i attribute absent")
    clock_network = {
        "internal_mw": float(clock_match.group(1)),
        "switching_mw": float(clock_match.group(2)),
        "leakage_mw": float(clock_match.group(3)),
        "total_mw": float(clock_match.group(4)),
        "includes_register_clock_pin_internal": True,
        "includes_cts_buffers_or_extracted_clock_interconnect": False,
    }
    close(clock_network["total_mw"], 4.58843946, 1e-12)

    scope = (r4 / "reports" / "ptpx_scope.rpt").read_text()
    clock = (r4 / "reports" / "ptpx_clock.rpt").read_text()
    engine = (hw_root / "dc_handoff" / "scripts" /
              "run_ptpx_m448r3_m431_m438_prelayout_stdcell_tt0p9v25c.tcl").read_text()
    require(re.search(r"(?m)^core_clk\s+3\.00\s+\{0 1\.5\}", clock) is not None,
            "3.0 ns ideal clock not observed")
    for scope_line in ("power_corner=tt0p9v25c", "voltage_v=0.9", "temperature_c=25",
                       "clock_network=ideal_no_cts", "wireload=ZeroWireload", "spef=false",
                       "macros=0", "sram=false", "interconnect_extracted=false",
                       "claim_scope=M416_selected_slice_only"):
        require(scope_line in scope, f"scope boundary missing: {scope_line}")
    require("set_wire_load_model -name ZeroWireload" in engine, "engine lacks ZeroWireload command")
    require("read_spef" not in engine and "read_parasitics" not in engine,
            "engine unexpectedly reads extracted parasitics")
    require("set_propagated_clock" not in engine, "engine unexpectedly propagates clock")

    ptlog = (r4 / "ptpx.log").read_text(errors="replace")
    sdc_warnings = len(re.findall(
        r"Warning: SDC version in file \(2\.1\) does not match the version you requested", ptlog))
    require(sdc_warnings == 2, f"expected two SDC-2 version warnings, got {sdc_warnings}")
    require(re.search(r"(?m)^Error:|^Fatal:", ptlog) is None, "R4 ptpx.log contains Error/Fatal")
    timing_check = (r4 / "reports" / "ptpx_check_timing.rpt").read_text()
    require("There are 1 ports with no clock-relative input delay specified." in timing_check and
            re.search(r"(?m)^reset_n\s*$", timing_check) is not None,
            "reset_n unconstrained timing warning absent")
    mapped_sdc = hw_root / "dc_handoff" / "runs" / "m431_m414_saif_tracked_dc_3p000ns_r1_20260826" / "netlist" / "m405_q32_elastic_selected_slice_mapped.sdc"
    sdc_text = mapped_sdc.read_text()
    require("set sdc_version 2.1" in sdc_text, "mapped SDC is not version 2.1")
    require("set_false_path   -from [get_ports reset_n]" in sdc_text,
            "reset_n false path missing")

    receipt_path = r4 / "m448r4_m431_m438_prelayout_stdcell_ptpx_receipt_r4.json"
    receipt = json.loads(receipt_path.read_text())
    receipt_mismatches = []
    if receipt["primary_100ps_prelayout_standard_cell_power_mw"] != {
        "internal": power["100ps"]["internal"],
        "leakage": power["100ps"]["leakage"],
        "net_switching": power["100ps"]["net_switching"],
        "total": power["100ps"]["total"],
    }:
        receipt_mismatches.append("primary_power")
    for field, value in energy.items():
        if not math.isclose(receipt["primary_100ps_prelayout_standard_cell_energy_per_measured_cycle_pj"][field],
                            value, rel_tol=0.0, abs_tol=1e-12):
            receipt_mismatches.append(f"energy_{field}")
    if not math.isclose(receipt["input_slew_sensitivity_total_power_mw"]["max_abs_delta_vs_primary_percent"],
                        sensitivity["max_abs_delta_percent"], rel_tol=0.0, abs_tol=1e-14):
        receipt_mismatches.append("sensitivity")
    require(not receipt_mismatches, f"R4 receipt mismatch: {receipt_mismatches}")

    r4_after = snapshot(r4)
    require(r4_before == r4_after, "R4 run changed during independent hammer")
    docs359 = hw_root / "docs" / "359_DATE终局冻结_20260813.md"
    require(sha256(docs359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs/359 changed")

    result = {
        "schema": "m456_m448r4_independent_recomputation_v1",
        "status": "PASS_M456_INDEPENDENT_R4_PTPX_SCOPE_ADMISSIBLE",
        "supersession": {
            "M448_R1": {"status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "outer_exit_code": 22,
                         "pt_shell_exit_code": 0, "out_of_range_ramps": 4139},
            "M448R2": {"status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "outer_exit_code": 25,
                        "pt_shell_exit_code": 0, "observed_source_echo_update_power_lines": 1,
                        "invalid_expected_lines": 3},
            "M448R3": {"status": "FAILED_INVALID_VACUOUS_SEAL_DO_NOT_CITE",
                        "manifest_entries": 1, "manifest_target": "-"},
            "numeric_outputs_reused_from_obsolete_runs": 0,
        },
        "r4_manifest": {
            "entries": len(manifest_lines), "relative_targets": len(targets),
            "dash_targets": sum(target == "-" for target in targets),
            "duplicate_targets": len(targets) - len(set(targets)),
            "work_targets": sum(target == "./work" or target.startswith("./work/") for target in targets),
            "hash_mismatches": 0, "required_targets_missing": 0,
            "manifest_sha256": sha256(manifest_path),
            "manifest_self_check_from_r4_cwd_rc": seal_process.returncode,
            "outer_seal_self_check_from_r4_cwd_rc": outer_process.returncode,
            "r4_regular_files_before_after": len(r4_before),
            "r4_tree_changed_during_hammer": False,
        },
        "ptpx": {"pt_shell_exit_code": 0, "runtime_ledger": ledger,
                  "runtime_ledger_lines": len(ledger), "check_power": check_power,
                  "tool": "Synopsys PrimeTime PX W-2024.09-SP3"},
        "activity": {"reports": report_activity, "raw_saif": raw_activity,
                     "nonzero_toggle_coverage_percent_exact": 21827 / 22800 * 100.0,
                     "reset_n_raw_saif": {"t0_ns": 0, "t1_ns": 6288009, "tx_ns": 0, "toggle_count": 0}},
        "power_mw": power,
        "energy_per_measured_cycle_pj": {
            "measurement_cycles": measurement_cycles, "saif_duration_ns": duration_ns,
            "effective_ns_per_cycle": ns_per_measured_cycle, **energy,
        },
        "sensitivity": sensitivity,
        "corner_and_model": {
            "corner": "tt0p9v25c", "voltage_v": 0.9, "temperature_c": 25,
            "clock_period_ns": 3.0, "clock_network": "ideal_no_cts",
            "wireload": "ZeroWireload", "spef": False, "macro_count": 0,
            "clock_network_power_group": clock_network,
        },
        "warnings_and_boundaries": {
            "sdc_file_version": "2.1", "read_sdc_requested_version": "2.2",
            "sdc_version_warning_count": sdc_warnings,
            "reset_n_has_clock_relative_input_delay": False,
            "reset_n_false_pathed": True, "reset_n_static_during_saif": True,
            "reset_signoff": False,
            "clock_network_interpretation": "Includes register clock-pin internal power; excludes CTS buffers and extracted clock interconnect.",
        },
        "receipt_crosscheck_mismatches": receipt_mismatches,
        "claim_boundary": {
            "prelayout_standard_cell_m416_selected_slice_power": True,
            "prelayout_standard_cell_m416_selected_slice_energy_per_measured_cycle": True,
            "input_slew_sensitivity": True,
            "reset_signoff": False, "sram_power": False, "macro_power": False,
            "extracted_interconnect_power": False, "full_conv_power": False,
            "full_network_power": False, "system_energy": False,
            "system_speedup": False, "paper_ppa_ready": False, "headline": False,
        },
        "identity": {
            "contract_sha256": sha256(contract_path), "auditor_sha256": sha256(script),
            "r4_receipt_sha256": sha256(receipt_path),
            "docs359_sha256": sha256(docs359),
        },
    }
    (output_dir / "m456_independent_recomputation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "m456_power_reparse.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "input_slew_ps", "internal_mw", "net_switching_mw", "leakage_mw", "total_mw"])
        writer.writeheader()
        for point, slew in (("50ps", 50), ("100ps", 100), ("200ps", 200)):
            writer.writerow({
                "input_slew_ps": slew,
                "internal_mw": power[point]["internal"],
                "net_switching_mw": power[point]["net_switching"],
                "leakage_mw": power[point]["leakage"],
                "total_mw": power[point]["total"],
            })
    print("PASS_M456", f"manifest={len(manifest_lines)}", "ledger=9",
          f"saif={raw_activity['nonzero_toggle_entries']}/{raw_activity['entries']}",
          f"energy_pj={energy['total']:.12f}",
          f"sensitivity_pct={sensitivity['max_abs_delta_percent']:.12f}")


if __name__ == "__main__":
    main()
