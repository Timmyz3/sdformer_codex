#!/usr/bin/env python3
"""Read-only forensic audit of the sealed M1649 C1 DC quarantine."""
from __future__ import print_function

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
Q = HW / ("dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_dc_"
          "r1_20260901.failed_or_incomplete.519344.quarantine")
ATTEMPT = HW / ("dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_"
                "dc_attempt_consumed")
CANONICAL = HW / ("dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_"
                  "dc_r1_20260901")
LOCK = HW / ("dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_"
             "dc_launch_lock")
WORK = HW / ("dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_"
             "dc_work.519344")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_Q_MANIFEST = "e94ffc3680513cb2f374676037cc7c3b14b77a7bc47b9d35edb812f17a9ae843"
EXPECTED_Q_OUTER_FILE = "c221bb79e4950780c6db04ef54ed1ea809ac880ad054f9316f7bba702a49ff44"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_ARTIFACTS = {
    "dc.log": "a02a10adf0de69ad863445290ac95554399b8401842542868b11191a0e2d1b4a",
    "dc.rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "TCL_INTERNAL_COMPLETE.txt": "07ed11af7c64167f0054f119350ae6d798c3c00cfe7c331041316fa6dba30649",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc": "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.svf": "7c15c1a30827df74c0da35f24f7e88723484c2a211edd3d6c049f52e21dec274",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc": "5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v": "842d100f6a3fc26684e13a8065191028af7840685aaf4b7cfa77a4fe998c46ee",
    "reports/setup_posthold_summary_machine.txt": "123d8653bf0800934857325fa77e6759fdff93f78e099c9411b4c689d4d0647d",
    "reports/hold_posthold_summary_machine.txt": "db11b098828b57fd61b6a4ef8bff2b3302b332bca78f04c7ea442c41b46d519f",
    "reports/area_posthold.rpt": "66f18b4890ec68ec9c4b7e69e004cc326063efe4b6b62d6f95d544228ee60333",
    "reports/qor_posthold.rpt": "268909e6433b799bf59909f670c28f2697a1b8fcfbcdcb8d96cff2b06fbd872a",
    "reports/macro_binding_audit.txt": "2e21f34b7263596729746460c27663ed469b410178a9753b791ef4429fc08742",
}


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode), "nonregular: " + str(path))


def key_values(path):
    output = {}
    for line in Path(path).read_text(encoding="utf-8", errors="strict").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in output, "duplicate key in " + str(path))
            output[key] = value
    return output


def verify_tree():
    manifest = Q / "SHA256SUMS"
    outer = Q / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    require(sha(manifest) == EXPECTED_Q_MANIFEST,
            "quarantine manifest identity drift")
    require(sha(outer) == EXPECTED_Q_OUTER_FILE,
            "quarantine outer file identity drift")
    require(outer.read_text(encoding="ascii") ==
            EXPECTED_Q_MANIFEST + "  SHA256SUMS\n",
            "quarantine outer seal content drift")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        require(re.match(r"^[0-9a-f]{64}  [^/\n][^\n]*$", line),
                "malformed quarantine manifest")
        digest, name = line.split("  ", 1)
        require(name not in rows and not Path(name).is_absolute() and
                all(part not in ("", ".", "..") for part in Path(name).parts),
                "unsafe/duplicate quarantine member")
        rows[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(Q), followlinks=False):
        for name in list(dirs) + list(files):
            path = Path(base) / name
            require(not path.is_symlink(), "symlink in quarantine")
            rel = path.relative_to(Q).as_posix()
            if path.is_file() and rel not in (
                    "SHA256SUMS", "SHA256SUMS.seal.sha256"):
                actual.add(rel)
    require(actual == set(rows) and len(rows) == 39,
            "quarantine topology drift")
    for name, digest in rows.items():
        regular(Q / name)
        require(sha(Q / name) == digest,
                "quarantine member SHA drift: " + name)
    for name, digest in EXPECTED_ARTIFACTS.items():
        require(rows.get(name) == digest, "key artifact identity drift: " + name)
    return rows


def timing(name, phase, delay_type, expected_wns):
    row = key_values(Q / "reports" / name)
    require(row == {"phase": phase, "delay_type": delay_type,
                    "status": "MET", "wns_ns": expected_wns,
                    "tns_ns": "0.000000000", "violating_paths": "0",
                    "negative_path_ceiling": "200000"},
            "timing summary drift: " + name)
    return {"phase": phase, "status": "MET", "wns_ns": float(expected_wns),
            "tns_ns": 0.0, "violating_paths": 0}


def audit_log():
    text = (Q / "dc.log").read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    errors = [(index + 1, line) for index, line in enumerate(lines)
              if re.match(r"^(?:Error|Fatal):", line)]
    require(errors == [(32, "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl")],
            "not exactly the admitted startup GUI error")
    require("no such variable\n    (read trace on \"::env(HOME)\")" in text and
            text.index(errors[0][1]) < text.index("Current time:") and
            text.index("Current time:") < text.index(
                "# M1630 additive source-only residual-hold closure candidate."),
            "HOME-unset startup error location/signature drift")
    require(not any(re.match(r"^(?:Error|Fatal):", line)
                    for line in lines[lines.index(next(line for line in lines
                        if line.startswith("Current time:"))) + 1:]),
            "in-flow Error/Fatal found")
    require("Writing verilog file '" in text and "Writing ddc file '" in text and
            "set_svf -off" in text and "Memory usage for this session" in text and
            "Thank you..." in text,
            "normal DC completion markers missing")
    warnings = re.findall(r"Warning:.*?\(([A-Z]+-[0-9]+)\)", text)
    require(sorted(set(warnings)) ==
            ["PWR-428", "TIM-104", "TIM-134", "TIM-164"],
            "warning code population drift")
    return {"error_count": 1, "fatal_count": 0,
            "only_error": "startup_gui_dv_HOME_unset_before_flow",
            "warning_codes": sorted(set(warnings)),
            "normal_completion_markers": True}


def audit_reports_and_netlist():
    require((Q / "dc.rc").read_text(encoding="ascii") == "0\n",
            "dc return code is not zero")
    terminal = key_values(Q / "TCL_INTERNAL_COMPLETE.txt")
    require(terminal.get("status") ==
            "M1630_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED" and
            terminal.get("input_generation") ==
            "original_m993_m1006_admitted_ddc" and
            terminal.get("failed_m1614_output_used") == "false" and
            terminal.get("set_fix_hold_count") == "1" and
            terminal.get("hold_only_incremental_mapping_count") == "1" and
            terminal.get("mapped_identity_modified") == "true" and
            terminal.get("formality_required") == "true" and
            terminal.get("independent_pt_required") == "true" and
            terminal.get("paper_citable") == "false",
            "TCL terminal contract drift")
    flow = key_values(Q / "reports/flow_contract.rpt")
    require(flow.get("input_generation") ==
            "original_m993_m1006_admitted_ddc" and
            flow.get("failed_m1614_output_used") == "false" and
            flow.get("clock_period_ns") == "3.000" and
            flow.get("setup_uncertainty_ns") == "0.200" and
            flow.get("reported_hold_uncertainty_ns") == "0.050" and
            flow.get("optimization_hold_guardband_ns") == "0.051" and
            flow.get("all_compile_command_count") == "1" and
            flow.get("hold_only_incremental_mapping_count") == "1" and
            flow.get("false_path_count") == "0" and
            flow.get("multicycle_path_count") == "0" and
            flow.get("disabled_timing_arc_count") == "0" and
            flow.get("case_analysis_count") == "0",
            "flow contract drift")

    setup = timing("setup_posthold_summary_machine.txt",
                   "POST_RESTORE_REPORTED", "max", "0.002221110")
    hold = timing("hold_posthold_summary_machine.txt",
                  "POST_RESTORE_REPORTED", "min", "0.000999451")
    area_text = (Q / "reports/area_posthold.rpt").read_text(
        encoding="utf-8", errors="replace")
    match = re.search(r"Total cell area:\s*([0-9.]+)", area_text)
    require(match is not None, "area missing")
    area = float(match.group(1)); baseline = 147246.392090
    overhead = (area / baseline - 1.0) * 100.0
    require(math.isfinite(area) and area == 152898.625984 and overhead < 5.0,
            "area gate drift")
    macro = key_values(Q / "reports/macro_binding_audit.txt")
    require(macro.get("status") ==
            "PASS_M1630_RESOLVED_LIBRARY_MACRO_STRUCTURE" and
            macro.get("macro_count_pre") == macro.get("macro_count_post") ==
            macro.get("expected_macro_count") == "9" and
            macro.get("behavioral_macro_verilog_read_by_dc") == "false" and
            macro.get("inferred_parent_array_allowed") == "false",
            "macro audit drift")
    qor = (Q / "reports/qor_posthold.rpt").read_text(
        encoding="utf-8", errors="replace")
    require(re.search(r"Nets With Violations:\s+0(?:\.00)?\s*$", qor,
                      re.MULTILINE), "DRC violations nonzero/missing")

    mapped_v = (Q / "netlist/m935_m912_three_stage_exact_parent_match_"
                "product_capture_island_m1630_residual_hold_closed_mapped.v")
    vtext = mapped_v.read_text(encoding="utf-8", errors="replace")
    require(vtext.rstrip().endswith("endmodule") and
            len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\b", vtext)) == 9,
            "mapped netlist completion/macro population drift")
    sdc = (Q / "netlist/m935_m912_three_stage_exact_parent_match_"
           "product_capture_island_m1630_residual_hold_closed_mapped.sdc")
    stext = sdc.read_text(encoding="utf-8", errors="replace")
    require(len(re.findall(r"(?m)^create_clock\b", stext)) == 1 and
            re.search(r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)", stext) and
            re.search(r"set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b", stext) and
            re.search(r"set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b", stext) and
            not re.search(r"set_clock_uncertainty\s+-hold\s+0?\.051(?:0+)?\b", stext),
            "mapped SDC clock/uncertainty drift")
    for command in ("set_false_path", "set_multicycle_path", "set_min_delay",
                    "set_max_delay", "set_disable_timing", "set_case_analysis"):
        require(not re.search(r"(?m)^\s*" + command + r"\b", stext),
                "forbidden mapped SDC command: " + command)
    for name in EXPECTED_ARTIFACTS:
        require((Q / name).stat().st_size > 0,
                "empty key artifact: " + name)
    return {"setup": setup, "hold": hold,
            "area_um2": area, "baseline_area_um2": baseline,
            "area_overhead_percent": overhead, "within_five_percent": True,
            "macro_count": 9, "drc_violating_nets": 0,
            "netlist_ddc_sdc_svf_present": True}


def main():
    rows = verify_tree()
    regular(DOCS359)
    require(sha(DOCS359) == EXPECTED_DOCS359, "docs359 drift")
    require(ATTEMPT.is_dir() and not ATTEMPT.is_symlink(),
            "consumed attempt missing")
    require(not CANONICAL.exists() and not LOCK.exists() and not WORK.exists(),
            "canonical/lock/work state drift")
    attempt_manifest = ATTEMPT / "SHA256SUMS"
    attempt_outer = ATTEMPT / "SHA256SUMS.seal.sha256"
    regular(attempt_manifest); regular(attempt_outer)
    require(attempt_outer.read_text(encoding="ascii") ==
            sha(attempt_manifest) + "  SHA256SUMS\n",
            "attempt outer seal drift")
    for line in attempt_manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        regular(ATTEMPT / name)
        require(sha(ATTEMPT / name) == digest, "attempt member drift")

    log = audit_log()
    physical = audit_reports_and_netlist()
    output = {
        "schema": "m1655_m1649_c1_quarantine_forensic_r1_v1",
        "status": "PASS_FORENSIC_RECOVERABLE_DC_ARTIFACT_SET__CANONICAL_RECOVERY_SOURCE_REQUIRED",
        "python": sys.version.split()[0],
        "quarantine": {"manifest_sha256": EXPECTED_Q_MANIFEST,
            "outer_seal_file_sha256": EXPECTED_Q_OUTER_FILE,
            "members": len(rows), "topology_exact": True,
            "runner_status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
            "runner_exit_code": 3, "dc_rc": 0},
        "log_classification": log,
        "physical": physical,
        "recovery": {"forensic_recovery_feasible": True,
            "dc_rerun_allowed": False,
            "current_quarantine_citable": False,
            "canonical_copy_created": False,
            "recovery_source_review_release_required": True,
            "formality_gate_to_gate_required": True,
            "formality_direct_rtl_required": True,
            "independent_pt_required": True,
            "power_required_for_paper_ppa": True},
        "claim_boundary": {"dc_internal_completion_evidence": True,
            "dc_canonical": False, "formality": False,
            "independent_pt": False, "power": False,
            "paper_ppa_ready": False, "cycle_speedup": False,
            "system_speedup": False, "headline": False},
        "eda_launched_by_review": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
