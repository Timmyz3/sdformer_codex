#!/usr/bin/env python3
"""Read-only fail-closed result hammer for the future canonical M962 DC result.

The default invocation is inert and does not inspect any DC output.  An
independent reviewer may pass ``--review-complete-canonical`` only after the
M962 runner has atomically published the canonical, recursively sealed result
directory.  The script never invokes EDA, never reads the transient work
prefix, and treats a complete setup-negative run as valid negative evidence.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
CANON = HW / "dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829"
ATTEMPT = HW / "dc_handoff/runs/.m962_m935_three_stage_match_macro_aware_dc_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m962_m935_three_stage_match_macro_aware_dc_launch_lock"
SOURCE = HW / "contracts/m962_m960_m959_m935_three_stage_match_macro_aware_dc_source_contract_r1_20260829.json"
RELEASE = HW / "contracts/m964_m963_m962_m960_m935_c1_macro_aware_dc_launch_release_r1_20260829.json"
M963 = HW / "reviews/m963_m962_m960_m935_c1_macro_aware_dc_source_hammer_r1_20260829"
M965 = HW / "reviews/m965_m964_m963_m962_m935_c1_macro_aware_dc_final_launch_release_hammer_r1_20260829"
RUNNER = HW / "dc_handoff/scripts/run_dc_m962_m935_three_stage_match_macro_aware_exact_sha_r1.sh"
TCL = HW / "dc_handoff/scripts/run_dc_m962_m935_three_stage_match_macro_aware_candidate.tcl"
SDC = HW / "dc_handoff/constraints/date_m962_m935_three_stage_match_macro_aware_3ns.sdc"
FILELIST = HW / "dc_handoff/filelists/date_m962_m935_three_stage_match_macro_aware_dc.f"
RTL = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_WRAPPER = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "0fb76a04b7eddba81a094b4a204f7b2227f642d384da8c574805b2f0b4cf00df",
    "release": "9d47a2c204bf89204ec124214ed64935a8fcc401d2ed34f5a881006f8c3bb1d2",
    "runner": "7ec1138696c40b923d6841dc21749aed35e93da266e00910b6715278c51da7fd",
    "tcl": "43be734a82b5061af39e66304e5fbf9bd34c36af45184509317c479ea59367df",
    "sdc": "a05e95e59611a74b239274d579befe1ab8d04f7684ad15ec85012c05d72b3014",
    "filelist": "e6d9d1ead574e7c4cc446981888aa404d2d92ecd321a6855a43ea498c501e75c",
    "rtl": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "macro_wrapper": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m963_review": "14e89cbd134844da81f6da2946feb07c7dada8e2dc9e9635ba4cd42aa8ada812",
    "m963_manifest": "eb71568744eab1dc92976baa6e52e8f8441e706fe6cc86ec93803968130387e7",
    "m963_outer": "767f6eec5a69cc3b9b69545249e263b5bf9c14486568d8ec44db56082a3e9b10",
    "m965_report": "ef84db7a8af9c27b91c409ded426ab0ef31ad89fbeb89462dec3dc32b5332cf2",
    "m965_manifest": "78150563b64be943afe1d9cd670458fa9c3d35ee940de17f32e7e393916977d1",
    "m965_outer": "409ce8edc54afbf86b8bccd7232b14bbce32c554ba988290bd5f2c7de5014621",
}

MACRO_CELL = "TS1N28HPCPHVTB128X128M4S"
DESIGN = "m935_m912_three_stage_exact_parent_match_product_capture_island"
REQUIRED = [
    "admission.txt", "dc.log", "dc.rc", "m962_dc_receipt.json",
    "RUN_COMPLETE.txt", "TCL_PASS_TERMINAL.txt",
    "reports/link.rpt", "reports/macro_binding_audit.txt",
    "reports/check_design_precompile.rpt", "reports/check_design_postcompile.rpt",
    "reports/check_timing_precompile.rpt", "reports/check_timing_postcompile.rpt",
    "reports/resources_precompile.rpt", "reports/resources_postcompile.rpt",
    "reports/references_precompile.rpt", "reports/references_postcompile.rpt",
    "reports/hierarchy_postcompile.rpt", "reports/qor.rpt",
    "reports/area_hierarchy.rpt", "reports/clocks.rpt",
    "reports/timing_setup_top100.rpt", "reports/constraint_setup_all.rpt",
    "reports/constraint_max_capacitance.rpt",
    "reports/constraint_max_transition.rpt", "reports/constraint_max_fanout.rpt",
    "reports/flow_contract.rpt", "reports/precompile_loop_gate.rpt",
    "reports/setup_summary_machine.txt",
    "netlist/%s_mapped.v" % DESIGN, "netlist/%s_mapped.sdc" % DESIGN,
    "netlist/%s.ddc" % DESIGN, "netlist/%s.svf" % DESIGN,
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    return path.is_file() and not path.is_symlink()


def kv(path):
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def verify_file_sidecars(payload, errors, label):
    sidecar = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    for path, name in ((payload, "payload"), (sidecar, "sidecar"), (outer, "outer")):
        require(regular(path), "%s missing/nonregular %s" % (label, name), errors)
    if not all(regular(p) for p in (payload, sidecar, outer)):
        return
    fields = sidecar.read_text().strip().split()
    require(len(fields) == 2 and fields[1].lstrip("*") == payload.name,
            "%s malformed sidecar" % label, errors)
    if fields:
        require(fields[0] == sha(payload), "%s payload SHA mismatch" % label, errors)
    outer_fields = outer.read_text().strip().split()
    require(len(outer_fields) == 2 and outer_fields[1].lstrip("*") == sidecar.name,
            "%s malformed outer sidecar" % label, errors)
    if outer_fields:
        require(outer_fields[0] == sha(sidecar), "%s outer sidecar mismatch" % label, errors)


def verify_directory(directory, errors, label):
    """Verify a recursively exact sealed directory and return seal metadata."""
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(),
            "%s directory missing/symlink" % label, errors)
    require(regular(manifest), "%s SHA256SUMS missing/nonregular" % label, errors)
    require(regular(outer), "%s outer seal missing/nonregular" % label, errors)
    if not regular(manifest) or not regular(outer):
        return {}
    outer_fields = outer.read_text().strip().split()
    require(len(outer_fields) == 2 and outer_fields[1].lstrip("*") == "SHA256SUMS",
            "%s malformed outer seal" % label, errors)
    if outer_fields:
        require(outer_fields[0] == sha(manifest), "%s outer seal mismatch" % label, errors)
    listed = {}
    for line_no, line in enumerate(manifest.read_text().splitlines(), 1):
        fields = line.split(None, 1)
        require(len(fields) == 2, "%s malformed manifest line %d" % (label, line_no), errors)
        if len(fields) != 2:
            continue
        digest, rel = fields
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
                "%s unsafe manifest path %s" % (label, rel), errors)
        require(rel not in listed, "%s duplicate manifest path %s" % (label, rel), errors)
        listed[rel] = digest
        target = directory / rel
        require(regular(target), "%s missing/nonregular %s" % (label, rel), errors)
        if regular(target):
            require(sha(target) == digest, "%s digest mismatch %s" % (label, rel), errors)
    actual = set()
    symlinks = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            symlinks.append(str(path.relative_to(directory)))
        elif path.is_file():
            actual.add(str(path.relative_to(directory)))
    expected = set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(actual == expected,
            "%s exact-set mismatch missing=%s extra=%s" %
            (label, sorted(expected - actual), sorted(actual - expected)), errors)
    require(not symlinks, "%s symlinks present %s" % (label, symlinks), errors)
    return {
        "manifest_sha256": sha(manifest),
        "outer_seal_file_sha256": sha(outer),
        "manifest_entries": len(listed),
        "actual_regular_files_including_seals": len(actual),
        "symlink_count": len(symlinks),
    }


def exact_float(pattern, text, label, errors):
    match = re.search(pattern, text, re.M)
    require(match is not None, "missing %s" % label, errors)
    return float(match.group(1)) if match else float("nan")


def exact_int(pattern, text, label, errors):
    match = re.search(pattern, text, re.M)
    require(match is not None, "missing %s" % label, errors)
    return int(match.group(1)) if match else None


def prepublication_gate():
    """Return WAIT without opening result payloads unless publication is complete."""
    if LOCK.exists():
        return "WAIT_M962_RUNNER_LOCK_PRESENT"
    if not CANON.is_dir() or CANON.is_symlink():
        return "WAIT_M962_CANONICAL_RESULT_NOT_PUBLISHED"
    for name in ("RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256"):
        if not regular(CANON / name):
            return "WAIT_M962_CANONICAL_RESULT_NOT_DOUBLE_SEALED"
    return None


def review_complete_result():
    errors = []
    result_seal = verify_directory(CANON, errors, "M962 canonical result")
    attempt_seal = verify_directory(ATTEMPT, errors, "M962 attempt")
    verify_directory(M963, errors, "M963 source hammer")
    verify_directory(M965, errors, "M965 launch-release hammer")
    verify_file_sidecars(SOURCE, errors, "M962 source contract")
    verify_file_sidecars(RELEASE, errors, "M964 release")

    identities = {
        "source": SOURCE, "release": RELEASE, "runner": RUNNER, "tcl": TCL,
        "sdc": SDC, "filelist": FILELIST, "rtl": RTL,
        "macro_wrapper": MACRO_WRAPPER, "docs359": DOC359,
        "m963_review": M963 / "review.json", "m963_manifest": M963 / "SHA256SUMS",
        "m963_outer": M963 / "SHA256SUMS.seal.sha256",
        "m965_report": M965 / "hammer_report.json", "m965_manifest": M965 / "SHA256SUMS",
        "m965_outer": M965 / "SHA256SUMS.seal.sha256",
    }
    for key, path in identities.items():
        require(regular(path), "%s identity missing/nonregular" % key, errors)
        if regular(path):
            require(sha(path) == EXPECTED[key], "%s identity SHA mismatch" % key, errors)

    for rel in REQUIRED:
        require(regular(CANON / rel), "required result artifact missing/nonregular: %s" % rel, errors)
    if errors:
        return {"status": "FAIL_M973_RESULT_INTEGRITY", "errors": errors}

    attempt = kv(ATTEMPT / "ATTEMPT_CONSUMED.txt")
    require(attempt.get("status") == "M962_ATTEMPT_CONSUMED", "attempt status", errors)
    require(attempt.get("max_dc_runs") == "1", "attempt max_dc_runs", errors)
    require(attempt.get("retry") == "false", "attempt retry", errors)
    require((CANON / "dc.rc").read_text().strip() == "0", "dc.rc is not zero", errors)

    admission = kv(CANON / "admission.txt")
    require(admission.get("status") == "M962_DC_ATTEMPT_ADMITTED", "admission status", errors)
    require(admission.get("clock_period_ns") == "3.000", "admission clock", errors)
    require(admission.get("macro_count") == "9", "admission macro count", errors)
    require(admission.get("false_paths") == "0", "admission false path count", errors)

    flow = kv(CANON / "reports/flow_contract.rpt")
    expected_flow = {
        "flow": "m962_m935_three_stage_match_macro_aware_candidate",
        "clock_period_ns": "3.000", "ideal_clock": "true", "wireload": "ZeroWireload",
        "compile_ultra_count": "1", "incremental_compile_count": "0",
        "hold_fix_command_count": "0", "false_path_count": "0",
        "multicycle_path_count": "0", "disabled_timing_arc_count": "0",
        "setup_failure_is_sealed_negative_evidence": "true", "hold_diagnostic_only": "true",
    }
    for key, expected in expected_flow.items():
        require(flow.get(key) == expected, "flow contract %s" % key, errors)

    loop = kv(CANON / "reports/precompile_loop_gate.rpt")
    require(loop.get("TIM-209") == "0", "precompile TIM-209", errors)
    require(loop.get("OPT-150") == "0", "precompile OPT-150", errors)
    require(loop.get("status") == "PASS_PRECOMPILE_LOOP_GATE", "precompile loop gate", errors)

    terminal = kv(CANON / "TCL_PASS_TERMINAL.txt")
    require(terminal.get("status") == "PASS_M962_DC_EXECUTION_AND_REPORT_CLOSURE",
            "Tcl terminal closure", errors)
    require(terminal.get("TIM-209") == "0" and terminal.get("OPT-150") == "0",
            "Tcl terminal loop diagnostics", errors)
    require(terminal.get("macro_count_pre") == "9" and terminal.get("macro_count_post") == "9",
            "Tcl terminal macro counts", errors)
    require(terminal.get("hold_signoff") == "false", "terminal hold boundary", errors)
    require(terminal.get("power_measured") == "false", "terminal power boundary", errors)

    macro = kv(CANON / "reports/macro_binding_audit.txt")
    require(macro.get("status") == "PASS_M962_RESOLVED_LIBRARY_MACRO_STRUCTURE",
            "macro audit status", errors)
    require(macro.get("macro_cell") == MACRO_CELL, "macro cell identity", errors)
    require(macro.get("macro_count_pre") == "9" and macro.get("macro_count_post") == "9",
            "macro pre/post count", errors)
    require(macro.get("expected_macro_count") == "9", "expected macro count", errors)
    require(macro.get("behavioral_macro_verilog_read_by_dc") == "false",
            "behavioral macro entered DC", errors)
    require(macro.get("inferred_parent_array_allowed") == "false",
            "inferred parent array allowed", errors)

    mapped = (CANON / ("netlist/%s_mapped.v" % DESIGN)).read_text(errors="replace")
    mapped_macro_instances = len(re.findall(r"\b%s\b" % re.escape(MACRO_CELL), mapped))
    require(mapped_macro_instances == 9,
            "mapped netlist macro instances=%d expected=9" % mapped_macro_instances, errors)

    wrapper = MACRO_WRAPPER.read_text(errors="replace")
    require("wire [6:0] macro_address" in wrapper, "macro wrapper lacks 7-bit physical address", errors)
    require("wire [6:0] macro_address = {1'b0, address};" in wrapper,
            "macro wrapper logical 64-row binding changed", errors)
    require("slice < 9" in wrapper, "macro wrapper nine-slice geometry changed", errors)
    require("[1151:0]" in wrapper, "macro wrapper 9x128 width changed", errors)

    summary = kv(CANON / "reports/setup_summary_machine.txt")
    status = summary.get("status")
    require(status in ("MET", "VIOLATED_CAPTURED"), "setup summary status", errors)
    try:
        wns = float(summary["setup_wns_ns"])
        tns = float(summary["setup_tns_ns"])
        violations = int(summary["setup_violating_paths"])
    except (KeyError, ValueError):
        errors.append("setup summary numeric parse")
        wns = tns = float("nan")
        violations = -1
    require(math.isfinite(wns) and math.isfinite(tns), "nonfinite WNS/TNS", errors)
    setup_met = status == "MET"
    if setup_met:
        require(violations == 0 and wns >= 0.0 and abs(tns) < 1e-9,
                "MET setup tuple inconsistent", errors)
    else:
        require(violations > 0 and wns < 0.0 and tns < 0.0,
                "VIOLATED_CAPTURED setup tuple inconsistent", errors)

    top100 = (CANON / "reports/timing_setup_top100.rpt").read_text(errors="replace")
    violated_reports = len(re.findall(r"slack \(VIOLATED\)", top100))
    met_reports = len(re.findall(r"slack \(MET\)", top100))
    reported_paths = violated_reports + met_reports
    require(reported_paths > 0 and reported_paths <= 100, "top100 report path count", errors)
    if setup_met:
        require(violated_reports == 0 and met_reports > 0,
                "MET result top100 contains no MET path or has violation", errors)
    else:
        require(violated_reports == min(100, violations),
                "negative result did not preserve min(100, violating_paths) paths", errors)
        require(met_reports == 0, "negative top100 unexpectedly mixes MET paths", errors)

    area_text = (CANON / "reports/area_hierarchy.rpt").read_text(errors="replace")
    total_cells = exact_int(r"^Number of cells:\s+([0-9]+)\s*$", area_text,
                            "hierarchical cell count", errors)
    comb_area = exact_float(r"^Combinational area:\s+([0-9.]+)\s*$", area_text,
                            "combinational area", errors)
    noncomb_area = exact_float(r"^Noncombinational area:\s+([0-9.]+)\s*$", area_text,
                               "noncombinational area", errors)
    macro_area = exact_float(r"^Macro/Black Box area:\s+([0-9.]+)\s*$", area_text,
                             "macro area", errors)
    total_cell_area = exact_float(r"^Total cell area:\s+([0-9.]+)\s*$", area_text,
                                  "total cell area", errors)
    require(all(math.isfinite(x) and x >= 0 for x in
                (comb_area, noncomb_area, macro_area, total_cell_area)),
            "invalid area tuple", errors)
    require(macro_area > 0.0 and total_cell_area > macro_area,
            "macro/total area incomplete", errors)
    require(abs((comb_area + noncomb_area + macro_area) - total_cell_area) < 0.01,
            "area components do not sum to total cell area", errors)

    receipt = json.loads((CANON / "m962_dc_receipt.json").read_text())
    expected_receipt_status = (
        "PASS_RAW_M962_3NS_SETUP_AREA_CANDIDATE_PENDING_RESULT_HAMMER"
        if setup_met else "SEALED_NEGATIVE_M962_3NS_SETUP_VIOLATION_PENDING_RESULT_HAMMER"
    )
    require(receipt.get("status") == expected_receipt_status, "receipt status", errors)
    require(receipt.get("macro_cell") == MACRO_CELL and receipt.get("macro_count") == 9,
            "receipt macro identity", errors)
    require(abs(float(receipt.get("total_cell_area_um2_dc_reported", -1)) - total_cell_area) < 1e-6,
            "receipt area mismatch", errors)
    receipt_setup = receipt.get("setup", {})
    require(receipt_setup.get("met") is setup_met, "receipt setup.met mismatch", errors)
    require(abs(float(receipt_setup.get("wns_ns", float("nan"))) - wns) < 1e-9,
            "receipt WNS mismatch", errors)
    require(abs(float(receipt_setup.get("tns_ns", float("nan"))) - tns) < 1e-9,
            "receipt TNS mismatch", errors)
    require(receipt_setup.get("violating_paths") == violations, "receipt violation count", errors)
    require(receipt_setup.get("top100_report_preserved") is True,
            "receipt top100 preservation claim", errors)
    receipt_identity = receipt.get("identity", {})
    require(receipt_identity.get("runner_sha256") == EXPECTED["runner"], "receipt runner SHA", errors)
    require(receipt_identity.get("source_contract_sha256") == EXPECTED["source"],
            "receipt source SHA", errors)
    require(receipt_identity.get("release_sha256") == EXPECTED["release"],
            "receipt release SHA", errors)

    complete = kv(CANON / "RUN_COMPLETE.txt")
    require(complete.get("status") == expected_receipt_status, "RUN_COMPLETE status", errors)
    require(complete.get("setup_met") == str(setup_met).lower(), "RUN_COMPLETE setup_met", errors)
    for key in ("hold_signoff", "power", "speedup", "paper_ppa_ready"):
        require(complete.get(key) == "false", "RUN_COMPLETE %s" % key, errors)

    dc_log = (CANON / "dc.log").read_text(errors="replace")
    fatal = re.search(
        r"(^|[^A-Za-z])(Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+)",
        dc_log, re.I | re.M)
    require(fatal is None, "fatal/link evidence in dc.log", errors)

    result_integrity = not errors
    physical_admission = result_integrity and setup_met
    return {
        "schema": "m973_m962_m935_c1_macro_aware_dc_result_hammer_recompute_v1",
        "status": ("PASS_M973_RESULT_INTEGRITY_SETUP_MET" if physical_admission else
                   "PASS_M973_RESULT_INTEGRITY_SETUP_VIOLATED" if result_integrity else
                   "FAIL_M973_RESULT_INTEGRITY"),
        "errors": errors,
        "identity": {
            "canonical_result": str(CANON.relative_to(HW)),
            "canonical_seal": result_seal,
            "attempt_seal": attempt_seal,
            "source_sha256": EXPECTED["source"],
            "release_sha256": EXPECTED["release"],
            "m965_report_sha256": EXPECTED["m965_report"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "physical_point": {
            "technology_nm": 28, "clock_period_ns": 3.0,
            "ideal_clock": True, "wireload": "ZeroWireload",
            "macro_cell": MACRO_CELL, "mapped_macro_instances": mapped_macro_instances,
            "physical_macro_geometry": "9 x 128 rows x 128 bits, single-port 1RW",
            "physical_parent_macro_capacity_bytes": 9 * 128 * 128 // 8,
            "logical_addressable_rows": 64,
            "logical_parent_payload_bytes": 9 * 64 * 128 // 8,
            "full_213376B_storage_obligation_integrated": False,
        },
        "area": {
            "hierarchical_cells": total_cells,
            "combinational_area_um2": comb_area,
            "noncombinational_area_um2": noncomb_area,
            "macro_area_um2": macro_area,
            "total_cell_area_um2": total_cell_area,
            "net_area": "undefined_under_ZeroWireload",
        },
        "setup": {
            "met": setup_met, "wns_ns": wns, "tns_ns": tns,
            "violating_paths": violations, "top100_reported_paths": reported_paths,
            "top100_violated_paths": violated_reports,
        },
        "admission": {
            "result_integrity": result_integrity,
            "setup_area_component_candidate": physical_admission,
            "complete_negative_evidence": result_integrity and not setup_met,
        },
        "claim_boundary": {
            "cpu_same_ledger_1p746753_promoted_to_rtl_cycle": False,
            "rtl_cycles_measured": False, "speedup": False, "system_speedup": False,
            "hold_signoff": False, "power": False, "energy": False,
            "full_storage_macro_integrated": False, "ppa": False,
            "paper_ppa_ready": False, "headline": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-complete-canonical", action="store_true")
    args = parser.parse_args()
    if not args.review_complete_canonical:
        print(json.dumps({
            "status": "STATIC_SOURCE_ONLY_NO_RESULT_READ",
            "canonical_result_read": False,
            "eda_started": False,
            "instruction": "Wait for an independent completion notice, then pass --review-complete-canonical.",
        }, indent=2, sort_keys=True))
        return 0
    wait = prepublication_gate()
    if wait:
        print(json.dumps({"status": wait, "canonical_result_read": False,
                          "eda_started": False}, indent=2, sort_keys=True))
        return 3
    result = review_complete_result()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status", "").startswith("PASS_M973_RESULT_INTEGRITY") else 2


if __name__ == "__main__":
    sys.exit(main())
