#!/usr/bin/env python3
"""Build a fail-closed standalone M43-r3 DC/STA/Formality receipt."""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import pathlib
import re


TOP = "qfit_parent_delta_p8_l96_multicontext"
CANDIDATE_SHA = "e70239b1ec9a7d4541b0ae8d0a8f55e252fa6c804b364ab126d8201e108e0deb"
VCS_RECEIPT_SHA = "3e416d615829c9b82206547ef3ab23178bfe3e01eeb0b0ff5a789bec116fe51a"
REVIEW_SHA = "8151f8f5ab0d1038fcdfc601a78da97300613304e46991ebfac2520d180d181a"
REVIEW_VALIDATOR_SHA = "60e1488cff5e867005f32d168e1b66e62533d815d94e753777a4a2b397b3bd87"
DC_RESOLVED_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"
FM_RESOLVED_SHA = "aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b"
SLOW_LIB_SHA = "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
FAST_LIB_SHA = "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            "missing, empty, or symlink evidence: {}".format(path))
    return path.read_text(encoding="utf-8", errors="replace")


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one_number(pattern, source, cast, label):
    found = re.findall(pattern, source, re.MULTILINE)
    require(len(found) == 1, "ambiguous {}: {} matches".format(label, len(found)))
    value = cast(found[0])
    require(type(value) is cast, label + " type")
    return value


def min_slack(source, label):
    values = [float(value) for value in re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)", source)]
    require(values, label + " timing report has no slack")
    return min(values)


def point_count(path, kind):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    values = [int(value) for value in re.findall(
        r"^\s*([0-9]+)\s+{} (?:compare )?points\s*$".format(kind),
        text(path), re.MULTILINE | re.IGNORECASE)]
    return max(values) if values else 0


def unmatched_count(source, label):
    no_unmatched = len(re.findall(r"^No unmatched points\.$", source, re.MULTILINE))
    require(no_unmatched <= 1, "ambiguous no-unmatched terminal")
    if no_unmatched == 1:
        require(not re.search(r"Unmatched reference\(implementation\)", source),
                "contradictory unmatched report")
        return 0
    values = [int(value) for value in re.findall(
        r"^\s*([0-9]+)\([0-9]+\) Unmatched reference\(implementation\) "
        + label + r"\s*$", source, re.MULTILINE)]
    require(values, "unmatched row missing: " + label)
    return values[-1]


def audit_value(source, name):
    return one_number(r"^{}=([0-9]+)$".format(re.escape(name)), source, int, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite receipt")
    run = args.run
    snapshot = args.snapshot

    for stage in ("dc", "sta", "formality"):
        require((run / (stage + ".rc")).read_bytes() == b"0\n", stage + " rc")
    area = text(run / "reports/sta_area.rpt")
    setup_report = text(run / "reports/sta_setup.rpt")
    hold_report = text(run / "reports/sta_hold.rpt")
    dc_log = text(run / "dc.raw.log")
    sta_log = text(run / "sta.raw.log")
    fm_log = text(run / "formality.raw.log")
    unmatched = text(run / "reports/formality_unmatched.rpt")
    audit = text(run / "reports/m43_r3_structural_audit.rpt")
    for label, source in (("DC", dc_log), ("STA", sta_log), ("Formality", fm_log)):
        require(not re.search(r"^(Error|Fatal):", source, re.MULTILINE),
                label + " Error/Fatal")
        require("Thank you" in source, label + " terminal marker")

    cells = one_number(r"^Number of cells:\s+([0-9]+)", area, int, "cells")
    comb_cells = one_number(
        r"^Number of combinational cells:\s+([0-9]+)", area, int,
        "combinational cells")
    seq_cells = one_number(
        r"^Number of sequential cells:\s+([0-9]+)", area, int,
        "sequential cells")
    macros = one_number(
        r"^Number of macros/black boxes:\s+([0-9]+)", area, int,
        "macro/black-box cells")
    comb_area = one_number(
        r"^Combinational area:\s+([0-9.]+)", area, float,
        "combinational area")
    seq_area = one_number(
        r"^Noncombinational area:\s+([0-9.]+)", area, float,
        "noncombinational area")
    total_area = one_number(
        r"^Total cell area:\s+([0-9.]+)", area, float, "total cell area")
    setup_wns = min_slack(setup_report, "setup")
    hold_wns = min_slack(hold_report, "hold")

    succeeded = len(re.findall(r"^Verification SUCCEEDED$", fm_log, re.MULTILINE))
    passing = [int(value) for value in re.findall(
        r"^\s*([0-9]+) Passing compare points\s*$", fm_log, re.MULTILINE)]
    failing_rows = re.findall(
        r"^Failing \(not equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$",
        fm_log, re.MULTILINE)
    require(succeeded == 1 and len(passing) == 1 and failing_rows,
            "ambiguous Formality terminal result")
    failing_columns = [int(value) for value in failing_rows[-1].split()]
    formality = {
        "verification_succeeded_terminal_count": succeeded,
        "passing_compare_points": passing[-1],
        "failing_compare_points": failing_columns[-1],
        "failing_result_columns": failing_columns,
        "aborted_compare_points": point_count(
            run / "reports/formality_aborted.rpt", "Aborted"),
        "unverified_compare_points": point_count(
            run / "reports/formality_unverified.rpt", "Unverified"),
        "unmatched_compare_points": unmatched_count(unmatched, r"compare points"),
        "unmatched_primary_or_blackbox_points": unmatched_count(
            unmatched, r"primary inputs, black-box outputs"),
        "unmatched_unread_points_diagnostic_only": unmatched_count(
            unmatched, r"unread points"),
        "fmr_elab_147_count": len(re.findall(r"FMR_ELAB-147", fm_log)),
        "message_filters_used": False,
    }
    formality["all_gates_pass"] = all([
        formality["passing_compare_points"] > 0,
        formality["failing_compare_points"] == 0,
        formality["aborted_compare_points"] == 0,
        formality["unverified_compare_points"] == 0,
        formality["unmatched_compare_points"] == 0,
        formality["unmatched_primary_or_blackbox_points"] == 0,
        formality["fmr_elab_147_count"] == 0,
    ])

    structural = {
        "physical_multiplier_hit_count": audit_value(
            audit, "physical_multiplier_hit_total"),
        "postcompile_reference_blackbox_attribute_count": audit_value(
            audit, "postcompile_reference_blackbox_attribute_count"),
        "area_macro_or_blackbox_cell_count": audit_value(
            audit, "area_macro_or_blackbox_cell_count"),
        "unresolved_link_signature_count": audit_value(
            audit, "unresolved_link_signature_count"),
    }
    structural["all_gates_pass"] = all(value == 0 for value in structural.values())

    warnings = [line for line in dc_log.splitlines() if line.startswith("Warning:")]
    warning_codes = collections.Counter()
    for line in warnings:
        match = re.search(r"\(([A-Z][A-Z0-9_-]*-[0-9]+)\)\s*$", line)
        warning_codes[match.group(1) if match else "UNCLASSIFIED"] += 1

    identity_paths = {
        "candidate_rtl_sha256":
            "hw_autoresearch_nts07/rtl_m43/qfit_parent_delta_p8_l96_multicontext.sv",
        "vcs_receipt_sha256":
            "hw_autoresearch_nts07/contracts/m43_r2_exact_sha_vcs_receipt_r1_20260823.json",
        "independent_review_sha256":
            "hw_autoresearch_nts07/results/m43_r2_independent_hammer_review_20260823/m43_r2_independent_hammer_review.json",
        "independent_review_validator_sha256":
            "hw_autoresearch_nts07/results/m43_r2_independent_hammer_review_20260823/validate_m43_r2_independent_hammer_review.py",
        "dc_resolved_binary_sha256": "tools/dc_resolved_binary",
        "formality_resolved_binary_sha256": "tools/formality_resolved_binary",
        "setup_library_sha256": "libraries/tcbn28hpcplusbwp35p140ssg0p9v125c.db",
        "hold_library_sha256": "libraries/tcbn28hpcplusbwp35p140ffg1p05vm40c.db",
    }
    exact_identity = dict((key, sha(snapshot / relative))
                          for key, relative in identity_paths.items())
    expected_identity = {
        "candidate_rtl_sha256": CANDIDATE_SHA,
        "vcs_receipt_sha256": VCS_RECEIPT_SHA,
        "independent_review_sha256": REVIEW_SHA,
        "independent_review_validator_sha256": REVIEW_VALIDATOR_SHA,
        "dc_resolved_binary_sha256": DC_RESOLVED_SHA,
        "formality_resolved_binary_sha256": FM_RESOLVED_SHA,
        "setup_library_sha256": SLOW_LIB_SHA,
        "hold_library_sha256": FAST_LIB_SHA,
    }
    require(exact_identity == expected_identity, "exact identity drift")

    all_pass = bool(
        macros == 0 and setup_wns >= 0.0 and hold_wns >= 0.0
        and formality["all_gates_pass"] and structural["all_gates_pass"])
    peak_adds = 768
    conditional_destination_adds = 1536
    receipt = {
        "schema": "m43_r3_exact_sha_synopsys_receipt_v1",
        "date": "2026-08-23",
        "status": "PASS_EXACT_SHA_FRESH_M43_R3_DC_STA_FORMALITY"
                  if all_pass else "FAIL_GATE_DO_NOT_CITE",
        "candidate_changed": False,
        "exact_identity": exact_identity,
        "flow_contract": {
            "top": TOP,
            "technology_nm": 28,
            "clock_period_ns": 3.0,
            "clock_frequency_mhz_nominal": 333.3333333333333,
            "setup_corner": "ssg0p9v125c",
            "hold_corner": "ffg1p05vm40c",
            "physical_scope": "standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro",
            "fresh_dc_sta_formality": True,
            "foreground_sequential_no_tee_no_background": True,
        },
        "logic_only_ppa": {
            "cell_count": cells,
            "combinational_cell_count": comb_cells,
            "sequential_cell_count": seq_cells,
            "macro_or_blackbox_cell_count": macros,
            "combinational_area_um2": comb_area,
            "noncombinational_area_um2": seq_area,
            "total_cell_area_um2": total_area,
            "setup_wns_ns_slow_ssg0p9v125c": setup_wns,
            "hold_wns_ns_fast_ffg1p05vm40c": hold_wns,
            "dc_warning_count": len(warnings),
            "dc_warning_codes": dict(sorted(warning_codes.items())),
        },
        "formality": formality,
        "structural_audit": structural,
        "peak_compute_contract": {
            "implemented_peak_signed_adds_per_cycle": peak_adds,
            "implemented_peak_signed_adds_per_cycle_derivation":
                "8 accepted weight banks times 96 output lanes",
            "implemented_peak_signed_adds_per_cycle_per_mm2":
                peak_adds * 1000000.0 / total_area,
            "implemented_area_um2_per_peak_signed_add_per_cycle":
                total_area / peak_adds,
            "conditional_k2_dual_destination_adds_per_cycle":
                conditional_destination_adds,
            "conditional_k2_dual_destination_adds_per_cycle_per_mm2":
                conditional_destination_adds * 1000000.0 / total_area,
            "conditional_k2_area_um2_per_destination_add_per_cycle":
                total_area / conditional_destination_adds,
            "conditional_k2_status":
                "ARCHITECTURAL_PROJECTION_NOT_IMPLEMENTED_OR_MEASURED_IN_THIS_BLOCK",
            "conditional_k2_assumption":
                "the same accepted parent delta is broadcast/reused for two destinations without duplicating this engine or adding a second read",
        },
        "gates": {
            "all_pass": all_pass,
            "dc_rc_zero": True,
            "sta_rc_zero": True,
            "formality_rc_zero": True,
            "setup_wns_nonnegative": setup_wns >= 0.0,
            "hold_wns_nonnegative": hold_wns >= 0.0,
            "formality_all_pass": bool(formality["all_gates_pass"]),
            "zero_multiplier_blackbox_unresolved": bool(structural["all_gates_pass"]),
        },
        "claim_boundary": {
            "permitted":
                "exact-SHA standalone logic-only fresh 3ns DC/STA/Formality area, cell, timing, equivalence, zero-multiplier/black-box/link audit, implemented peak-add area density, and explicitly conditional K2 area-density projection",
            "not_admitted":
                "placed/routed or SRAM-macro-inclusive PPA, clock-tree or routed-wire accuracy, power, energy, parent DAG/memory integration, measured K2 implementation, full-network cycles or speedup, accuracy, external accelerator comparison, DATE headline, or best-paper status",
            "paper_ppa_ready": False,
            "system_speedup_admitted": False,
            "power_or_energy_admitted": False,
        },
    }
    require(all_pass, "one or more strict M43-r3 gates failed")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
