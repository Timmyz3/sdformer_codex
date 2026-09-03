#!/usr/bin/env python3
"""Unsealed independent-result-hammer source draft for future M1808 evidence.

Static mode reads only this draft and its checklist.  Canonical audit mode is
inert unless a caller explicitly supplies five exact SHA256 pins after the
M1808 canonical result exists.  This draft never launches EDA, creates a
review, writes a PASS token, or seals output.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
CHECKLIST = HERE / "checklist.json"
HW = HERE.parent.parent
RESULT = HW / "results/m1808_c3_mapped_energy_r1_20260902"
ATTEMPT = HW / "results/.m1808_c3_mapped_energy_attempt_consumed"
M1808_FAILURE = HW / "results/m1808_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1808_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"
M1815_REVIEW = (HW /
    "reviews/m1815_m1808_c3_m1454_fixed_t10_mapped_energy_source_hammer_r1_20260902/review.json")

RESULT_STATUS = "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER"
RUN_TOKEN = "PASS_M1808_C3_M1454_FIXED_T10_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER"
RUNTIME_TOKEN = "PASS_M1808_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY"
TAG_TOKEN = "PASS_M1808_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD"
PTPX_TOKEN = "PASS_M1790_C3_M1454_FIXED_T10_MAPPED_COMPONENT_PTPX_TOOL_COMPLETE"
TOP = "tb_m1808_c3_m1454_fixed_t10_mapped_energy"
SAIF_SCOPE = TOP + ".dut"
DESIGN = "m518_matched_fixed_t10_atlif"
OLD_FAILURE_SHA = "aea36bcd319f89be78ba7a6b26f0ec02acf17f095601ad227ac194759f063427"

EXPECTED_IDENTITY = {
    "netlist_sha256": "7c01af42322b8feed904df2862aac6e21cbe165b988f1b248f2e94d23f23a7a7",
    "sdc_sha256": "bb3697e833cb987e4a85ab2a62b4f40946a8c3d6b7eaba08504570f5a862f23f",
    "runner_sha256": "17262b329a130c027d3be4b0a912ac75a34d63bc29c568372433a5126d6d6e51",
    "source_contract_sha256": "cfba88c6866dbcd67a97680f0276dba53443b95bd44d00732aa134c67cb11c92",
    "source_review_json_sha256": "5a5ecdd93d78033c842b5985028b243eea71361b360e27513d2e9361a6870092",
    "launch_release_sha256": "c948d79fb6fd93a2d4f33b6c16c83c33b6a2985cdaef7d928e63fc292dc3549f",
}
ONE_SHOT = {
    "vcs_compiles": 1, "simv_runs": 1, "saif_files": 1,
    "ptpx_runs": 1, "automatic_retry": False,
    "reuse_prior_simv_saif_ptpx": False,
}
ATTEMPT_UNIQUENESS = {
    "attempt_latch": "results/.m1808_c3_mapped_energy_attempt_consumed",
    "canonical_result": "results/m1808_c3_mapped_energy_r1_20260902",
    "failure_result": "results/m1808_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine",
    "private_build": "results/m1808_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite",
    "prelaunch_namespaces_required_absent": True,
    "no_replace_atomic_publish": True,
    "automatic_retry": False,
}
EXPECTED_CLAIM = {
    "directed_component_workload": True,
    "prelayout_logic_only": True,
    "tt_0p9v_25c": True,
    "ideal_clock": True,
    "zero_wireload": True,
    "spef": False,
    "macro_count": 0,
    "component_power": True,
    "directed_window_energy": True,
    "energy_per_frame": False,
    "speedup": False,
    "system_speedup": False,
    "silicon": False,
    "signoff": False,
    "paper_ppa_ready": False,
    "headline": False,
}
DRAFT_CLAIM = {
    "mapped_component_functionality": False,
    "dut_only_saif": False,
    "component_power": False,
    "directed_window_energy": False,
    "prelayout_logic_only": True,
    "ideal_clock": True,
    "spef": False,
    "macro_count": 0,
    "energy_per_frame": False,
    "speedup": False,
    "system_speedup": False,
    "silicon": False,
    "signoff": False,
    "paper_ppa_ready": False,
    "headline": False,
}
EXPECTED_MEMBERS = {
    "RUN_COMPLETE.txt",
    "compile.log",
    "runtime.json",
    "metrics.json",
    "receipt.json",
    "candidate/m1808_c3_fixed_t10_component.saif",
    "candidate/mapped_sim.log",
    "candidate/ptpx/PTPX_INTERNAL_COMPLETE.txt",
    "candidate/ptpx/ptpx.log",
    "candidate/ptpx/reports/saif_annotation_summary.rpt",
    "candidate/ptpx/reports/inconsistent_annotation.rpt",
    "candidate/ptpx/reports/switching_coverage.rpt",
    "candidate/ptpx/reports/switching_annotated.rpt",
    "candidate/ptpx/reports/switching_unannotated.rpt",
    "candidate/ptpx/reports/check_timing.rpt",
    "candidate/ptpx/reports/check_power.rpt",
    "candidate/ptpx/reports/ptpx_whole_mapped_c3_logic.rpt",
    "candidate/ptpx/reports/ptpx_hierarchy_diagnostic.rpt",
    "candidate/ptpx/reports/ptpx_verbose.rpt",
    "candidate/ptpx/reports/scope_and_boundary.rpt",
}


class AuditFailure(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_pin(value, name):
    need(type(value) is str and len(value) == 64
         and all(char in "0123456789abcdef" for char in value),
         "invalid caller pin " + name)
    return value


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AuditFailure("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root " + str(path))
    return value


def validate_checklist_data(value, script_text):
    need(value.get("schema") ==
         "m1808_c3_mapped_energy_result_hammer_source_draft_r0_v1",
         "checklist schema")
    need(value.get("status") ==
         "INCOMPLETE_RESULT_HAMMER_SOURCE_DRAFT__CANONICAL_ABSENT__NO_PASS_NO_SEAL_NO_REVIEW",
         "checklist status")
    need(value.get("canonical_result") ==
         "results/m1808_c3_mapped_energy_r1_20260902", "canonical result")
    need(value.get("canonical_attempt") ==
         "results/.m1808_c3_mapped_energy_attempt_consumed", "canonical attempt")
    need(value.get("old_failure_exclusion") ==
         "results/m1798_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine",
         "old failure exclusion")
    need(value.get("future_caller_pins") == [
        "result_manifest_sha256", "result_outer_file_sha256",
        "attempt_json_sha256", "attempt_manifest_sha256",
        "attempt_outer_file_sha256"], "future pins")
    need(value.get("claim_boundary") == DRAFT_CLAIM, "draft claim boundary")
    checks = value.get("checks")
    need(type(checks) is list and len(checks) == 39, "exactly 39 checks")
    need([row.get("id") for row in checks] ==
         ["HR%02d" % index for index in range(1, 40)], "check IDs/order")
    need(len(set(row.get("id") for row in checks)) == 39, "duplicate check ID")
    for row in checks:
        need(type(row) is dict and row.get("domain")
             and row.get("requirement") and row.get("evidence")
             and row.get("severity") in ("P0", "P1"),
             "bounded checklist row " + str(row.get("id")))
    need(value.get("authorization") == {
        "read_canonical_now": False, "create_review": False,
        "create_pass": False, "seal_review": False,
        "run_eda": False, "query_license": False}, "draft authorization")
    for token in (
            "EXPECTED_MEMBERS", "len(checks) == 39",
            "M1808_RESULT_EVIDENCE_VALIDATED__FORMAL_REVIEW_AND_SEAL_STILL_REQUIRED"):
        need(token in script_text, "script omits " + token)
    for function_name in (
            "verify_dir_seal", "validate_runtime", "validate_saif",
            "validate_ptpx", "validate_power_and_energy", "isolate_m1798"):
        anchor = "def " + function_name + "("
        need(script_text.count(anchor) == 1,
             "script function anchor " + function_name)
    return {
        "status": "INCOMPLETE_M1808_RESULT_HAMMER_DRAFT_STATICALLY_CHECKED__NO_CANONICAL_READ_NO_PASS",
        "checks": 39, "canonical_read": False, "eda_runs": 0,
        "license_queries": 0, "review_created": False, "seal_created": False,
    }


def validate_static():
    for path in (CHECKLIST, SCRIPT):
        need(path.is_file() and not path.is_symlink(), "draft source absent")
    return validate_checklist_data(strict_json(CHECKLIST), SCRIPT.read_text())


def verify_dir_seal(root, manifest_pin, outer_pin):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "canonical directory")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "manifest input")
    need(outer.is_file() and not outer.is_symlink(), "outer seal input")
    need(sha(manifest) == exact_pin(manifest_pin, "manifest"),
         "manifest caller pin")
    need(sha(outer) == exact_pin(outer_pin, "outer"), "outer caller pin")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2 and len(fields[0]) == 64
             and all(char in "0123456789abcdef" for char in fields[0]),
             "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts
             and name not in mapping and name not in
             ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
             "manifest path")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == fields[0],
             "manifest member " + name)
        mapping[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed directory")
        if (path.is_file() and path.name not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256")):
            actual.add(path.relative_to(root).as_posix())
    need(actual == set(mapping), "unlisted/missing sealed member")
    return mapping


def bad_log(text):
    patterns = (r"(?m)^Error-", r"(?m)^Error:", r"(?m)^Fatal:",
                r"(?m)^\*\* Error", r"Assertion failed", r"\$fatal")
    return any(re.search(pattern, text) for pattern in patterns)


def validate_runtime(path):
    text = Path(path).read_text(errors="strict")
    need(not bad_log(text), "runtime fatal/error signature")
    need(text.count(RUNTIME_TOKEN) == 1, "runtime PASS cardinality")
    need(text.count(TAG_TOKEN) == 1, "tag PASS cardinality")
    reset = re.findall(
        r"M1808_RESET_SETTLING_GATE cycles=([0-9]+) debug=([0-9]+) binary=([01]) zero=([01])",
        text)
    need(reset == [("3", "11", "1", "1")], "reset-settling boundary")
    window = re.findall(r"M1808_SAIF_WINDOW_STOP cycles=([0-9]+)", text)
    need(len(window) == 1 and int(window[0]) > 0, "measurement window")
    result = re.findall(
        r"M1808_PUBLIC_RESULT_CHECK tiles=([0-9]+) beats=([0-9]+) mismatches=([0-9]+) xz=([0-9]+)",
        text)
    need(result == [("8", "40", "0", "0")], "numeric/public scoreboard")
    counters = re.findall(
        r"M1808_PUBLIC_COUNTER_DELTAS raw_beats=([0-9]+) tiles=([0-9]+) issues=([0-9]+) done=([0-9]+) pushes=([0-9]+) departures=([0-9]+)",
        text)
    need(counters == [("40", "8", "136", "8", "40", "40")],
         "public counter conservation")
    cover = re.findall(
        r"M1808_PUBLIC_COVERAGE result_stall_cycles=([0-9]+) raw_stall_cycles=([0-9]+) retire_cycles=([0-9]+)",
        text)
    need(len(cover) == 1 and all(int(item) > 0 for item in cover[0]),
         "runtime coverage")
    tags = re.findall(
        r"M1808_TILE_DONE_TAG_CHECK total=([0-9]+) warmup=([0-9]+) measured=([0-9]+) mismatches=([0-9]+) raw_stall=([0-9]+) result_stall=([0-9]+)",
        text)
    need(len(tags) == 1 and tags[0][0:4] == ("9", "1", "8", "0")
         and int(tags[0][4]) > 0 and int(tags[0][5]) > 0,
         "ordered tag scoreboard")
    return {
        "status": "PASS_M1808_RESET_SETTLING_PUBLIC_RUNTIME",
        "measurement_cycles": int(window[0]), "measured_tiles": 8,
        "result_beats": 40, "tile_done_tags_checked": 9,
        "reset_settling_cycles": 3, "debug_counters_checked": 11,
        "result_stall_cycles": int(cover[0][0]),
        "raw_stall_cycles": int(cover[0][1]),
        "retire_cycles": int(cover[0][2]),
        "numeric_mismatches": 0, "public_xz": 0,
    }


def sexpr_tokens(text):
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def parse_saif(text):
    tokens = sexpr_tokens(text)
    position = [0]
    def parse_one():
        need(position[0] < len(tokens) and tokens[position[0]] == "(",
             "SAIF syntax")
        position[0] += 1
        value = []
        while position[0] < len(tokens) and tokens[position[0]] != ")":
            if tokens[position[0]] == "(":
                value.append(parse_one())
            else:
                value.append(tokens[position[0]])
                position[0] += 1
        need(position[0] < len(tokens), "SAIF unterminated")
        position[0] += 1
        return value
    root = parse_one()
    need(position[0] == len(tokens) and root and root[0] == "SAIFILE",
         "SAIF root")
    return root


def forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def all_forms(node, tag):
    output = []
    if isinstance(node, list):
        if node and node[0] == tag:
            output.append(node)
        for item in node:
            if isinstance(item, list):
                output.extend(all_forms(item, tag))
    return output


def direct_instance(node, name):
    hits = [item for item in forms(node, "INSTANCE")
            if len(item) >= 2 and item[1].lstrip("\\") == name]
    need(len(hits) == 1, "SAIF instance " + name)
    return hits[0]


def validate_saif(path, cycles):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         "SAIF input")
    root = parse_saif(path.read_text(errors="strict"))
    duration = forms(root, "DURATION")
    need(len(duration) == 1 and len(duration[0]) == 2,
         "SAIF duration field")
    duration_ns = float(duration[0][1])
    need(math.isfinite(duration_ns) and abs(duration_ns-cycles*3.0) <= 1e-6,
         "SAIF duration/cycles")
    dut = direct_instance(direct_instance(root, TOP), "dut")
    groups = dict((tag, all_forms(dut, tag))
                  for tag in ("T0", "T1", "TX", "TC", "IG"))
    count = len(groups["T0"])
    need(count > 0 and all(len(value) == count for value in groups.values()),
         "SAIF DUT form cardinality")
    need(all(len(item) == 2 and float(item[1]) == 0.0
             for item in groups["TX"]), "SAIF TX nonzero")
    for t0, t1, tx in zip(groups["T0"], groups["T1"], groups["TX"]):
        values = [float(item[1]) for item in (t0, t1, tx)]
        need(all(math.isfinite(value) and value >= 0.0 for value in values)
             and abs(sum(values)-duration_ns) <= 1e-6,
             "SAIF activity conservation")
    toggles = [float(item[1]) for item in groups["TC"] if len(item) == 2]
    need(toggles and all(math.isfinite(value) and value >= 0.0
                         for value in toggles) and any(value > 0.0 for value in toggles),
         "SAIF nonvacuity")
    return {"cycles": cycles, "duration_ns": duration_ns,
            "activity_forms_per_tag": count, "tx_nonzero": 0,
            "saif_scope": SAIF_SCOPE, "saif_sha256": sha(path)}


def parse_annotation(path):
    text = Path(path).read_text(errors="strict")
    patterns = {
        "total_nets": r"Total number of nets = ([0-9]+)",
        "annotated_nets": r"Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)",
        "total_leaf": r"Total number of leaf cells = ([0-9]+)",
        "annotated_leaf": r"Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)",
    }
    hits = dict((key, re.findall(pattern, text)) for key, pattern in patterns.items())
    need(all(len(value) == 1 for value in hits.values()), "annotation fields")
    total_nets = int(hits["total_nets"][0])
    ann_nets, ann_net_percent = hits["annotated_nets"][0]
    total_leaf = int(hits["total_leaf"][0])
    ann_leaf, ann_leaf_percent = hits["annotated_leaf"][0]
    need(total_nets > 0 and int(ann_nets) == total_nets
         and float(ann_net_percent) == 100.0, "net annotation")
    need(total_leaf > 0 and int(ann_leaf) == total_leaf
         and float(ann_leaf_percent) == 100.0, "leaf annotation")
    return {"total_nets": total_nets, "annotated_nets": int(ann_nets),
            "total_leaf_cells": total_leaf,
            "annotated_leaf_cells": int(ann_leaf)}


def parse_switching_coverage(path):
    text = Path(path).read_text(errors="strict")
    pattern = (r"(?m)^" + re.escape(DESIGN)
               + r"\s+([0-9.]+)\s+([0-9]+)\s+([0-9]+)\s*$")
    hits = re.findall(pattern, text)
    need(len(hits) == 1 and "Coverage is defined as" in text,
         "switching coverage row")
    percent, covered, total = float(hits[0][0]), int(hits[0][1]), int(hits[0][2])
    need(math.isfinite(percent) and 0.0 < percent <= 100.0
         and 0 < covered <= total, "switching coverage domain")
    need(abs(percent - 100.0*covered/total) <= 0.011,
         "switching coverage arithmetic")
    return {"percent": percent, "covered_nets": covered, "total_nets": total}


def parse_key_values(path):
    output = {}
    for raw in Path(path).read_text(errors="strict").splitlines():
        need(raw.count("=") == 1, "scope syntax")
        key, value = raw.split("=", 1)
        need(key and key not in output, "scope duplicate")
        output[key] = value
    return output


def validate_ptpx(root, cycles, annotation):
    root = Path(root)
    marker = (root / "PTPX_INTERNAL_COMPLETE.txt").read_text(errors="strict")
    need(marker.count(PTPX_TOKEN) == 1 and marker.count("macro_count=0") == 1
         and marker.count("claim_boundary=prelayout_logic_only_component_energy") == 1,
         "PTPX marker")
    need(marker.count("exact_net_annotation=%d/%d=100.0%%" %
                      (annotation["annotated_nets"], annotation["total_nets"])) == 1,
         "PTPX net marker")
    need(marker.count("exact_leaf_annotation=%d/%d=100.0%%" %
                      (annotation["annotated_leaf_cells"],
                       annotation["total_leaf_cells"])) == 1,
         "PTPX leaf marker")
    check_power = (root / "reports/check_power.rpt").read_text(errors="strict")
    pt_log = (root / "ptpx.log").read_text(errors="strict")
    need(check_power.count("check_power succeeded.") == 1
         and not bad_log(check_power) and not bad_log(pt_log),
         "check_power/PTPX log")
    scope = parse_key_values(root / "reports/scope_and_boundary.rpt")
    expected_scope = {
        "milestone": "M1790", "design": DESIGN,
        "analysis": "averaged_prelayout_mapped_gate_activity",
        "power_corner": "TT_0p9V_25C", "clock_period_ns": "3.000",
        "measurement_cycles": str(cycles),
        "saif_duration_ns": str(cycles*3), "saif_scope": SAIF_SCOPE,
        "public_port_only_testbench": "true",
        "hierarchical_drive_or_read": "false",
        "clock_network": "ideal_no_cts", "wireload": "ZeroWireload",
        "spef": "false", "macro_count": "0", "component_only": "true",
        "not_speedup": "true", "not_system_or_frame_energy": "true",
        "not_silicon_or_signoff": "true",
    }
    need(scope == expected_scope, "scope/claim boundary")
    return scope


POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")


def validate_power_and_energy(path, cycles, metrics, receipt):
    text = Path(path).read_text(errors="strict")
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "power mode/unit")
    power = {}
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "power field " + field)
        power[field] = float(hits[0])
        need(math.isfinite(power[field]) and power[field] >= 0.0,
             "power domain " + field)
    need(power["Total Power"] > 0.0, "zero total power")
    subtotal = sum(power[field] for field in POWER_FIELDS[:3])
    tolerance = max(1e-8, 1e-6*max(1.0, power["Total Power"]))
    need(abs(subtotal-power["Total Power"]) <= tolerance,
         "power conservation")
    duration_ns = cycles*3.0
    energy_pj = power["Total Power"]*duration_ns
    expected = {
        "status": "PASS_M1808_COMPONENT_METRIC_PENDING_RESULT_HAMMER",
        "cycles": cycles, "duration_ns": duration_ns,
        "net_switching_power_mw": power[POWER_FIELDS[0]],
        "cell_internal_power_mw": power[POWER_FIELDS[1]],
        "cell_leakage_power_mw": power[POWER_FIELDS[2]],
        "total_power_mw": power[POWER_FIELDS[3]],
        "directed_window_energy_pj": energy_pj,
        "component_total_conserved": True, "macro_count": 0,
        "claim_boundary": EXPECTED_CLAIM,
    }
    need(metrics == expected, "metrics independent recomputation")
    need(receipt.get("metrics") == expected
         and receipt.get("claim_boundary") == EXPECTED_CLAIM,
         "receipt metric/claim drift")
    return {"total_power_mw": power["Total Power"],
            "directed_window_energy_pj": energy_pj}


def isolate_m1798(root, members):
    forbidden = (
        "m1798_c3_mapped_energy_r1_20260902.failed_or_incomplete",
        "m1798_c3_mapped_energy", OLD_FAILURE_SHA,
    )
    need(not any("m1798" in name.lower() or name.endswith("failure.json")
                 for name in members), "M1798/failure member copied")
    for name in members:
        path = Path(root) / name
        text = path.read_text(errors="strict")
        need(not any(token in text for token in forbidden),
             "M1798 failure content copied into " + name)


def validate_receipt(receipt, metrics):
    need(receipt.get("schema") ==
         "m1808_c3_m1454_fixed_t10_mapped_energy_candidate_receipt_r1_v1",
         "receipt schema")
    need(receipt.get("status") == RESULT_STATUS, "receipt status")
    need(receipt.get("one_shot") == ONE_SHOT, "receipt one-shot")
    need(receipt.get("identity") == EXPECTED_IDENTITY, "receipt identity")
    need(receipt.get("workload") == {
        "warmup_tiles_outside_saif": 1, "measured_dense_tiles": 8,
        "ordered_tile_done_tags_checked": 9,
        "checkpoint_capture": False, "public_port_only": True},
        "receipt workload")
    need(receipt.get("gate_simulation") == {
        "mode": "zero_delay_mapped_functional", "timing_simulation": False,
        "independent_timing_authority": "M1456"}, "gate simulation boundary")
    need(receipt.get("timing_authority") == {
        "clock_period_ns": 3.0, "pt_setup_wns_ns": 0.000299,
        "pt_hold_wns_ns": 0.030474}, "timing authority")
    need(receipt.get("metrics") == metrics, "receipt metrics")


def validate_attempt(pins):
    mapping = verify_dir_seal(
        ATTEMPT, pins["attempt_manifest_sha256"],
        pins["attempt_outer_file_sha256"])
    need(set(mapping) == {"attempt.json"}, "attempt inventory")
    need(mapping["attempt.json"] ==
         exact_pin(pins["attempt_json_sha256"], "attempt_json_sha256"),
         "attempt JSON caller pin")
    attempt = strict_json(ATTEMPT / "attempt.json")
    need(attempt.get("status") == "M1808_ATTEMPT_CONSUMED", "attempt status")
    need(attempt.get("budget") == {
        "vcs_compiles": 1, "simv_runs": 1,
        "saif_files": 1, "ptpx_runs": 1}, "attempt budget")
    need(attempt.get("automatic_retry") is False
         and attempt.get("reuse_prior_simv_saif_ptpx") is False,
         "attempt retry/reuse")
    need(attempt.get("workload") ==
         "one_directed_warmup_plus_eight_measured_fixed_t10_tiles",
         "attempt workload")
    need(attempt.get("attempt_uniqueness") == ATTEMPT_UNIQUENESS,
         "attempt namespace binding")


def audit_canonical(pins):
    validate_static()
    need(RESULT.is_dir() and not RESULT.is_symlink(), "canonical absent")
    members = verify_dir_seal(
        RESULT, pins["result_manifest_sha256"],
        pins["result_outer_file_sha256"])
    need(set(members) == EXPECTED_MEMBERS, "exact sealed result inventory")
    need(len([name for name in members if name == "compile.log"]) == 1
         and len([name for name in members if name.endswith("mapped_sim.log")]) == 1
         and len([name for name in members if name.endswith(".saif")]) == 1
         and len([name for name in members if name.endswith("ptpx.log")]) == 1
         and len([name for name in members
                  if name.endswith("ptpx_whole_mapped_c3_logic.rpt")]) == 1,
         "one-shot result artifact cardinality")
    validate_attempt(pins)
    need(not os.path.lexists(str(M1808_FAILURE)), "M1808 failure coexists")
    need(PRIVATE.is_dir() and not PRIVATE.is_symlink(), "private build namespace")
    compile_text = (RESULT / "compile.log").read_text(errors="strict")
    need(not bad_log(compile_text), "compile error signature")
    runtime = validate_runtime(RESULT / "candidate/mapped_sim.log")
    runtime_json = strict_json(RESULT / "runtime.json")
    need(runtime_json == runtime, "runtime JSON independent parse")
    receipt = strict_json(RESULT / "receipt.json")
    metrics = strict_json(RESULT / "metrics.json")
    validate_receipt(receipt, metrics)
    need((RESULT / "RUN_COMPLETE.txt").read_text().splitlines() == [RUN_TOKEN],
         "RUN_COMPLETE token")

    source_review = strict_json(M1815_REVIEW)
    need(sha(M1815_REVIEW) == EXPECTED_IDENTITY["source_review_json_sha256"],
         "M1815 source-review identity")
    reset = source_review.get("reset_settling_hammer", {})
    need(reset.get("immediate_architectural_control_fields") == 28
         and reset.get("debug_counter_fields_delayed") == 11
         and reset.get("all_39_public_fields_covered_at_boundary") is True
         and reset.get("complete_39_field_aggregate_every_later_edge") is True,
         "39-field bounded source review")

    saif = validate_saif(
        RESULT / "candidate/m1808_c3_fixed_t10_component.saif",
        runtime["measurement_cycles"])
    need(receipt.get("saif_check", {}).get("status") == "PASS_M1808_DUT_ONLY_SAIF"
         and receipt.get("saif_check", {}).get("cycles") == saif["cycles"]
         and receipt.get("saif_check", {}).get("duration_ns") == saif["duration_ns"]
         and receipt.get("saif_check", {}).get("activity_forms_per_tag") ==
             saif["activity_forms_per_tag"]
         and receipt.get("saif_check", {}).get("tx_nonzero") == 0
         and receipt.get("saif_check", {}).get("saif_scope") == SAIF_SCOPE
         and receipt.get("saif_check", {}).get("saif_sha256") == saif["saif_sha256"],
         "receipt SAIF independent parse")
    pt_root = RESULT / "candidate/ptpx"
    annotation = parse_annotation(pt_root / "reports/saif_annotation_summary.rpt")
    coverage = parse_switching_coverage(pt_root / "reports/switching_coverage.rpt")
    validate_ptpx(pt_root, runtime["measurement_cycles"], annotation)
    energy = validate_power_and_energy(
        pt_root / "reports/ptpx_whole_mapped_c3_logic.rpt",
        runtime["measurement_cycles"], metrics, receipt)
    isolate_m1798(RESULT, members)
    checks = strict_json(CHECKLIST)["checks"]
    need(len(checks) == 39, "39-check report")
    return {
        "status": "M1808_RESULT_EVIDENCE_VALIDATED__FORMAL_REVIEW_AND_SEAL_STILL_REQUIRED",
        "checks_executed": 39,
        "check_ids": [row["id"] for row in checks],
        "runtime": runtime, "saif": saif, "annotation": annotation,
        "switching_coverage": coverage, "energy": energy,
        "claim_boundary": EXPECTED_CLAIM,
        "review_created": False, "seal_created": False,
        "eda_runs": 0, "license_queries": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static", action="store_true")
    modes.add_argument("--audit-canonical", action="store_true")
    parser.add_argument("--result-manifest-sha256")
    parser.add_argument("--result-outer-file-sha256")
    parser.add_argument("--attempt-json-sha256")
    parser.add_argument("--attempt-manifest-sha256")
    parser.add_argument("--attempt-outer-file-sha256")
    args = parser.parse_args()
    if args.static:
        output = validate_static()
    else:
        pins = {
            "result_manifest_sha256": args.result_manifest_sha256,
            "result_outer_file_sha256": args.result_outer_file_sha256,
            "attempt_json_sha256": args.attempt_json_sha256,
            "attempt_manifest_sha256": args.attempt_manifest_sha256,
            "attempt_outer_file_sha256": args.attempt_outer_file_sha256,
        }
        for name, value in pins.items():
            exact_pin(value, name)
        output = audit_canonical(pins)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
