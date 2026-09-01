#!/usr/bin/env python3
"""Read-only M1789 result hammer for the sealed M1782 C1 energy result.

This program intentionally does not invoke VCS, PrimeTime, a license query, or
the M1782 runner.  It independently parses the sealed evidence and prints a
single JSON object to stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, getcontext
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
RESULT = HW / "results/m1782_c1_expected_macro_leaf_blackbox_energy_r1_20260902"
ATTEMPT = HW / "results/.m1782_c1_expected_macro_leaf_blackbox_energy_attempt_consumed"
M1782 = HW / "contracts/m1782_m1772_c1_expected_macro_leaf_blackbox_energy_source_contract_r1_20260902.json"
M1783 = HW / "reviews/m1783_m1782_m1772_c1_expected_macro_leaf_blackbox_energy_source_hammer_r1_20260902"
M1784 = HW / "contracts/m1784_m1783_m1782_m1772_c1_expected_macro_leaf_blackbox_energy_launch_release_r1_20260902.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_MACRO_REF = "TS1N28HPCPHVTB128X128M4S"
EXPECTED_MACROS = {
    f"u_parent_scratch/g_slice_{index}__u_parent_sram" for index in range(9)
}
EXPECTED_SOURCES = {
    "dc_handoff/tb/tb_m1772_c1_m1701_two_bank_public_warmup_energy.sv":
        "21ead36213d89a425a170fce85823994562e8410c9bd24b338b7cf29f02a750d",
    "dc_handoff/filelists/date_m1772_c1_m1701_two_bank_public_warmup_energy.f":
        "9da54a6a3b60a05602adbb0bb4440d0ac95c035c73a1b69d6589dab2f8664906",
    "dc_handoff/scripts/m1772_c1_m1701_two_bank_public_warmup_energy.ucli.tcl":
        "beaa724867c28198d600840b2b8fe7dcbe665ad7cf6ee9449c92be6ccafccef7",
    "dc_handoff/scripts/run_ptpx_m1782_c1_m1701_expected_macro_leaf_blackbox_energy.tcl":
        "e5c1b5157eba7a58dc7ef3326ba4aab8012a4da8d7dd09ff20a837f4664a4e16",
    "dc_handoff/scripts/run_m1782_m1772_c1_expected_macro_leaf_blackbox_energy_one_shot.py":
        "4fd47d6ad137eb56f16eca92c6e716e18f9f6fb9f93f0e9f6e962b4ed8418a16",
    "system_simulator/scripts/check_m1782_c1_expected_macro_leaf_blackbox_energy_source.py":
        "4f99a631adbe8e72af77023dd3a1f8941586609cce1b7108979b2ef87232323b",
    "system_simulator/tests/test_m1782_c1_expected_macro_leaf_blackbox_energy_source.py":
        "41116179f8c4f5689896b2381c2d311f45910c9da570907d2c0c62563c315016",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path):
    def reject_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )


def manifest_entries(path: Path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed manifest line in {path}: {line!r}")
        digest, relative = match.groups()
        if relative in entries:
            raise ValueError(f"duplicate manifest path in {path}: {relative}")
        entries[relative] = digest
    return entries


def verify_directory_seal(directory: Path):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    outer_entries = manifest_entries(outer)
    if outer_entries != {"SHA256SUMS": sha256(manifest)}:
        raise ValueError(f"outer seal mismatch: {directory}")
    entries = manifest_entries(manifest)
    for relative, digest in entries.items():
        member = directory / relative
        if not member.is_file() or member.is_symlink() or sha256(member) != digest:
            raise ValueError(f"sealed member mismatch: {member}")
    disk = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    expected = set(entries) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    if disk != expected:
        raise ValueError(
            f"sealed population mismatch: missing={sorted(expected - disk)} "
            f"extra={sorted(disk - expected)}"
        )
    symlinks = [str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"symlinks under sealed directory {directory}: {symlinks}")
    return {
        "manifest_entries": len(entries),
        "disk_regular_files": len(disk),
        "missing": 0,
        "extra": 0,
        "symlinks": 0,
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
    }


def verify_double_sealed_file(path: Path):
    digest_file = Path(str(path) + ".sha256")
    outer_file = Path(str(digest_file) + ".seal.sha256")
    if manifest_entries(digest_file) != {path.name: sha256(path)}:
        raise ValueError(f"digest sidecar mismatch: {path}")
    if manifest_entries(outer_file) != {digest_file.name: sha256(digest_file)}:
        raise ValueError(f"outer sidecar mismatch: {path}")
    return {
        "payload_sha256": sha256(path),
        "digest_file_sha256": sha256(digest_file),
        "outer_seal_file_sha256": sha256(outer_file),
    }


def parse_saif(path: Path):
    text = path.read_text(encoding="utf-8")
    duration = Decimal(re.search(r"\(DURATION ([0-9.]+)\)", text).group(1))
    activity = re.findall(
        r"\(T0 ([0-9.]+)\)\s+\(T1 ([0-9.]+)\)\s+\(TX ([0-9.]+)\)\s+"
        r"\(TC ([0-9.]+)\)\s+\(IG ([0-9.]+)\)",
        text,
    )
    if len(activity) != 117690:
        raise ValueError(f"unexpected SAIF activity population: {len(activity)}")
    tx_nonzero = 0
    duration_mismatch = 0
    for t0, t1, tx, _tc, _ig in activity:
        values = tuple(map(Decimal, (t0, t1, tx)))
        tx_nonzero += values[2] != 0
        duration_mismatch += sum(values) != duration
    tag_counts = {
        tag: len(re.findall(rf"\({tag} [0-9.]+\)", text))
        for tag in ("T0", "T1", "TX", "TC", "IG")
    }
    instance_paths = re.findall(r"\(INSTANCE\s+([^\s()]+)", text)
    if instance_paths[:2] != ["tb_m1772_c1_m1701_two_bank_public_warmup_energy", "dut"]:
        raise ValueError(f"unexpected SAIF root/scope: {instance_paths[:2]}")
    return {
        "sha256": sha256(path),
        "duration_ns": str(duration),
        "activity_records": len(activity),
        "tag_counts": tag_counts,
        "tx_nonzero": tx_nonzero,
        "t0_t1_tx_duration_mismatches": duration_mismatch,
        "root_instance": instance_paths[0],
        "strip_scope": ".".join(instance_paths[:2]),
    }


def parse_runtime(log_path: Path):
    text = log_path.read_text(encoding="utf-8")
    pass_token = "PASS_M1772_C1_M1701_TWO_BANK_WARMUP_MAPPED_DIRECTED_COMPONENT_ACTIVITY"
    counter_match = re.search(
        r"M1772_PUBLIC_COUNTERS cycles=(\d+) issue_accepts=(\d+) parent_edges=(\d+) "
        r"macro_reads=(\d+) macro_writes=(\d+) forwards=(\d+) "
        r"dead_write_elisions=(\d+) psum_commits=(\d+) row_completions=(\d+)",
        text,
    )
    if counter_match is None:
        raise ValueError("missing unique public counter line")
    counters = tuple(map(int, counter_match.groups()))
    expected = (253, 96, 48, 46, 34, 2, 30, 64, 64)
    if counters != expected:
        raise ValueError(f"runtime counters differ: {counters}")
    coverage = "COVERAGE_M1772_TWO_BANK_PUBLIC_WARMUP bank0_epoch=5943 bank1_epoch=5944 public_backpressure=1 hierarchy_drive=0"
    if text.count(pass_token) != 1 or text.count(coverage) != 1:
        raise ValueError("unique PASS or two-bank warmup coverage token missing")
    if text.count("M1772_SAIF_WINDOW_START epoch=5945") != 1:
        raise ValueError("measurement epoch start is not unique")
    if text.count("M1772_SAIF_WINDOW_STOP cycles=253") != 1:
        raise ValueError("measurement stop is not unique")
    return {
        "sha256": sha256(log_path),
        "unique_pass": 1,
        "two_bank_public_warmup_cover": 1,
        "measurement_epoch": 5945,
        "cycles": counters[0],
        "issue_accepts": counters[1],
        "parent_edges": counters[2],
        "macro_reads": counters[3],
        "macro_writes": counters[4],
        "forwards": counters[5],
        "dead_write_elisions": counters[6],
        "psum_commits": counters[7],
        "row_completions": counters[8],
    }


def parse_inventory(path: Path):
    text = path.read_text(encoding="utf-8")
    if "black_box_count=9" not in text or "expected_macro_count=9" not in text:
        raise ValueError("black-box inventory count mismatch")
    rows = re.findall(
        r"^name=(\S+) ref=(\S+) is_hierarchical=(\S+) is_black_box=(\S+)$",
        text,
        re.MULTILINE,
    )
    names = {row[0] for row in rows}
    if len(rows) != 9 or names != EXPECTED_MACROS:
        raise ValueError("black-box identity set mismatch")
    if any(row[1:] != (EXPECTED_MACRO_REF, "false", "true") for row in rows):
        raise ValueError("black-box ref/leaf/attribute mismatch")
    return {
        "count": len(rows),
        "ref_name": EXPECTED_MACRO_REF,
        "names": sorted(names),
        "missing_expected": 0,
        "unexpected": 0,
        "all_leaf": True,
        "all_black_box_attribute_true": True,
    }


def parse_power(report_path: Path, hierarchy_path: Path):
    getcontext().prec = 30
    text = report_path.read_text(encoding="utf-8")
    hierarchy = hierarchy_path.read_text(encoding="utf-8")
    patterns = {
        "internal_mw": r"Cell Internal Power\s+=\s+([0-9.eE+-]+)",
        "switching_mw": r"Net Switching Power\s+=\s+([0-9.eE+-]+)",
        "leakage_mw": r"Cell Leakage Power\s+=\s+([0-9.eE+-]+)",
        "total_mw": r"Total Power\s+=\s+([0-9.eE+-]+)",
    }
    values = {key: Decimal(re.search(pattern, text).group(1)) for key, pattern in patterns.items()}
    expected = {
        "internal_mw": Decimal("26.660183"),
        "switching_mw": Decimal("1.74465036"),
        "leakage_mw": Decimal("0.671468437"),
        "total_mw": Decimal("29.0763016"),
    }
    if values != expected:
        raise ValueError(f"whole power values differ: {values}")
    component_sum = values["internal_mw"] + values["switching_mw"] + values["leakage_mw"]
    conservation_delta = component_sum - values["total_mw"]
    if abs(conservation_delta) > Decimal("0.000001"):
        raise ValueError(f"whole power conservation failure: {conservation_delta}")
    energy = values["total_mw"] * Decimal("759")
    if energy != Decimal("22068.9129144"):
        raise ValueError(f"directed-window energy differs: {energy}")
    hierarchy_match = re.search(
        r"u_parent_scratch \([^\n]+\)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.]+)",
        hierarchy,
    )
    if hierarchy_match is None:
        raise ValueError("cannot parse parent scratch hierarchy")
    hierarchy_values = tuple(Decimal(value) for value in hierarchy_match.groups())
    hierarchy_total = hierarchy_values[3]
    if hierarchy_total != Decimal("10.5071545"):
        raise ValueError(f"parent scratch hierarchy power differs: {hierarchy_total}")
    memory_match = re.search(
        r"^memory\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)",
        text,
        re.MULTILINE,
    )
    memory_group_total = Decimal(memory_match.group(1))
    if memory_group_total != Decimal("10.5068808"):
        raise ValueError(f"memory power-group total differs: {memory_group_total}")
    alternative_dynamic = Decimal(46) * Decimal("94.57074") + Decimal(34) * Decimal("90.65763")
    alternative_leakage = Decimal("0.54009423") * Decimal("759")
    alternative_total = alternative_dynamic + alternative_leakage
    if alternative_total != Decimal("7842.54498057"):
        raise ValueError(f"datasheet alternative differs: {alternative_total}")
    return {
        "internal_power_mw": str(values["internal_mw"]),
        "switching_power_mw": str(values["switching_mw"]),
        "leakage_power_mw": str(values["leakage_mw"]),
        "total_power_mw": str(values["total_mw"]),
        "component_sum_mw": str(component_sum),
        "component_sum_minus_report_total_mw": str(conservation_delta),
        "directed_window_energy_pj": str(energy),
        "parent_scratch_hierarchy_total_mw": str(hierarchy_total),
        "parent_scratch_share_percent_recomputed": str(hierarchy_total / values["total_mw"] * 100),
        "memory_power_group_total_mw": str(memory_group_total),
        "hierarchy_minus_memory_group_mw": str(hierarchy_total - memory_group_total),
        "datasheet_alternative_dynamic_pj": str(alternative_dynamic),
        "datasheet_alternative_leakage_pj": str(alternative_leakage),
        "datasheet_alternative_total_pj": str(alternative_total),
        "datasheet_alternative_added_to_whole_ptpx": False,
    }


def main():
    result_seal = verify_directory_seal(RESULT)
    attempt_seal = verify_directory_seal(ATTEMPT)
    m1783_seal = verify_directory_seal(M1783)
    m1782_seal = verify_double_sealed_file(M1782)
    m1784_seal = verify_double_sealed_file(M1784)

    result_receipt = strict_json(RESULT / "receipt.json")
    metrics = strict_json(RESULT / "metrics.json")
    runtime_json = strict_json(RESULT / "runtime.json")
    attempt_json = strict_json(ATTEMPT / "attempt.json")
    contract = strict_json(M1782)
    source_review = strict_json(M1783 / "review.json")
    release = strict_json(M1784)
    inventory_json = strict_json(RESULT / "black_box_inventory.json")

    source_hashes = {relative: sha256(HW / relative) for relative in EXPECTED_SOURCES}
    if source_hashes != EXPECTED_SOURCES:
        raise ValueError("current source hashes do not match frozen M1782 identities")
    testbench_text = (HW / "dc_handoff/tb/tb_m1772_c1_m1701_two_bank_public_warmup_energy.sv").read_text(encoding="utf-8")
    if testbench_text.count("always #1.5 clk_core = ~clk_core;") != 1:
        raise ValueError("testbench 3 ns clock identity differs")
    contract_hashes = {item["path"]: item["sha256"] for item in contract["source_files"]}
    if contract_hashes != EXPECTED_SOURCES:
        raise ValueError("M1782 contract source identity differs")
    if release["identity"]["runner_sha256"] != EXPECTED_SOURCES[
        "dc_handoff/scripts/run_m1782_m1772_c1_expected_macro_leaf_blackbox_energy_one_shot.py"
    ]:
        raise ValueError("M1784 release runner identity differs")
    if release["identity"]["source_contract_sha256"] != sha256(M1782):
        raise ValueError("M1784 does not bind exact M1782 contract")
    if release["identity"]["m1783_review_sha256"] != sha256(M1783 / "review.json"):
        raise ValueError("M1784 does not bind exact M1783 review")

    runtime = parse_runtime(RESULT / "candidate/mapped_sim.log")
    saif = parse_saif(RESULT / "candidate/m1782_c1_directed_component.saif")
    inventory = parse_inventory(RESULT / "candidate/ptpx/reports/black_box_inventory_machine.rpt")
    power = parse_power(
        RESULT / "candidate/ptpx/reports/ptpx_whole_mapped_c1_including_9macro_liberty.rpt",
        RESULT / "candidate/ptpx/reports/ptpx_hierarchy_diagnostic_including_9macro_liberty.rpt",
    )

    annotation = (RESULT / "candidate/ptpx/reports/saif_annotation_summary.rpt").read_text(encoding="utf-8")
    if "Number of annotated nets = 115377 (100.00%)" not in annotation:
        raise ValueError("net annotation is not exact 100%")
    if "Number of fully annotated leaf cells = 107371 (100.00%)" not in annotation:
        raise ValueError("leaf annotation is not exact 100%")
    inconsistent = (RESULT / "candidate/ptpx/reports/inconsistent_annotation.rpt").read_text(encoding="utf-8")
    if len([line for line in inconsistent.splitlines()[2:] if line.strip()]) != 0:
        raise ValueError("inconsistent SAIF annotation rows present")
    coverage = (RESULT / "candidate/ptpx/reports/switching_coverage.rpt").read_text(encoding="utf-8")
    if not re.search(r"61\.92\s+71439\s+115377", coverage):
        raise ValueError("nonzero switching coverage differs")
    check_power = (RESULT / "candidate/ptpx/reports/check_power.rpt").read_text(encoding="utf-8")
    ramp_out_of_range = len(re.findall(r"^  ramp .* is out of ramp range", check_power, re.MULTILINE))
    load_out_of_range = len(re.findall(r"^  load .* is out of load range", check_power, re.MULTILINE))
    if (ramp_out_of_range, load_out_of_range) != (12628, 2304):
        raise ValueError("check_power diagnostic population differs")
    ptpx_log = (RESULT / "candidate/ptpx/ptpx.log").read_text(encoding="utf-8")
    startup_pt063 = ptpx_log.count("Error: Library Compiler executable path is not set. (PT-063)")
    if startup_pt063 != 1:
        raise ValueError("PT-063 startup diagnostic population differs")

    if strict_json(RESULT / "runtime.json") != runtime_json or runtime_json["measurement_cycles"] != runtime["cycles"]:
        raise ValueError("runtime JSON/log mismatch")
    if result_receipt["saif_check"]["cycles"] != runtime["cycles"]:
        raise ValueError("receipt/log cycle mismatch")
    if Decimal(str(result_receipt["saif_check"]["duration_ns"])) != Decimal(saif["duration_ns"]):
        raise ValueError("receipt/SAIF duration mismatch")
    if inventory_json["names"] != inventory["names"]:
        raise ValueError("inventory JSON/report identity mismatch")
    if Decimal(str(metrics["ptpx_whole_mapped_c1_including_9macro_liberty"]["total_power_mw"])) != Decimal(power["total_power_mw"]):
        raise ValueError("metrics/report total power mismatch")
    if Decimal(str(metrics["ptpx_whole_mapped_c1_including_9macro_liberty"]["directed_window_energy_pj"])) != Decimal(power["directed_window_energy_pj"]):
        raise ValueError("metrics/recomputed energy mismatch")

    fresh_counts = {
        "attempt_consumed": attempt_json["status"] == "M1782_ATTEMPT_CONSUMED",
        "automatic_retry": attempt_json["automatic_retry"],
        "vcs_compiles": result_receipt["one_shot"]["vcs_compiles"],
        "simv_runs": result_receipt["one_shot"]["simv_runs"],
        "saif_files": result_receipt["one_shot"]["saif_files"],
        "ptpx_runs": result_receipt["one_shot"]["ptpx_runs"],
        "compile_top_markers": (RESULT / "compile.log").read_text(encoding="utf-8").count("Top Level Modules:"),
        "mapped_pass_tokens": runtime["unique_pass"],
        "ptpx_version_headers": (RESULT / "candidate/ptpx/ptpx.log").read_text(encoding="utf-8").count("PrimeTime (R)"),
        "saif_payload_files": len(list((RESULT / "candidate").glob("*.saif"))),
        "failure_coordinate_present": any((HW / "results").glob("m1782_c1_expected_macro_leaf_blackbox_energy_r1_20260902.failed*")),
    }
    if fresh_counts != {
        "attempt_consumed": True,
        "automatic_retry": False,
        "vcs_compiles": 1,
        "simv_runs": 1,
        "saif_files": 1,
        "ptpx_runs": 1,
        "compile_top_markers": 1,
        "mapped_pass_tokens": 1,
        "ptpx_version_headers": 1,
        "saif_payload_files": 1,
        "failure_coordinate_present": False,
    }:
        raise ValueError(f"fresh-one execution evidence differs: {fresh_counts}")

    if sha256(DOCS359) != EXPECTED_DOCS359:
        raise ValueError("docs359 identity changed")

    output = {
        "status": "PASS_M1789_READ_ONLY_INDEPENDENT_HAMMER",
        "strict_json_documents": 8,
        "seals": {
            "result": result_seal,
            "attempt": attempt_seal,
            "m1783": m1783_seal,
            "m1782_contract": m1782_seal,
            "m1784_release": m1784_seal,
        },
        "source_hashes": source_hashes,
        "fresh_one": fresh_counts,
        "runtime": runtime,
        "saif": saif,
        "black_box_inventory": inventory,
        "annotation": {
            "nets": "115377/115377=100.00%",
            "leaf_cells": "107371/107371=100.00%",
            "inconsistent_rows": 0,
            "nonzero_toggle_coverage_percent": "61.92",
            "nonzero_toggle_nets": 71439,
        },
        "prelayout_diagnostics": {
            "check_power_ramp_out_of_range_rows": ramp_out_of_range,
            "check_power_load_out_of_range_rows": load_out_of_range,
            "pt063_library_compiler_startup_rows": startup_pt063,
            "ptpx_completed_after_startup_diagnostic": True,
            "classification": "declared_prelayout_mixed_corner_component_estimate_not_signoff",
        },
        "power": power,
        "identity": {
            "m1782_contract_sha256": sha256(M1782),
            "m1783_review_sha256": sha256(M1783 / "review.json"),
            "m1784_release_sha256": sha256(M1784),
            "result_receipt_sha256": sha256(RESULT / "receipt.json"),
            "result_manifest_sha256": sha256(RESULT / "SHA256SUMS"),
            "result_outer_seal_file_sha256": sha256(RESULT / "SHA256SUMS.seal.sha256"),
            "attempt_json_sha256": sha256(ATTEMPT / "attempt.json"),
            "docs359_sha256": sha256(DOCS359),
        },
        "reviewer_execution": {
            "eda_runs": 0,
            "license_queries": 0,
            "canonical_result_writes": 0,
            "docs359_writes": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
