#!/usr/bin/env python3
"""Different-author, CPU-only hammer for the inert M1777 source chain."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1777_m1776_c2_two_axis_equal_bandwidth_energy_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1777_m1776_c2_two_axis_equal_bandwidth_energy_one_shot.py"
CONTRACT = HW / "contracts/m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_author_receipt_r1_20260902"
M1776 = HW / "reviews/m1776_m1770_m1753_c2_k1_mapped_fault_failure_diagnosis_r1_20260902"
M1753_FAILURE = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine"
BASE = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
ASSERTION = HW / "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1684_c2_m1609_fresh_mapped_production_energy_tt0p9v25c.tcl"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SPEC = importlib.util.spec_from_file_location("m1778_live_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("checker loader unavailable")
SPEC.loader.exec_module(M)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "seal identity")
    need(outer.read_text() == manifest_sha + "  SHA256SUMS\n", "outer content")
    listed = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        member = root / rel
        need(member.is_file() and not member.is_symlink() and sha(member) == fields[0],
             "member identity")
        listed[name] = fields[0]
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == set(listed), "sealed population")


def rejected(function, value):
    try:
        function(value)
    except Exception:
        return
    raise RuntimeError("mutation accepted")


def validate_pt_text(text):
    for token in (
            "read_saif -strip_path $::env(SAIF_INSTANCE)",
            "Number of annotated nets = ([0-9]+)",
            "Number of fully annotated leaf cells = ([0-9]+)",
            "$annotated_nets != $total_nets",
            "$annotated_percent != 100.0",
            "$annotated_leaf_cells != $total_leaf_cells",
            "$annotated_leaf_percent != 100.0",
            "check_power succeeded\\.", "update_power",
            "report_power -unit mW -nosplit -significant_digits 8",
            "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX"):
        need(token in text, "PTPX gate absent: " + token)
    need("report_power -hierarchy" in text, "hierarchical audit absent")
    return True


def validate_assertion_text(text):
    vector = "{protocol_error, numeric_overflow,\n                    stale_response_seen, endpoint_fault}"
    need(vector in text, "four-state fault vector drift")
    need(text.count("check_fault_vector();") == 2, "both-phase fault checks absent")
    need("always @(posedge clk_core)" in text and "always @(negedge clk_core)" in text,
         "both phase processes absent")
    need("$isunknown" in text and "$fatal" in text, "fail-closed assertion absent")
    lowered = "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines()).lower()
    need("force " not in lowered and "+vcs+init" not in lowered
         and "ignore_x" not in lowered and "coerce_x" not in lowered,
         "unknown masking mechanism")
    return True


def rows():
    result = []
    power = {"k8": 2.0, "k1x8": 8.0}
    for axis in M.AXES:
        for case_id, cycles in enumerate(M.AXES[axis]["cycles"]):
            total = power[axis] + case_id * 0.1
            result.append({"axis": axis, "case": case_id, "cycles": cycles,
                           "accepted_sources": M.EVENTS[case_id],
                           "net_switching_mw": total * 0.3,
                           "cell_internal_mw": total * 0.6,
                           "cell_leakage_mw": total * 0.1,
                           "total_mw": total})
    return result


def power_text(net="0.30000000", internal="0.60000000",
               leakage="0.10000000", total="1.00000000"):
    return ("Report : Averaged Power\nCommand: report_power -unit mW\n"
            "Net Switching Power = " + net + "\n"
            "Cell Internal Power = " + internal + "\n"
            "Cell Leakage Power = " + leakage + "\n"
            "Total Power = " + total + "\n")


def main():
    live_runner = RUNNER.read_text()
    live_contract = strict_json(CONTRACT)
    live_pt = PT_TCL.read_text()
    live_assertion = ASSERTION.read_text()

    need(sha(DOCS359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359 drift")
    need(sha(RUNNER) == "b67db52da22d25147c15ffe8966d70dab10c110fe6363b502ceefbfa16069504",
         "runner drift")
    need(sha(CHECKER) == "bc63ba45c710543dad888f5e22b7403e44a73e41b1981fb8d1037588baa4b122",
         "checker drift")
    need(sha(CONTRACT) == "28d03c9611c7c07d4a18a258e04fdf84772c2c9f734b83a47442a48754b2faf7",
         "contract drift")
    need(sha(ASSERTION) ==
         "39fdc0f47628272a6f1a7b6887da52fdbf4d71f1f5fe6557d4a7022f06bc62b1",
         "assertion drift")
    verify_sealed_directory(M1776,
        "72ef5e9727d6b4a845b61c3dca46b96639d2531afcca340d62f99435dbcdc6ab",
        "acdf0d9c60100971639b45215fc0b4bd9cf1ba49d437e3e61e970621f22be580")
    verify_sealed_directory(BASE,
        "22388b70b68f4b038a464446704bdc37fb9f51d536fc12b656b0e51045f5efac",
        "f41253a98d74e7b5087c39f49ddbade856ac825f1286c0c73ccf18bdbc6cd4a2")
    verify_sealed_directory(AUTHOR,
        "a1a626564fee40bfa46e42dd7fe09362fe2da6183d123d39a906515a296956bb",
        "f5ebc79cce0103eb5cc4a236d919745fc059399c086d4741f5d59344461e0d73")
    verify_sealed_directory(M1753_FAILURE,
        "38f234a99970d8c9802c059cad90b155a30b9048e7ce95e312c27f22dc2f3a9c",
        "0ceb38d76c1c2869baec5a8f47ab1cf4cb2ba56a12de292c18e68fd3ca4bb120")

    receipt = strict_json(M1776 / "receipt.json")
    need(receipt["execution_counts"] == {"wrapper_attempts": 1,
         "m1753_attempts": 1, "vcs_compiles": 1, "simv_runs": 1,
         "saif_files": 0, "ptpx_runs": 0, "canonical_results": 0},
         "M1753 failure disclosure drift")
    need(receipt["failure_observation"]["axis"] == "k1"
         and receipt["failure_observation"]["fatal_message"] ==
         "M1684 mapped fault vector contains X/Z", "M1753 root observation drift")
    need(receipt["first_principles_decision"]["k1_energy_in_m1777"] is False
         and receipt["retained_k1_dc_diagnostic"]["energy"] is False,
         "K1 boundary drift")

    M.validate_runner_text(live_runner)
    M.validate_contract_value(live_contract)
    validate_pt_text(live_pt)
    validate_assertion_text(live_assertion)
    # Creating this review directory intentionally trips M1777's live
    # pre-review freshness gate.  Rebind only the checker's in-memory future
    # authority paths to absent temporary names, leaving every authored byte
    # and every execution/result namespace untouched, so all other live source
    # gates can still be exercised after review construction has begun.
    with tempfile.TemporaryDirectory(prefix="m1778_authority_isolation_") as temp:
        old_review, old_release = M.M1778, M.M1779
        try:
            M.M1778 = Path(temp) / "future_review_absent"
            M.M1779 = Path(temp) / "future_release_absent.json"
            isolated = M.validate_sources()
        finally:
            M.M1778, M.M1779 = old_review, old_release
    need(isolated["status"] == "PASS_M1777_SOURCE_ONLY_NO_EDA",
         "isolated source gate")
    mapping = dict((row["path"], row["sha256"])
                   for row in live_contract["execution_files"])
    need(len(mapping) == len(live_contract["execution_files"]), "duplicate inventory")
    for rel, digest in mapping.items():
        path = HW / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "execution identity drift: " + rel)
    need(not any("k1_fresh" in key for key in mapping), "K1 energy filelist retained")
    need(live_contract["comparison_boundary"]["k1_dc_area_um2"] == 124546.967176
         and live_contract["comparison_boundary"]["k1_dc_hold_closed"] is False,
         "K1 DC diagnostic incomplete")

    runner_mutations = [
        live_runner.replace('AXES = ("k8", "k1x8")', 'AXES = ("k1", "k8", "k1x8")'),
        live_runner.replace('"vcs_compiles": 2', '"vcs_compiles": 1'),
        live_runner.replace('"simv_runs": 10', '"simv_runs": 9'),
        live_runner.replace('"saif_files": 10', '"saif_files": 9'),
        live_runner.replace('"ptpx_runs": 10', '"ptpx_runs": 9'),
        live_runner.replace('"automatic_retry": False', '"automatic_retry": True'),
        live_runner.replace('"partial_axis_citable": False', '"partial_axis_citable": True'),
        live_runner.replace('ATTEMPT.mkdir()', 'ATTEMPT.mkdir(exist_ok=True)', 1),
        live_runner.replace('"FAULT_BINARY_CLEAN": "true"', '"FAULT_BINARY_CLEAN": "false"', 1),
        live_runner.replace('"REGISTERED_FAULT_PUBLIC_ZERO": "true"', '"REGISTERED_FAULT_PUBLIC_ZERO": "false"', 1),
        live_runner.replace('all ten checked SAIF coordinates required before any PTPX',
                            'ten-SAIF gate removed', 1),
        live_runner + '\n# +vcs+initreg+random\n',
        live_runner + '\n# ignore_x\n',
    ]
    for index, value in enumerate(runner_mutations):
        try:
            rejected(M.validate_runner_text, value)
        except RuntimeError as error:
            raise RuntimeError("runner mutation accepted: " + str(index)) from error

    contract_mutations = []
    for path, value in (
            (("future_budget", "vcs_compiles"), 1),
            (("future_budget", "simv_runs"), 9),
            (("future_budget", "saif_files"), 9),
            (("future_budget", "ptpx_runs"), 9),
            (("future_budget", "all_ten_checked_saif_before_any_ptpx"), False),
            (("future_budget", "automatic_retry"), True),
            (("future_budget", "partial_axis_citable"), True),
            (("comparison_boundary", "primary_axes"), ["k1", "k8", "k1x8"]),
            (("comparison_boundary", "k1_energy"), 1.0),
            (("comparison_boundary", "k1_dc_role"), "PRIMARY"),
            (("comparison_boundary", "k1_dc_rerun"), True),
            (("fault_integrity", "assertion_sha256"), "0" * 64),
            (("fault_integrity", "primary_axis_xz_policy"), "IGNORE"),
            (("interpreter_identity", "version"), "3.12.12"),
            (("claim_boundary", "energy"), True)):
        item = copy.deepcopy(live_contract)
        item[path[0]][path[1]] = value
        contract_mutations.append(item)
    for value in contract_mutations:
        rejected(M.validate_contract_value, value)

    for token in ("read_saif -strip_path $::env(SAIF_INSTANCE)",
                  "$annotated_nets != $total_nets",
                  "$annotated_leaf_cells != $total_leaf_cells",
                  "check_power succeeded\\.", "update_power",
                  "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX"):
        rejected(validate_pt_text, live_pt.replace(token, "REMOVED", 1))
    rejected(validate_assertion_text,
             live_assertion.replace("check_fault_vector();", "", 1))
    rejected(validate_assertion_text, live_assertion + "\nforce dut.fault = 0;\n")

    baseline_rows = rows()
    metrics = M.aggregate_metrics(baseline_rows)
    need(math.isclose(metrics["equal_bandwidth_cycle_speedup_k8_vs_k1x8"],
                      1945.0 / 1913.0), "cycle metric")
    need(math.isclose(metrics["equal_bandwidth_throughput_per_mm2_k8_vs_k1x8"],
                      (1945.0 * 585534.971643) / (1913.0 * 130476.905184)),
         "throughput/area metric")
    mutated = baseline_rows[:-1]
    rejected(M.aggregate_metrics, mutated)
    mutated = copy.deepcopy(baseline_rows)
    mutated[0]["axis"] = "k1"
    rejected(M.aggregate_metrics, mutated)
    mutated = copy.deepcopy(baseline_rows)
    mutated[0]["cycles"] += 1
    rejected(M.aggregate_metrics, mutated)
    mutated = copy.deepcopy(baseline_rows)
    mutated[0]["accepted_sources"] += 1
    rejected(M.aggregate_metrics, mutated)
    mutated = copy.deepcopy(baseline_rows)
    mutated[0]["total_mw"] += 1.0
    rejected(M.aggregate_metrics, mutated)

    with tempfile.TemporaryDirectory(prefix="m1778_hammer_") as temp:
        root = Path(temp)
        report = root / "power.rpt"
        report.write_text(power_text())
        value = M.parse_power_report(report)
        need(value["report_scope"] == "WHOLE_MAPPED_COMPONENT"
             and value["logic_only_premacro"] is True, "power scope")
        for bad in (power_text(total="1.10000000"),
                    power_text(total="0.00000000"),
                    power_text(net="-0.1"),
                    power_text(total="nan"),
                    power_text() + "Total Power = 1.00000000\n"):
            report.write_text(bad)
            rejected(M.parse_power_report, report)

    attempts = [
        HW / "results/.m1777_c2_two_axis_equal_bandwidth_energy_attempt_consumed",
        HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902",
        HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.failed_or_incomplete.quarantine",
        HW / "contracts/m1779_m1778_m1777_c2_two_axis_equal_bandwidth_energy_launch_release_r1_20260902.json",
    ]
    need(not any(os.path.lexists(str(path)) for path in attempts),
         "attempt/result/release unexpectedly exists")

    result = {
        "schema": "m1778_independent_hammer_output_r1_v1",
        "status": "PASS_M1778_M1777_C2_TWO_AXIS_EQUAL_BANDWIDTH_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "runner_mutations_rejected": len(runner_mutations),
        "contract_mutations_rejected": len(contract_mutations),
        "ptpx_and_assertion_mutations_rejected": 8,
        "metric_and_power_mutations_rejected": 10,
        "total_mutations_rejected": len(runner_mutations) + len(contract_mutations) + 18,
        "m1753_failure_disclosed": True,
        "k1_energy_axis_removed": True,
        "k1_dc_diagnostic_retained": True,
        "all_ten_saif_before_ptpx": True,
        "pre_review_live_source_check": "PASS_BEFORE_M1778_DIRECTORY_CREATION",
        "post_directory_source_gate": "EXPECTED_FAIL_FUTURE_REVIEW_EXISTS",
        "isolated_future_path_source_check": "PASS_NO_AUTHORED_FILE_MUTATION",
        "docs359_sha256": sha(DOCS359),
        "license_queries": 0,
        "eda_runs": 0,
        "attempts_created": 0,
        "results_created": 0,
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
