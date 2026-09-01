#!/usr/bin/env python3
"""CPU-only source/result checker for the M1777 two-axis C2 campaign."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
OLD_CHECKER = HW / "system_simulator/scripts/check_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1684_for_m1777", str(OLD_CHECKER))
OLD = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("M1684 checker import unavailable")
SPEC.loader.exec_module(OLD)

RUNNER = HW / "dc_handoff/scripts/run_m1777_m1776_c2_two_axis_equal_bandwidth_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1777_m1776_c2_two_axis_equal_bandwidth_energy_source.py"
CONTRACT = HW / "contracts/m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_contract_r1_20260902.json"
SOURCE_SPEC = HW / "contracts/m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_spec_r1_20260902.json"
M1776 = HW / "reviews/m1776_m1770_m1753_c2_k1_mapped_fault_failure_diagnosis_r1_20260902"
M1778 = HW / "reviews/m1778_m1777_c2_two_axis_equal_bandwidth_energy_source_hammer_r1_20260902"
M1779 = HW / "contracts/m1779_m1778_m1777_c2_two_axis_equal_bandwidth_energy_launch_release_r1_20260902.json"

BASE = OLD.BASE
DESIGN = OLD.DESIGN
NET_REL = OLD.NET_REL
SDC_REL = OLD.SDC_REL
ASSERT = OLD.ASSERT
TOP_TB = OLD.TOP_TB
UCLI = OLD.UCLI
PT_TCL = OLD.PT_TCL
FILELISTS = dict(OLD.FILELISTS)
AXES = {
    "k8": {"cycles": [51, 131, 486, 1231, 14],
           "area_um2": 130476.905184,
           "net_sha": "6c62d99b444ba25f8eb3f1e491479b44f5613b0323e032af8150e81c84f393c4",
           "sdc_sha": "852c62c1ed8d4a6c69a8fdd17ac7c3b18f0cdee271fb4aaa25fba6a2f77535eb"},
    "k1x8": {"cycles": [53, 133, 499, 1246, 14],
             "area_um2": 585534.971643,
             "net_sha": "5316db453f0ca70524ea18091e0924f79d116afd46d5432906f3182d1ccfd704",
             "sdc_sha": "17414d50eda57b2ba6f1ff3f376c24d2be6c70e9b625f717202cc72ce53c49f2"},
}
EVENTS = [20, 41, 90, 110, 0]
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))


def need(condition, message):
    if not condition:
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
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "JSON not regular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def validate_runner_text(text):
    need(text.startswith("#!/usr/bin/python3.12\n"), "interpreter shebang drift")
    need('PYTHON312 = Path("/usr/bin/python3.12")' in text,
         "interpreter path absent")
    need('AXES = ("k8", "k1x8")' in text, "exact two-axis tuple absent")
    need('"k1":' not in text, "K1 execution axis reintroduced")
    for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                  '"saif_files": 10', '"ptpx_runs": 10',
                  'for axis in AXES:', 'for case_id in CASES:',
                  'ATTEMPT.mkdir()', '"automatic_retry": False',
                  '"partial_axis_citable": False',
                  '"all ten checked SAIF coordinates required before any PTPX"',
                  '"M1684_SAIF_FILE": str(saif)',
                  '"FAULT_BINARY_CLEAN": "true"',
                  '"REGISTERED_FAULT_PUBLIC_ZERO": "true"',
                  'fault_localization_required_if_xz'):
        need(token in text, "runner gate absent: " + token)
    saif_marker = "all ten checked SAIF coordinates required before any PTPX"
    pt_marker = 'state["phase"] = "PTPX_"'
    need(saif_marker in text and pt_marker in text, "SAIF/PTPX order anchors absent")
    saif_gate = text.index(saif_marker)
    pt_loop = text.index(pt_marker)
    need(saif_gate < pt_loop, "PTPX precedes ten-SAIF gate")
    need(text.count("ATTEMPT.mkdir()") == 1, "attempt count drift")
    need("exist_ok=True" not in text and "exist_ok = True" not in text,
         "replaceable attempt namespace")
    need("automatic_retry\": True" not in text, "retry enabled")
    need("partial_axis_citable\": True" not in text, "partial publication enabled")
    need("+vcs+init" not in text.lower(), "initialization switch present")
    need("+warn=no" not in text.lower(), "warning suppression present")
    need("ignore_x" not in text.lower() and "coerce_x" not in text.lower(),
         "unknown coercion present")
    return True


def validate_contract_value(contract):
    need(contract.get("schema") ==
         "m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1778_REVIEW_AND_M1779_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    need(contract.get("execution_geometry") == {
        "axes": ["k8", "k1x8"], "cases": [0, 1, 2, 3, 4],
        "accepted_sources_per_case": EVENTS, "accepted_sources_per_axis": 261,
        "clock_period_ns": 3.0,
        "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION"},
        "execution geometry drift")
    need(contract.get("future_budget") == {
        "attempts": 1, "vcs_compiles": 2, "simv_runs": 10,
        "saif_files": 10, "ptpx_runs": 10,
        "all_ten_checked_saif_before_any_ptpx": True,
        "automatic_retry": False, "partial_axis_citable": False},
        "budget/order/retry drift")
    interpreter = contract.get("interpreter_identity", {})
    need(interpreter == {"path": "/usr/bin/python3.12",
        "implementation": "CPython", "version": "3.12.13",
        "sha256": "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814",
        "validated_before_authority_or_license": True},
        "interpreter identity drift")
    comparison = contract.get("comparison_boundary", {})
    need(comparison.get("primary_axes") == ["k8", "k1x8"]
         and comparison.get("k1_energy") ==
            "NOT_MEASURED__M1753_K1_MAPPED_XZ_FAILURE_DISCLOSED"
         and comparison.get("k1_dc_role") == "DIAGNOSTIC_ONLY"
         and comparison.get("k1_dc_rerun") is False,
         "comparison boundary drift")
    fault = contract.get("fault_integrity", {})
    need(fault.get("assertion_sha256") ==
         "39fdc0f47628272a6f1a7b6887da52fdbf4d71f1f5fe6557d4a7022f06bc62b1"
         and fault.get("assertion_changed") is False
         and fault.get("primary_axis_xz_policy") ==
            "FAIL_CLOSED_AND_REQUIRE_LOCALIZED_REPAIR_BEFORE_ANY_ENERGY_RESULT",
         "fault integrity drift")
    return True


def validate_sources():
    contract = strict_json(CONTRACT)
    validate_contract_value(contract)
    validate_runner_text(RUNNER.read_text())
    rows = contract.get("execution_files")
    need(isinstance(rows, list), "execution files absent")
    mapping = {}
    for row in rows:
        need(type(row) is dict and set(row) == {"path", "sha256"},
             "execution row malformed")
        need(row["path"] not in mapping, "duplicate execution file")
        mapping[row["path"]] = row["sha256"]
    expected = {path.relative_to(HW).as_posix() for path in (
        RUNNER, CHECKER, TEST, OLD_CHECKER, ASSERT, TOP_TB, UCLI, PT_TCL,
        FILELISTS["k8"], FILELISTS["k1x8"], OLD.MEM, OLD.CASE_TB,
        OLD.OLD_ASSERT)}
    need(expected.issubset(set(mapping)), "execution inventory incomplete")
    for rel, digest in mapping.items():
        path = HW / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "execution identity drift: " + rel)
    need(mapping[ASSERT.relative_to(HW).as_posix()] ==
         "39fdc0f47628272a6f1a7b6887da52fdbf4d71f1f5fe6557d4a7022f06bc62b1",
         "fault assertion weakened/replaced")
    for axis in AXES:
        mapped = contract.get("mapped_axes", {}).get(axis, {})
        need(mapped.get("cycles") == AXES[axis]["cycles"], "cycle drift")
        need(math.isclose(mapped.get("area_um2"), AXES[axis]["area_um2"],
                          rel_tol=0.0, abs_tol=1e-9), "area drift")
        for key, anchor in (("netlist", "net_sha"), ("sdc", "sdc_sha")):
            row = mapped.get(key, {})
            path = HW / row.get("path", "__missing__")
            need(path.is_file() and not path.is_symlink()
                 and row.get("sha256") == AXES[axis][anchor]
                 and sha(path) == AXES[axis][anchor], "mapped identity drift")
    need(not os.path.lexists(str(M1778)), "future review exists")
    need(not os.path.lexists(str(M1779)), "future release exists")
    for path in (HW / "results/.m1777_c2_two_axis_equal_bandwidth_energy_attempt_consumed",
                 HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902",
                 HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.failed_or_incomplete.quarantine"):
        need(not os.path.lexists(str(path)), "attempt/result namespace exists")
    return {"schema": "m1777_c2_two_axis_energy_source_check_r1_v1",
            "status": "PASS_M1777_SOURCE_ONLY_NO_EDA",
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "accepted_sources_per_axis": 261,
            "k1_energy": "NOT_MEASURED__M1753_K1_MAPPED_XZ_FAILURE_DISCLOSED",
            "whole_component_ptpx": True,
            "claim_boundary": CLAIMS}


def validate_runtime_log(path, axis, case_id):
    need(axis in AXES, "primary axis only")
    return OLD.validate_runtime_log(path, axis, case_id)


def validate_saif(path, axis, case_id, cycles):
    need(axis in AXES and case_id in range(5), "axis/case")
    need(cycles == AXES[axis]["cycles"][case_id], "cycle anchor")
    value = OLD.validate_saif(path, axis, case_id, cycles)
    value["status"] = "PASS_M1777_DIRECTED_COMPONENT_DUT_ONLY_SAIF"
    value["workload_class"] = "DIRECTED_COMPONENT_NOT_PRODUCTION"
    return value


def parse_power_report(path):
    value = OLD.parse_power_report(path)
    value["report_scope"] = "WHOLE_MAPPED_COMPONENT"
    value["logic_only_premacro"] = True
    return value


def aggregate_metrics(entries):
    need(len(entries) == 10, "metrics require ten coordinates")
    need(set((row["axis"], row["case"]) for row in entries) ==
         set((axis, case) for axis in AXES for case in range(5)),
         "metrics Cartesian product")
    axes = {}
    for axis in AXES:
        rows = sorted((row for row in entries if row["axis"] == axis),
                      key=lambda row: row["case"])
        need([row["cycles"] for row in rows] == AXES[axis]["cycles"],
             "cycle anchor drift")
        need([row["accepted_sources"] for row in rows] == EVENTS,
             "accepted-source denominator drift")
        total_cycles = sum(row["cycles"] for row in rows)
        total_sources = sum(row["accepted_sources"] for row in rows)
        internal_pj = sum(row["cell_internal_mw"] * row["cycles"] * 3.0
                          for row in rows)
        switching_pj = sum(row["net_switching_mw"] * row["cycles"] * 3.0
                           for row in rows)
        leakage_pj = sum(row["cell_leakage_mw"] * row["cycles"] * 3.0
                         for row in rows)
        energy_pj = sum(row["total_mw"] * row["cycles"] * 3.0
                        for row in rows)
        need(math.isclose(internal_pj + switching_pj + leakage_pj, energy_pj,
                          rel_tol=1e-4, abs_tol=1e-6), "energy decomposition")
        axes[axis] = {"cycles": total_cycles,
                      "accepted_sources": total_sources,
                      "duration_ns": total_cycles * 3.0,
                      "cell_internal_energy_pj": internal_pj,
                      "net_switching_energy_pj": switching_pj,
                      "cell_leakage_energy_pj": leakage_pj,
                      "total_energy_pj": energy_pj,
                      "cycle_weighted_average_power_mw":
                          energy_pj / (total_cycles * 3.0),
                      "energy_pj_per_accepted_source": energy_pj / total_sources,
                      "area_um2": AXES[axis]["area_um2"]}
    cycle_speedup = axes["k1x8"]["cycles"] / axes["k8"]["cycles"]
    throughput_area = ((axes["k1x8"]["cycles"] * AXES["k1x8"]["area_um2"])
                       / (axes["k8"]["cycles"] * AXES["k8"]["area_um2"]))
    energy_ratio = axes["k1x8"]["total_energy_pj"] / axes["k8"]["total_energy_pj"]
    return {"axes": axes,
            "equal_bandwidth_cycle_speedup_k8_vs_k1x8": cycle_speedup,
            "equal_bandwidth_throughput_per_mm2_k8_vs_k1x8": throughput_area,
            "equal_bandwidth_energy_ratio_k1x8_over_k8": energy_ratio,
            "equal_bandwidth_k8_energy_saving_fraction": 1.0 - 1.0 / energy_ratio,
            "joint_disclosure_required": True,
            "k1_energy": "NOT_MEASURED__M1753_K1_MAPPED_XZ_FAILURE_DISCLOSED",
            "k1_dc_role": "DIAGNOSTIC_ONLY",
            "k8_vs_single_k1_headline_forbidden": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "saif", "power"), required=True)
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--case", dest="case_id", type=int)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--saif", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--power-report", type=Path)
    args = parser.parse_args()
    if args.mode == "source":
        output = validate_sources()
    elif args.mode == "saif":
        need(args.axis is not None and args.case_id is not None
             and args.cycles is not None and args.saif and args.log, "SAIF args")
        output = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
        output["runtime"] = validate_runtime_log(args.log, args.axis, args.case_id)
    else:
        need(args.power_report is not None, "power args")
        output = parse_power_report(args.power_report)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
