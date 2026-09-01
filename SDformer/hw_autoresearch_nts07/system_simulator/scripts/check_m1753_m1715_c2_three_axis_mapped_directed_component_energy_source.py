#!/usr/bin/env python3
"""No-EDA source/result checker for the M1753 three-axis C2 campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import tokenize
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
OLD_CHECKER = HW / "system_simulator/scripts/check_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1684_for_m1753", OLD_CHECKER)
OLD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(OLD)

RUNNER = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
M1760 = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"

AXES = {
    "k1": {"define": "M979_AXIS_K1",
           "net_sha": "750a4d8f7fb9aa8ecca4f748e29a18ae400af3c06b8df2b87200fd345a525e5f",
           "sdc_sha": "8050e66102d865e9223b65660a9a649b2eb6c4a4e098bd0ceed13940ef31f1d3",
           "cycles": [259, 737, 3153, 7569, 14]},
    "k8": {"define": "M979_AXIS_K8", "net_sha": OLD.AXES["k8"]["net_sha"],
           "sdc_sha": OLD.AXES["k8"]["sdc_sha"], "cycles": OLD.AXES["k8"]["cycles"]},
    "k1x8": {"define": "M979_AXIS_K1X8", "net_sha": OLD.AXES["k1x8"]["net_sha"],
             "sdc_sha": OLD.AXES["k1x8"]["sdc_sha"], "cycles": OLD.AXES["k1x8"]["cycles"]},
}
# OLD.validate_saif is generic once the missing K1 axis is supplied.
OLD.AXES["k1"] = AXES["k1"]
EVENTS = [20, 41, 90, 110, 0]
PACKETS = [1, 2, 4, 8, 1]
AREAS_UM2 = {"k1": 124546.967176, "k8": 130476.905184,
             "k1x8": 585534.971643}
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))
COUNTS = {"vcs_compiles": 3, "simv_runs": 15,
          "saif_files": 15, "ptpx_runs": 15}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    need(path.is_file() and not path.is_symlink(), "JSON not regular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def _strip_c_comments_and_strings(text: str) -> str:
    out = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                state = "line"; out.extend("  "); index += 2; continue
            if char == "/" and nxt == "*":
                state = "block"; out.extend("  "); index += 2; continue
            if char == '"':
                state = "string"; out.append(" "); index += 1; continue
            out.append(char); index += 1; continue
        if state == "line":
            if char == "\n": state = "code"; out.append("\n")
            else: out.append(" ")
            index += 1; continue
        if state == "block":
            if char == "*" and nxt == "/":
                state = "code"; out.extend("  "); index += 2
            else:
                out.append("\n" if char == "\n" else " "); index += 1
            continue
        if char == "\\" and nxt:
            out.extend("  "); index += 2
        elif char == '"':
            state = "code"; out.append(" "); index += 1
        else:
            out.append("\n" if char == "\n" else " "); index += 1
    return "".join(out)


def active_force_present(path: Path) -> bool:
    text = path.read_text()
    if path.suffix == ".py":
        try:
            return any(item.type == tokenize.NAME and item.string == "force"
                       for item in tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError):
            return True
    if path.suffix in {".sv", ".v"}:
        return re.search(r"\bforce\b", _strip_c_comments_and_strings(text)) is not None
    cleaned = "\n".join(line for line in text.splitlines()
                          if not line.lstrip().startswith("#"))
    return re.search(r"(?<![A-Za-z0-9_$])force(?![A-Za-z0-9_$])",
                     cleaned) is not None


def validate_runtime_log(path: Path, axis: str, case_id: int) -> dict[str, Any]:
    text = path.read_text(errors="strict")
    forbidden = ("Assertion failed", "Fatal:", "$fatal", "Error-[",
                 "contains X/Z", "fault asserted", "coverage incomplete")
    need(not any(token in text for token in forbidden), "runtime fatal/assertion")
    monitor = (r"PASS M1684 M1609 binary-clean production case=" + str(case_id)
               + r" accepted_sources=([0-9]+) source_packets=([0-9]+)"
               + r" endpoint_accepts=([0-9]+) result_accepts=([1-9][0-9]*)"
               + r" done_accepts=1 fault_binary_clean=1 registered_fault_public_zero=1")
    hits = re.findall(monitor, text)
    need(len(hits) == 1, "M1684 runtime PASS absent/duplicated")
    need(int(hits[0][0]) == EVENTS[case_id]
         and int(hits[0][1]) == PACKETS[case_id], "source denominator drift")
    endpoint = int(hits[0][2])
    need((case_id < 4 and endpoint > 0) or (case_id == 4 and endpoint == 0),
         "endpoint activity drift")
    display = {"k1": "K1", "k8": "K8", "k1x8": "K1x8"}[axis]
    exact_pass = ("PASS M979 mapped replay axis=" + display
                  + " case=" + str(case_id) + " events=" + str(EVENTS[case_id])
                  + " cycles=" + str(AXES[axis]["cycles"][case_id]))
    need(text.count(exact_pass) == 1, "M979 exact axis/cycle PASS absent/duplicated")
    need(text.count("PASS M1334 coverage case=" + str(case_id)) == 1,
         "M1334 coverage PASS absent/duplicated")
    return {"log_sha256": sha(path), "accepted_sources": EVENTS[case_id],
            "endpoint_accepts": endpoint}


def validate_saif(path: Path, axis: str, case_id: int, cycles: int) -> dict[str, Any]:
    need(axis in AXES and case_id in range(5), "axis/case")
    need(cycles == AXES[axis]["cycles"][case_id], "cycle anchor")
    value = OLD.validate_saif(path, axis, case_id, cycles)
    value["status"] = "PASS_M1753_DIRECTED_COMPONENT_DUT_ONLY_SAIF"
    value["workload_class"] = "DIRECTED_COMPONENT_NOT_PRODUCTION"
    return value


def parse_power_report(path: Path) -> dict[str, Any]:
    value = OLD.parse_power_report(path)
    value["report_scope"] = "WHOLE_MAPPED_COMPONENT"
    value["logic_only_premacro"] = True
    return value


def aggregate_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    need(len(entries) == 15, "metrics require fifteen coordinates")
    need(set((row["axis"], row["case"]) for row in entries) ==
         set((axis, case) for axis in AXES for case in range(5)),
         "metrics Cartesian product")
    axes: dict[str, dict[str, float | int]] = {}
    for axis in AXES:
        rows = sorted((row for row in entries if row["axis"] == axis),
                      key=lambda row: row["case"])
        need([row["cycles"] for row in rows] == AXES[axis]["cycles"],
             "cycle anchor drift")
        need([row["accepted_sources"] for row in rows] == EVENTS,
             "accepted-source denominator drift")
        total_cycles = sum(row["cycles"] for row in rows)
        total_sources = sum(row["accepted_sources"] for row in rows)
        energy_pj = sum(row["total_mw"] * row["cycles"] * 3.0 for row in rows)
        internal_pj = sum(row["cell_internal_mw"] * row["cycles"] * 3.0 for row in rows)
        switching_pj = sum(row["net_switching_mw"] * row["cycles"] * 3.0 for row in rows)
        leakage_pj = sum(row["cell_leakage_mw"] * row["cycles"] * 3.0 for row in rows)
        need(math.isclose(internal_pj + switching_pj + leakage_pj, energy_pj,
                          rel_tol=1e-4, abs_tol=1e-6), "energy decomposition")
        axes[axis] = {"cycles": total_cycles, "accepted_sources": total_sources,
                      "duration_ns": total_cycles * 3.0,
                      "cell_internal_energy_pj": internal_pj,
                      "net_switching_energy_pj": switching_pj,
                      "cell_leakage_energy_pj": leakage_pj,
                      "total_energy_pj": energy_pj,
                      "cycle_weighted_average_power_mw": energy_pj / (total_cycles * 3.0),
                      "energy_pj_per_accepted_source": energy_pj / total_sources,
                      "area_um2": AREAS_UM2[axis]}
    cycle_speedup = axes["k1x8"]["cycles"] / axes["k8"]["cycles"]
    throughput_area = ((axes["k1x8"]["cycles"] * AREAS_UM2["k1x8"])
                       / (axes["k8"]["cycles"] * AREAS_UM2["k8"]))
    energy_ratio = axes["k1x8"]["total_energy_pj"] / axes["k8"]["total_energy_pj"]
    return {"axes": axes,
            "equal_bandwidth_cycle_speedup_k8_vs_k1x8": cycle_speedup,
            "equal_bandwidth_throughput_per_mm2_k8_vs_k1x8": throughput_area,
            "equal_bandwidth_energy_ratio_k1x8_over_k8": energy_ratio,
            "equal_bandwidth_k8_energy_saving_fraction": 1.0 - 1.0 / energy_ratio,
            "joint_disclosure_required": True,
            "k8_vs_single_k1_headline_forbidden": True}


def validate_sources() -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1760_REVIEW_AND_M1761_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    rows = contract.get("execution_files")
    need(isinstance(rows, list), "execution files absent")
    mapping = dict((row.get("path"), row.get("sha256")) for row in rows)
    need(len(mapping) == len(rows), "duplicate execution file")
    for rel, digest in mapping.items():
        path = HW / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "execution identity drift: " + rel)
        if path.suffix in {".py", ".sv", ".tcl", ".f"}:
            need("init" + "reg" not in path.read_text().lower(),
                 "forbidden initialization token: " + rel)
            need(not active_force_present(path), "active force: " + rel)
    expected_authored = {RUNNER.relative_to(HW).as_posix(),
                         CHECKER.relative_to(HW).as_posix(),
                         TEST.relative_to(HW).as_posix(),
                         "dc_handoff/filelists/iscas_m1753_c2_m1609_k1_mapped_directed_energy.f"}
    need(expected_authored.issubset(set(mapping)), "authored source inventory")
    for axis in AXES:
        mapped = contract.get("mapped_axes", {}).get(axis, {})
        for key in ("netlist", "sdc"):
            row = mapped.get(key, {})
            path = HW / row.get("path", "__missing__")
            need(path.is_file() and not path.is_symlink()
                 and sha(path) == row.get("sha256"), "mapped axis drift")
        need(mapped.get("cycles") == AXES[axis]["cycles"], "axis cycle drift")
        need(math.isclose(mapped.get("area_um2"), AREAS_UM2[axis],
                          rel_tol=0.0, abs_tol=1e-9), "axis area drift")
    runner = RUNNER.read_text()
    for token in ('AXES = ("k1", "k8", "k1x8")',
                  '"vcs_compiles": 3', '"simv_runs": 15',
                  '"saif_files": 15', '"ptpx_runs": 15',
                  '"DIRECTED_COMPONENT_NOT_PRODUCTION"',
                  '"whole_component_report_power": True',
                  '"must_be_same_table_and_sentence": True'):
        need(token in runner, "runner gate absent: " + token)
    need("m1730_for_m1753" not in runner
         and "run_m1730_m1715" not in runner,
         "drifted M1730 executable identity inherited")
    pt_text = OLD.PT_TCL.read_text()
    need("report_power -unit mW" in pt_text, "whole report_power absent")
    need("report_power $" not in pt_text, "cell collection report_power forbidden")
    need("read_saif -strip_path" in pt_text
         and "annotated_nets != $total_nets" in pt_text
         and "annotated_leaf_cells != $total_leaf_cells" in pt_text,
         "exact SAIF annotation gates absent")
    for path in (M1760, M1761, Path(str(M1761) + ".sha256"),
                 Path(str(M1761) + ".sha256.seal.sha256"),
                 HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed",
                 HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901"):
        need(not os.path.lexists(path), "future/result namespace exists: " + str(path))
    need(contract.get("workload_boundary") == {
        "class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
        "cases_per_axis": 5, "accepted_sources_per_case": EVENTS,
        "accepted_sources_per_axis": 261, "clock_period_ns": 3.0,
        "trace_or_system_energy": False}, "workload boundary drift")
    return {"schema": "m1753_c2_three_axis_energy_source_check_r1_v1",
            "status": "PASS_M1753_SOURCE_ONLY_NO_EDA",
            "axes": list(AXES), "cases_per_axis": 5,
            "accepted_sources_per_axis": 261,
            "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
            "whole_component_ptpx": True,
            "joint_cycle_area_efficiency_disclosure": True,
            "claim_boundary": CLAIMS}


def main() -> None:
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
