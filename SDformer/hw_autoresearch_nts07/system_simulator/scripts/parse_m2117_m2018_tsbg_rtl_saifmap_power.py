#!/opt/anaconda3/bin/python3
"""Fail-closed parser for the M2117 RTL-SAIF/saif_map/PTPX campaign.

This parser never invokes EDA.  It validates raw RTL SAIF conservation and
unknown time, classifies Synopsys transformation maps, and admits matched
power only when annotation, nonzero activity, critical-cone activity, power
component arithmetic, and energy arithmetic all close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


AXES = {
    "ordinary_lru4": {"mode": 0, "cycles": 20292, "reads": 14304},
    "tsbg_b4": {"mode": 1, "cycles": 7569, "reads": 4608},
}
CRITICAL = (
    "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
    "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
)
POWER_FIELDS = {
    "switching_mw": "Net Switching Power",
    "internal_mw": "Cell Internal Power",
    "leakage_mw": "Cell Leakage Power",
    "total_mw": "Total Power",
}


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlink: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def one_float(pattern: str, text: str, label: str) -> float:
    values = re.findall(pattern, text, flags=re.MULTILINE)
    need(len(values) == 1, f"nonunique/missing {label}: {len(values)}")
    return float(values[0])


def parse_saif(path: Path, axis: str) -> dict[str, object]:
    need(axis in AXES, "unknown axis")
    text = read(path)
    scale_match = re.findall(
        r"\(TIMESCALE\s+([0-9.eE+-]+)\s+([A-Za-z]+)\)", text)
    duration_match = re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", text)
    need(len(scale_match) == 1 and len(duration_match) == 1,
         "nonunique SAIF header")
    scale, unit = float(scale_match[0][0]), scale_match[0][1]
    ns_per_unit = {
        "s": 1.0e9, "ms": 1.0e6, "us": 1.0e3,
        "ns": 1.0, "ps": 1.0e-3, "fs": 1.0e-6,
    }
    need(unit in ns_per_unit, f"unsupported SAIF unit: {unit}")
    duration_raw = float(duration_match[0])
    duration_ns = duration_raw * scale * ns_per_unit[unit]
    expected_ns = AXES[axis]["cycles"] * 3.0
    need(math.isclose(duration_ns, expected_ns, rel_tol=0.0, abs_tol=1.0e-6),
         f"duration {duration_ns} != {expected_ns}")

    records = re.findall(
        r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
        r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
    need(len(records) >= 100, f"too few SAIF records: {len(records)}")
    tx_sum = 0.0
    toggled = 0
    conservation_failures = 0
    for t0s, t1s, txs, tcs in records:
        t0, t1, tx, tc = map(float, (t0s, t1s, txs, tcs))
        need(min(t0, t1, tx, tc) >= 0.0, "negative SAIF field")
        tx_sum += tx
        toggled += int(tc > 0.0)
        if not math.isclose(t0 + t1 + tx, duration_raw,
                            rel_tol=0.0, abs_tol=max(1.0, scale)):
            conservation_failures += 1
    need(tx_sum == 0.0, f"SAIF TX sum nonzero: {tx_sum}")
    need(conservation_failures == 0,
         f"SAIF T0/T1/TX conservation failures: {conservation_failures}")
    need(toggled >= 20, f"insufficient nonzero-toggle records: {toggled}")
    for token in CRITICAL:
        token_records = re.findall(
            rf"\({re.escape(token)}(?:\\?\[[^\]]+\])?\s+"
            rf"\(T0\s+[0-9.eE+-]+\)\s*\(T1\s+[0-9.eE+-]+\)\s*"
            rf"\(TX\s+0(?:\.0+)?\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
        need(token_records and any(float(value) > 0 for value in token_records),
             f"missing/zero RTL critical activity: {token}")
    return {
        "axis": axis,
        "sha256": sha256(path),
        "duration_raw": duration_raw,
        "duration_ns": duration_ns,
        "expected_cycles": AXES[axis]["cycles"],
        "record_count": len(records),
        "nonzero_toggle_record_count": toggled,
        "tx_sum": tx_sum,
        "conservation_failures": conservation_failures,
    }


MAP_RE = re.compile(
    r"^set_rtl_to_gate_name\s+-rtl\s+\{([^}]+)\}\s+-gate\s+(.+?)\s*$",
    flags=re.MULTILINE,
)


def map_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for rtl, gate in MAP_RE.findall(read(path)):
        need(rtl not in rows or rows[rtl] == gate,
             f"intra-map conflict for {rtl}: {path}")
        rows[rtl] = gate
    need(rows, f"empty transformation map: {path}")
    return rows


def classify_maps(default_path: Path, essential_path: Path,
                  output: Path | None = None) -> dict[str, object]:
    default = map_rows(default_path)
    essential = map_rows(essential_path)
    intersection = set(default) & set(essential)
    target_differences = sorted(
        name for name in intersection if default[name] != essential[name])
    union = set(default) | set(essential)
    need(len(union) == len(default) + len(essential) - len(intersection),
         "map union arithmetic")
    value = {
        "default": {"path": str(default_path), "sha256": sha256(default_path),
                    "entries": len(default)},
        "essential": {"path": str(essential_path),
                      "sha256": sha256(essential_path),
                      "entries": len(essential)},
        "intersection_entries": len(intersection),
        "union_entries": len(union),
        # Synopsys can intentionally map one RTL state name to a sequential
        # cell in the default class and to a Q pin/net in -essential.  These
        # are complementary mapping types, not an error; PT sources default
        # first and essential second.  Preserve the exact disagreements for
        # review instead of silently calling them conflicts.
        "intersection_target_difference_entries": len(target_differences),
        "intersection_rtl_names": sorted(intersection),
        "intersection_target_difference_rtl_names": target_differences,
    }
    if output is not None:
        write_json(output, value)
    return value


def parse_annotation(path: Path) -> dict[str, float | int]:
    text = read(path)
    total = int(one_float(r"Total number of nets\s*=\s*([0-9]+)",
                          text, "total nets"))
    annotated_match = re.findall(
        r"Number of annotated nets\s*=\s*([0-9]+)\s*\(([0-9.]+)%\)", text)
    need(len(annotated_match) == 1, "annotated nets parse")
    annotated, percent = int(annotated_match[0][0]), float(annotated_match[0][1])
    leaf = int(one_float(r"Total number of leaf cells\s*=\s*([0-9]+)",
                         text, "total leaf"))
    leaf_match = re.findall(
        r"Number of fully annotated leaf cells\s*=\s*([0-9]+)\s*"
        r"\(([0-9.]+)%\)", text)
    need(len(leaf_match) == 1, "annotated leaf parse")
    annotated_leaf = int(leaf_match[0][0])
    leaf_percent = float(leaf_match[0][1])
    need(total > 0 and leaf > 0 and annotated <= total and annotated_leaf <= leaf,
         "annotation count domain")
    need(percent >= 95.0 and leaf_percent >= 95.0,
         f"annotation below 95%: net={percent}, leaf={leaf_percent}")
    return {"total_nets": total, "annotated_nets": annotated,
            "net_percent": percent, "total_leaf_cells": leaf,
            "fully_annotated_leaf_cells": annotated_leaf,
            "leaf_percent": leaf_percent}


def parse_switching_coverage(path: Path) -> dict[str, float | int]:
    text = read(path)
    rows = re.findall(r"^m2018[^\s]*\s+([0-9.]+)\s+([0-9]+)\s+([0-9]+)\s*$",
                      text, flags=re.MULTILINE)
    need(len(rows) == 1, f"switching coverage row count: {len(rows)}")
    percent, covered, total = float(rows[0][0]), int(rows[0][1]), int(rows[0][2])
    need(total > 0 and 0 < covered <= total and percent >= 20.0,
         f"nonzero-toggle coverage below gate: {percent}%")
    return {"percent": percent, "covered_nets": covered, "total_nets": total}


def parse_critical(path: Path, name: str) -> dict[str, object]:
    text = read(path)
    need(name in text, f"critical report identity missing: {name}")
    # Restrict the numeric test to object rows containing the cone name.  This
    # prevents dates, tool versions, or a bus index in the report header from
    # masquerading as switching activity.  Strip bus indices before parsing.
    object_rows = [line for line in text.splitlines()
                   if name in line and not line.lstrip().startswith((
                       "Report", "Design", "Date", "Version"))]
    row_toggle_values = []
    for line in object_rows:
        without_indices = re.sub(r"\\?\[[0-9]+\]", "", line)
        values = [float(value) for value in re.findall(
            r"(?<![A-Za-z_])[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?",
            without_indices, flags=re.IGNORECASE)]
        if values:
            row_toggle_values.append(values[0])
    need(object_rows and any(value > 0.0 for value in row_toggle_values),
         f"critical cone zero/unreported: {name}")
    return {"name": name, "sha256": sha256(path),
            "object_row_count": len(object_rows), "has_nonzero_numeric": True}


def parse_power(path: Path, duration_ns: float) -> dict[str, float]:
    text = read(path)
    values: dict[str, float] = {}
    for key, label in POWER_FIELDS.items():
        values[key] = one_float(
            rf"^\s*{re.escape(label)}\s*=\s*([0-9.eE+-]+)", text, label)
        need(math.isfinite(values[key]) and values[key] >= 0.0,
             f"invalid power {key}")
    subtotal = values["switching_mw"] + values["internal_mw"] + values["leakage_mw"]
    need(math.isclose(subtotal, values["total_mw"], rel_tol=2.0e-4,
                      abs_tol=1.0e-6),
         f"power components do not sum: {subtotal} vs {values['total_mw']}")
    values["duration_ns"] = duration_ns
    values["energy_nj"] = values["total_mw"] * duration_ns * 1.0e-3
    values["dynamic_energy_nj"] = (
        values["switching_mw"] + values["internal_mw"]
    ) * duration_ns * 1.0e-3
    values["leakage_energy_nj"] = values["leakage_mw"] * duration_ns * 1.0e-3
    need(math.isclose(values["energy_nj"],
                      values["dynamic_energy_nj"] + values["leakage_energy_nj"],
                      rel_tol=2.0e-4, abs_tol=1.0e-9), "energy components")
    return values


def parse_axis(axis_root: Path, axis: str) -> dict[str, object]:
    cfg = AXES[axis]
    saif = parse_saif(axis_root / "rtl_execute.saif", axis)
    map_audit = classify_maps(
        axis_root / "dc/netlist/m2018_axis.ptpx_map.default.tcl",
        axis_root / "dc/netlist/m2018_axis.ptpx_map.essential.tcl",
        axis_root / "map_classification.json")
    pt = axis_root / "ptpx/reports"
    annotation = parse_annotation(pt / "saif_annotation_summary.rpt")
    switching = parse_switching_coverage(pt / "switching_coverage.rpt")
    inconsistent = read(pt / "inconsistent_annotation.rpt")
    need(not re.search(r"\b(inconsistent|error|failed)\b", inconsistent,
                       flags=re.IGNORECASE), "inconsistent SAIF annotation")
    critical = [parse_critical(pt / f"critical_{name}_activity.rpt", name)
                for name in CRITICAL]
    power = parse_power(pt / "power.rpt", cfg["cycles"] * 3.0)
    scope = read(pt / "scope_and_boundary.rpt")
    for token in (
        f"axis={axis}", f"measurement_cycles={cfg['cycles']}",
        f"scalar_weight_reads={cfg['reads']}",
        "weight_sram_capacity_bytes=294912",
        "activity=mapped_netlist_power_driven_by_transformation_mapped_RTL_SAIF",
        "mapped_gate_vcs_activity=false", "macro_count=0",
    ):
        need(token in scope, f"scope boundary missing: {token}")
    need((axis_root / "ptpx/PTPX_INTERNAL_COMPLETE.txt").is_file(),
         "PTPX terminal absent")
    return {"axis": axis, "schedule_mode": cfg["mode"],
            "measurement_cycles": cfg["cycles"],
            "measurement_duration_ns": cfg["cycles"] * 3.0,
            "scalar_weight_reads": cfg["reads"],
            "weight_sram_capacity_bytes": 294912,
            "rtl_saif": saif, "map_classification": map_audit,
            "annotation": annotation, "nonzero_toggle_coverage": switching,
            "critical_cones": critical, "logic_power": power}


def final_result(root: Path, output: Path) -> dict[str, object]:
    axes = {axis: parse_axis(root / axis, axis) for axis in AXES}
    base, cand = axes["ordinary_lru4"], axes["tsbg_b4"]
    bp, cp = base["logic_power"], cand["logic_power"]
    result = {
        "schema": "m2117_m2018_tsbg_rtl_saifmap_power_result_r1_v1",
        "status": "PASS_RAW_M2117_MATCHED_RTL_SAIFMAP_PTPX_PENDING_INDEPENDENT_RESULT_HAMMER",
        "axes": axes,
        "comparison": {
            "cycle_speedup": base["measurement_cycles"] / cand["measurement_cycles"],
            "scalar_weight_read_reduction_fraction": 1.0 - cand["scalar_weight_reads"] / base["scalar_weight_reads"],
            "logic_energy_reduction_fraction": 1.0 - cp["energy_nj"] / bp["energy_nj"],
            "logic_dynamic_energy_reduction_fraction": 1.0 - cp["dynamic_energy_nj"] / bp["dynamic_energy_nj"],
            "common_weight_sram_capacity_bytes": 294912,
            "external_sram_dynamic_energy_included": False,
            "external_sram_dynamic_energy_model_ready_reads": {
                "ordinary_lru4": 14304, "tsbg_b4": 4608,
            },
        },
        "claim_boundary": {
            "mapped_netlist_power_driven_by_transformation_mapped_rtl_activity": True,
            "mapped_gate_vcs_activity": False,
            "same_fixed_workload": True,
            "schedule_mode_only_axis_difference": True,
            "logic_only": True,
            "external_weight_sram_area_included": False,
            "external_weight_sram_dynamic_energy_included": False,
            "prelayout": True, "ideal_clock": True,
            "wireload": "ZeroWireload", "system_speedup": False,
            "energy_frame": False, "paper_ppa_ready": False,
        },
    }
    write_json(output, result)
    return result


def static_check() -> dict[str, object]:
    source = Path(__file__).read_text()
    checks = {
        "two_axes_exact": set(AXES) == {"ordinary_lru4", "tsbg_b4"},
        "schedule_mode_only_values": [AXES[a]["mode"] for a in AXES] == [0, 1],
        "fixed_read_counts": AXES["ordinary_lru4"]["reads"] == 14304
            and AXES["tsbg_b4"]["reads"] == 4608,
        "tx_gate_present": "tx_sum == 0.0" in source,
        "duration_gate_present": "duration_ns, expected_ns" in source,
        "intra_map_conflict_gate_present": "intra-map conflict" in source,
        "annotation_95_gate_present": "percent >= 95.0" in source,
        "nonzero_coverage_gate_present": "percent >= 20.0" in source,
        "power_energy_arithmetic_present": "power components do not sum" in source
            and "energy components" in source,
        "sram_reads_disclosed": "external_sram_dynamic_energy_model_ready_reads" in source,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2117_STATIC_PARSER", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    saif_p = sub.add_parser("saif")
    saif_p.add_argument("--axis", required=True, choices=AXES)
    saif_p.add_argument("--path", required=True, type=Path)
    maps_p = sub.add_parser("maps")
    maps_p.add_argument("--default", required=True, type=Path)
    maps_p.add_argument("--essential", required=True, type=Path)
    maps_p.add_argument("--output", required=True, type=Path)
    final_p = sub.add_parser("final")
    final_p.add_argument("--root", required=True, type=Path)
    final_p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "static":
            value = static_check()
        elif args.command == "saif":
            value = parse_saif(args.path, args.axis)
        elif args.command == "maps":
            value = classify_maps(args.default, args.essential, args.output)
        else:
            value = final_result(args.root, args.output)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except Failure as exc:
        print(f"M2117_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
