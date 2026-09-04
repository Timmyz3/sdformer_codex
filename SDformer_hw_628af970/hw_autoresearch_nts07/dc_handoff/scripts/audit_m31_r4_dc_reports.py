#!/usr/bin/env python3
"""Fail-closed report audit for fresh M31-r4 3.000 ns logic-only DC."""

import argparse
import hashlib
import json
import re
from pathlib import Path


DESIGN = "qfit_atlif_unified_t10_t2_stream_core"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing M31 DC {}".format(label))
    return path


def exact_int(text, pattern, label):
    rows = re.findall(pattern, text, re.MULTILINE)
    if len(rows) != 1:
        raise ValueError("M31 DC {} population drift".format(label))
    return int(rows[0])


def exact_float(text, pattern, label):
    rows = re.findall(pattern, text, re.MULTILINE)
    if len(rows) != 1:
        raise ValueError("M31 DC {} population drift".format(label))
    return float(rows[0])


def parse_resource_audit(path):
    text = require_file(path, "resource audit").read_text(encoding="utf-8")
    scalar = {}
    leaf_rows = []
    for line in text.splitlines():
        if line.startswith("leaf="):
            match = re.match(
                r"^leaf=(\S+) ref=(\S+) mapped_cells=(\d+) "
                r"mapped_area=([0-9]+(?:\.[0-9]+)?)$", line)
            if not match:
                raise ValueError("M31 DC malformed multiplier leaf row")
            leaf_rows.append({
                "path": match.group(1), "ref": match.group(2),
                "mapped_cells": int(match.group(3)),
                "mapped_area": float(match.group(4)),
            })
        elif "=" in line:
            key, value = line.split("=", 1)
            if key in scalar:
                raise ValueError("M31 DC duplicate resource audit field")
            scalar[key] = value
    expected_scalar = {
        "stage", "pool_count", "leaf_count", "pool_path",
        "pool_external_leaf_count", "empty_mapped_leaf_count", "status",
    }
    if set(scalar) != expected_scalar:
        raise ValueError("M31 DC resource audit key population drift")
    if scalar != {
            "stage": "postcompile", "pool_count": "1", "leaf_count": "96",
            "pool_path": "u_mul_pool", "pool_external_leaf_count": "0",
            "empty_mapped_leaf_count": "0",
            "status": "PASS_EXACT_ONE_POOL_96_LEAVES"}:
        raise ValueError("M31 DC exact multiplier resource contract failed")
    if len(leaf_rows) != 96 or len(set(row["path"] for row in leaf_rows)) != 96:
        raise ValueError("M31 DC multiplier leaf row population drift")
    for row in leaf_rows:
        if (not row["path"].startswith("u_mul_pool/")
                or not row["ref"].startswith("qfit_signed_int8_mul_leaf")
                or row["mapped_cells"] <= 0 or row["mapped_area"] <= 0.0):
            raise ValueError("M31 DC empty or external multiplier leaf")
    return {
        "pool_hierarchy_instances": 1,
        "multiplier_leaf_hierarchy_instances": len(leaf_rows),
        "pool_external_multiplier_leaf_instances": 0,
        "empty_mapped_multiplier_leaf_instances": 0,
        "minimum_mapped_cells_per_multiplier_leaf": min(
            row["mapped_cells"] for row in leaf_rows),
        "minimum_mapped_area_per_multiplier_leaf_um2": min(
            row["mapped_area"] for row in leaf_rows),
    }


def parse_clock_report(path, period):
    text = require_file(path, "clock report").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if re.match(r"^core_clk\s+", line):
            match = re.match(
                r"^core_clk\s+([0-9]+(?:\.[0-9]+)?)\s+"
                r"\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$", line)
            if not match:
                raise ValueError("M31 DC malformed core clock row")
            rows.append((float(match.group(1)), match.group(2)))
    if len(rows) != 1 or abs(rows[0][0] - period) > 1e-9:
        raise ValueError("M31 DC exact core clock population drift")
    attributes = rows[0][1]
    if "p" in attributes or "G" in attributes or "g" in attributes:
        raise ValueError("M31 DC clock unexpectedly propagated or generated")
    return {
        "clock_count": 1,
        "clock_period_ns": rows[0][0],
        "clock_attributes": attributes,
        "clock_network_model": "IDEAL_UNPROPAGATED",
    }


def parse_slack(path, delay_type):
    text = require_file(path, "{} timing report".format(delay_type)).read_text(
        encoding="utf-8")
    if "slack (VIOLATED)" in text:
        raise ValueError("M31 DC {} timing violation".format(delay_type))
    values = re.findall(r"^\s*slack \(MET\)\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$",
                        text, re.MULTILINE)
    if not values:
        raise ValueError("M31 DC {} MET slack is missing".format(delay_type))
    return min(float(value) for value in values)


def build(run_dir, period=3.000):
    run = Path(run_dir).resolve()
    reports = run / "reports"
    qor_path = require_file(reports / "qor.rpt", "QoR report")
    area_path = require_file(reports / "area.rpt", "area report")
    clocks_path = require_file(reports / "clocks.rpt", "clock report")
    resource_path = require_file(
        reports / "m31_resource_audit_postcompile.rpt", "resource audit")
    references_path = require_file(
        reports / "references_postcompile.rpt", "reference report")
    setup_path = require_file(reports / "timing_setup.rpt", "setup report")
    hold_path = require_file(reports / "timing_hold.rpt", "hold report")
    dc_log_path = require_file(run / "dc.log", "log")
    netlist_path = require_file(run / "netlist/{}_mapped.v".format(DESIGN),
                                "mapped netlist")
    svf_path = require_file(run / "netlist/{}.svf".format(DESIGN), "SVF")

    qor = qor_path.read_text(encoding="utf-8")
    area = area_path.read_text(encoding="utf-8")
    references = references_path.read_text(encoding="utf-8")
    dc_log = dc_log_path.read_text(encoding="utf-8")
    if "Design : {}".format(DESIGN) not in qor:
        raise ValueError("M31 DC QoR design identity drift")
    hierarchical_cells = exact_int(
        qor, r"^\s*Hierarchical Cell Count:\s+(\d+)\s*$", "hierarchical cells")
    leaf_cells = exact_int(
        qor, r"^\s*Leaf Cell Count:\s+(\d+)\s*$", "leaf cells")
    qor_macros = exact_int(qor, r"^\s*Macro Count:\s+(\d+)\s*$", "QoR macros")
    area_total_cells = exact_int(
        area, r"^Number of cells:\s+(\d+)\s*$", "area total cells")
    combinational_cells = exact_int(
        area, r"^Number of combinational cells:\s+(\d+)\s*$",
        "combinational cells")
    sequential_cells = exact_int(
        area, r"^Number of sequential cells:\s+(\d+)\s*$",
        "sequential cells")
    area_macros = exact_int(
        area, r"^Number of macros/black boxes:\s+(\d+)\s*$", "area macros")
    if area_total_cells != hierarchical_cells + leaf_cells:
        raise ValueError("M31 DC total/hierarchical/leaf cell accounting drift")
    if leaf_cells != combinational_cells + sequential_cells:
        raise ValueError("M31 DC leaf/combinational/sequential accounting drift")
    if qor_macros != 0 or area_macros != 0:
        raise ValueError("M31 DC unexpected macro or black-box population")
    cell_area = exact_float(
        area, r"^Total cell area:\s+([0-9]+(?:\.[0-9]+)?)\s*$",
        "total cell area")
    if not re.search(
            r"^Net Interconnect area:\s+undefined\s+"
            r"\(Wire load has zero net area\)\s*$", area, re.MULTILINE):
        raise ValueError("M31 DC is not the admitted zero-wire area model")
    if exact_float(qor, r"^\s*Net Area:\s+([0-9]+(?:\.[0-9]+)?)\s*$",
                   "QoR net area") != 0.0:
        raise ValueError("M31 DC QoR net area is nonzero")
    if re.search(r"(^|[^A-Za-z])(GTECH|DW_[A-Za-z0-9_]*mult|unresolved)",
                 references, re.IGNORECASE):
        raise ValueError("M31 DC unresolved arithmetic reference")
    if re.search(r"^\s*(Error|Fatal):", dc_log, re.MULTILINE):
        raise ValueError("M31 DC log contains an error or fatal")

    resource = parse_resource_audit(resource_path)
    clock = parse_clock_report(clocks_path, float(period))
    setup_slack = parse_slack(setup_path, "setup")
    hold_slack = parse_slack(hold_path, "hold")
    return {
        "schema": "m31_r4_fresh_dc_report_audit_v1",
        "status": "PASS_M31_R4_EXACT96_ZERO_WIRE_IDEAL_CLOCK_3NS_LOGIC_ONLY",
        "identity": {
            "run_directory": str(run),
            "mapped_netlist_sha256": sha256(netlist_path),
            "svf_sha256": sha256(svf_path),
            "report_sha256": {
                path.name: sha256(path) for path in (
                    qor_path, area_path, clocks_path, resource_path,
                    references_path, setup_path, hold_path, dc_log_path)
            },
        },
        "resource_audit": resource,
        "cell_accounting": {
            "total_cell_instances_including_hierarchy": area_total_cells,
            "hierarchical_cell_instances": hierarchical_cells,
            "leaf_mapped_cell_instances": leaf_cells,
            "combinational_leaf_cell_instances": combinational_cells,
            "sequential_leaf_cell_instances": sequential_cells,
            "macro_or_black_box_instances": 0,
        },
        "physical_assumptions": dict(clock,
            interconnect_area_model="ZERO_WIRE_LOAD",
            net_interconnect_area_um2=0.0,
            macro_timing_models="NONE"),
        "timing": {
            "setup_wns_ns": setup_slack,
            "hold_wns_ns": hold_slack,
            "setup_and_hold_met": setup_slack >= 0.0 and hold_slack >= 0.0,
        },
        "area": {
            "total_cell_area_um2": cell_area,
            "placed_or_routed_area_admitted": False,
        },
        "admission": {
            "fresh_current_source_dc_sta_admitted": True,
            "formality_admitted": False,
            "paper_ppa_ready": False,
            "power_energy_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False,
        },
    }


def write_output(path, result):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite M31 DC machine audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--period", type=float, default=3.000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.run_dir, args.period)
    write_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
