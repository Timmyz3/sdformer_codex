#!/usr/bin/env python3
"""Fail-closed parser for the M2121 matched macro-free ICC2 experiment.

This parser admits only a pair of routed logic islands.  It deliberately does
not promote the result to macro-inclusive, accelerator-wide, or signoff PPA.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict


AXES = ("ordinary_lru4", "tsbg_b4")
PASS_TOKEN = "PASS_M2121_MATCHED_MACROFREE_ICC2_AXIS"
REQUIRED_EQUAL = (
    "top",
    "public_port_count",
    "input_master_count",
    "tt_master_coverage",
    "ss_master_coverage",
    "ff_master_coverage",
    "physical_master_coverage",
    "floorplan_policy",
    "pin_policy",
    "route_layers",
    "cts_cell_policy",
    "hold_cell_policy",
    "clock_period_ns",
    "setup_uncertainty_ns",
    "hold_uncertainty_ns",
    "parasitic_tech",
    "parasitic_corner_scope",
    "common_external_sram_bytes",
    "common_external_sram_integrated",
    "physical_sdc_sha256",
    "flow_tcl_sha256",
    "routing_policy_sha256",
    "scenario_policy_sha256",
    "floorplan_actual_sha256",
    "die_boundary_actual",
    "core_bbox_actual",
    "setup_scenario_actual",
    "hold_scenario_actual",
    "power_scenario_actual",
)


OPEN_PATTERNS = (
    re.compile(r"^\s*Total number of open nets\s*=\s*(\d+)\s*\.?\s*$", re.I),
    re.compile(r"^\s*TOTAL OPEN NETS\s*[:=]\s*(\d+)\s*$", re.I),
    re.compile(r"^\s*Total\s+(\d+)\s+nets have their routing open(?:\s|\().*$", re.I),
    re.compile(r"^\s*Total open nets\s*[:=]\s*(\d+)\s*$", re.I),
)
DRC_PATTERNS = (
    re.compile(r"^\s*Total number of DRC(?:s| violations)\s*=\s*(\d+)\s*\.?\s*$", re.I),
    re.compile(r"^\s*TOTAL (?:DRC )?VIOLATIONS\s*[:=]\s*(\d+)\s*$", re.I),
)


def parse_route_counts(path):
    """Parse real check_routes numeric summaries; status/exit code is insufficient."""
    text = path.read_text(encoding="utf-8", errors="strict")
    values = {"open": [], "drc": []}
    for line in text.splitlines():
        for pattern in OPEN_PATTERNS:
            match = pattern.match(line)
            if match:
                values["open"].append(int(match.group(1)))
                break
        for pattern in DRC_PATTERNS:
            match = pattern.match(line)
            if match:
                values["drc"].append(int(match.group(1)))
                break
    if not values["open"] or not values["drc"]:
        raise ValueError(f"{path}: missing anchored check_routes open/DRC summaries")
    if set(values["open"]) != {0} or set(values["drc"]) != {0}:
        raise ValueError(f"{path}: nonzero or contradictory route counts {values}")
    return 0, 0


def parse_def_physical_identity(path):
    """Return actual die and full placed pin inventory from routed DEF."""
    text = path.read_text(encoding="utf-8", errors="strict")
    unit = re.search(r"^UNITS DISTANCE MICRONS\s+(\d+)\s*;", text, re.M)
    die = re.search(
        r"^DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+"
        r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;", text, re.M)
    header = re.search(r"^PINS\s+(\d+)\s*;\s*$", text, re.M)
    section = re.search(r"^PINS\s+\d+\s*;\s*$(.*?)^END PINS\s*$", text, re.M | re.S)
    if not unit or not die or not header or not section:
        raise ValueError(f"{path}: incomplete DEF units/die/pins")
    pin_blocks = re.findall(r"^\s*-\s+(\S+)(.*?);\s*$", section.group(1), re.M | re.S)
    if len(pin_blocks) != int(header.group(1)) or len(pin_blocks) != 4551:
        raise ValueError(f"{path}: DEF pin cardinality mismatch")
    pins = []
    for name, body in pin_blocks:
        layer = re.search(r"\+\s+LAYER\s+(\S+)", body)
        placed = re.search(r"\+\s+(?:FIXED|PLACED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)", body)
        if not layer or not placed:
            raise ValueError(f"{path}: pin {name} lacks actual layer/location")
        pins.append((name, layer.group(1), int(placed.group(1)), int(placed.group(2)), placed.group(3)))
    if len({p[0] for p in pins}) != 4551:
        raise ValueError(f"{path}: duplicate DEF pin")
    return {
        "dbu_per_micron": int(unit.group(1)),
        "diearea_dbu": tuple(map(int, die.groups())),
        "pins": tuple(sorted(pins)),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_facts(path):
    facts: Dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            raise ValueError(f"{path}:{lineno}: malformed fact")
        key, value = raw.split("=", 1)
        if not key or key in facts:
            raise ValueError(f"{path}:{lineno}: duplicate/empty key {key!r}")
        facts[key] = value
    return facts


def require_file(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty, or symlink artifact: {path}")


def parse_axis(root, expected_axis):
    required = [
        root / "RUN_COMPLETE.txt",
        root / "machine_facts.txt",
        root / "reports" / "ports_sorted.txt",
        root / "reports" / "actual_floorplan.txt",
        root / "reports" / "actual_routing_layers.rpt",
        root / "reports" / "actual_cts_cells.txt",
        root / "reports" / "actual_hold_cells.txt",
        root / "reports" / "actual_scenarios.rpt",
        root / "reports" / "reference_libraries.rpt",
        root / "reports" / "design_mismatch.rpt",
        root / "reports" / "pre_placement_check.rpt",
        root / "reports" / "pre_clock_check.rpt",
        root / "reports" / "pre_route_check.rpt",
        root / "reports" / "route_check.rpt",
        root / "reports" / "qor.rpt",
        root / "reports" / "timing_setup.rpt",
        root / "reports" / "timing_hold.rpt",
        root / "reports" / "clock_qor.rpt",
        root / "reports" / "congestion.rpt",
        root / "reports" / "wirelength.rpt",
        root / "output" / "routed.v",
        root / "output" / "routed.sdc",
        root / "output" / "routed.def",
    ]
    for path in required:
        require_file(path)
    strict_spefs = [p for p in (root / "output" / "routed.spef",
                                root / "output" / "routed.spef.gz") if p.exists()]
    if len(strict_spefs) != 1:
        raise ValueError(f"{expected_axis}: require exactly routed.spef or routed.spef.gz")
    spef = strict_spefs[0]
    require_file(spef)

    if root.joinpath("RUN_COMPLETE.txt").read_text(encoding="utf-8").strip() != PASS_TOKEN:
        raise ValueError(f"bad terminal token for {expected_axis}")
    facts = read_facts(root / "machine_facts.txt")
    if facts.get("status") != PASS_TOKEN or facts.get("axis") != expected_axis:
        raise ValueError(f"identity/status mismatch for {expected_axis}")

    exact = {
        "top": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend",
        "public_port_count": "4551",
        "input_master_count": "94",
        "tt_master_coverage": "94/94",
        "ss_master_coverage": "94/94",
        "ff_master_coverage": "94/94",
        "physical_master_coverage": "94/94",
        "unresolved_reference_count": "0",
        "accepted_mismatch_count": "0",
        "logical_physical_mismatch_count": "0",
        "routing_layer_gate_count": "9",
        "via_layer_gate_count": "8",
        "route_check_return": "1",
        "pre_placement_check_return": "1",
        "pre_clock_check_return": "1",
        "pre_route_check_return": "1",
        "die_bbox_um": "0,0,800,800",
        "core_bbox_um": "40,40,760,760",
        "floorplan_policy": "fixed_die_core_800_720um_v1",
        "pin_policy": "sorted_four_side_round_robin_exact_location_v1",
        "route_layers": "M2:M8",
        "cts_cell_policy": "CKBD_and_CKND_only_v1",
        "hold_cell_policy": "DEL_BUFF_INV_only_v1",
        "clock_period_ns": "3.000",
        "setup_uncertainty_ns": "0.200",
        "hold_uncertainty_ns": "0.050",
        "parasitic_tech": "n28_1p9m_6x1z1u_typ",
        "parasitic_corner_scope": "same_typical_rc_on_ss_ff_tt",
        "common_external_sram_bytes": "294912",
        "common_external_sram_integrated": "false",
        "propagated_clock": "true",
        "macro_instances": "0",
    }
    for key, expected in exact.items():
        if facts.get(key) != expected:
            raise ValueError(f"{expected_axis}: {key}={facts.get(key)!r}, expected {expected!r}")

    setup = float(facts["setup_wns_ns"])
    hold = float(facts["hold_wns_ns"])
    area = float(facts["routed_standard_cell_area_um2"])
    leaf = int(facts["routed_leaf_cell_count"])
    seq = int(facts["routed_sequential_cell_count"])
    clock_like = int(facts["clock_like_cell_count"])
    hold_like = int(facts["hold_like_cell_count"])
    for name, value in (("setup", setup), ("hold", hold), ("area", area)):
        if not math.isfinite(value):
            raise ValueError(f"{expected_axis}: non-finite {name}")
    if setup < 0.0 or hold < 0.0 or area <= 0.0:
        raise ValueError(f"{expected_axis}: timing/area admission failed")
    if leaf <= 0 or seq != 74460 or clock_like <= 0 or hold_like <= 0:
        raise ValueError(f"{expected_axis}: impl cell census admission failed")

    ports = (root / "reports" / "ports_sorted.txt").read_text(encoding="utf-8").splitlines()
    if len(ports) != 4551 or ports != sorted(ports) or len(set(ports)) != 4551:
        raise ValueError(f"{expected_axis}: invalid deterministic port inventory")
    if facts.get("port_inventory_sha256") != sha256(root / "reports" / "ports_sorted.txt"):
        raise ValueError(f"{expected_axis}: port signature mismatch")
    if facts.get("floorplan_actual_sha256") != sha256(root / "reports" / "actual_floorplan.txt"):
        raise ValueError(f"{expected_axis}: actual floorplan signature mismatch")
    floorplan_lines = (root / "reports" / "actual_floorplan.txt").read_text(encoding="utf-8").splitlines()
    if floorplan_lines != ["die_boundary=" + facts.get("die_boundary_actual", ""),
                           "core_bbox=" + facts.get("core_bbox_actual", "")]:
        raise ValueError(f"{expected_axis}: queried floorplan facts/report mismatch")
    policy_bytes = b"".join((root / "reports" / name).read_bytes() for name in (
        "actual_routing_layers.rpt", "actual_cts_cells.txt", "actual_hold_cells.txt"))
    if facts.get("routing_policy_sha256") != hashlib.sha256(policy_bytes).hexdigest():
        raise ValueError(f"{expected_axis}: actual routing/CTS/hold policy signature mismatch")
    if facts.get("scenario_policy_sha256") != sha256(root / "reports" / "actual_scenarios.rpt"):
        raise ValueError(f"{expected_axis}: actual scenario signature mismatch")
    scenario_text = (root / "reports" / "actual_scenarios.rpt").read_text(encoding="utf-8")
    scenario_names = [facts.get("setup_scenario_actual", ""), facts.get("hold_scenario_actual", ""),
                      facts.get("power_scenario_actual", "")]
    if any(not name or name not in scenario_text for name in scenario_names) or len(set(scenario_names)) != 3:
        raise ValueError(f"{expected_axis}: actual scenario report is not semantically bound")
    routing_text = (root / "reports" / "actual_routing_layers.rpt").read_text(encoding="utf-8")
    if "M2" not in routing_text or "M8" not in routing_text:
        raise ValueError(f"{expected_axis}: actual route-layer report lacks M2/M8")
    if not re.fullmatch(r"[0-9a-f]{64}", facts.get("flow_tcl_sha256", "")):
        raise ValueError(f"{expected_axis}: invalid flow Tcl identity")
    open_count, drc_count = parse_route_counts(root / "reports" / "route_check.rpt")
    if facts.get("route_open_net_count") != str(open_count):
        raise ValueError(f"{expected_axis}: route open-net fact/report mismatch")
    if facts.get("route_drc_violation_count") != str(drc_count):
        raise ValueError(f"{expected_axis}: route DRC fact/report mismatch")
    physical_identity_raw = parse_def_physical_identity(root / "output" / "routed.def")
    physical_identity = {
        "dbu_per_micron": physical_identity_raw["dbu_per_micron"],
        "diearea_dbu": physical_identity_raw["diearea_dbu"],
        "pin_count": len(physical_identity_raw["pins"]),
        "pin_inventory_sha256": hashlib.sha256(
            json.dumps(physical_identity_raw["pins"], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }

    return {
        "facts": facts,
        "setup_wns_ns": setup,
        "hold_wns_ns": hold,
        "routed_standard_cell_area_um2": area,
        "routed_leaf_cell_count": leaf,
        "routed_sequential_cell_count": seq,
        "clock_like_cell_count": clock_like,
        "hold_like_cell_count": hold_like,
        "port_inventory_sha256": facts["port_inventory_sha256"],
        "physical_identity": physical_identity,
        "routed_netlist_sha256": sha256(root / "output" / "routed.v"),
        "routed_sdc_sha256": sha256(root / "output" / "routed.sdc"),
        "routed_def_sha256": sha256(root / "output" / "routed.def"),
        "routed_spef_sha256": {spef.name: sha256(spef)},
    }


def parse_pair(ordinary_dir, tsbg_dir):
    axes = {
        "ordinary_lru4": parse_axis(ordinary_dir, "ordinary_lru4"),
        "tsbg_b4": parse_axis(tsbg_dir, "tsbg_b4"),
    }
    base = axes["ordinary_lru4"]["facts"]
    cand = axes["tsbg_b4"]["facts"]
    for key in REQUIRED_EQUAL:
        if base.get(key) != cand.get(key):
            raise ValueError(f"unmatched physical axis: {key}: {base.get(key)!r} != {cand.get(key)!r}")
    if axes["ordinary_lru4"]["port_inventory_sha256"] != axes["tsbg_b4"]["port_inventory_sha256"]:
        raise ValueError("ordinary/TSBG port inventories differ")
    if axes["ordinary_lru4"]["physical_identity"] != axes["tsbg_b4"]["physical_identity"]:
        raise ValueError("ordinary/TSBG actual DEF die/pin physical identities differ")

    base_area = axes["ordinary_lru4"]["routed_standard_cell_area_um2"]
    cand_area = axes["tsbg_b4"]["routed_standard_cell_area_um2"]
    return {
        "schema": "m2123_m2029_m2018_matched_macrofree_icc2_pnr_raw_receipt_r1_v1",
        "status": "PASS_RAW_M2123_MATCHED_MACROFREE_ICC2_PNR_PENDING_M2124_INDEPENDENT_RESULT_HAMMER",
        "axes": axes,
        "comparison": {
            "tsbg_over_ordinary_routed_logic_area_ratio": cand_area / base_area,
            "tsbg_routed_logic_area_overhead_fraction": cand_area / base_area - 1.0,
            "both_setup_met": True,
            "both_hold_met": True,
            "same_floorplan_pins_constraints_corners_route_effort": True,
        },
        "common_external_sram_model": {
            "capacity_bytes": 294912,
            "integrated_in_pnr": False,
            "area_leakage_equal_between_axes": True,
            "dynamic_read_energy_requires_separate_request_count_model": True,
        },
        "claim_boundary": {
            "matched_macro_free_post_route_logic_islands": True,
            "common_external_288kib_sram_model": True,
            "macro_inclusive": False,
            "sram_integrated_timing_or_power": False,
            "whole_accelerator": False,
            "whole_network": False,
            "emir_lvs_tapeout_signoff": False,
            "paper_ppa_ready": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinary-dir", type=Path, required=True)
    ap.add_argument("--tsbg-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ns = ap.parse_args()
    receipt = parse_pair(ns.ordinary_dir, ns.tsbg_dir)
    ns.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
