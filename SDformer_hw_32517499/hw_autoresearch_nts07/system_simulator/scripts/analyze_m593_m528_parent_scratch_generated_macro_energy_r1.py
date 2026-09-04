#!/usr/bin/env python3
"""Build a bounded generated-macro energy subtotal for M528 parent scratch.

This analyzer intentionally prices only the nine 128x128-bit 1RW parent-scratch
macros.  It is not a C1, chip, or system energy model.  It combines the frozen
M528 access/cycle ledger with the checksum-audited foundry datasheet current at
the slow 0.9-V corner.  Interconnect, clock tree, adapters, other memories,
logic, DRAM, and PTPX are outside the result.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
WORD_BYTES = 144
MACRO_COUNT = 9
VOLTAGE_V = 0.9
CLOCK_PERIOD_NS = 3.0
SAMPLE_COUNT = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{path}[{index}]")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON must be an object: {path}")
    assert_finite(value)
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_design(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    matches = [row for row in rows if row.get("design") == name]
    require(len(matches) == 1, f"expected one traffic row {name}, got {len(matches)}")
    return matches[0]


def energy_row(
    name: str,
    cycles: int,
    read_bytes: int,
    write_bytes: int,
    read_energy_pj: float,
    write_energy_pj: float,
    leakage_power_mw: float,
) -> Dict[str, Any]:
    require(read_bytes % WORD_BYTES == 0, f"{name} read bytes are not 144-B aligned")
    require(write_bytes % WORD_BYTES == 0, f"{name} write bytes are not 144-B aligned")
    reads = read_bytes // WORD_BYTES
    writes = write_bytes // WORD_BYTES
    dynamic_mj_per_frame = (
        reads * read_energy_pj + writes * write_energy_pj
    ) / SAMPLE_COUNT / 1.0e9
    latency_ms_per_frame = cycles * CLOCK_PERIOD_NS / SAMPLE_COUNT / 1.0e6
    leakage_mj_per_frame = leakage_power_mw * latency_ms_per_frame / 1000.0
    return {
        "design": name,
        "cycles_s10": cycles,
        "latency_ms_per_frame_at_3ns": latency_ms_per_frame,
        "read_bytes_s10": read_bytes,
        "write_bytes_s10": write_bytes,
        "read_accesses_s10": reads,
        "write_accesses_s10": writes,
        "dynamic_energy_mj_per_frame": dynamic_mj_per_frame,
        "leakage_energy_mj_per_frame": leakage_mj_per_frame,
        "modeled_parent_scratch_energy_mj_per_frame": (
            dynamic_mj_per_frame + leakage_mj_per_frame
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m528-result", type=Path, required=True)
    parser.add_argument("--m528-result-sha256", required=True)
    parser.add_argument("--m528-hammer", type=Path, required=True)
    parser.add_argument("--m528-hammer-sha256", required=True)
    parser.add_argument("--macro-map", type=Path, required=True)
    parser.add_argument("--macro-map-sha256", required=True)
    parser.add_argument("--docs359", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "m528_result": (args.m528_result, args.m528_result_sha256),
        "m528_hammer": (args.m528_hammer, args.m528_hammer_sha256),
        "macro_map": (args.macro_map, args.macro_map_sha256),
        "docs359": (args.docs359, DOCS359_SHA256),
    }
    identity: Dict[str, Dict[str, str]] = {}
    for name, (path, expected) in inputs.items():
        actual = sha256_file(path)
        require(actual == expected, f"{name} SHA mismatch: {actual} != {expected}")
        identity[name] = {"path": str(path), "sha256": actual}

    m528 = load_json(args.m528_result)
    hammer = load_json(args.m528_hammer)
    macro = load_json(args.macro_map)

    require(m528.get("schema") == "m528_h67_single_port_same_ledger_recompute_result_v1", "M528 schema drift")
    require(m528.get("claim_boundary", {}).get("exact_cpu_cycle_recompute") is True, "M528 CPU scope not admitted")
    require(m528.get("claim_boundary", {}).get("energy") is False, "M528 unexpectedly claims energy")
    require(hammer.get("score_100") == 99, "M528 result hammer score drift")
    require(hammer.get("claim_boundary", {}).get("admitted_exact_cpu_cycle_candidate") is True, "M528 hammer admission missing")
    require(hammer.get("claim_boundary", {}).get("traffic_is_logical_bytes_not_energy") is True, "M528 traffic boundary missing")
    require(macro.get("schema") == "tsmc28_sram_macro_mapping_audit_v1", "macro-map schema drift")
    require(macro.get("docs359_sha256") == DOCS359_SHA256, "macro-map docs359 identity drift")

    inventory = macro.get("generated_view_inventory", {})
    slow = inventory.get("slow", {})
    require(inventory.get("cell") == "TS1N28HPCPHVTB128X128M4S", "generated macro cell drift")
    require(inventory.get("logical_shape") == "128x128b 1RW SP", "generated macro shape drift")
    require(inventory.get("checksum_verification") == "13/13 OK on 2026-08-27", "generated view checksum status drift")
    require(slow.get("corner") == "ssg0p9v125c", "slow corner drift")
    require(float(slow.get("area_um2")) == 8758.3606, "macro area drift")
    require(float(slow.get("cycle_ns")) == 0.616, "macro cycle drift")
    require(float(slow.get("access_ns")) == 0.4679, "macro access drift")
    require(float(slow.get("readc_uA_per_MHz")) == 11.6754, "read current drift")
    require(float(slow.get("writec_uA_per_MHz")) == 11.1923, "write current drift")
    require(float(slow.get("leakage_uA")) == 66.6783, "leakage current drift")

    generated = m528["capacity"]["m505_dead_write_only_1rw"]["generated_parent_scratch"]
    require(generated["organization"] == "9 x 128x128-bit 1RW SP; lower 64 rows used", "M528 macro organization drift")
    require(float(generated["area_um2"]) == MACRO_COUNT * float(slow["area_um2"]), "M528 macro area does not match nine generated views")

    traffic = m528.get("traffic", {}).get("rows", [])
    require(isinstance(traffic, list), "M528 traffic rows missing")
    all_write_traffic = unique_design(traffic, "m473_fused_concurrent_1r1w_ceiling")
    dead_traffic = unique_design(traffic, "m505_dead_write_only_1rw")
    cycles = m528["aggregate_cycles"]

    read_energy_pj = MACRO_COUNT * float(slow["readc_uA_per_MHz"]) * VOLTAGE_V
    write_energy_pj = MACRO_COUNT * float(slow["writec_uA_per_MHz"]) * VOLTAGE_V
    leakage_power_mw = MACRO_COUNT * float(slow["leakage_uA"]) * VOLTAGE_V / 1000.0

    rows = [
        energy_row(
            "all_write_1rw_parent_scratch",
            int(cycles["m504_all_write_1rw_cycles"]),
            int(all_write_traffic["parent_scratch_read_bytes"]),
            int(all_write_traffic["parent_scratch_write_bytes"]),
            read_energy_pj,
            write_energy_pj,
            leakage_power_mw,
        ),
        energy_row(
            "dead_write_only_1rw_parent_scratch",
            int(cycles["m505_dead_write_only_1rw_cycles"]),
            int(dead_traffic["parent_scratch_read_bytes"]),
            int(dead_traffic["parent_scratch_write_bytes"]),
            read_energy_pj,
            write_energy_pj,
            leakage_power_mw,
        ),
    ]
    baseline, candidate = rows
    cycle_speedup = baseline["cycles_s10"] / candidate["cycles_s10"]
    energy_reduction = 1.0 - (
        candidate["modeled_parent_scratch_energy_mj_per_frame"]
        / baseline["modeled_parent_scratch_energy_mj_per_frame"]
    )
    require(abs(cycle_speedup - float(cycles["m504_to_dead_write_speedup"])) < 1e-15, "cycle ablation drift")

    result = {
        "schema": "m593_m528_parent_scratch_generated_macro_energy_result_v1",
        "status": "PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_INDEPENDENT_RESULT_HAMMER",
        "identity": identity,
        "scope": {
            "checkpoint": "H67 ep35",
            "sequence_count": 1,
            "sample_count": SAMPLE_COUNT,
            "operators": "four bottleneck Conv3x3 only",
            "component": "nine generated 128x128-bit 1RW parent-scratch macros only",
            "corner": "ssg0p9v125c at 0.9 V",
            "clock_period_ns": CLOCK_PERIOD_NS,
        },
        "macro": {
            "cell": inventory["cell"],
            "count": MACRO_COUNT,
            "area_um2": MACRO_COUNT * float(slow["area_um2"]),
            "cycle_ns": float(slow["cycle_ns"]),
            "access_ns": float(slow["access_ns"]),
            "full_1152b_read_energy_pj_per_access": read_energy_pj,
            "full_1152b_write_energy_pj_per_access": write_energy_pj,
            "leakage_power_mw": leakage_power_mw,
            "model_note": "datasheet current model; nine slices activated per logical 1152-bit access",
        },
        "rows": rows,
        "ablation": {
            "dead_write_only_cycle_speedup_vs_all_write": cycle_speedup,
            "dead_write_only_parent_scratch_energy_reduction_fraction": energy_reduction,
            "dead_write_only_parent_scratch_energy_reduction_percent": energy_reduction * 100.0,
            "dead_write_only_parent_scratch_energy_saved_mj_per_frame": (
                baseline["modeled_parent_scratch_energy_mj_per_frame"]
                - candidate["modeled_parent_scratch_energy_mj_per_frame"]
            ),
        },
        "claim_boundary": {
            "allowed_label": "generated-macro datasheet component model for M528 parent scratch",
            "component_energy_model": True,
            "exact_trace_access_counts": True,
            "generated_macro_area_and_current": True,
            "rtl_integrated_macro_ppa": False,
            "interconnect_or_clock_tree_energy": False,
            "logic_energy": False,
            "other_sram_energy": False,
            "dram_energy": False,
            "c1_total_energy": False,
            "energy_per_full_network_frame": False,
            "system_energy": False,
            "silicon_measurement": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "m593_m528_parent_scratch_generated_macro_energy_result_r1.json"
    csv_path = args.output_dir / "m593_parent_scratch_energy_rows_r1.csv"
    with json_path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
