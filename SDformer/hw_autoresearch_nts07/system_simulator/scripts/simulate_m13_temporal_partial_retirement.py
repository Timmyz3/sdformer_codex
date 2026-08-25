#!/usr/bin/env python3
"""Event model for M4 destination retirement into a recurrent ATLIF engine.

The simulator compares three causal schedules on stratified real Local/Motion
bitmaps.  It preserves the admitted M7 no-overlap total and uses the M7 hybrid
operator speed only to normalize each sampled producer timeline; it does not
turn sampled rows into a cycle-accurate full-network claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_bits(records: list[dict[str, str]], current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    current_bits = np.unpackbits(current, axis=1, bitorder="little").astype(bool)
    previous_bits = np.unpackbits(previous, axis=1, bitorder="little").astype(bool)
    use_motion = np.asarray(
        [row["row_use_motion"].lower() == "true" for row in records], dtype=bool
    )[:, None]
    return np.where(use_motion, current_bits ^ previous_bits, current_bits)


def physical_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"],
        row["operator_call_index"], row["weight_group"], row["row_id"],
    )


def geometry_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["weight_group"], row["source_width"],
        row["chunks_per_row"], row["output_channel_fanout"],
    )


def build_patterns(
    records: list[dict[str, str]], current: np.ndarray, previous: np.ndarray,
    wall: Any, *, contexts: int, issue_width: int, reduce_slots: int,
    output_lanes: int, sample_id: int,
) -> dict[str, list[dict[str, Any]]]:
    bits = selected_bits(records, current, previous)
    bundles = wall.ordered_row_bundles(records)
    bundle_by_key: dict[tuple[tuple[str, ...], int], list[int]] = {}
    physical_by_geometry: OrderedDict[tuple[str, ...], list[tuple[str, ...]]] = OrderedDict()
    for bundle in bundles:
        row = records[bundle[0]]
        if int(row["sample_id"]) != sample_id:
            continue
        physical = physical_key(row)
        step = int(row["temporal_step"])
        bundle_by_key[(physical, step)] = bundle
        physical_by_geometry.setdefault(geometry_key(row), [])
        if physical not in physical_by_geometry[geometry_key(row)]:
            physical_by_geometry[geometry_key(row)].append(physical)

    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for geometry, physical_rows in physical_by_geometry.items():
        physical_rows.sort(key=lambda key: int(key[-1]))
        name = geometry[2]
        chunks = int(geometry[7])
        fanout = int(geometry[8])
        lane_tiles = math.ceil(fanout / output_lanes)
        for start in range(0, len(physical_rows), contexts):
            group = physical_rows[start : start + contexts]
            steps = sorted(
                step for physical, step in bundle_by_key if physical == group[0]
            )
            if steps != list(range(len(steps))):
                raise ValueError(f"non-contiguous temporal steps for {name}")
            step_items = []
            for step in steps:
                step_bundles = [bundle_by_key[(physical, step)] for physical in group]
                descriptor_cycles = sum(len(bundle) for bundle in step_bundles)
                lane_compute = 0
                for chunk in range(chunks):
                    counts = np.zeros((len(group), issue_width), dtype=np.int64)
                    for context, bundle in enumerate(step_bundles):
                        index = bundle[chunk]
                        counts[context] = [
                            int(bits[index, bank::issue_width].sum())
                            for bank in range(issue_width)
                        ]
                    lane_compute += wall.compact_issue_cycles(counts, reduce_slots) + 2
                step_items.append({
                    "descriptor_cycles": descriptor_cycles,
                    "lane_compute_cycles": lane_compute,
                    "contexts": len(group),
                })
            by_module[name].append({
                "steps": step_items,
                "temporal_steps": len(step_items),
                "fanout": fanout,
                "lane_tiles": lane_tiles,
                "chunks": chunks,
                "sampled_row_ids": [int(key[-1]) for key in group],
            })
    return dict(by_module)


def pattern_full_cost(pattern: dict[str, Any], output_lanes: int) -> int:
    del output_lanes
    return sum(
        item["descriptor_cycles"]
        + pattern["lane_tiles"] * item["lane_compute_cycles"]
        + pattern["lane_tiles"] * item["contexts"]
        for item in pattern["steps"]
    )


def simulate_patterns(
    patterns: list[dict[str, Any]], *, group_count: int, scale: float,
    variant: str, output_lanes: int = 96, atlif_lanes: int = 16,
) -> dict[str, Any]:
    if variant not in {"full_context", "lane_cache", "lane_replay"}:
        raise ValueError(f"unknown schedule: {variant}")
    producer_time = 0.0
    atlif_time = 0.0
    producer_stall = 0.0
    descriptor_raw = 0
    compute_raw = 0
    output_raw = 0
    atlif_tasks = 0

    def retire_lane(pattern: dict[str, Any], lane: int, step: dict[str, int]) -> None:
        nonlocal producer_time, atlif_time, compute_raw, output_raw, atlif_tasks
        producer_time += step["lane_compute_cycles"] * scale
        compute_raw += step["lane_compute_cycles"]
        valid_lanes = min(output_lanes, pattern["fanout"] - lane * output_lanes)
        neuron_groups = math.ceil(valid_lanes / atlif_lanes)
        for _ in range(step["contexts"]):
            producer_time += scale
            output_raw += 1
            atlif_time = max(atlif_time, producer_time) + neuron_groups
            atlif_tasks += neuron_groups

    for group_index in range(group_count):
        pattern = patterns[group_index % len(patterns)]
        if variant == "full_context":
            for step in pattern["steps"]:
                producer_time += step["descriptor_cycles"] * scale
                descriptor_raw += step["descriptor_cycles"]
                for lane in range(pattern["lane_tiles"]):
                    retire_lane(pattern, lane, step)
        else:
            # Only one 96-lane tile owns ATLIF contexts at a time.  Starting a
            # new lane/spatial group waits for t=T-1 retirement of the old one.
            if producer_time < atlif_time:
                producer_stall += atlif_time - producer_time
                producer_time = atlif_time
            if variant == "lane_cache":
                cached = sum(item["descriptor_cycles"] for item in pattern["steps"])
                producer_time += cached * scale
                descriptor_raw += cached
            for lane in range(pattern["lane_tiles"]):
                if producer_time < atlif_time:
                    producer_stall += atlif_time - producer_time
                    producer_time = atlif_time
                for step in pattern["steps"]:
                    if variant == "lane_replay":
                        producer_time += step["descriptor_cycles"] * scale
                        descriptor_raw += step["descriptor_cycles"]
                    retire_lane(pattern, lane, step)
    finish = max(producer_time, atlif_time)
    return {
        "producer_finish_cycles": producer_time,
        "atlif_finish_cycles": atlif_time,
        "fused_finish_cycles": finish,
        "producer_context_stall_cycles": producer_stall,
        "descriptor_raw_cycles": descriptor_raw,
        "compute_raw_cycles": compute_raw,
        "output_raw_cycles": output_raw,
        "atlif_tasks": atlif_tasks,
    }


def load_csv_by_name(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def state_bits(patterns: list[dict[str, Any]], variant: str, contexts: int, output_lanes: int) -> int:
    peak = 0
    for pattern in patterns:
        steps = pattern["temporal_steps"]
        chunks = pattern["chunks"]
        fanout = pattern["fanout"]
        acc = contexts * output_lanes * 32
        one_step_descriptors = contexts * chunks * 2 * 256
        if variant == "full_context":
            atlif = contexts * fanout * steps * 24
            descriptors = one_step_descriptors
        elif variant == "lane_cache":
            atlif = contexts * min(output_lanes, fanout) * steps * 24
            descriptors = one_step_descriptors * steps
        else:
            atlif = contexts * min(output_lanes, fanout) * steps * 24
            descriptors = one_step_descriptors
        peak = max(peak, atlif + acc + descriptors)
    return peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--dependency-audit", type=Path, required=True)
    parser.add_argument("--operator-ledger", type=Path, required=True)
    parser.add_argument("--m7-envelope", type=Path, required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--issue-width", type=int, default=16)
    parser.add_argument("--reduce-slots", type=int, default=4)
    parser.add_argument("--output-lanes", type=int, default=96)
    parser.add_argument("--atlif-lanes", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    wall = load_module("m13_wall", script_dir / "analyze_m4_descriptor_resident_wall_cycles.py")
    validator = wall.load_validator()
    manifest, records, current, previous = validator.validate(args.tile_dir.resolve())
    patterns = build_patterns(
        records, current, previous, wall, contexts=args.contexts,
        issue_width=args.issue_width, reduce_slots=args.reduce_slots,
        output_lanes=args.output_lanes, sample_id=args.sample_id,
    )
    dependency = json.loads(args.dependency_audit.read_text(encoding="utf-8"))
    if dependency.get("status") != "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION":
        raise ValueError(
            "M13 quantitative timing requires a fully admitted dependency audit; "
            f"got {dependency.get('status')}"
        )
    operator_rows = load_csv_by_name(args.operator_ledger)
    m7 = json.loads(args.m7_envelope.read_text(encoding="utf-8"))
    system = m7["system_envelope"]
    hybrid = system["variants"]["hybrid"]
    hybrid_speed = float(hybrid["effective_m4_speedup_vs_local_p1"])
    point = next(item for item in hybrid["stream_points"] if item["stream_lanes"] == args.atlif_lanes)
    no_overlap = int(point["no_overlap_cycles"])
    fixed = int(system["fixed_baseline_cycles"])

    direct_rows = [
        row for row in dependency["rows"]
        if row["category"] == "direct_m4"
        and row["live"] is True
        and row.get("admitted_for_overlap") is True
    ]
    variants: dict[str, dict[str, Any]] = {
        name: {"modules": [], "hidden_cycles": 0.0, "peak_state_bits": 0}
        for name in ("full_context", "lane_cache", "lane_replay")
    }
    for edge in direct_rows:
        producer = edge["producers"][0]
        if producer not in patterns or producer not in operator_rows:
            raise ValueError(f"missing real producer pattern/ledger: {producer}")
        module_patterns = patterns[producer]
        operator = operator_rows[producer]
        temporal_steps = int(edge["temporal_steps"])
        fanout = module_patterns[0]["fanout"]
        output_elements = int(operator["output_elements_per_frame"])
        denominator = temporal_steps * fanout
        if output_elements % denominator:
            raise ValueError(f"non-integral output rows for {producer}")
        rows_per_step = output_elements // denominator
        group_count = math.ceil(rows_per_step / args.contexts)
        raw_full = sum(
            pattern_full_cost(module_patterns[index % len(module_patterns)], args.output_lanes)
            for index in range(group_count)
        )
        target_producer = float(operator["activity_cycles_at_config_lanes"]) / hybrid_speed
        scale = target_producer / raw_full
        atlif_service = int(edge[f"service_cycles_l{args.atlif_lanes}"])
        expected_tasks = math.ceil(output_elements / temporal_steps / args.atlif_lanes) * temporal_steps
        for variant in variants:
            sim = simulate_patterns(
                module_patterns, group_count=group_count, scale=scale, variant=variant,
                output_lanes=args.output_lanes, atlif_lanes=args.atlif_lanes,
            )
            if sim["atlif_tasks"] != expected_tasks:
                raise ValueError(
                    f"ATLIF task identity failed for {producer}: {sim['atlif_tasks']} != {expected_tasks}"
                )
            serial = target_producer + atlif_service
            hidden = serial - sim["fused_finish_cycles"]
            item = {
                "producer": producer,
                "atlif": edge["name"],
                "rows_per_step": rows_per_step,
                "group_count": group_count,
                "sample_patterns": len(module_patterns),
                "target_producer_cycles": target_producer,
                "atlif_service_cycles": atlif_service,
                "raw_full_cycles": raw_full,
                "normalization_scale": scale,
                **sim,
                "hidden_cycles": hidden,
            }
            variants[variant]["modules"].append(item)
            variants[variant]["hidden_cycles"] += hidden
            variants[variant]["peak_state_bits"] = max(
                variants[variant]["peak_state_bits"],
                state_bits(module_patterns, variant, args.contexts, args.output_lanes),
            )

    for variant, item in variants.items():
        hidden = float(item["hidden_cycles"])
        system_cycles = no_overlap - hidden
        item.update({
            "system_cycles": system_cycles,
            "speedup_vs_fixed": fixed / system_cycles,
            "direct_m4_atlif_service_cycles": sum(
                module["atlif_service_cycles"] for module in item["modules"]
            ),
            "claim_boundary": "stratified real-row causal event estimate; joins and non-M4 ATLIF edges remain serial",
        })

    payload = {
        "schema": "m13_temporal_partial_retirement_event_model_v1",
        "status": "PASS_STRATIFIED_CAUSAL_EVENT_ESTIMATE_NOT_FULL_SPATIAL_CYCLE_PROOF",
        "architecture": {
            "contexts": args.contexts, "issue_width": args.issue_width,
            "reduce_slots": args.reduce_slots, "output_lanes": args.output_lanes,
            "atlif_lanes": args.atlif_lanes,
            "full_context_schedule": "spatial-group/timestep/lane-tile with all fanout ATLIF contexts",
            "lane_cache_schedule": "spatial-group/lane-tile/timestep with T descriptors cached",
            "lane_replay_schedule": "spatial-group/lane-tile/timestep with descriptor replay",
        },
        "system_identity": {
            "fixed_baseline_cycles": fixed,
            "m7_no_overlap_cycles": no_overlap,
            "hybrid_m4_speed": hybrid_speed,
        },
        "variants": variants,
        "identities": {
            "tile_manifest_sha256": sha256(args.tile_dir / "manifest.json"),
            "tile_records_sha256": sha256(args.tile_dir / "tile_records.csv"),
            "packed_tiles_sha256": sha256(args.tile_dir / "packed_tiles.npz"),
            "dependency_audit_sha256": sha256(args.dependency_audit),
            "operator_ledger_sha256": sha256(args.operator_ledger),
            "m7_envelope_sha256": sha256(args.m7_envelope),
            "source_sha256": sha256(Path(__file__).resolve()),
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
        },
        "claim_boundary": (
            "Causal producer/ATLIF queue and context-credit timing on stratified real rows; "
            "producer totals are normalized to the admitted M7 hybrid speed. Not a full-spatial "
            "trace, join/non-M4 overlap proof, SRAM contention model, RTL result, or PPA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, item in variants.items():
        print(
            f"{name}: hidden={item['hidden_cycles']:.0f} cycles "
            f"system={item['system_cycles']:.0f} speedup={item['speedup_vs_fixed']:.6f}x "
            f"state={item['peak_state_bits']}b"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
