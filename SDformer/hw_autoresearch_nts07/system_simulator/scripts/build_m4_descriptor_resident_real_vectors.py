#!/usr/bin/env python3
"""Build bounded real Local/Motion descriptor batches for the M4 VCS miter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_tag(key: tuple[str, ...]) -> int:
    digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF) + 1


def packed_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    return f"{int.from_bytes(packed.tobytes(), 'little'):0{len(packed)*2}x}"


def candidates(
    records: list[dict[str, str]], wall: Any, contexts: int, availability_mode: str
) -> list[tuple[tuple[str, ...], list[list[int]]]]:
    rows = wall.ordered_row_bundles(records)
    grouped: OrderedDict[tuple[str, ...], list[list[int]]] = OrderedDict()
    for bundle in rows:
        grouped.setdefault(
            wall.batch_key(records[bundle[0]], availability_mode), []
        ).append(bundle)
    result = []
    for key, bundles in grouped.items():
        for start in range(0, len(bundles), contexts):
            result.append((key, bundles[start : start + contexts]))
    return result


def stratified_select(
    values: list[tuple[tuple[str, ...], list[list[int]]]], limit: int
) -> list[tuple[tuple[str, ...], list[list[int]]]]:
    by_sample: dict[int, list[tuple[tuple[str, ...], list[list[int]]]]] = defaultdict(list)
    for item in values:
        by_sample[int(item[0][0])].append(item)
    samples = sorted(by_sample)
    if limit < len(samples):
        raise ValueError("batch limit must admit every sample")
    base, extra = divmod(limit, len(samples))
    selected = []
    for order, sample_id in enumerate(samples):
        population = by_sample[sample_id]
        count = min(len(population), base + (order < extra))
        indices = np.linspace(0, len(population) - 1, count, dtype=np.int64)
        selected.extend(population[int(index)] for index in indices)
    selected.sort(key=lambda item: (
        int(item[0][0]), item[0][1], int(item[1][0][0])
    ))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True
    )
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument(
        "--availability-mode",
        choices=("temporal_fenced", "layer_materialized_greedy"),
        default="temporal_fenced",
    )
    parser.add_argument("--batches-per-identity-line", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.contexts <= 0:
        raise ValueError("contexts must be positive")
    validator = load_module("build_dual_line_tile_memory_trace.py", "m4_validator")
    wall = load_module("analyze_m4_descriptor_resident_wall_cycles.py", "m4_wall")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "real_descriptors.txt"
    lines: list[str] = []
    manifest: dict[str, Any] = {
        "schema": "m4_descriptor_resident_real_vectors_v1",
        "status": "PASS_CHECKPOINT_BOUND_REAL_BITMAP_DESCRIPTOR_BATCHES",
        "contexts": args.contexts,
        "availability_mode": args.availability_mode,
        "requires_upstream_materialized_activation_rows": (
            args.availability_mode == "layer_materialized_greedy"
        ),
        "requires_spatial_c4_row_buffer": args.availability_mode == "temporal_fenced",
        "batches_per_identity_line": args.batches_per_identity_line,
        "identities": {},
        "population": defaultdict(int),
        "sample_batches": defaultdict(int),
        "object_tags": {},
    }
    tag_to_key: dict[int, tuple[str, ...]] = {}
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        source_manifest, records, current, previous = validator.validate(directory)
        current_bits = np.unpackbits(current, axis=1, bitorder="little").astype(bool)
        previous_bits = np.unpackbits(previous, axis=1, bitorder="little").astype(bool)
        manifest["identities"][label] = {
            "directory": str(directory),
            "records": len(records),
            "checkpoint_sha256": source_manifest["run_context"]["artifact_identity"][
                "checkpoint_sha256"
            ],
            "source_sha256": source_manifest["sha256"],
        }
        all_candidates = candidates(records, wall, args.contexts, args.availability_mode)
        for line_name in ("local", "hybrid"):
            chosen = stratified_select(all_candidates, args.batches_per_identity_line)
            for key, batch in chosen:
                first_row = records[batch[0][0]]
                physical_key = (
                    first_row["name"], first_row["operator"],
                    first_row["weight_group"], first_row["source_width"],
                    first_row["chunks_per_row"], first_row["output_channel_fanout"],
                )
                tag = object_tag(physical_key)
                if tag in tag_to_key and tag_to_key[tag] != physical_key:
                    raise ValueError("object-tag hash collision")
                tag_to_key[tag] = physical_key
                manifest["object_tags"][f"{tag:08x}"] = list(physical_key)
                batch_lane_tiles = int(records[batch[0][0]]["output_lane_tile_count_96"])
                manifest["population"]["batches"] += 1
                manifest["population"]["outputs"] += len(batch) * batch_lane_tiles
                manifest["sample_batches"][f"{label}:{line_name}:s{key[0]}"] += 1
                batch_selected: list[list[np.ndarray]] = []
                for context, bundle in enumerate(batch):
                    chunks = len(bundle)
                    context_selected: list[np.ndarray] = []
                    for chunk, index in enumerate(bundle):
                        row = records[index]
                        use_motion = (
                            line_name == "hybrid" and row["row_use_motion"].lower() == "true"
                        )
                        if use_motion:
                            selected = current_bits[index] ^ previous_bits[index]
                            negative = np.logical_and(previous_bits[index], ~current_bits[index])
                        else:
                            selected = current_bits[index]
                            negative = np.zeros_like(selected)
                        if np.any(np.logical_and(negative, ~selected)):
                            raise ValueError("negative bitmap escaped selected bitmap")
                        context_selected.append(selected)
                        is_last = chunk == chunks - 1
                        is_batch_last = context == len(batch) - 1 and is_last
                        lines.append(
                            f"{tag:08x} {context + 1:08x} {chunk} {chunks} "
                            f"{batch_lane_tiles} {int(use_motion)} "
                            f"{packed_hex(selected)} {packed_hex(negative)} "
                            f"{int(chunk == 0)} {int(is_last)} {int(is_batch_last)}\n"
                        )
                        manifest["population"]["descriptors"] += 1
                        manifest["population"]["selected_sources"] += int(selected.sum())
                        manifest["population"]["negative_sources"] += int(negative.sum())
                        manifest["population"][
                            "motion_descriptors" if use_motion else "local_descriptors"
                        ] += 1
                    batch_selected.append(context_selected)
                chunks = len(batch_selected[0])
                for _lane_tile in range(batch_lane_tiles):
                    for chunk in range(chunks):
                        counts = np.asarray(
                            [
                                [
                                    int(context_bits[chunk][bank::16].sum())
                                    for bank in range(16)
                                ]
                                for context_bits in batch_selected
                            ],
                            dtype=np.int64,
                        )
                        manifest["population"]["compact_issue_cycles"] += (
                            wall.compact_issue_cycles(counts, reduce_slots=4)
                        )
                        manifest["population"]["chunk_control_cycles"] += 2
                        manifest["population"]["lane_expanded_selected_sources"] += int(
                            counts.sum()
                        )
                        manifest["population"]["same_width_dense_issue_cycles"] += sum(
                            math.ceil(int(records[bundle[chunk]]["valid_bits"]) / 16)
                            for bundle in batch
                        )

    output.write_text("".join(lines), encoding="utf-8")
    manifest["population"] = dict(manifest["population"])
    population = manifest["population"]
    population["m4_wall_cycles"] = (
        population["descriptors"]
        + population["compact_issue_cycles"]
        + population["chunk_control_cycles"]
        + population["outputs"]
    )
    population["p1_sparse_wall_cycles"] = (
        population["descriptors"]
        + population["lane_expanded_selected_sources"]
        + population["chunk_control_cycles"]
        + population["outputs"]
    )
    population["same_width_dense_wall_cycles"] = (
        population["descriptors"]
        + population["same_width_dense_issue_cycles"]
        + population["chunk_control_cycles"]
        + population["outputs"]
    )
    population["speedup_vs_p1_sparse_wall"] = (
        population["p1_sparse_wall_cycles"] / population["m4_wall_cycles"]
    )
    population["speedup_vs_same_width_dense_wall"] = (
        population["same_width_dense_wall_cycles"] / population["m4_wall_cycles"]
    )
    manifest["sample_batches"] = dict(sorted(manifest["sample_batches"].items()))
    manifest["sha256"] = {
        "real_descriptors.txt": sha256(output),
        "generator": sha256(Path(__file__)),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "PASS: wrote "
        f"{manifest['population']['batches']} batches, "
        f"{manifest['population']['descriptors']} descriptors, "
        f"{manifest['population']['outputs']} outputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
