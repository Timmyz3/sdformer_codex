#!/usr/bin/env python3
"""Cycle-exact M4 descriptor-resident Local/Motion kernel model.

The model mirrors the M4 single-buffer controller: a batch contains up to C
rows with the same immutable weight geometry, every 256-bit chunk descriptor
is loaded once, and the batch then walks output-lane tiles and source chunks.
Each source chunk has one PREP cycle, bank-conflict issue cycles, and one DRAIN
cycle.  Completed 96-lane accumulators leave at one context per cycle.

Unlike the M3 optimistic bound, all command, controller, and output cycles are
charged.  The compact reducer admits at most ``reduce_slots`` banks for one
context per cycle, which is the rule implemented by the M4 RTL and prevents a
CONTEXTS x ISSUE_WIDTH replication of the SIMD adder tree.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_validator() -> Any:
    path = Path(__file__).with_name("build_dual_line_tile_memory_trace.py")
    spec = importlib.util.spec_from_file_location("dual_line_tile_memory_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tile validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_bits(
    records: list[dict[str, str]], current: np.ndarray, previous: np.ndarray, line: str
) -> np.ndarray:
    current_bits = np.unpackbits(current, axis=1, bitorder="little").astype(bool)
    if line == "local":
        return current_bits
    previous_bits = np.unpackbits(previous, axis=1, bitorder="little").astype(bool)
    use_motion = np.asarray(
        [row["row_use_motion"].lower() == "true" for row in records], dtype=bool
    )[:, None]
    return np.where(use_motion, current_bits ^ previous_bits, current_bits)


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"],
        row["sequence_key"],
        row["name"],
        row["operator_call_index"],
        row["row_id"],
        row["temporal_step"],
    )


def batch_key(
    row: dict[str, str], availability_mode: str = "layer_materialized_greedy"
) -> tuple[str, ...]:
    # sample_id is an explicit scheduling fence.  The remaining fields are
    # exactly the geometry that the descriptor-resident batch holds constant.
    key = (
        row["sample_id"],
        row["sequence_key"],
        row["name"],
        row["operator"],
        row["weight_group"],
        row["source_width"],
        row["chunks_per_row"],
        row["output_channel_fanout"],
    )
    if availability_mode == "layer_materialized_greedy":
        return key
    if availability_mode == "temporal_fenced":
        # Do not borrow contexts from a future timestep or another dynamic
        # invocation of the same named operator.  Rows within one invocation
        # may still fill C contexts after that invocation has materialized.
        return key + (row["operator_call_index"], row["temporal_step"])
    raise ValueError(f"unsupported availability mode: {availability_mode}")


def ordered_row_bundles(records: list[dict[str, str]]) -> list[list[int]]:
    grouped: OrderedDict[tuple[str, ...], list[int]] = OrderedDict()
    for index, row in enumerate(records):
        grouped.setdefault(row_key(row), []).append(index)
    bundles: list[list[int]] = []
    for indices in grouped.values():
        indices.sort(key=lambda index: int(records[index]["chunk_index"]))
        chunks = int(records[indices[0]]["chunks_per_row"])
        if len(indices) != chunks:
            raise ValueError("M4 requires a complete chunk bundle for every row")
        if [int(records[index]["chunk_index"]) for index in indices] != list(range(chunks)):
            raise ValueError("M4 chunk indices are not dense and ordered")
        bundles.append(indices)
    return bundles


def compact_issue_cycles(bank_counts: np.ndarray, reduce_slots: int) -> int:
    """Run the RTL's deterministic bank-first/context-first compact scheduler."""
    if bank_counts.ndim != 2 or reduce_slots <= 0:
        raise ValueError("bank_counts must be 2-D and reduce_slots must be positive")
    remaining = bank_counts.astype(np.int64, copy=True)
    cycles = 0
    while np.any(remaining):
        used = np.zeros(remaining.shape[0], dtype=np.int64)
        issued = 0
        for bank in range(remaining.shape[1]):
            for context in range(remaining.shape[0]):
                if remaining[context, bank] > 0 and used[context] < reduce_slots:
                    remaining[context, bank] -= 1
                    used[context] += 1
                    issued += 1
                    break
        if issued == 0:
            raise RuntimeError("compact scheduler made no progress")
        cycles += 1
    return cycles


def analyze_identity(
    records: list[dict[str, str]],
    current: np.ndarray,
    previous: np.ndarray,
    *,
    line: str,
    issue_width: int,
    contexts: int,
    reduce_slots: int,
    output_lanes: int = 96,
    availability_mode: str = "temporal_fenced",
) -> dict[str, Any]:
    if issue_width <= 0 or issue_width & (issue_width - 1):
        raise ValueError("issue_width must be a positive power of two")
    if contexts <= 0 or reduce_slots <= 0 or reduce_slots > issue_width or output_lanes <= 0:
        raise ValueError("invalid context/reducer geometry")
    bits = selected_bits(records, current, previous, line)
    bundles = ordered_row_bundles(records)
    scheduling_groups: OrderedDict[tuple[str, ...], list[list[int]]] = OrderedDict()
    for bundle in bundles:
        scheduling_groups.setdefault(
            batch_key(records[bundle[0]], availability_mode), []
        ).append(bundle)

    totals: defaultdict[str, int] = defaultdict(int)
    samples: dict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    batches = 0
    cross_temporal_batches = 0
    cross_spatial_row_batches = 0
    cross_operator_call_batches = 0
    cross_sequence_batches = 0
    partial_context_batches = 0
    for key, group_bundles in scheduling_groups.items():
        for start in range(0, len(group_bundles), contexts):
            batch = group_bundles[start : start + contexts]
            chunks = len(batch[0])
            fanout = int(records[batch[0][0]]["output_channel_fanout"])
            lane_tiles = math.ceil(fanout / output_lanes)
            if any(len(bundle) != chunks for bundle in batch):
                raise ValueError("chunk geometry changed inside an M4 batch")
            if any(
                int(records[bundle[0]]["output_channel_fanout"]) != fanout
                for bundle in batch
            ):
                raise ValueError("lane-tile geometry changed inside an M4 batch")
            sample_id = int(key[0])
            batches += 1
            batch_rows = [records[bundle[0]] for bundle in batch]
            if len({row["temporal_step"] for row in batch_rows}) > 1:
                cross_temporal_batches += 1
            if len({row["row_id"] for row in batch_rows}) > 1:
                cross_spatial_row_batches += 1
            if len({row["operator_call_index"] for row in batch_rows}) > 1:
                cross_operator_call_batches += 1
            if len({row["sequence_key"] for row in batch_rows}) > 1:
                cross_sequence_batches += 1
            if len(batch) < contexts:
                partial_context_batches += 1
            descriptor_cycles = sum(len(bundle) for bundle in batch)
            totals["descriptor_load_cycles"] += descriptor_cycles
            samples[sample_id]["descriptor_load_cycles"] += descriptor_cycles
            # Every output-lane tile replays exactly the same source descriptor.
            # Evaluate the deterministic compact schedule once per chunk and
            # scale its cycle/work counts by the number of lane tiles.
            for chunk in range(chunks):
                counts = np.zeros((len(batch), issue_width), dtype=np.int64)
                valid_dense_cycles = 0
                for context, bundle in enumerate(batch):
                    index = bundle[chunk]
                    counts[context] = [
                        int(bits[index, bank::issue_width].sum())
                        for bank in range(issue_width)
                    ]
                    valid_dense_cycles += math.ceil(
                        int(records[index]["valid_bits"]) / issue_width
                    )
                selected_sources = int(counts.sum())
                issue_cycles = compact_issue_cycles(counts, reduce_slots)
                totals["selected_sources"] += selected_sources * lane_tiles
                totals["same_width_dense_issue_cycles"] += valid_dense_cycles * lane_tiles
                totals["compact_issue_cycles"] += issue_cycles * lane_tiles
                totals["chunk_control_cycles"] += 2 * lane_tiles
                samples[sample_id]["selected_sources"] += selected_sources * lane_tiles
                samples[sample_id]["same_width_dense_issue_cycles"] += (
                    valid_dense_cycles * lane_tiles
                )
                samples[sample_id]["compact_issue_cycles"] += issue_cycles * lane_tiles
                samples[sample_id]["chunk_control_cycles"] += 2 * lane_tiles
            totals["output_cycles"] += len(batch) * lane_tiles
            samples[sample_id]["output_cycles"] += len(batch) * lane_tiles

    def cycles(item: dict[str, int], issue_field: str) -> int:
        return (
            item["descriptor_load_cycles"]
            + item[issue_field]
            + item["chunk_control_cycles"]
            + item["output_cycles"]
        )

    per_sample: dict[str, Any] = {}
    for sample_id in sorted(samples):
        item = samples[sample_id]
        m4_cycles = cycles(item, "compact_issue_cycles")
        p1_cycles = cycles(item, "selected_sources")
        dense_cycles = cycles(item, "same_width_dense_issue_cycles")
        per_sample[str(sample_id)] = {
            **dict(item),
            "m4_wall_cycles": m4_cycles,
            "p1_sparse_wall_cycles": p1_cycles,
            "same_width_dense_wall_cycles": dense_cycles,
            "speedup_vs_p1_sparse_wall": p1_cycles / m4_cycles if m4_cycles else 1.0,
            "speedup_vs_same_width_dense_wall": (
                dense_cycles / m4_cycles if m4_cycles else 1.0
            ),
        }

    m4_cycles = cycles(totals, "compact_issue_cycles")
    p1_cycles = cycles(totals, "selected_sources")
    dense_cycles = cycles(totals, "same_width_dense_issue_cycles")
    p1_samples = [item["speedup_vs_p1_sparse_wall"] for item in per_sample.values()]
    dense_samples = [
        item["speedup_vs_same_width_dense_wall"] for item in per_sample.values()
    ]
    return {
        "records": len(records),
        "row_bundles": len(bundles),
        "batches": batches,
        "samples": len(per_sample),
        "availability_mode": availability_mode,
        "cross_temporal_batches": cross_temporal_batches,
        "cross_spatial_row_batches": cross_spatial_row_batches,
        "cross_operator_call_batches": cross_operator_call_batches,
        "cross_sequence_batches": cross_sequence_batches,
        "partial_context_batches": partial_context_batches,
        "resident_context_utilization": len(bundles) / (batches * contexts),
        "cross_sample_contexts": False,
        **dict(totals),
        "m4_wall_cycles": m4_cycles,
        "p1_sparse_wall_cycles": p1_cycles,
        "same_width_dense_wall_cycles": dense_cycles,
        "speedup_vs_p1_sparse_wall": p1_cycles / m4_cycles if m4_cycles else 1.0,
        "speedup_vs_same_width_dense_wall": dense_cycles / m4_cycles if m4_cycles else 1.0,
        "p1_sparse_sample_speedup_min": min(p1_samples, default=1.0),
        "p1_sparse_sample_speedup_median": float(np.median(p1_samples)) if p1_samples else 1.0,
        "p1_sparse_sample_speedup_max": max(p1_samples, default=1.0),
        "same_width_dense_sample_speedup_min": min(dense_samples, default=1.0),
        "same_width_dense_sample_speedup_median": (
            float(np.median(dense_samples)) if dense_samples else 1.0
        ),
        "same_width_dense_sample_speedup_max": max(dense_samples, default=1.0),
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True
    )
    parser.add_argument("--issue-width", type=int, default=16)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--reduce-slots", type=int, default=4)
    parser.add_argument("--output-lanes", type=int, default=96)
    parser.add_argument(
        "--availability-mode",
        choices=("temporal_fenced", "layer_materialized_greedy"),
        default="temporal_fenced",
        help=(
            "temporal_fenced forbids context fill across dynamic calls/timesteps; "
            "layer_materialized_greedy is a non-causal legacy-order sensitivity"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validator = load_validator()

    identities: dict[str, Any] = {}
    loaded = []
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        manifest, records, current, previous = validator.validate(directory)
        sample_ids = sorted({int(row["sample_id"]) for row in records})
        if len(sample_ids) < 2:
            raise ValueError("M4 requires a cross-sample trace")
        loaded.append((label, directory, manifest, records, current, previous))
        identities[label] = {
            "directory": str(directory),
            "records": len(records),
            "sample_ids": sample_ids,
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"][
                "checkpoint_sha256"
            ],
            "trace_source_sha256": manifest["run_context"].get("source_sha256", {}),
        }

    variants: dict[str, Any] = {}
    for line in ("local", "hybrid"):
        per_identity = {
            label: analyze_identity(
                records,
                current,
                previous,
                line=line,
                issue_width=args.issue_width,
                contexts=args.contexts,
                reduce_slots=args.reduce_slots,
                output_lanes=args.output_lanes,
                availability_mode=args.availability_mode,
            )
            for label, _directory, _manifest, records, current, previous in loaded
        }
        m4_cycles = sum(item["m4_wall_cycles"] for item in per_identity.values())
        p1_cycles = sum(item["p1_sparse_wall_cycles"] for item in per_identity.values())
        dense_cycles = sum(
            item["same_width_dense_wall_cycles"] for item in per_identity.values()
        )
        p1_samples = [
            sample["speedup_vs_p1_sparse_wall"]
            for item in per_identity.values()
            for sample in item["per_sample"].values()
        ]
        dense_samples = [
            sample["speedup_vs_same_width_dense_wall"]
            for item in per_identity.values()
            for sample in item["per_sample"].values()
        ]
        variants[line] = {
            "m4_wall_cycles": m4_cycles,
            "p1_sparse_wall_cycles": p1_cycles,
            "same_width_dense_wall_cycles": dense_cycles,
            "speedup_vs_p1_sparse_wall": p1_cycles / m4_cycles,
            "speedup_vs_same_width_dense_wall": dense_cycles / m4_cycles,
            "p1_sparse_sample_speedup_min": min(p1_samples),
            "p1_sparse_sample_speedup_median": float(np.median(p1_samples)),
            "same_width_dense_sample_speedup_min": min(dense_samples),
            "same_width_dense_sample_speedup_median": float(np.median(dense_samples)),
            "per_identity": per_identity,
        }

    payload = {
        "schema": "m4_descriptor_resident_wall_cycles_v1",
        "status": "PASS_M4_EXECUTABLE_SINGLE_BUFFER_WALL_CYCLE_MODEL",
        "claim_boundary": (
            "cycle-exact single-buffer Local/Motion source-kernel controller on admitted "
            "real bitmaps; includes descriptor load, PREP/DRAIN, compact bank issue, and "
            "96-lane output cycles; temporal_fenced forbids batching across dynamic "
            "operator calls/timesteps while layer_materialized_greedy assumes an upstream "
            "activation store and preserves legacy greedy order (it is not an optimized "
            "upper bound); excludes "
            "SRAM/DRAM contention, unrelated network "
            "operators, clock-domain crossings, and full-system overlap"
        ),
        "architecture": {
            "issue_width": args.issue_width,
            "contexts": args.contexts,
            "reduce_slots_per_context": args.reduce_slots,
            "output_lanes": args.output_lanes,
            "availability_mode": args.availability_mode,
            "max_chunks": 12,
            "max_lane_tiles": 32,
            "descriptor_state_bits": args.contexts * 12 * 2 * 256,
            "max_descriptor_buffer_bytes": args.contexts * 12 * 2 * 256 // 8,
            "descriptor_input_width_bits": 2 * 256,
            "weight_response_width_bits": args.issue_width * args.output_lanes * 8,
            "accumulator_output_width_bits": args.output_lanes * 32,
            "accumulator_state_bits": args.contexts * args.output_lanes * 32,
            "shared_reducer_signed_adders": (
                args.contexts * args.reduce_slots * args.output_lanes
            ),
            "m3_naive_reducer_signed_adders": (
                args.contexts * args.issue_width * args.output_lanes
            ),
        },
        "identities": identities,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# M4 descriptor-resident full-wall-cycle DSE\n\n",
        "| line | identity | M4 cycles | vs P1 sparse wall | vs dense same-P wall | P1 sample min | dense sample min |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for line, line_item in variants.items():
        for label, item in line_item["per_identity"].items():
            report.append(
                f"| {line} | {label} | {item['m4_wall_cycles']:,} | "
                f"{item['speedup_vs_p1_sparse_wall']:.6f}x | "
                f"{item['speedup_vs_same_width_dense_wall']:.6f}x | "
                f"{item['p1_sparse_sample_speedup_min']:.6f}x | "
                f"{item['same_width_dense_sample_speedup_min']:.6f}x |\n"
            )
    report.append(
        "\nThese are source-kernel wall cycles under an always-ready external weight "
        "interface, not full-network FPS or end-to-end acceleration.\n"
    )
    args.output.with_suffix(".md").write_text("".join(report), encoding="utf-8")
    print(f"PASS: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
