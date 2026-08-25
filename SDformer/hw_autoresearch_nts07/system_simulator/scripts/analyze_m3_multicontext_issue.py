#!/usr/bin/env python3
"""DSE bank-aware multi-context issue on admitted Local/Motion tile traces.

This is an issue-only lower-bound model.  A context owns one activation tile
and one resident Acc32 output vector.  Banks may serve different contexts in
the same cycle, but a bank still performs at most one weight-column read.  The
model never combines different weight objects (operator/group/source chunk/
output-lane tile), so every issued local bank address remains unambiguous.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ADDRESS_FIELDS = (
    "name",
    "operator",
    "weight_group",
    "source_base",
    "source_width",
    "chunk_index",
    "output_lane_tile_count_96",
    "valid_bits",
    "chunks_per_row",
)


def load_validator() -> Any:
    path = Path(__file__).with_name("build_dual_line_tile_memory_trace.py")
    spec = importlib.util.spec_from_file_location("dual_line_tile_memory_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tile validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule_object_bank_counts(bank_counts: np.ndarray, contexts: int) -> int:
    """Return issue cycles for ordered batches of contexts sharing one weight object."""
    if bank_counts.ndim != 2 or contexts <= 0:
        raise ValueError("bank_counts must be 2-D and contexts must be positive")
    cycles = 0
    for start in range(0, len(bank_counts), contexts):
        batch = bank_counts[start : start + contexts]
        if len(batch):
            cycles += int(batch.sum(axis=0).max(initial=0))
    return cycles


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


def weight_object(row: dict[str, str], lane_tile: int) -> tuple[str, ...]:
    return (
        row["name"],
        row["operator"],
        row["weight_group"],
        row["source_base"],
        row["source_width"],
        row["chunk_index"],
        str(lane_tile),
    )


def analyze_identity(
    records: list[dict[str, str]],
    current: np.ndarray,
    previous: np.ndarray,
    *,
    line: str,
    issue_width: int,
    contexts: int,
) -> dict[str, Any]:
    if issue_width <= 0 or issue_width & (issue_width - 1):
        raise ValueError("issue_width must be a power of two")
    for field in REQUIRED_ADDRESS_FIELDS:
        if records and field not in records[0]:
            raise ValueError(f"trace lacks required weight-address field: {field}")
    bits = selected_bits(records, current, previous, line)
    # The sample id is a scheduling fence, not part of the physical weight
    # address.  This prevents a batch from borrowing concurrency across two
    # independent evaluation samples while still reporting the true number of
    # resident weight objects.
    schedule_counts: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)
    physical_objects: set[tuple[str, ...]] = set()
    sample_sources: dict[int, int] = defaultdict(int)
    sample_transactions: dict[int, int] = defaultdict(int)
    sample_records: dict[int, int] = defaultdict(int)
    sample_dense_cycles: dict[int, int] = defaultdict(int)
    sources = 0
    transactions = 0
    dense_cycles = 0
    for index, row in enumerate(records):
        bank_counts = np.asarray(
            [bits[index, bank::issue_width].sum() for bank in range(issue_width)],
            dtype=np.int64,
        )
        lane_tiles = int(row["output_lane_tile_count_96"])
        if lane_tiles <= 0:
            raise ValueError("output_lane_tile_count_96 must be positive")
        sample_id = int(row["sample_id"])
        sample_records[sample_id] += 1
        valid_bits = int(row["valid_bits"])
        if valid_bits <= 0 or valid_bits > bits.shape[1]:
            raise ValueError(f"invalid valid_bits={valid_bits}")
        row_dense_cycles = math.ceil(valid_bits / issue_width) * lane_tiles
        dense_cycles += row_dense_cycles
        sample_dense_cycles[sample_id] += row_dense_cycles
        for lane_tile in range(lane_tiles):
            physical_object = weight_object(row, lane_tile)
            selected_count = int(bank_counts.sum())
            physical_objects.add(physical_object)
            schedule_counts[(str(sample_id), *physical_object)].append(bank_counts)
            sources += selected_count
            transactions += 1
            sample_sources[sample_id] += selected_count
            sample_transactions[sample_id] += 1
    cycles = sum(
        schedule_object_bank_counts(np.stack(values), contexts)
        for values in schedule_counts.values()
        if values
    )
    sample_cycles: dict[int, int] = defaultdict(int)
    for key, values in schedule_counts.items():
        sample_cycles[int(key[0])] += schedule_object_bank_counts(np.stack(values), contexts)
    per_sample = {
        str(sample_id): {
            "selected_sources": sample_sources[sample_id],
            "lane_expanded_transactions": sample_transactions[sample_id],
            "issue_cycles": sample_cycles[sample_id],
            "same_width_dense_issue_cycles": sample_dense_cycles[sample_id],
            "speedup_vs_same_width_dense_issue": (
                sample_dense_cycles[sample_id] / sample_cycles[sample_id]
                if sample_cycles[sample_id]
                else 1.0
            ),
            "speedup_vs_p1_source_cycles": (
                sample_sources[sample_id] / sample_cycles[sample_id]
                if sample_cycles[sample_id]
                else 1.0
            ),
            "bank_utilization": (
                sample_sources[sample_id] / (issue_width * sample_cycles[sample_id])
                if sample_cycles[sample_id]
                else 1.0
            ),
            "speedup_vs_p1_serialized_service_lower_bound": (
                (sample_sources[sample_id] + sample_transactions[sample_id])
                / (sample_cycles[sample_id] + sample_transactions[sample_id])
                if sample_cycles[sample_id] + sample_transactions[sample_id]
                else 1.0
            ),
            "descriptor_load_cycles_if_reused_across_output_lanes": (
                sample_records[sample_id]
            ),
            "speedup_vs_p1_descriptor_residency_optimistic_bound": (
                (sample_sources[sample_id] + sample_transactions[sample_id])
                / (sample_cycles[sample_id] + sample_records[sample_id])
                if sample_cycles[sample_id] + sample_records[sample_id]
                else 1.0
            ),
        }
        for sample_id in sorted(sample_sources)
    }
    sample_speedups = [item["speedup_vs_p1_source_cycles"] for item in per_sample.values()]
    serialized_service_cycles = cycles + transactions
    p1_serialized_service_cycles = sources + transactions
    descriptor_reuse_cycles = cycles + len(records)
    max_chunks_per_row = max((int(row["chunks_per_row"]) for row in records), default=0)
    return {
        "records": len(records),
        "max_chunks_per_row": max_chunks_per_row,
        "samples": len({int(row["sample_id"]) for row in records}),
        "cross_sample_contexts": False,
        "weight_objects": len(physical_objects),
        "sample_weight_scheduling_groups": len(schedule_counts),
        "lane_expanded_transactions": transactions,
        "selected_sources": sources,
        "issue_cycles": cycles,
        "same_width_dense_issue_cycles": dense_cycles,
        "speedup_vs_same_width_dense_issue": dense_cycles / cycles if cycles else 1.0,
        "speedup_vs_p1_source_cycles": sources / cycles if cycles else 1.0,
        "serialized_command_cycles": transactions,
        "serialized_service_lower_bound_cycles": serialized_service_cycles,
        "speedup_vs_p1_serialized_service_lower_bound": (
            p1_serialized_service_cycles / serialized_service_cycles
            if serialized_service_cycles
            else 1.0
        ),
        "descriptor_load_cycles_if_reused_across_output_lanes": len(records),
        "descriptor_residency_optimistic_cycles": descriptor_reuse_cycles,
        "speedup_vs_p1_descriptor_residency_optimistic_bound": (
            p1_serialized_service_cycles / descriptor_reuse_cycles
            if descriptor_reuse_cycles
            else 1.0
        ),
        "bank_utilization": sources / (issue_width * cycles) if cycles else 1.0,
        "sample_speedup_min": min(sample_speedups, default=1.0),
        "sample_speedup_median": float(np.median(sample_speedups)) if sample_speedups else 1.0,
        "sample_speedup_max": max(sample_speedups, default=1.0),
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True
    )
    parser.add_argument(
        "--issue-width", action="append", type=int, choices=(4, 8, 16), required=True
    )
    parser.add_argument(
        "--contexts", action="append", type=int, choices=(1, 2, 4, 8, 16), required=True
    )
    parser.add_argument("--eligible-target", type=float, default=7.687553)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validator = load_validator()

    loaded: list[tuple[str, Path, dict[str, Any], list[dict[str, str]], np.ndarray, np.ndarray]] = []
    identities: dict[str, Any] = {}
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        manifest, records, current, previous = validator.validate(directory)
        sample_ids = sorted({int(row["sample_id"]) for row in records})
        if len(sample_ids) < 2:
            raise ValueError(f"M3 DSE requires cross-sample tiles, got {sample_ids}: {directory}")
        loaded.append((label, directory, manifest, records, current, previous))
        identities[label] = {
            "directory": str(directory),
            "records": len(records),
            "sample_ids": sample_ids,
            "checkpoint_sha256": manifest.get("run_context", {})
            .get("artifact_identity", {})
            .get("checkpoint_sha256"),
            "trace_source_sha256": manifest.get("run_context", {}).get("source_sha256", {}),
        }

    variants: dict[str, Any] = {}
    for line in ("local", "hybrid"):
        line_variants: dict[str, Any] = {}
        for issue_width in sorted(set(args.issue_width)):
            for contexts in sorted(set(args.contexts)):
                per_identity = {
                    label: analyze_identity(
                        records,
                        current,
                        previous,
                        line=line,
                        issue_width=issue_width,
                        contexts=contexts,
                    )
                    for label, _directory, _manifest, records, current, previous in loaded
                }
                sources = sum(item["selected_sources"] for item in per_identity.values())
                cycles = sum(item["issue_cycles"] for item in per_identity.values())
                transactions = sum(
                    item["lane_expanded_transactions"] for item in per_identity.values()
                )
                descriptor_loads = sum(item["records"] for item in per_identity.values())
                max_chunks_per_row = max(
                    (item["max_chunks_per_row"] for item in per_identity.values()),
                    default=0,
                )
                dense_cycles = sum(
                    item["same_width_dense_issue_cycles"] for item in per_identity.values()
                )
                speedup = sources / cycles if cycles else 1.0
                serialized_speedup = (
                    (sources + transactions) / (cycles + transactions)
                    if cycles + transactions
                    else 1.0
                )
                descriptor_residency_bound = (
                    (sources + transactions) / (cycles + descriptor_loads)
                    if cycles + descriptor_loads
                    else 1.0
                )
                serialized_sample_speedups = [
                    sample["speedup_vs_p1_serialized_service_lower_bound"]
                    for identity in per_identity.values()
                    for sample in identity["per_sample"].values()
                ]
                descriptor_sample_speedups = [
                    sample["speedup_vs_p1_descriptor_residency_optimistic_bound"]
                    for identity in per_identity.values()
                    for sample in identity["per_sample"].values()
                ]
                key = f"p{issue_width}_c{contexts}"
                line_variants[key] = {
                    "issue_width": issue_width,
                    "contexts": contexts,
                    "selected_sources": sources,
                    "issue_cycles": cycles,
                    "same_width_dense_issue_cycles": dense_cycles,
                    "speedup_vs_same_width_dense_issue": (
                        dense_cycles / cycles if cycles else 1.0
                    ),
                    "speedup_vs_same_width_dense_serialized_service_lower_bound": (
                        (dense_cycles + transactions) / (cycles + transactions)
                        if cycles + transactions
                        else 1.0
                    ),
                    "speedup_vs_p1_source_cycles": speedup,
                    "serialized_command_cycles": transactions,
                    "serialized_service_lower_bound_cycles": cycles + transactions,
                    "speedup_vs_p1_serialized_service_lower_bound": serialized_speedup,
                    "serialized_service_sample_speedup_min": min(
                        serialized_sample_speedups, default=1.0
                    ),
                    "serialized_service_sample_speedup_median": float(
                        np.median(serialized_sample_speedups)
                    ) if serialized_sample_speedups else 1.0,
                    "serialized_service_sample_speedup_max": max(
                        serialized_sample_speedups, default=1.0
                    ),
                    "descriptor_load_cycles_if_reused_across_output_lanes": descriptor_loads,
                    "descriptor_residency_optimistic_cycles": cycles + descriptor_loads,
                    "speedup_vs_p1_descriptor_residency_optimistic_bound": (
                        descriptor_residency_bound
                    ),
                    "descriptor_residency_sample_speedup_min": min(
                        descriptor_sample_speedups, default=1.0
                    ),
                    "descriptor_residency_sample_speedup_median": float(
                        np.median(descriptor_sample_speedups)
                    ) if descriptor_sample_speedups else 1.0,
                    "descriptor_residency_sample_speedup_max": max(
                        descriptor_sample_speedups, default=1.0
                    ),
                    "bank_utilization": sources / (issue_width * cycles) if cycles else 1.0,
                    "context_state_bits_lower_bound": contexts * (2 * 256 + 96 * 32),
                    "descriptor_resident_state_bits_lower_bound": contexts * (
                        max_chunks_per_row * 2 * 256 + 96 * 32
                    ),
                    "descriptor_resident_peak_chunks_per_context": max_chunks_per_row,
                    "meets_eligible_engine_target": speedup >= args.eligible_target,
                    "meets_target_with_current_serialized_command_interface": (
                        serialized_speedup >= args.eligible_target
                    ),
                    "meets_target_with_descriptor_residency_optimistic_bound": (
                        descriptor_residency_bound >= args.eligible_target
                    ),
                    "per_identity": per_identity,
                }
        variants[line] = line_variants

    p16c4_local = variants["local"].get("p16_c4")
    p16c4_hybrid = variants["hybrid"].get("p16_c4")
    payload = {
        "schema": "m3_bank_aware_multicontext_issue_dse_v2",
        "status": "PASS_M3_COMMAND_BOTTLENECK_EXPOSED_M4_DESCRIPTOR_GATE",
        "claim_boundary": (
            "exact lane-expanded source/bank issue lower bound on admitted cross-sample tiles; "
            "one weight read per bank/cycle; contexts never cross operator/group/source-chunk/"
            "output-lane weight identity or sample boundary; excludes scheduler/response latency, SRAM timing/power, "
            "other operators, and full-system overlap; the serialized-service metric adds one "
            "non-overlapped command cycle per lane-expanded transaction, while the descriptor-"
            "residency metric is an optimistic bound that requires reusing each activation "
            "descriptor across output-lane objects and overlapping the object stream"
        ),
        "eligible_engine_target": args.eligible_target,
        "m3_candidate_gate": {
            "candidate": "p16_c4" if p16c4_local and p16c4_hybrid else None,
            "current_serialized_interface_meets_target": bool(
                p16c4_local
                and p16c4_hybrid
                and p16c4_local[
                    "meets_target_with_current_serialized_command_interface"
                ]
                and p16c4_hybrid[
                    "meets_target_with_current_serialized_command_interface"
                ]
            ),
            "descriptor_residency_all_sample_min_meets_target": bool(
                p16c4_local
                and p16c4_hybrid
                and p16c4_local["descriptor_residency_sample_speedup_min"]
                >= args.eligible_target
                and p16c4_hybrid["descriptor_residency_sample_speedup_min"]
                >= args.eligible_target
            ),
            "decision": (
                "M3 P16C4 is a correctness/synthesis milestone, not the final engine; "
                "M4 must implement cross-output-lane activation descriptor residency "
                "and overlap the weight-object stream"
            ),
        },
        "identities": identities,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# M3 bank-aware multi-context issue DSE\n\n",
        "| line | P | contexts | vs active-serial | vs dense same-P | serialized current interface | descriptor-resident optimistic | bank utilization | target by current interface |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for line, line_variants in variants.items():
        for item in line_variants.values():
            report.append(
                f"| {line} | {item['issue_width']} | {item['contexts']} | "
                f"{item['speedup_vs_p1_source_cycles']:.6f}x | "
                f"{item['speedup_vs_same_width_dense_issue']:.6f}x | "
                f"{item['speedup_vs_p1_serialized_service_lower_bound']:.6f}x | "
                f"{item['speedup_vs_p1_descriptor_residency_optimistic_bound']:.6f}x | "
                f"{item['bank_utilization']:.4%} | "
                f"{'PASS' if item['meets_target_with_current_serialized_command_interface'] else 'NO'} |\n"
            )
    report.append(
        "\nThe current RTL serializes one wide command per lane-expanded transaction; "
        "the descriptor-resident column is an optimistic M4 gate, not implemented speedup. "
        "Neither column is full-network acceleration.\n"
    )
    args.output.with_suffix(".md").write_text("".join(report), encoding="utf-8")
    print(f"PASS: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
