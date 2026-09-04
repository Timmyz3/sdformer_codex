#!/usr/bin/env python3
"""Audit cross-context source coalescing on admitted Local/Motion bitmaps.

The existing M4 scheduler permits one context/source update from each weight
bank per cycle.  Spatial contexts share the same weight object, so identical
source indices can instead use one bank read and broadcast the signed INT8
column to every requesting context.  This model preserves the existing
per-context reducer-slot limit and total reducer population; it changes only
the bank request scheduler.
"""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coalesced_issue(
    active: np.ndarray, issue_width: int, reduce_slots: int
) -> dict[str, int]:
    """Schedule non-regressing broadcasts around the frozen M4 grants.

    Each cycle first computes exactly the baseline bank-first/context-first
    primary grants.  Only after every primary grant is frozen does it broadcast
    that source to matching contexts with otherwise unused reducer slots.  The
    baseline work of the cycle is therefore never displaced, no look-ahead or
    per-chunk oracle is required, and one chosen source still consumes one
    weight-bank read even when several contexts update.
    """

    if active.ndim != 2 or issue_width <= 0 or reduce_slots <= 0:
        raise ValueError("invalid coalescing geometry")
    remaining = active.astype(bool, copy=False)
    contexts, source_width = remaining.shape
    # A source is represented by a four-bit requesting-context mask.  The
    # descriptor width is at most 256 bits, so sparse dictionaries avoid the
    # repeated NumPy slicing that would otherwise dominate this audit.
    initial_sources: list[dict[int, int]] = [dict() for _ in range(issue_width)]
    for source in range(source_width):
        context_mask = 0
        for context in range(contexts):
            if remaining[context, source]:
                context_mask |= 1 << context
        if context_mask:
            initial_sources[source % issue_width][source] = context_mask
    shadow_sources = [dict(sources) for sources in initial_sources]
    actual_sources = [dict(sources) for sources in initial_sources]
    shadow_updates = int(remaining.sum())
    actual_updates = shadow_updates
    cycles = 0
    bank_reads = 0
    context_updates = 0
    multi_context_reads = 0
    while actual_updates:
        if shadow_updates <= 0:
            raise RuntimeError("shadow M4 retired before actual coalesced work")
        shadow_used = [0] * contexts
        primary_grants: list[tuple[int, int, int]] = []
        # Advance an independent shadow copy of the exact frozen M4 schedule.
        # Early multicast never mutates this state, so its cycle contract is
        # identical to the admitted baseline even when actual work disappears.
        for bank in range(issue_width):
            for context in range(contexts):
                if shadow_used[context] >= reduce_slots:
                    continue
                sources = [
                    source
                    for source, context_mask in shadow_sources[bank].items()
                    if context_mask & (1 << context)
                ]
                if not sources:
                    continue
                source = min(sources)
                primary_grants.append((bank, source, context))
                shadow_used[context] += 1
                break
        if not primary_grants:
            raise RuntimeError("shadow M4 scheduler made no progress")

        for bank, source, context in primary_grants:
            context_bit = 1 << context
            if not shadow_sources[bank].get(source, 0) & context_bit:
                raise RuntimeError("shadow primary grant disappeared")
            shadow_sources[bank][source] &= ~context_bit
            shadow_updates -= 1
            if shadow_sources[bank][source] == 0:
                del shadow_sources[bank][source]

        # Commit the still-live actual instances of the frozen shadow grants.
        # A request consumed by an earlier multicast uses no reducer or bank in
        # its later shadow slot, which can only create additional slack.
        used = [0] * contexts
        actual_primary_reads: list[tuple[int, int, int]] = []
        issued_this_cycle = 0
        for bank, source, context in primary_grants:
            context_bit = 1 << context
            if not actual_sources[bank].get(source, 0) & context_bit:
                continue
            actual_sources[bank][source] &= ~context_bit
            actual_primary_reads.append((bank, source, context))
            used[context] += 1
            context_updates += 1
            issued_this_cycle += 1
            actual_updates -= 1
            bank_reads += 1
            if actual_sources[bank][source] == 0:
                del actual_sources[bank][source]

        # A broadcast may use only reducer capacity left idle by the complete
        # set of still-live shadow-due updates.  Thus no due actual update is
        # traded for a shared one, and every request is guaranteed to retire no
        # later than its independent shadow-M4 grant.
        for bank, source, primary_context in actual_primary_reads:
            del primary_context
            context_mask = actual_sources[bank].get(source, 0)
            broadcast_count = 0
            for context in range(contexts):
                context_bit = 1 << context
                if context_mask & context_bit and used[context] < reduce_slots:
                    actual_sources[bank][source] &= ~context_bit
                    used[context] += 1
                    context_updates += 1
                    issued_this_cycle += 1
                    actual_updates -= 1
                    broadcast_count += 1
            if broadcast_count:
                multi_context_reads += 1
            if source in actual_sources[bank] and actual_sources[bank][source] == 0:
                del actual_sources[bank][source]
        if issued_this_cycle == 0 and actual_updates:
            # All grants in this shadow cycle were consumed early.  Advancing
            # one empty actual cycle is still no slower than the baseline.
            pass
        cycles += 1
    return {
        "cycles": cycles,
        "bank_reads": bank_reads,
        "context_updates": context_updates,
        "multi_context_reads": multi_context_reads,
    }


def analyze_identity(
    records: list[dict[str, str]],
    current: np.ndarray,
    previous: np.ndarray,
    *,
    line: str,
    issue_width: int,
    contexts: int,
    reduce_slots: int,
    output_lanes: int,
) -> dict[str, Any]:
    wall = load_module(
        "analyze_m4_descriptor_resident_wall_cycles.py", "m10_wall_model"
    )
    bits = wall.selected_bits(records, current, previous, line)
    bundles = wall.ordered_row_bundles(records)
    groups: OrderedDict[tuple[str, ...], list[list[int]]] = OrderedDict()
    for bundle in bundles:
        groups.setdefault(
            wall.batch_key(records[bundle[0]], "temporal_fenced"), []
        ).append(bundle)

    totals: defaultdict[str, int] = defaultdict(int)
    samples: dict[int, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    batches = 0
    for key, group_bundles in groups.items():
        sample_id = int(key[0])
        for start in range(0, len(group_bundles), contexts):
            batch = group_bundles[start : start + contexts]
            chunks = len(batch[0])
            fanout = int(records[batch[0][0]]["output_channel_fanout"])
            lane_tiles = math.ceil(fanout / output_lanes)
            if any(len(bundle) != chunks for bundle in batch):
                raise ValueError("chunk geometry changed inside a batch")
            batches += 1
            descriptor_cycles = sum(len(bundle) for bundle in batch)
            totals["descriptor_load_cycles"] += descriptor_cycles
            samples[sample_id]["descriptor_load_cycles"] += descriptor_cycles
            for chunk in range(chunks):
                active = np.stack(
                    [bits[bundle[chunk]].astype(bool, copy=False) for bundle in batch]
                )
                counts = np.stack(
                    [
                        np.asarray(
                            [
                                int(active[context, bank::issue_width].sum())
                                for bank in range(issue_width)
                            ],
                            dtype=np.int64,
                        )
                        for context in range(len(batch))
                    ]
                )
                baseline_cycles = wall.compact_issue_cycles(counts, reduce_slots)
                coalesced = coalesced_issue(active, issue_width, reduce_slots)
                updates = int(active.sum())
                if coalesced["context_updates"] != updates:
                    raise ValueError("coalesced scheduler lost source updates")
                if coalesced["cycles"] > baseline_cycles:
                    raise ValueError("coalescing regressed the deterministic M4 schedule")
                scaled = {
                    "selected_source_updates": updates * lane_tiles,
                    "baseline_issue_cycles": baseline_cycles * lane_tiles,
                    "coalesced_issue_cycles": coalesced["cycles"] * lane_tiles,
                    "baseline_weight_reads": updates * lane_tiles,
                    "coalesced_weight_reads": coalesced["bank_reads"] * lane_tiles,
                    "multi_context_weight_reads": (
                        coalesced["multi_context_reads"] * lane_tiles
                    ),
                    "chunk_replays": lane_tiles,
                    "chunk_control_cycles": 2 * lane_tiles,
                }
                for field, value in scaled.items():
                    totals[field] += value
                    samples[sample_id][field] += value
            output_cycles = len(batch) * lane_tiles
            totals["output_cycles"] += output_cycles
            samples[sample_id]["output_cycles"] += output_cycles

    def wall_cycles(item: dict[str, int], issue_field: str) -> int:
        return (
            item["descriptor_load_cycles"]
            + item[issue_field]
            + item["chunk_control_cycles"]
            + item["output_cycles"]
        )

    per_sample: dict[str, Any] = {}
    for sample_id in sorted(samples):
        item = samples[sample_id]
        baseline = wall_cycles(item, "baseline_issue_cycles")
        coalesced = wall_cycles(item, "coalesced_issue_cycles")
        per_sample[str(sample_id)] = {
            **dict(item),
            "baseline_wall_cycles": baseline,
            "coalesced_wall_cycles": coalesced,
            "speedup_vs_m4": baseline / coalesced,
            "weight_read_reduction_fraction": 1.0
            - item["coalesced_weight_reads"] / item["baseline_weight_reads"],
        }
    baseline = wall_cycles(totals, "baseline_issue_cycles")
    coalesced = wall_cycles(totals, "coalesced_issue_cycles")
    sample_speedups = [item["speedup_vs_m4"] for item in per_sample.values()]
    return {
        "records": len(records),
        "row_bundles": len(bundles),
        "batches": batches,
        **dict(totals),
        "baseline_wall_cycles": baseline,
        "coalesced_wall_cycles": coalesced,
        "speedup_vs_m4": baseline / coalesced,
        "issue_cycle_reduction_fraction": 1.0
        - totals["coalesced_issue_cycles"] / totals["baseline_issue_cycles"],
        "weight_read_reduction_fraction": 1.0
        - totals["coalesced_weight_reads"] / totals["baseline_weight_reads"],
        "sample_speedup_min": min(sample_speedups),
        "sample_speedup_median": float(np.median(sample_speedups)),
        "sample_speedup_max": max(sample_speedups),
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=2, required=True)
    parser.add_argument("--issue-width", type=int, default=16)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--reduce-slots", type=int, default=4)
    parser.add_argument("--output-lanes", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validator = load_module(
        "build_dual_line_tile_memory_trace.py", "m10_tile_validator"
    )
    loaded = []
    identities: dict[str, Any] = {}
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        manifest, records, current, previous = validator.validate(directory)
        loaded.append((label, directory, manifest, records, current, previous))
        identities[label] = {
            "directory": str(directory),
            "records": len(records),
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"][
                "checkpoint_sha256"
            ],
            "tile_records_sha256": sha256(directory / "tile_records.csv"),
            "packed_tiles_sha256": sha256(directory / "packed_tiles.npz"),
        }

    variants: dict[str, Any] = {}
    for line in ("local", "hybrid"):
        variants[line] = {
            "per_identity": {
                label: analyze_identity(
                    records,
                    current,
                    previous,
                    line=line,
                    issue_width=args.issue_width,
                    contexts=args.contexts,
                    reduce_slots=args.reduce_slots,
                    output_lanes=args.output_lanes,
                )
                for label, _directory, _manifest, records, current, previous in loaded
            }
        }
        rows = variants[line]["per_identity"].values()
        baseline = sum(item["baseline_wall_cycles"] for item in rows)
        rows = variants[line]["per_identity"].values()
        coalesced = sum(item["coalesced_wall_cycles"] for item in rows)
        variants[line]["combined_speedup_vs_m4"] = baseline / coalesced

    payload = {
        "schema": "m10_correlated_source_coalescing_v1",
        "status": "PASS_CYCLE_EXACT_BITMAP_DSE_PRE_RTL",
        "claim_boundary": (
            "Cycle-exact source-kernel scheduler comparison on admitted real bitmaps. "
            "It preserves M4 reducer count and one read per weight bank per cycle, but "
            "freezes every bank-first/context-first M4 primary grant before using "
            "otherwise-idle reducer slots for matching-context broadcasts, so no "
            "look-ahead or oracle fallback is assumed. It does not prove RTL "
            "correctness, SRAM/DRAM timing, or full-network speedup."
        ),
        "architecture": {
            "issue_width": args.issue_width,
            "contexts": args.contexts,
            "reduce_slots_per_context": args.reduce_slots,
            "output_lanes": args.output_lanes,
            "signed_vector_adders_unchanged": (
                args.contexts * args.reduce_slots * args.output_lanes
            ),
            "weight_bank_read_ports_unchanged": args.issue_width,
            "scheduler": (
                "freeze the baseline bank-first/context-first primary grants, then "
                "broadcast each selected source into matching contexts using only "
                "reducer slots left idle by that complete baseline grant set"
            ),
        },
        "identities": identities,
        "variants": variants,
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "correlated_source_coalescing.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Correlated-source coalescing DSE\n\n",
        "| line | identity | M4 wall cycles | coalesced cycles | speedup | weight reads saved | sample min |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for line, variant in variants.items():
        for label, item in variant["per_identity"].items():
            lines.append(
                f"| {line} | {label} | {item['baseline_wall_cycles']:,} | "
                f"{item['coalesced_wall_cycles']:,} | {item['speedup_vs_m4']:.6f}x | "
                f"{item['weight_read_reduction_fraction']:.4%} | "
                f"{item['sample_speedup_min']:.6f}x |\n"
            )
    lines.append(
        "\nThese are source-kernel scheduler cycles, not full-network FPS.\n"
    )
    (args.output / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    print(f"PASS: wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
