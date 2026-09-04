#!/usr/bin/env python3
"""Build stratified real-bitmap VCS commands for the M3 P16C4 engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict, deque
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_object(row: dict[str, str], lane_tile: int) -> tuple[str, ...]:
    return (
        row["name"], row["operator"], row["weight_group"], row["source_base"],
        row["source_width"], row["chunk_index"], str(lane_tile),
    )


def object_tag(identity: tuple[str, ...]) -> int:
    digest = hashlib.sha256("\x1f".join(identity).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") or 1


def hex_bitmap(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    return f"{int.from_bytes(packed.tobytes(), 'little'):064x}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True
    )
    parser.add_argument("--max-commands", type=int, default=20_000)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_commands <= 0 or args.contexts <= 0:
        raise ValueError("max-commands and contexts must be positive")

    validator = load_validator()
    commands: list[dict[str, Any]] = []
    identities: dict[str, Any] = {}
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        manifest, records, current, previous = validator.validate(directory)
        current_bits = np.unpackbits(current, axis=1, bitorder="little").astype(bool)
        previous_bits = np.unpackbits(previous, axis=1, bitorder="little").astype(bool)
        for index, row in enumerate(records):
            use_motion = row["row_use_motion"].lower() == "true"
            selected = current_bits[index] ^ previous_bits[index] if use_motion else current_bits[index]
            negative = previous_bits[index] & ~current_bits[index] if use_motion else np.zeros(256, dtype=bool)
            for lane_tile in range(int(row["output_lane_tile_count_96"])):
                identity = physical_object(row, lane_tile)
                commands.append({
                    "label": label,
                    "sample_id": int(row["sample_id"]),
                    "object": identity,
                    "object_tag": object_tag(identity),
                    "use_motion": use_motion,
                    "source_bits": hex_bitmap(selected),
                    "negative_bits": hex_bitmap(negative),
                    "selected_count": int(selected.sum()),
                })
        identities[label] = {
            "directory": str(directory),
            "records": len(records),
            "sample_ids": sorted({int(row["sample_id"]) for row in records}),
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
            "manifest_sha256": sha256(directory / "manifest.json"),
            "tile_records_sha256": sha256(directory / "tile_records.csv"),
            "packed_tiles_sha256": sha256(directory / "packed_tiles.npz"),
        }

    counts = np.asarray([row["selected_count"] for row in commands], dtype=np.int64)
    low_cut, high_cut = np.quantile(counts, [1 / 3, 2 / 3], method="nearest")
    for row in commands:
        count = row["selected_count"]
        row["density_tier"] = "low" if count <= low_cut else ("mid" if count <= high_cut else "high")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in commands:
        groups[(row["label"], row["sample_id"], *row["object"])].append(row)
    buckets: dict[tuple[Any, ...], deque[list[dict[str, Any]]]] = defaultdict(deque)
    for key in sorted(groups):
        values = groups[key]
        for start in range(0, len(values), args.contexts):
            batch = values[start : start + args.contexts]
            motion_class = "motion" if any(row["use_motion"] for row in batch) else "local"
            tier = max((row["density_tier"] for row in batch), key=("low", "mid", "high").index)
            bucket_key = (batch[0]["label"], batch[0]["sample_id"], tier, motion_class)
            buckets[bucket_key].append(batch)

    selected_batches: list[list[dict[str, Any]]] = []
    selected_count = 0
    ordered_keys = sorted(buckets)
    while selected_count < args.max_commands:
        progress = False
        for key in ordered_keys:
            if not buckets[key]:
                continue
            batch = buckets[key][0]
            if selected_count + len(batch) > args.max_commands:
                continue
            selected_batches.append(buckets[key].popleft())
            selected_count += len(batch)
            progress = True
            if selected_count >= args.max_commands:
                break
        if not progress:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    vector_path = args.output / "real_commands.txt"
    tag = 1
    strata: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    motion_commands = 0
    with vector_path.open("w", encoding="ascii") as handle:
        for batch in selected_batches:
            for index, row in enumerate(batch):
                batch_last = int(index == len(batch) - 1)
                use_motion = int(row["use_motion"])
                handle.write(
                    f"{row['object_tag']:016x} {tag} {batch_last} {use_motion} "
                    f"{row['source_bits']} {row['negative_bits']}\n"
                )
                strata[f"{row['label']}:{row['density_tier']}:{'motion' if use_motion else 'local'}"] += 1
                sample_counts[f"{row['label']}:s{row['sample_id']}"] += 1
                motion_commands += use_motion
                tag += 1

    payload = {
        "schema": "m3_p16c4_real_vcs_vectors_v1",
        "status": "PASS_STRATIFIED_REAL_BITMAPS_NOT_PERFORMANCE_DISTRIBUTION",
        "claim_boundary": (
            "stratified correctness population from exact admitted H67/Local tiles; "
            "round-robin strata sampling is intentionally not a workload-frequency estimate"
        ),
        "contexts": args.contexts,
        "max_commands": args.max_commands,
        "commands": selected_count,
        "batches": len(selected_batches),
        "motion_commands": motion_commands,
        "local_commands": selected_count - motion_commands,
        "density_cut_selected_sources": {"low_mid": int(low_cut), "mid_high": int(high_cut)},
        "strata": dict(sorted(strata.items())),
        "sample_counts": dict(sorted(sample_counts.items())),
        "identities": identities,
        "sha256": {"real_commands.txt": sha256(vector_path)},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"PASS: wrote {selected_count} commands in {len(selected_batches)} batches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
