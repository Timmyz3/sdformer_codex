#!/usr/bin/env python3
"""Build real-bitmap temporal sequences for the integrated M4 state miter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_tag(parts: tuple[str, ...]) -> int:
    value = int.from_bytes(
        hashlib.sha256("\x1f".join(parts).encode()).digest()[:4], "big"
    )
    return (value & 0x7FFF_FFFF) + 1


def packed_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    return f"{int.from_bytes(packed.tobytes(), 'little'):0{len(packed)*2}x}"


def physical_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["row_id"], row["weight_group"],
    )


def geometry_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["weight_group"], row["source_width"],
        row["chunks_per_row"], row["output_channel_fanout"],
        row["output_lane_tile_count_96"],
    )


def load_identity(directory: Path) -> tuple[dict[str, Any], list[dict[str, str]],
                                              np.ndarray, np.ndarray]:
    manifest_path = directory / "manifest.json"
    records_path = directory / "tile_records.csv"
    packed_path = directory / "packed_tiles.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(records_path.open(encoding="utf-8")))
    packed = np.load(packed_path)
    current = np.unpackbits(
        packed["packed_current_bits"], axis=1, bitorder="little"
    ).astype(bool)
    previous = np.unpackbits(
        packed["packed_previous_bits"], axis=1, bitorder="little"
    ).astype(bool)
    if not rows or current.shape != previous.shape or current.shape[0] != len(rows):
        raise ValueError(f"tile package shape mismatch: {directory}")
    return manifest, rows, current, previous


def choose_groups(
    rows: list[dict[str, str]], contexts: int, groups_per_sample: int
) -> list[list[tuple[str, ...]]]:
    physical: dict[tuple[str, ...], list[int]] = defaultdict(list)
    geometry: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for index, row in enumerate(rows):
        physical[physical_key(row)].append(index)
    for key, indices in physical.items():
        first = rows[indices[0]]
        geometry[geometry_key(first)].append(key)

    candidates_by_sample: dict[int, list[tuple[int, int, int, list[tuple[str, ...]]]]] = defaultdict(list)
    for geometry_value, keys in geometry.items():
        keys.sort(key=lambda item: int(item[5]))
        for start in range(0, len(keys), contexts):
            group = keys[start:start + contexts]
            motion = 0
            fanout = int(geometry_value[8])
            width = int(geometry_value[6])
            for key in group:
                motion += sum(
                    rows[index]["row_use_motion"].lower() == "true"
                    and rows[index]["chunk_index"] == "0"
                    for index in physical[key]
                )
            candidates_by_sample[int(geometry_value[0])].append(
                (motion, fanout, width, group)
            )
    selected = []
    for sample in sorted(candidates_by_sample):
        candidates = candidates_by_sample[sample]
        motion_candidates = [item for item in candidates if item[0] > 0]
        if not motion_candidates:
            raise ValueError(f"sample {sample} has no Motion-covered context group")
        ranked_motion = sorted(
            motion_candidates, key=lambda item: item[:3], reverse=True
        )
        # Also cover multi-lane-tile drain order without selecting the largest
        # 16-tile geometry, which would turn this bounded correctness miter
        # into a long throughput run.
        ranked_geometry = sorted(
            (item for item in candidates if 96 < item[1] <= 384),
            key=lambda item: (item[1], item[0], item[2]), reverse=True,
        )
        chosen: list[list[tuple[str, ...]]] = []
        for item in ranked_motion + ranked_geometry:
            if item[3] not in chosen:
                chosen.append(item[3])
            if len(chosen) == groups_per_sample:
                break
        if len(chosen) != groups_per_sample:
            raise ValueError(f"sample {sample} lacks geometry-diverse groups")
        selected.extend(chosen)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2,
        metavar=("LABEL", "TILE_DIR"), required=True,
    )
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--groups-per-sample", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.contexts <= 0 or args.groups_per_sample <= 0:
        raise ValueError("contexts and groups-per-sample must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "stateful_real_descriptors.txt"
    lines: list[str] = []
    ledger: dict[str, int] = defaultdict(int)
    manifest: dict[str, Any] = {
        "schema": "m4_stateful_real_vectors_v1",
        "status": "PASS_CHECKPOINT_BOUND_TEMPORAL_STATE_SEQUENCES",
        "claim_boundary": (
            "real checkpoint bitmap/state transitions with a deterministic "
            "synthetic INT8 weight function; not checkpoint-weight Acc32"
        ),
        "contexts": args.contexts,
        "temporal_modes": ["local_only", "hybrid_local_motion"],
        "identities": {},
        "sample_groups": {},
        "population": ledger,
        "trace_format": (
            "object_tag tag chunk chunks lane_tiles motion selected negative "
            "golden_current row_first row_last batch_last state_context "
            "state_base epoch temporal_step temporal_length temporal_first "
            "temporal_last sequence_id"
        ),
    }
    epoch = 1
    sequence_id = 0
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        source_manifest, rows, current, previous = load_identity(directory)
        groups = choose_groups(rows, args.contexts, args.groups_per_sample)
        index_by_key_step_chunk: dict[tuple[tuple[str, ...], int, int], int] = {}
        for index, row in enumerate(rows):
            identity = (physical_key(row), int(row["temporal_step"]),
                        int(row["chunk_index"]))
            if identity in index_by_key_step_chunk:
                raise ValueError(f"duplicate temporal record: {identity}")
            index_by_key_step_chunk[identity] = index
        manifest["identities"][label] = {
            "directory": str(directory),
            "tile_manifest_sha256": sha256(directory / "manifest.json"),
            "tile_records_sha256": sha256(directory / "tile_records.csv"),
            "packed_tiles_sha256": sha256(directory / "packed_tiles.npz"),
            "checkpoint_sha256": source_manifest["run_context"][
                "artifact_identity"]["checkpoint_sha256"],
            "selected_sample_groups": len(groups),
        }
        for group in groups:
            sample = int(group[0][0])
            manifest["sample_groups"].setdefault(
                f"{label}:s{sample}", []
            ).append([list(key) for key in group])
            first_index = index_by_key_step_chunk[(group[0], 0, 0)]
            temporal_length = len({
                int(row["temporal_step"]) for row in rows
                if physical_key(row) == group[0]
            })
            if temporal_length not in (2, 10):
                raise ValueError("state engine admits only T2/T10")
            for mode in ("local_only", "hybrid_local_motion"):
                sequence_id += 1
                sequence_epoch = epoch
                epoch += 1
                for step in range(temporal_length):
                    batch_indices = []
                    for context, key in enumerate(group):
                        anchor = rows[index_by_key_step_chunk[(key, step, 0)]]
                        chunks = int(anchor["chunks_per_row"])
                        lane_tiles = int(anchor["output_lane_tile_count_96"])
                        object_parts = (
                            anchor["name"], anchor["operator"],
                            anchor["weight_group"], anchor["source_width"],
                            anchor["chunks_per_row"],
                            anchor["output_channel_fanout"],
                        )
                        object_value = stable_tag(object_parts)
                        tag_value = stable_tag(key)
                        row_motion = mode == "hybrid_local_motion" and \
                            anchor["row_use_motion"].lower() == "true"
                        if step == 0 and row_motion:
                            raise ValueError("temporal first step cannot use Motion")
                        for chunk in range(chunks):
                            index = index_by_key_step_chunk[(key, step, chunk)]
                            if step > 0:
                                prior_index = index_by_key_step_chunk[(key, step - 1, chunk)]
                                if not np.array_equal(previous[index], current[prior_index]):
                                    raise ValueError(
                                        f"previous-state chain mismatch {label} {key} "
                                        f"step={step} chunk={chunk}"
                                    )
                            selected = current[index] ^ previous[index] if row_motion \
                                else current[index]
                            negative = np.logical_and(previous[index], ~current[index]) \
                                if row_motion else np.zeros_like(selected)
                            if np.any(negative & ~selected):
                                raise ValueError("negative descriptor escaped selected set")
                            is_last = chunk == chunks - 1
                            is_batch_last = context == len(group) - 1 and is_last
                            lines.append(
                                f"{object_value:08x} {tag_value:08x} {chunk} "
                                f"{chunks} {lane_tiles} {int(row_motion)} "
                                f"{packed_hex(selected)} {packed_hex(negative)} "
                                f"{packed_hex(current[index])} {int(chunk == 0)} "
                                f"{int(is_last)} {int(is_batch_last)} {context} 0 "
                                f"{sequence_epoch} {step} {temporal_length} "
                                f"{int(step == 0)} {int(step == temporal_length-1)} "
                                f"{sequence_id}\n"
                            )
                            batch_indices.append(index)
                            ledger["descriptors"] += 1
                            ledger["selected_sources"] += int(selected.sum())
                            ledger["negative_sources"] += int(negative.sum())
                        outputs = lane_tiles
                        ledger["outputs"] += outputs
                        ledger["motion_outputs" if row_motion else "local_outputs"] += outputs
                    ledger["batches"] += 1
                ledger["sequences"] += 1
                ledger["local_only_sequences" if mode == "local_only"
                       else "hybrid_sequences"] += 1
            del first_index

    if ledger["motion_outputs"] <= 0 or ledger["negative_sources"] <= 0:
        raise ValueError("selected population lacks signed Motion coverage")
    trace_path.write_text("".join(lines), encoding="utf-8")
    manifest["population"] = dict(ledger)
    manifest["sha256"] = {
        "stateful_real_descriptors.txt": sha256(trace_path),
        "generator": sha256(Path(__file__)),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: {ledger['sequences']} sequences {ledger['batches']} batches "
        f"{ledger['outputs']} outputs Motion={ledger['motion_outputs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
