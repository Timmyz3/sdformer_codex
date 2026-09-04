#!/usr/bin/env python3
"""把参数化 Local5 多 tile 原始 trace 编码为结构模板与有类型 tile patch。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "cycle", "event", "tile", "head", "source", "lane", "out", "delay",
    "index", "origin", "payload",
]
CLASS_ORDER = (
    "prefix", "head_seed", "inter_head_gap", "head_accumulate",
    "tile_tail", "tile_transition", "suffix",
)
CLASS_CODE = {name: index for index, name in enumerate(CLASS_ORDER)}
NUMERIC_FIELDS = ("cycle", "tile", "head", "source", "lane", "out", "delay", "index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("source trace is not the frozen 11-column v2 schema")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("source trace is empty or has malformed rows")
    return rows


def find_one(rows: list[dict[str, str]], event: str, tile: int, head: int) -> int:
    matches = [
        index for index, row in enumerate(rows)
        if row["event"] == event
        and int(row["tile"]) == tile
        and int(row["head"]) == head
    ]
    if len(matches) != 1:
        raise ValueError(f"boundary {event}/{tile}/{head} is not unique")
    return matches[0]


def segment_rows(
    rows: list[dict[str, str]], heads: int
) -> list[dict[str, int | str]]:
    starts = {
        (tile, head): find_one(rows, "head_start", tile, head)
        for tile in range(heads) for head in range(heads)
    }
    dones = {
        (tile, head): find_one(rows, "head_done", tile, head)
        for tile in range(heads) for head in range(heads)
    }
    tile_dones = {
        tile: find_one(rows, "tile_done", tile, -1) for tile in range(heads)
    }
    segments: list[dict[str, int | str]] = []

    def add(name: str, start: int, stop: int, tile: int, head: int) -> None:
        if start > stop or (start == stop and name not in ("prefix", "suffix")):
            raise ValueError(f"empty or reversed segment {name}/{tile}/{head}")
        segments.append({
            "class": name, "start": start, "stop": stop,
            "tile": tile, "head": head,
        })

    cursor = 0
    add("prefix", cursor, starts[(0, 0)], -1, -1)
    cursor = starts[(0, 0)]
    for tile in range(heads):
        for head in range(heads):
            start = starts[(tile, head)]
            done = dones[(tile, head)]
            if cursor != start or done < start:
                raise ValueError("head boundary coverage is discontinuous")
            add("head_seed" if head == 0 else "head_accumulate",
                start, done + 1, tile, head)
            cursor = done + 1
            if head + 1 < heads:
                next_start = starts[(tile, head + 1)]
                add("inter_head_gap", cursor, next_start, tile, head)
                cursor = next_start
            else:
                add("tile_tail", cursor, tile_dones[tile] + 1, tile, head)
                cursor = tile_dones[tile] + 1
        if tile + 1 < heads:
            next_start = starts[(tile + 1, 0)]
            add("tile_transition", cursor, next_start, tile + 1, -1)
            cursor = next_start
    add("suffix", cursor, len(rows), -1, -1)
    if segments[0]["start"] != 0 or segments[-1]["stop"] != len(rows):
        raise ValueError("segment endpoints do not cover the source trace")
    if any(segments[i]["stop"] != segments[i + 1]["start"] for i in range(len(segments) - 1)):
        raise ValueError("segments overlap or leave gaps")
    return segments


def fixed_bytes(values: list[str], width: int) -> np.ndarray:
    encoded = [value.encode("ascii") for value in values]
    if any(len(value) > width for value in encoded):
        raise ValueError(f"dictionary entry exceeds S{width}")
    return np.asarray(encoded, dtype=f"S{width}")


def build_archive(
    rows: list[dict[str, str]], trace_sha: str, identity: dict[str, int]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    heads = identity["heads"]
    if heads < 1 or heads > 32:
        raise ValueError("heads must be in 1..32")
    segments = segment_rows(rows, heads)
    by_class: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for segment in segments:
        by_class[str(segment["class"])].append(segment)
    if set(by_class) != set(CLASS_ORDER):
        raise ValueError("template class coverage differs")

    event_values = sorted({row["event"] for row in rows})
    origin_values = sorted({row["origin"] for row in rows})
    payload_values = sorted({row["payload"] for row in rows})
    event_code = {value: index for index, value in enumerate(event_values)}
    origin_code = {value: index for index, value in enumerate(origin_values)}
    payload_code = {value: index for index, value in enumerate(payload_values)}
    if max(len(event_values), len(origin_values)) > 255:
        raise ValueError("event/origin dictionary exceeds uint8")

    template_offsets = [0]
    template_event: list[int] = []
    template_origin: list[int] = []
    class_stats: dict[str, Any] = {}
    for name in CLASS_ORDER:
        instances = by_class[name]
        first = instances[0]
        base = rows[int(first["start"]):int(first["stop"])]
        skeleton = [(row["event"], row["origin"]) for row in base]
        for instance in instances[1:]:
            candidate = rows[int(instance["start"]):int(instance["stop"])]
            if [(row["event"], row["origin"]) for row in candidate] != skeleton:
                raise ValueError(f"{name} instances do not share one structural template")
        template_event.extend(event_code[row["event"]] for row in base)
        template_origin.extend(origin_code[row["origin"]] for row in base)
        template_offsets.append(len(template_event))
        class_stats[name] = {
            "instances": len(instances),
            "template_rows": len(base),
            "expanded_rows": len(base) * len(instances),
        }

    instance_class: list[int] = []
    instance_tile: list[int] = []
    instance_head: list[int] = []
    patch_offsets = [0]
    numeric: dict[str, list[int]] = {name: [] for name in NUMERIC_FIELDS}
    patch_payload: list[int] = []
    for segment in segments:
        name = str(segment["class"])
        selected = rows[int(segment["start"]):int(segment["stop"])]
        instance_class.append(CLASS_CODE[name])
        instance_tile.append(int(segment["tile"]))
        instance_head.append(int(segment["head"]))
        for row in selected:
            for field in NUMERIC_FIELDS:
                numeric[field].append(int(row[field]))
            patch_payload.append(payload_code[row["payload"]])
        patch_offsets.append(len(patch_payload))

    arrays = {
        "schema_version": np.asarray([1], dtype=np.uint16),
        "heads": np.asarray([heads], dtype=np.uint16),
        "source_trace_sha256": fixed_bytes([trace_sha], 64),
        "class_name": fixed_bytes(list(CLASS_ORDER), 32),
        "event_dictionary": fixed_bytes(event_values, 40),
        "origin_dictionary": fixed_bytes(origin_values, 64),
        "payload_dictionary": fixed_bytes(payload_values, 64),
        "template_offsets": np.asarray(template_offsets, dtype=np.int64),
        "template_event_code": np.asarray(template_event, dtype=np.uint8),
        "template_origin_code": np.asarray(template_origin, dtype=np.uint8),
        "instance_class_code": np.asarray(instance_class, dtype=np.uint8),
        "instance_tile": np.asarray(instance_tile, dtype=np.int16),
        "instance_head": np.asarray(instance_head, dtype=np.int16),
        "patch_offsets": np.asarray(patch_offsets, dtype=np.int64),
        "patch_cycle": np.asarray(numeric["cycle"], dtype=np.uint32),
        "patch_tile": np.asarray(numeric["tile"], dtype=np.int16),
        "patch_head": np.asarray(numeric["head"], dtype=np.int16),
        "patch_source": np.asarray(numeric["source"], dtype=np.int16),
        "patch_lane": np.asarray(numeric["lane"], dtype=np.int16),
        "patch_out": np.asarray(numeric["out"], dtype=np.int16),
        "patch_delay": np.asarray(numeric["delay"], dtype=np.int16),
        "patch_index": np.asarray(numeric["index"], dtype=np.int32),
        "patch_payload_code": np.asarray(patch_payload, dtype=np.uint32),
    }
    template_rows = len(template_event)
    expanded_rows = len(rows)
    array_bytes = {name: int(value.nbytes) for name, value in arrays.items()}
    report = {
        "schema": "local5_phase_template_patch_manifest_v2",
        "status": "GENERATED_PENDING_INDEPENDENT_VERIFY_NOT_G0",
        "evidence": "[rtl-trace-derived]",
        "formal_g0": "DENY",
        "identity": identity,
        "source_trace_sha256": trace_sha,
        "expanded_rows": expanded_rows,
        "template_rows": template_rows,
        "base_event_reuse_factor": expanded_rows / template_rows,
        "instances": len(segments),
        "class_stats": class_stats,
        "payload_dictionary_entries": len(payload_values),
        "array_bytes": array_bytes,
        "array_bytes_total": sum(array_bytes.values()),
        "boundary": [
            "tile patch 保存逐行 cycle/identity/payload；没有假设不同 tile 服务时序相同",
            "这是单窗参数化 trace archive canary，不是 formal G0 或架构性能",
        ],
    }
    return arrays, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--sample", type=int, default=2)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--window", type=int, default=249)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace = args.trace.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True)
    identity = {
        "sample": args.sample, "stage": args.stage, "block": args.block,
        "window": args.window, "heads": args.heads,
    }
    if any(value < 0 for value in identity.values()):
        raise ValueError("identity values must be non-negative")
    arrays, manifest = build_archive(load_rows(trace), sha256(trace), identity)
    archive_path = output / "phase_template_patch.npz"
    np.savez(archive_path, **arrays)
    manifest["archive_sha256"] = sha256(archive_path)
    manifest["archive_file_bytes"] = archive_path.stat().st_size
    manifest["source_trace_file_bytes"] = trace.stat().st_size
    manifest["file_size_reduction"] = trace.stat().st_size / archive_path.stat().st_size
    temporary = output / f"manifest.json.tmp.{os.getpid()}"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({
        "status": manifest["status"],
        "archive_sha256": manifest["archive_sha256"],
        "base_event_reuse_factor": manifest["base_event_reuse_factor"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
