#!/usr/bin/env python3
"""Generate a two-pass, memory-mapped Local5 phase array store."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

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
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "schema_version": np.dtype("uint16"),
    "identity_sample": np.dtype("uint32"),
    "identity_stage": np.dtype("uint32"),
    "identity_block": np.dtype("uint32"),
    "identity_window": np.dtype("uint32"),
    "heads": np.dtype("uint16"),
    "source_trace_sha256": np.dtype("S64"),
    "class_name": np.dtype("S32"),
    "event_dictionary": np.dtype("S40"),
    "origin_dictionary": np.dtype("S64"),
    "payload_dictionary": np.dtype("S64"),
    "template_offsets": np.dtype("int64"),
    "template_event_code": np.dtype("uint8"),
    "template_origin_code": np.dtype("uint8"),
    "instance_class_code": np.dtype("uint8"),
    "instance_tile": np.dtype("int16"),
    "instance_head": np.dtype("int16"),
    "patch_offsets": np.dtype("int64"),
    "patch_cycle": np.dtype("uint32"),
    "patch_tile": np.dtype("int16"),
    "patch_head": np.dtype("int16"),
    "patch_source": np.dtype("int16"),
    "patch_lane": np.dtype("int16"),
    "patch_out": np.dtype("int16"),
    "patch_delay": np.dtype("int16"),
    "patch_index": np.dtype("int32"),
    "patch_payload_code": np.dtype("uint32"),
}
PATCH_NUMERIC = {
    "cycle": "patch_cycle",
    "tile": "patch_tile",
    "head": "patch_head",
    "source": "patch_source",
    "lane": "patch_lane",
    "out": "patch_out",
    "delay": "patch_delay",
    "index": "patch_index",
}
FIELD_INDEX = {name: index for index, name in enumerate(FIELDS)}
PAGE_DROP_ROWS = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_bytes(values: list[str], width: int) -> np.ndarray:
    encoded = [value.encode("ascii") for value in values]
    if any(not value or len(value) > width or b"\x00" in value for value in encoded):
        raise ValueError(f"dictionary entry violates S{width}")
    return np.asarray(encoded, dtype=f"S{width}")


def trace_rows(path: Path) -> Iterator[tuple[str, ...]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        header = handle.readline()
        if header.rstrip("\r\n") != ",".join(FIELDS):
            raise ValueError("source trace is not the frozen 11-column schema")
        for line in handle:
            raw = line.rstrip("\r\n")
            if not raw or '"' in raw or "\r" in raw or "\n" in raw:
                raise ValueError("source trace row violates unquoted ASCII line schema")
            row = tuple(raw.split(","))
            if len(row) != len(FIELDS):
                raise ValueError("source trace row has a different column count")
            yield row


def parse_numeric(row: tuple[str, ...]) -> tuple[int, ...]:
    try:
        values = tuple(int(row[FIELD_INDEX[name]]) for name in PATCH_NUMERIC)
    except ValueError as error:
        raise ValueError("source trace numeric field is not an integer") from error
    if not 0 <= values[0] <= np.iinfo(np.uint32).max:
        raise ValueError("source trace cycle exceeds uint32")
    for index, name in enumerate(("tile", "head", "source", "lane", "out", "delay"), start=1):
        if not np.iinfo(np.int16).min <= values[index] <= np.iinfo(np.int16).max:
            raise ValueError(f"source trace {name} exceeds int16")
    if not np.iinfo(np.int32).min <= values[7] <= np.iinfo(np.int32).max:
        raise ValueError("source trace index exceeds int32")
    return values


def first_pass(path: Path) -> dict[str, Any]:
    events: set[str] = set()
    origins: set[str] = set()
    payloads: set[str] = set()
    head_starts: dict[tuple[int, int], int] = {}
    head_dones: dict[tuple[int, int], int] = {}
    tile_dones: dict[int, int] = {}
    event_counts: Counter[str] = Counter()
    row_count = 0
    previous_cycle = -1
    for row_count, row in enumerate(trace_rows(path), start=1):
        try:
            cycle = int(row[FIELD_INDEX["cycle"]])
            tile = int(row[FIELD_INDEX["tile"]])
            head = int(row[FIELD_INDEX["head"]])
        except ValueError as error:
            raise ValueError("source trace boundary numeric field is not an integer") from error
        if not 0 <= cycle <= np.iinfo(np.uint32).max:
            raise ValueError("source trace cycle exceeds uint32")
        if any(
            not np.iinfo(np.int16).min <= value <= np.iinfo(np.int16).max
            for value in (tile, head)
        ):
            raise ValueError("source trace tile/head exceeds int16")
        if cycle < previous_cycle:
            raise ValueError("source trace cycle order regressed")
        previous_cycle = cycle
        event = row[FIELD_INDEX["event"]]
        origin = row[FIELD_INDEX["origin"]]
        payload = row[FIELD_INDEX["payload"]]
        if not event or not origin or not payload:
            raise ValueError("source trace string field is empty")
        events.add(event)
        origins.add(origin)
        payloads.add(payload)
        event_counts[event] += 1
        index = row_count - 1
        if event in {"head_start", "head_done"}:
            key = (tile, head)
            target = head_starts if event == "head_start" else head_dones
            if key in target:
                raise ValueError(f"duplicate {event} boundary")
            target[key] = index
        elif event == "tile_done":
            if tile in tile_dones:
                raise ValueError("duplicate tile_done boundary")
            tile_dones[tile] = index
    if row_count == 0:
        raise ValueError("source trace is empty")
    return {
        "row_count": row_count,
        "events": sorted(events),
        "origins": sorted(origins),
        "payloads": sorted(payloads),
        "head_starts": head_starts,
        "head_dones": head_dones,
        "tile_dones": tile_dones,
        "event_counts": dict(sorted(event_counts.items())),
    }


def build_segments(scan: dict[str, Any], heads: int) -> list[dict[str, int | str]]:
    if not 2 <= heads <= 32:
        raise ValueError("phase array store requires heads in 2..32")
    starts = scan["head_starts"]
    dones = scan["head_dones"]
    tile_dones = scan["tile_dones"]
    expected_heads = {(tile, head) for tile in range(heads) for head in range(heads)}
    if set(starts) != expected_heads or set(dones) != expected_heads:
        raise ValueError("head boundary key set differs")
    if set(tile_dones) != set(range(heads)):
        raise ValueError("tile boundary key set differs")
    segments: list[dict[str, int | str]] = []

    def add(name: str, start: int, stop: int, tile: int, head: int) -> None:
        if start > stop or (start == stop and name not in {"prefix", "suffix"}):
            raise ValueError(f"empty/reversed segment {name}/{tile}/{head}")
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
            add(
                "head_seed" if head == 0 else "head_accumulate",
                start, done + 1, tile, head,
            )
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
    add("suffix", cursor, scan["row_count"], -1, -1)
    if segments[0]["start"] != 0 or segments[-1]["stop"] != scan["row_count"]:
        raise ValueError("segment endpoints differ")
    if any(
        segments[index]["stop"] != segments[index + 1]["start"]
        for index in range(len(segments) - 1)
    ):
        raise ValueError("segments overlap or leave gaps")
    if {str(segment["class"]) for segment in segments} != set(CLASS_ORDER):
        raise ValueError("template class coverage differs")
    return segments


def save_array(root: Path, name: str, value: np.ndarray) -> None:
    if value.dtype != ARRAY_DTYPES[name] or value.ndim != 1:
        raise ValueError(f"{name} dtype/rank differs before save")
    np.save(root / f"{name}.npy", value, allow_pickle=False)


def open_patch(root: Path, name: str, count: int) -> np.memmap:
    return np.lib.format.open_memmap(
        root / f"{name}.npy", mode="w+", dtype=ARRAY_DTYPES[name], shape=(count,)
    )


def flush_and_drop(value: np.memmap) -> None:
    value.flush()
    raw = getattr(value, "_mmap", None)
    if raw is None or not hasattr(raw, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError("platform lacks required mmap MADV_DONTNEED support")
    raw.madvise(mmap.MADV_DONTNEED)


def generate(
    trace: Path, output: Path, identity: dict[str, int], verifier_source: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > np.iinfo(np.uint32).max
        for value in identity.values()
    ):
        raise ValueError("identity values must fit unsigned 32-bit integers")
    staging = output.with_name(f"{output.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    arrays_dir = staging / "arrays"
    source_dir = staging / "source"
    arrays_dir.mkdir(parents=True)
    source_dir.mkdir()
    scan = first_pass(trace)
    heads = identity["heads"]
    segments = build_segments(scan, heads)
    row_count = scan["row_count"]
    trace_sha = sha256(trace)
    events = scan["events"]
    origins = scan["origins"]
    payloads = scan["payloads"]
    if len(events) > 255 or len(origins) > 255 or len(payloads) > np.iinfo(np.uint32).max:
        raise ValueError("dictionary code width exceeded")
    event_code = {value: index for index, value in enumerate(events)}
    origin_code = {value: index for index, value in enumerate(origins)}
    payload_code = {value: index for index, value in enumerate(payloads)}

    first_segment_by_class: dict[str, int] = {}
    for index, segment in enumerate(segments):
        first_segment_by_class.setdefault(str(segment["class"]), index)
    template_lengths = {
        name: int(segments[first_segment_by_class[name]]["stop"])
        - int(segments[first_segment_by_class[name]]["start"])
        for name in CLASS_ORDER
    }
    template_offsets = [0]
    for name in CLASS_ORDER:
        template_offsets.append(template_offsets[-1] + template_lengths[name])
    patch_offsets = [0]
    for segment in segments:
        patch_offsets.append(patch_offsets[-1] + int(segment["stop"]) - int(segment["start"]))
    if patch_offsets[-1] != row_count:
        raise ValueError("patch offsets do not cover trace")

    save_array(arrays_dir, "schema_version", np.asarray([2], dtype=np.uint16))
    for name in ("sample", "stage", "block", "window"):
        save_array(
            arrays_dir, f"identity_{name}",
            np.asarray([identity[name]], dtype=np.uint32),
        )
    save_array(arrays_dir, "heads", np.asarray([heads], dtype=np.uint16))
    save_array(arrays_dir, "source_trace_sha256", fixed_bytes([trace_sha], 64))
    save_array(arrays_dir, "class_name", fixed_bytes(list(CLASS_ORDER), 32))
    save_array(arrays_dir, "event_dictionary", fixed_bytes(events, 40))
    save_array(arrays_dir, "origin_dictionary", fixed_bytes(origins, 64))
    save_array(arrays_dir, "payload_dictionary", fixed_bytes(payloads, 64))
    save_array(
        arrays_dir, "template_offsets", np.asarray(template_offsets, dtype=np.int64)
    )
    save_array(
        arrays_dir, "instance_class_code",
        np.asarray([CLASS_CODE[str(row["class"])] for row in segments], dtype=np.uint8),
    )
    save_array(
        arrays_dir, "instance_tile",
        np.asarray([int(row["tile"]) for row in segments], dtype=np.int16),
    )
    save_array(
        arrays_dir, "instance_head",
        np.asarray([int(row["head"]) for row in segments], dtype=np.int16),
    )
    save_array(arrays_dir, "patch_offsets", np.asarray(patch_offsets, dtype=np.int64))

    writable: dict[str, np.memmap] = {
        **{
            array_name: open_patch(arrays_dir, array_name, row_count)
            for array_name in PATCH_NUMERIC.values()
        },
        "patch_payload_code": open_patch(arrays_dir, "patch_payload_code", row_count),
    }
    template_event = np.empty(template_offsets[-1], dtype=np.uint8)
    template_origin = np.empty(template_offsets[-1], dtype=np.uint8)
    numeric_buffers: dict[str, list[int]] = {
        array_name: [] for array_name in PATCH_NUMERIC.values()
    }
    payload_buffer: list[int] = []
    buffer_start = 0
    last_page_drop = 0
    segment_index = 0

    segment_events: list[int] = []
    segment_origins: list[int] = []

    def finish_segment(index: int) -> None:
        nonlocal segment_events, segment_origins
        segment = segments[index]
        class_name = str(segment["class"])
        class_code = CLASS_CODE[class_name]
        expected_length = int(segment["stop"]) - int(segment["start"])
        if len(segment_events) != expected_length or len(segment_origins) != expected_length:
            raise ValueError("second-pass segment length differs")
        template_start = template_offsets[class_code]
        template_stop = template_offsets[class_code + 1]
        observed_events = np.asarray(segment_events, dtype=np.uint8)
        observed_origins = np.asarray(segment_origins, dtype=np.uint8)
        if index == first_segment_by_class[class_name]:
            template_event[template_start:template_stop] = observed_events
            template_origin[template_start:template_stop] = observed_origins
        elif (
            not np.array_equal(template_event[template_start:template_stop], observed_events)
            or not np.array_equal(template_origin[template_start:template_stop], observed_origins)
        ):
            raise ValueError(f"{class_name} structural template differs")
        segment_events = []
        segment_origins = []

    def flush_buffers(stop: int) -> None:
        nonlocal buffer_start, payload_buffer, last_page_drop
        count = stop - buffer_start
        if count != len(payload_buffer):
            raise ValueError("second-pass write buffer length differs")
        if count == 0:
            return
        for array_name, buffer in numeric_buffers.items():
            if len(buffer) != count:
                raise ValueError(f"{array_name} write buffer length differs")
            writable[array_name][buffer_start:stop] = np.asarray(
                buffer, dtype=ARRAY_DTYPES[array_name]
            )
            buffer.clear()
        writable["patch_payload_code"][buffer_start:stop] = np.asarray(
            payload_buffer, dtype=np.uint32
        )
        payload_buffer = []
        buffer_start = stop
        if stop - last_page_drop >= PAGE_DROP_ROWS:
            for value in writable.values():
                flush_and_drop(value)
            last_page_drop = stop

    for row_index, row in enumerate(trace_rows(trace)):
        while row_index >= int(segments[segment_index]["stop"]):
            finish_segment(segment_index)
            segment_index += 1
        segment = segments[segment_index]
        if not int(segment["start"]) <= row_index < int(segment["stop"]):
            raise ValueError("second-pass segment position differs")
        observed_event = event_code[row[FIELD_INDEX["event"]]]
        observed_origin = origin_code[row[FIELD_INDEX["origin"]]]
        segment_events.append(observed_event)
        segment_origins.append(observed_origin)
        numeric = parse_numeric(row)
        for value, array_name in zip(numeric, PATCH_NUMERIC.values()):
            numeric_buffers[array_name].append(value)
        payload_buffer.append(payload_code[row[FIELD_INDEX["payload"]]])
        if len(payload_buffer) >= 65536:
            flush_buffers(row_index + 1)
    if row_index + 1 != row_count:
        raise ValueError("second-pass row count differs")
    while segment_index < len(segments):
        finish_segment(segment_index)
        segment_index += 1
    flush_buffers(row_count)
    for value in writable.values():
        flush_and_drop(value)
    del writable
    save_array(arrays_dir, "template_event_code", template_event)
    save_array(arrays_dir, "template_origin_code", template_origin)

    shutil.copy2(Path(__file__).resolve(), source_dir / Path(__file__).name)
    shutil.copy2(verifier_source, source_dir / verifier_source.name)
    arrays: dict[str, Any] = {}
    total_nbytes = 0
    total_file_bytes = 0
    for name, dtype in ARRAY_DTYPES.items():
        path = arrays_dir / f"{name}.npy"
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != dtype or value.ndim != 1:
            raise ValueError(f"written {name} dtype/rank differs")
        arrays[name] = {
            "file": f"arrays/{name}.npy",
            "dtype": dtype.str,
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        total_nbytes += int(value.nbytes)
        total_file_bytes += path.stat().st_size
    class_stats = {}
    for name in CLASS_ORDER:
        class_code = CLASS_CODE[name]
        selected = [
            segment for segment in segments if str(segment["class"]) == name
        ]
        class_stats[name] = {
            "instances": len(selected),
            "template_rows": template_lengths[name],
            "expanded_rows": sum(
                int(segment["stop"]) - int(segment["start"])
                for segment in selected
            ),
        }
    manifest = {
        "schema": "local5_phase_array_store_v2",
        "status": "GENERATED_PENDING_INDEPENDENT_VERIFY_NOT_G0",
        "evidence": "[rtl-trace-derived]",
        "formal_g0": "DENY",
        "identity": identity,
        "source_trace": str(trace),
        "source_trace_sha256": trace_sha,
        "source_trace_file_bytes": trace.stat().st_size,
        "expanded_rows": row_count,
        "template_rows": template_offsets[-1],
        "base_event_reuse_factor": row_count / template_offsets[-1],
        "instances": len(segments),
        "class_stats": class_stats,
        "event_counts": scan["event_counts"],
        "payload_dictionary_entries": len(payloads),
        "array_nbytes_total": total_nbytes,
        "array_file_bytes_total": total_file_bytes,
        "mmap_page_drop_rows": PAGE_DROP_ROWS,
        "arrays": arrays,
        "source_bindings": {
            "generator": {
                "file": f"source/{Path(__file__).name}",
                "sha256": sha256(source_dir / Path(__file__).name),
            },
            "independent_verifier": {
                "file": f"source/{verifier_source.name}",
                "sha256": sha256(source_dir / verifier_source.name),
            },
        },
        "boundary": [
            "two-pass generator stores dictionaries plus mmap typed arrays",
            "mmap patch arrays are flushed and MADV_DONTNEED-released every 1048576 rows",
            "array-store bytes are verification archive storage, not on-chip SRAM",
            "formal G0 and architecture performance remain unavailable",
        ],
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--verifier-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    identity = {
        "sample": args.sample, "stage": args.stage, "block": args.block,
        "window": args.window, "heads": args.heads,
    }
    manifest = generate(
        args.trace.resolve(), args.output_dir.resolve(), identity,
        args.verifier_source.resolve(),
    )
    print(json.dumps({
        "status": manifest["status"], "identity": identity,
        "expanded_rows": manifest["expanded_rows"],
        "array_file_bytes_total": manifest["array_file_bytes_total"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
