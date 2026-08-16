#!/usr/bin/env python3
"""Strict NPZ content replay for Local5 EREP RTL trace and Acc32 miter archives."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .local5_erep_ledger_replay_v4 import (
        BLOCK_ORDER,
        HEAD_FIELDS,
        ROLE_RESOURCES,
        STAGE_BLOCKS,
        STAGE_HEADS,
        STAGE_WEIGHTS,
        WINDOW_FIELDS,
        canonical_sha,
        decode_phase,
        strict_sha,
        strict_uint,
    )
else:
    from local5_erep_ledger_replay_v4 import (
        BLOCK_ORDER,
        HEAD_FIELDS,
        ROLE_RESOURCES,
        STAGE_BLOCKS,
        STAGE_HEADS,
        STAGE_WEIGHTS,
        WINDOW_FIELDS,
        canonical_sha,
        decode_phase,
        strict_sha,
        strict_uint,
    )


ARCHIVE_SCHEMA_VERSION = 4
TOKENS = 450
OUT_DIM = 32
FORMAL_SAMPLE_COUNT = 100
FORMAL_WINDOW_COUNT = FORMAL_SAMPLE_COUNT * sum(STAGE_BLOCKS.values())
FORMAL_HEAD_COUNT = FORMAL_SAMPLE_COUNT * sum(
    STAGE_BLOCKS[stage] * STAGE_HEADS[stage] for stage in STAGE_HEADS
)
FORMAL_PHASE_COUNT = FORMAL_SAMPLE_COUNT * sum(
    STAGE_BLOCKS[stage]
    * (2 * STAGE_HEADS[stage] + STAGE_HEADS[stage] * (1 + 2 * STAGE_HEADS[stage]))
    for stage in STAGE_HEADS
)
FORMAL_ACC32_VALUE_COUNT = FORMAL_HEAD_COUNT * TOKENS * OUT_DIM
ROLE_ORDER = ("prepare", "drain", "fill", "direct", "execute")
ROLE_TO_CODE = {role: index for index, role in enumerate(ROLE_ORDER)}
CODE_TO_ROLE = {value: key for key, value in ROLE_TO_CODE.items()}
RESOURCE_ORDER = tuple(
    sorted({resource for values in ROLE_RESOURCES.values() for resource in values})
)
RESOURCE_TO_CODE = {resource: index for index, resource in enumerate(RESOURCE_ORDER)}
CODE_TO_RESOURCE = {value: key for key, value in RESOURCE_TO_CODE.items()}

WINDOW_ARRAY_SPECS = {
    "window_sample": np.dtype("uint16"),
    "window_stage": np.dtype("uint8"),
    "window_block": np.dtype("uint8"),
    "window_token": np.dtype("uint16"),
    "window_weight": np.dtype("uint16"),
    "window_heads": np.dtype("uint8"),
}
TRACE_ARRAY_SPECS = {
    "schema_version": np.dtype("uint16"),
    **WINDOW_ARRAY_SPECS,
    "phase_window_index": np.dtype("uint16"),
    "phase_input_head": np.dtype("int16"),
    "phase_role": np.dtype("uint8"),
    "phase_output_tile": np.dtype("int16"),
    "phase_duration": np.dtype("uint32"),
    "phase_event_offsets": np.dtype("int64"),
    "event_resource": np.dtype("uint8"),
    "event_cycle": np.dtype("uint32"),
    "event_identity": np.dtype("S64"),
}
MITER_ARRAY_SPECS = {
    "schema_version": np.dtype("uint16"),
    **WINDOW_ARRAY_SPECS,
    "window_value_offsets": np.dtype("int64"),
    "expected_acc32": np.dtype("int32"),
    "actual_acc32": np.dtype("int32"),
}


def _strict_arrays(
    payload: Mapping[str, Any], specs: Mapping[str, np.dtype], name: str
) -> dict[str, np.ndarray]:
    if not isinstance(payload, Mapping) or set(payload) != set(specs):
        raise ValueError(f"{name} NPZ member set is not frozen")
    arrays: dict[str, np.ndarray] = {}
    for member, dtype in specs.items():
        value = payload[member]
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            raise ValueError(f"{name}/{member} dtype is not {dtype}")
        if value.ndim != 1:
            raise ValueError(f"{name}/{member} must be one-dimensional")
        arrays[member] = value
    version = arrays["schema_version"]
    if version.shape != (1,) or int(version[0]) != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(f"{name} schema version is invalid")
    return arrays


def _validate_npz_container(
    path: Path, specs: Mapping[str, np.dtype], name: str
) -> None:
    expected = [f"{member}.npy" for member in specs]
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            observed = [member.filename for member in members]
            if observed != expected or len(observed) != len(set(observed)):
                raise ValueError(
                    f"{name} ZIP member names/order/uniqueness are not frozen"
                )
            if archive.comment:
                raise ValueError(f"{name} ZIP archive comment is not allowed")
            for member in members:
                if (
                    member.is_dir()
                    or member.flag_bits != 0
                    or member.comment
                    or member.extra
                    or member.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                ):
                    raise ValueError(f"{name} ZIP member encoding is not allowed")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{name} NPZ container cannot be parsed safely") from error


def _window_metadata(arrays: Mapping[str, np.ndarray], *, formal: bool) -> list[dict[str, int]]:
    count = len(arrays["window_sample"])
    if formal and count != FORMAL_WINDOW_COUNT:
        raise ValueError("formal archive must contain exactly 1200 windows")
    if any(len(arrays[name]) != count for name in WINDOW_ARRAY_SPECS):
        raise ValueError("archive window metadata arrays are not aligned")
    rows = []
    for index in range(count):
        row = {
            "sample": int(arrays["window_sample"][index]),
            "stage": int(arrays["window_stage"][index]),
            "block": int(arrays["window_block"][index]),
            "window": int(arrays["window_token"][index]),
            "weight": int(arrays["window_weight"][index]),
            "heads": int(arrays["window_heads"][index]),
        }
        if (
            row["stage"] not in STAGE_HEADS
            or row["heads"] != STAGE_HEADS[row["stage"]]
            or row["weight"] != STAGE_WEIGHTS[row["stage"]]
        ):
            raise ValueError(f"archive window {index} has invalid stage/H/weight")
        if formal:
            stage, block = BLOCK_ORDER[index % len(BLOCK_ORDER)]
            if (
                row["sample"] != index // len(BLOCK_ORDER)
                or (row["stage"], row["block"]) != (stage, block)
            ):
                raise ValueError("formal archive window metadata is not canonical")
        rows.append(row)
    if len(
        {
            (row["sample"], row["stage"], row["block"], row["window"])
            for row in rows
        }
    ) != count:
        raise ValueError("archive contains duplicate window coordinates")
    return rows


def _expected_phase_descriptors(
    windows: list[dict[str, int]],
) -> list[tuple[int, int, str, int]]:
    descriptors = []
    for window_index, row in enumerate(windows):
        heads = row["heads"]
        descriptors.extend(
            (window_index, -1, "prepare", tile) for tile in range(heads)
        )
        descriptors.extend(
            (window_index, -1, "drain", tile) for tile in range(heads)
        )
        for head in range(heads):
            descriptors.append((window_index, head, "fill", -1))
            descriptors.extend(
                (window_index, head, "direct", tile) for tile in range(heads)
            )
            descriptors.extend(
                (window_index, head, "execute", tile) for tile in range(heads)
            )
    return descriptors


def _decode_identity(value: np.bytes_, name: str) -> str:
    raw = bytes(value).rstrip(b"\x00")
    if not raw or b"\x00" in raw:
        raise ValueError(f"{name} is empty or contains embedded NUL")
    try:
        identity = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not ASCII") from error
    if len(identity.encode("ascii")) > 64:
        raise ValueError(f"{name} exceeds S64")
    return identity


def parse_trace_archive(
    payload: Mapping[str, Any], *, formal: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = _strict_arrays(payload, TRACE_ARRAY_SPECS, "RTL trace archive")
    windows = _window_metadata(arrays, formal=formal)
    descriptors = _expected_phase_descriptors(windows)
    phase_count = len(descriptors)
    if formal and phase_count != FORMAL_PHASE_COUNT:
        raise ValueError("formal RTL trace archive must contain exactly 462600 phases")
    phase_names = (
        "phase_window_index", "phase_input_head", "phase_role",
        "phase_output_tile", "phase_duration",
    )
    if any(len(arrays[name]) != phase_count for name in phase_names):
        raise ValueError("RTL trace phase arrays do not match canonical phase count")
    offsets = arrays["phase_event_offsets"]
    event_count = len(arrays["event_resource"])
    if (
        offsets.shape != (phase_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != event_count
        or np.any(offsets[1:] < offsets[:-1])
        or len(arrays["event_cycle"]) != event_count
        or len(arrays["event_identity"]) != event_count
    ):
        raise ValueError("RTL trace event offsets/arrays are not exact")

    common: dict[int, dict[str, list[Any]]] = {
        index: {"prepare": [], "drain": []} for index in range(len(windows))
    }
    by_head: dict[tuple[int, int], dict[str, Any]] = {}
    for phase_index, expected in enumerate(descriptors):
        window_index, input_head, role, output_tile = expected
        observed = (
            int(arrays["phase_window_index"][phase_index]),
            int(arrays["phase_input_head"][phase_index]),
            CODE_TO_ROLE.get(int(arrays["phase_role"][phase_index])),
            int(arrays["phase_output_tile"][phase_index]),
        )
        if observed != expected:
            raise ValueError("RTL trace phases are not in canonical role/head/tile order")
        duration = int(arrays["phase_duration"][phase_index])
        resource_events = {resource: [] for resource in ROLE_RESOURCES[role]}
        start = int(offsets[phase_index])
        end = int(offsets[phase_index + 1])
        observed_order = []
        for event_index in range(start, end):
            resource = CODE_TO_RESOURCE.get(int(arrays["event_resource"][event_index]))
            if resource not in resource_events:
                raise ValueError("RTL trace event resource is illegal for its phase role")
            event = {
                "cycle": int(arrays["event_cycle"][event_index]),
                "identity": _decode_identity(
                    arrays["event_identity"][event_index],
                    f"event {event_index} identity",
                ),
            }
            resource_events[resource].append(event)
            observed_order.append(
                (ROLE_RESOURCES[role].index(resource), event["cycle"], event["identity"])
            )
        if observed_order != sorted(observed_order):
            raise ValueError("RTL trace events are not in canonical resource/cycle/identity order")
        body = {"duration": duration, "resource_events": resource_events}
        phase = {**body, "phase_event_sha256": canonical_sha(body)}
        decode_phase(
            phase,
            role,
            epoch_records=(
                None
                if role not in {"fill", "execute"}
                else len(resource_events[ROLE_RESOURCES[role][0]])
            ),
        )
        if input_head < 0:
            common[window_index][role].append(phase)
        else:
            row = by_head.setdefault(
                (window_index, input_head),
                {"fill": None, "direct_by_tile": [], "execute_by_tile": []},
            )
            if role == "fill":
                row["fill"] = phase
            else:
                row[f"{role}_by_tile"].append(phase)

    window_rows = []
    head_rows = []
    for window_index, metadata in enumerate(windows):
        heads = metadata["heads"]
        window_rows.append(
            {
                **metadata,
                "output_tiles": list(range(heads)),
                "prepare_by_tile": common[window_index]["prepare"],
                "drain_by_tile": common[window_index]["drain"],
            }
        )
        for head in range(heads):
            phases = by_head.get((window_index, head))
            if (
                phases is None
                or phases["fill"] is None
                or len(phases["direct_by_tile"]) != heads
                or len(phases["execute_by_tile"]) != heads
            ):
                raise ValueError("RTL trace archive has incomplete head phase coverage")
            fill_records = len(
                phases["fill"]["resource_events"]["relation_workspace_1rw"]
            )
            coordinate = {
                field: metadata[field]
                for field in ("sample", "stage", "block", "window")
            }
            trace_body = {
                "fill": phases["fill"],
                "direct_by_tile": phases["direct_by_tile"],
                "execute_by_tile": phases["execute_by_tile"],
            }
            head_rows.append(
                {
                    **coordinate,
                    "input_head": head,
                    "epoch_records": fill_records,
                    "rtl_trace_sha256": canonical_sha(trace_body),
                    **trace_body,
                }
            )
    if formal and len(head_rows) != FORMAL_HEAD_COUNT:
        raise ValueError("formal RTL trace archive must contain exactly 13800 heads")
    return window_rows, head_rows


def _miter_digest(
    metadata: Mapping[str, int], expected: np.ndarray, actual: np.ndarray
) -> str:
    digest = hashlib.sha256()
    identity = [
        metadata[field]
        for field in ("sample", "stage", "block", "window", "heads")
    ]
    digest.update(json.dumps(identity, separators=(",", ":")).encode("ascii"))
    digest.update(expected.tobytes(order="C"))
    digest.update(actual.tobytes(order="C"))
    return digest.hexdigest()


def parse_miter_archive(
    payload: Mapping[str, Any], *, formal: bool = False
) -> list[dict[str, Any]]:
    arrays = _strict_arrays(payload, MITER_ARRAY_SPECS, "Acc32 miter archive")
    windows = _window_metadata(arrays, formal=formal)
    offsets = arrays["window_value_offsets"]
    value_count = len(arrays["expected_acc32"])
    if formal and value_count != FORMAL_ACC32_VALUE_COUNT:
        raise ValueError(
            "formal Acc32 miter archive must contain exactly 198720000 scalars"
        )
    if (
        offsets.shape != (len(windows) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != value_count
        or np.any(offsets[1:] < offsets[:-1])
        or len(arrays["actual_acc32"]) != value_count
    ):
        raise ValueError("Acc32 miter offsets/arrays are not exact")
    rows = []
    for index, metadata in enumerate(windows):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        expected_count = metadata["heads"] * TOKENS * OUT_DIM
        if end - start != expected_count:
            raise ValueError("Acc32 miter window does not contain H*450*32 scalars")
        expected = arrays["expected_acc32"][start:end]
        actual = arrays["actual_acc32"][start:end]
        mismatch = int(np.count_nonzero(expected != actual))
        if mismatch != 0:
            raise ValueError("Acc32 miter archive contains a nonzero mismatch")
        rows.append(
            {
                **metadata,
                "acc32_miter_sha256": _miter_digest(metadata, expected, actual),
                "acc32_mismatch_count": mismatch,
            }
        )
    return rows


def validate_archive_contents(
    trace_payload: Mapping[str, Any],
    miter_payload: Mapping[str, Any],
    head_ledger: Mapping[str, Any],
    *,
    formal: bool = False,
) -> dict[str, Any]:
    trace_windows, trace_heads = parse_trace_archive(trace_payload, formal=formal)
    miter_rows = parse_miter_archive(miter_payload, formal=formal)
    if len(trace_windows) != len(miter_rows):
        raise ValueError("trace and miter archives have different window counts")
    expected_windows = []
    for trace, miter in zip(trace_windows, miter_rows, strict=True):
        for field in ("sample", "stage", "block", "window", "weight", "heads"):
            if trace[field] != miter[field]:
                raise ValueError("trace and miter archive window metadata mismatch")
        expected_windows.append(
            {
                **trace,
                "acc32_miter_sha256": miter["acc32_miter_sha256"],
                "acc32_mismatch_count": miter["acc32_mismatch_count"],
            }
        )
    if not isinstance(head_ledger, Mapping):
        raise ValueError("head phase ledger must be a mapping")
    if canonical_sha(head_ledger.get("windows")) != canonical_sha(expected_windows):
        raise ValueError("head ledger windows differ from parsed archive contents")
    if canonical_sha(head_ledger.get("heads")) != canonical_sha(trace_heads):
        raise ValueError("head ledger rows differ from parsed RTL trace archive")
    return {
        "window_count": len(expected_windows),
        "head_count": len(trace_heads),
        "phase_count": len(trace_payload["phase_duration"]),
        "event_count": len(trace_payload["event_cycle"]),
        "acc32_value_count": len(miter_payload["expected_acc32"]),
        "acc32_mismatch_count": 0,
        "window_rows_sha256": canonical_sha(expected_windows),
        "head_rows_sha256": canonical_sha(trace_heads),
    }


def validate_archive_files(
    trace_path: Path,
    miter_path: Path,
    head_ledger: Mapping[str, Any],
    *,
    formal: bool = False,
) -> dict[str, Any]:
    try:
        _validate_npz_container(trace_path, TRACE_ARRAY_SPECS, "RTL trace archive")
        _validate_npz_container(miter_path, MITER_ARRAY_SPECS, "Acc32 miter archive")
        with np.load(trace_path, allow_pickle=False) as trace_file:
            trace = {name: trace_file[name] for name in trace_file.files}
        with np.load(miter_path, allow_pickle=False) as miter_file:
            miter = {name: miter_file[name] for name in miter_file.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError("formal archive NPZ cannot be parsed safely") from error
    return validate_archive_contents(trace, miter, head_ledger, formal=formal)


def encode_trace_fixture(head_ledger: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Encode a small validated head ledger for synthetic parser tests only."""
    windows = head_ledger["windows"]
    heads = head_ledger["heads"]
    grouped = {}
    for row in heads:
        key = (row["sample"], row["stage"], row["block"], row["window"])
        grouped.setdefault(key, []).append(row)
    descriptors = []
    events = []
    offsets = [0]
    for window_index, window in enumerate(windows):
        key = tuple(window[field] for field in ("sample", "stage", "block", "window"))
        phase_rows = []
        phase_rows.extend((-1, "prepare", tile, phase) for tile, phase in enumerate(window["prepare_by_tile"]))
        phase_rows.extend((-1, "drain", tile, phase) for tile, phase in enumerate(window["drain_by_tile"]))
        for head, row in enumerate(grouped[key]):
            phase_rows.append((head, "fill", -1, row["fill"]))
            phase_rows.extend((head, "direct", tile, phase) for tile, phase in enumerate(row["direct_by_tile"]))
            phase_rows.extend((head, "execute", tile, phase) for tile, phase in enumerate(row["execute_by_tile"]))
        for input_head, role, tile, phase in phase_rows:
            descriptors.append((window_index, input_head, role, tile, phase["duration"]))
            for resource in ROLE_RESOURCES[role]:
                for event in phase["resource_events"][resource]:
                    identity = event["identity"]
                    if not isinstance(identity, str):
                        raise ValueError("fixture event identity must be a string")
                    try:
                        encoded_identity = identity.encode("ascii")
                    except UnicodeEncodeError as error:
                        raise ValueError("fixture event identity must be ASCII") from error
                    if not encoded_identity or len(encoded_identity) > 64 or b"\x00" in encoded_identity:
                        raise ValueError(
                            "fixture event identity must be nonempty NUL-free ASCII within S64"
                        )
                    events.append(
                        (RESOURCE_TO_CODE[resource], event["cycle"], encoded_identity)
                    )
            offsets.append(len(events))
    return {
        "schema_version": np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.uint16),
        "window_sample": np.asarray([row["sample"] for row in windows], dtype=np.uint16),
        "window_stage": np.asarray([row["stage"] for row in windows], dtype=np.uint8),
        "window_block": np.asarray([row["block"] for row in windows], dtype=np.uint8),
        "window_token": np.asarray([row["window"] for row in windows], dtype=np.uint16),
        "window_weight": np.asarray([row["weight"] for row in windows], dtype=np.uint16),
        "window_heads": np.asarray([row["heads"] for row in windows], dtype=np.uint8),
        "phase_window_index": np.asarray([row[0] for row in descriptors], dtype=np.uint16),
        "phase_input_head": np.asarray([row[1] for row in descriptors], dtype=np.int16),
        "phase_role": np.asarray([ROLE_TO_CODE[row[2]] for row in descriptors], dtype=np.uint8),
        "phase_output_tile": np.asarray([row[3] for row in descriptors], dtype=np.int16),
        "phase_duration": np.asarray([row[4] for row in descriptors], dtype=np.uint32),
        "phase_event_offsets": np.asarray(offsets, dtype=np.int64),
        "event_resource": np.asarray([row[0] for row in events], dtype=np.uint8),
        "event_cycle": np.asarray([row[1] for row in events], dtype=np.uint32),
        "event_identity": np.asarray([row[2] for row in events], dtype="S64"),
    }


def encode_miter_fixture(
    head_ledger: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create zero-mismatch synthetic Acc32 arrays and bind their window digests."""
    bound = copy.deepcopy(head_ledger)
    windows = bound["windows"]
    offsets = [0]
    expected_parts = []
    for index, row in enumerate(windows):
        count = row["heads"] * TOKENS * OUT_DIM
        values = np.arange(count, dtype=np.int64)
        values = ((values + index * 17) % 65521 - 32760).astype(np.int32)
        expected_parts.append(values)
        offsets.append(offsets[-1] + count)
    expected = np.concatenate(expected_parts) if expected_parts else np.asarray([], dtype=np.int32)
    actual = expected.copy()
    payload = {
        "schema_version": np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.uint16),
        "window_sample": np.asarray([row["sample"] for row in windows], dtype=np.uint16),
        "window_stage": np.asarray([row["stage"] for row in windows], dtype=np.uint8),
        "window_block": np.asarray([row["block"] for row in windows], dtype=np.uint8),
        "window_token": np.asarray([row["window"] for row in windows], dtype=np.uint16),
        "window_weight": np.asarray([row["weight"] for row in windows], dtype=np.uint16),
        "window_heads": np.asarray([row["heads"] for row in windows], dtype=np.uint8),
        "window_value_offsets": np.asarray(offsets, dtype=np.int64),
        "expected_acc32": expected,
        "actual_acc32": actual,
    }
    summaries = parse_miter_archive(payload)
    for row, summary in zip(windows, summaries, strict=True):
        row["acc32_miter_sha256"] = summary["acc32_miter_sha256"]
        row["acc32_mismatch_count"] = summary["acc32_mismatch_count"]
    return payload, bound
