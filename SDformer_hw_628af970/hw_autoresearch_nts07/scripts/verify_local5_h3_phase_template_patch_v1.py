#!/usr/bin/env python3
"""独立展开并验证参数化 Local5 phase-template + tile-patch canary。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
ARRAY_DTYPES = {
    "schema_version": np.dtype("uint16"),
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
PASS_PREFIX = "PASS Local5 "
GROUP_TAG = 0x5D5000
TOKENS = 450
LANES = 32
OUT_DIM = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_bytes(array: np.ndarray, name: str) -> list[str]:
    result = []
    for index, value in enumerate(array):
        raw = bytes(value).rstrip(b"\x00")
        if not raw or b"\x00" in raw:
            raise ValueError(f"{name}[{index}] is empty or has embedded NUL")
        try:
            result.append(raw.decode("ascii"))
        except UnicodeDecodeError as error:
            raise ValueError(f"{name}[{index}] is not ASCII") from error
    if len(result) != len(set(result)):
        raise ValueError(f"{name} dictionary is not unique")
    return result


def read_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        if set(handle.files) != set(ARRAY_DTYPES):
            raise ValueError("archive NPZ member set differs")
        arrays = {name: np.array(handle[name], copy=True) for name in handle.files}
    for name, dtype in ARRAY_DTYPES.items():
        value = arrays[name]
        if value.dtype != dtype or value.ndim != 1:
            raise ValueError(f"archive {name} dtype/rank differs")
    if arrays["schema_version"].tolist() != [1]:
        raise ValueError("archive schema version differs")
    if decode_bytes(arrays["class_name"], "class_name") != list(CLASS_ORDER):
        raise ValueError("archive class order differs")
    return arrays


def validate_offsets(offsets: np.ndarray, expected_last: int, name: str) -> None:
    if (
        len(offsets) < 2
        or int(offsets[0]) != 0
        or int(offsets[-1]) != expected_last
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"{name} offsets differ")


def expanded_rows(arrays: dict[str, np.ndarray]) -> Iterator[dict[str, str]]:
    events = decode_bytes(arrays["event_dictionary"], "event_dictionary")
    origins = decode_bytes(arrays["origin_dictionary"], "origin_dictionary")
    payloads = decode_bytes(arrays["payload_dictionary"], "payload_dictionary")
    template_offsets = arrays["template_offsets"]
    patch_offsets = arrays["patch_offsets"]
    instance_count = len(arrays["instance_class_code"])
    heads_array = arrays["heads"]
    if heads_array.shape != (1,) or not 1 <= int(heads_array[0]) <= 32:
        raise ValueError("archive heads contract differs")
    heads = int(heads_array[0])
    if (
        len(template_offsets) != len(CLASS_ORDER) + 1
        or len(patch_offsets) != instance_count + 1
        or len(arrays["instance_tile"]) != instance_count
        or len(arrays["instance_head"]) != instance_count
    ):
        raise ValueError("archive instance/template offset shapes differ")
    validate_offsets(
        template_offsets, len(arrays["template_event_code"]), "template"
    )
    if len(arrays["template_origin_code"]) != len(arrays["template_event_code"]):
        raise ValueError("template event/origin arrays differ")
    patch_count = len(arrays["patch_cycle"])
    validate_offsets(patch_offsets, patch_count, "patch")
    for name in (
        "patch_tile", "patch_head", "patch_source", "patch_lane", "patch_out",
        "patch_delay", "patch_index", "patch_payload_code",
    ):
        if len(arrays[name]) != patch_count:
            raise ValueError(f"{name} length differs")
    if (
        len(arrays["template_event_code"])
        and int(np.max(arrays["template_event_code"])) >= len(events)
    ) or (
        len(arrays["template_origin_code"])
        and int(np.max(arrays["template_origin_code"])) >= len(origins)
    ) or (
        patch_count and int(np.max(arrays["patch_payload_code"])) >= len(payloads)
    ):
        raise ValueError("archive dictionary code is out of range")

    expected_instances: list[tuple[int, int, int]] = [
        (CLASS_ORDER.index("prefix"), -1, -1)
    ]
    for tile in range(heads):
        for head in range(heads):
            expected_instances.append((
                CLASS_ORDER.index("head_seed" if head == 0 else "head_accumulate"),
                tile,
                head,
            ))
            if head + 1 < heads:
                expected_instances.append((
                    CLASS_ORDER.index("inter_head_gap"), tile, head
                ))
            else:
                expected_instances.append((
                    CLASS_ORDER.index("tile_tail"), tile, head
                ))
        if tile + 1 < heads:
            expected_instances.append((
                CLASS_ORDER.index("tile_transition"), tile + 1, -1
            ))
    expected_instances.append((CLASS_ORDER.index("suffix"), -1, -1))
    observed_instances = [
        (
            int(arrays["instance_class_code"][index]),
            int(arrays["instance_tile"][index]),
            int(arrays["instance_head"][index]),
        )
        for index in range(instance_count)
    ]
    if observed_instances != expected_instances:
        raise ValueError("instance class/tile/head typed metadata sequence differs")

    for instance in range(instance_count):
        class_code = int(arrays["instance_class_code"][instance])
        if not 0 <= class_code < len(CLASS_ORDER):
            raise ValueError("instance class code is out of range")
        template_start = int(template_offsets[class_code])
        template_stop = int(template_offsets[class_code + 1])
        patch_start = int(patch_offsets[instance])
        patch_stop = int(patch_offsets[instance + 1])
        if template_stop - template_start != patch_stop - patch_start:
            raise ValueError("instance patch length does not match its template")
        for relative, patch_index in enumerate(range(patch_start, patch_stop)):
            template_index = template_start + relative
            yield {
                "cycle": str(int(arrays["patch_cycle"][patch_index])),
                "event": events[int(arrays["template_event_code"][template_index])],
                "tile": str(int(arrays["patch_tile"][patch_index])),
                "head": str(int(arrays["patch_head"][patch_index])),
                "source": str(int(arrays["patch_source"][patch_index])),
                "lane": str(int(arrays["patch_lane"][patch_index])),
                "out": str(int(arrays["patch_out"][patch_index])),
                "delay": str(int(arrays["patch_delay"][patch_index])),
                "index": str(int(arrays["patch_index"][patch_index])),
                "origin": origins[int(arrays["template_origin_code"][template_index])],
                "payload": payloads[int(arrays["patch_payload_code"][patch_index])],
            }


def verify_expansion(
    source_trace: Path, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    bound_sha = decode_bytes(arrays["source_trace_sha256"], "source_trace_sha256")
    if bound_sha != [sha256(source_trace)]:
        raise ValueError("archive source trace SHA binding differs")
    digest = hashlib.sha256()
    digest.update((",".join(FIELDS) + "\n").encode("ascii"))
    row_count = 0
    expanded = expanded_rows(arrays)
    with source_trace.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("source trace header differs")
        for expected in reader:
            try:
                actual = next(expanded)
            except StopIteration as error:
                raise ValueError("archive expansion ended early") from error
            if actual != expected:
                raise ValueError(f"archive expansion differs at row {row_count}")
            digest.update((",".join(actual[name] for name in FIELDS) + "\n").encode("ascii"))
            row_count += 1
    try:
        next(expanded)
    except StopIteration:
        pass
    else:
        raise ValueError("archive expansion has extra rows")
    if digest.hexdigest() != sha256(source_trace):
        raise ValueError("expanded byte-stream SHA differs from source trace")
    return {"rows": row_count, "expanded_trace_sha256": digest.hexdigest()}


def semantic_ledgers(path: Path) -> dict[str, dict[str, Any]]:
    groups = {
        "handshake": ("rtl_handshake", hashlib.sha256(), 0),
        "boundary": ("rtl_boundary", hashlib.sha256(), 0),
        "state": ("rtl_internal_state", hashlib.sha256(), 0),
        "telemetry": ("rtl_protocol_telemetry", hashlib.sha256(), 0),
    }
    core_all_digest = hashlib.sha256()
    core_all_count = 0
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("semantic ledger trace header differs")
        for row in reader:
            for name, (origin, digest, count) in list(groups.items()):
                if row["origin"] != origin:
                    continue
                fields = [row[field] for field in FIELDS if field != "cycle"]
                digest.update(("\x1f".join(fields) + "\n").encode("ascii"))
                groups[name] = (origin, digest, count + 1)
                if origin != "rtl_protocol_telemetry":
                    core_all_digest.update(
                        ("\x1f".join(fields) + "\n").encode("ascii")
                    )
                    core_all_count += 1
                break
    result = {
        name: {"count": count, "cycle_free_ordered_sha256": digest.hexdigest()}
        for name, (_, digest, count) in groups.items()
    }
    result["core_all"] = {
        "count": core_all_count,
        "cycle_free_ordered_sha256": core_all_digest.hexdigest(),
    }
    return result


def verify_trace_bindings(path: Path, manifest: Path, receipt: Path) -> None:
    expected = {
        "manifest_binding": sha256(manifest),
        "receipt_binding": sha256(receipt),
    }
    observed: dict[str, str] = {}
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["event"] not in expected:
                continue
            if (
                row["event"] in observed
                or row["cycle"] != "0"
                or row["payload"] != "-"
            ):
                raise ValueError("trace package binding is duplicated or malformed")
            observed[row["event"]] = row["origin"]
    if observed != expected:
        raise ValueError("trace identity package SHA bindings differ")


def unpack(payload: str, fields_lsb: tuple[tuple[str, int], ...]) -> dict[str, int]:
    value = int(payload, 16)
    result = {}
    for name, width in fields_lsb:
        result[name] = value & ((1 << width) - 1)
        value >>= width
    if value:
        raise ValueError("payload has nonzero bits above its frozen width")
    return result


def load_inputs(
    path: Path, heads: int
) -> dict[tuple[int, int], tuple[int, tuple[int, ...], int]]:
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 11:
            raise ValueError("combined input row width differs")
        head, plane, y, x = map(int, fields[:4])
        source = (plane * 15 + y) * 15 + x
        q = int(fields[4], 16)
        k = tuple(int(value, 16) for value in fields[5:10])
        mask = int(fields[10], 16)
        key = (head, source)
        if key in result:
            raise ValueError("combined input identity duplicated")
        result[key] = (q, k, mask)
    if len(result) != heads * TOKENS:
        raise ValueError("combined input count differs")
    return result


def load_weights(path: Path, heads: int) -> dict[tuple[int, int, int, int], int]:
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("weight row width differs")
        head, tile, lane, out = map(int, fields[:4])
        key = (head, tile, lane, out)
        if key in result:
            raise ValueError("weight identity duplicated")
        result[key] = int(fields[4], 16)
    if len(result) != heads * heads * LANES * OUT_DIM:
        raise ValueError("weight count differs")
    return result


def verify_payloads_and_weight_stalls(
    path: Path, inputs_path: Path, weights_path: Path, actual_path: Path,
    heads: int,
) -> dict[str, Any]:
    inputs = load_inputs(inputs_path, heads)
    weights = load_weights(weights_path, heads)
    actual_rows = actual_path.read_text(encoding="ascii").splitlines()
    if len(actual_rows) != heads * TOKENS * OUT_DIM:
        raise ValueError("actual Acc32 count differs")
    actual = [int(value, 16) for value in actual_rows]
    relation_available = weight_available = final_request = 0
    weight_pending: dict[str, Any] | None = None
    stall_histogram: Counter[int] = Counter()
    valid_ready0_cycles = 0
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event = row["event"]
            if event == "relation_response_available":
                head = int(row["head"])
                source = int(row["source"])
                tile = relation_available // (heads * TOKENS)
                q, k_values, mask = inputs[(head, source)]
                value = unpack(row["payload"], (
                    ("error", 1), ("mask", 5), ("k", 160), ("q", 32),
                    ("source", 9), ("head", 5), ("tag", 24),
                ))
                packed_k = sum(k_value << (32 * index) for index, k_value in enumerate(k_values))
                expected = {
                    "error": 0, "mask": mask, "k": packed_k, "q": q,
                    "source": source, "head": head, "tag": GROUP_TAG + tile,
                }
                if value != expected:
                    raise ValueError("relation response payload fields differ from input table")
                relation_available += 1
            elif event == "weight_response_available":
                key = (
                    int(row["head"]), int(row["tile"]),
                    int(row["lane"]), int(row["out"]),
                )
                value = unpack(row["payload"], (
                    ("error", 1), ("data", 8), ("out", 5), ("lane", 5),
                    ("tile", 5), ("head", 5), ("tag", 24),
                ))
                expected = {
                    "error": 0, "data": weights[key], "out": key[3],
                    "lane": key[2], "tile": key[1], "head": key[0],
                    "tag": GROUP_TAG + key[1],
                }
                if value != expected or weight_pending is not None:
                    raise ValueError("weight response payload fields or ordering differ")
                weight_pending = {
                    "cycle": int(row["cycle"]), "payload": row["payload"],
                    "key": key, "stall_cycles": [],
                }
                weight_available += 1
            elif event == "weight_response_stall":
                key = (
                    int(row["head"]), int(row["tile"]),
                    int(row["lane"]), int(row["out"]),
                )
                if (
                    weight_pending is None
                    or row["origin"] != "rtl_protocol_telemetry"
                    or row["payload"] != weight_pending["payload"]
                    or key != weight_pending["key"]
                ):
                    raise ValueError("weight valid1/ready0 telemetry differs")
                weight_pending["stall_cycles"].append(int(row["cycle"]))
                valid_ready0_cycles += 1
            elif event == "weight_response_accept":
                if weight_pending is None or row["payload"] != weight_pending["payload"]:
                    raise ValueError("weight accept lacks matching available payload")
                delta = int(row["cycle"]) - weight_pending["cycle"]
                if weight_pending["stall_cycles"] != list(
                    range(weight_pending["cycle"], int(row["cycle"]))
                ):
                    raise ValueError("weight valid1/ready0 cycles are not contiguous")
                stall_histogram[delta] += 1
                weight_pending = None
            elif event == "final_request":
                tile = int(row["tile"])
                source = int(row["source"])
                out = int(row["out"])
                index = int(row["index"])
                plane, spatial = divmod(source, 225)
                y, x = divmod(spatial, 15)
                value = unpack(row["payload"], (
                    ("last", 1), ("data", 32), ("out", 5), ("x", 4),
                    ("y", 4), ("plane", 1), ("tile", 5), ("tag", 24),
                ))
                expected = {
                    "last": int(source == TOKENS - 1 and out == OUT_DIM - 1),
                    "data": actual[index], "out": out, "x": x, "y": y,
                    "plane": plane, "tile": tile, "tag": GROUP_TAG + tile,
                }
                if value != expected:
                    raise ValueError("final payload fields differ from actual Acc32 stream")
                final_request += 1
    if weight_pending is not None:
        raise ValueError("trace ended with pending weight response")
    if (
        relation_available != heads * heads * TOKENS
        or weight_available != heads * heads * LANES * OUT_DIM
        or final_request != heads * TOKENS * OUT_DIM
        or sum(stall_histogram.values()) != weight_available
        or not any(delta > 0 for delta in stall_histogram)
        or valid_ready0_cycles != sum(
            delta * count for delta, count in stall_histogram.items()
        )
    ):
        raise ValueError("payload count or directed weight backpressure coverage differs")
    return {
        "relation_payloads_decoded": relation_available,
        "weight_payloads_decoded": weight_available,
        "final_payloads_decoded": final_request,
        "weight_available_accept_delta_histogram": {
            str(key): value for key, value in sorted(stall_histogram.items())
        },
        "weight_held_valid_pairs": sum(
            value for key, value in stall_histogram.items() if key > 0
        ),
        "weight_valid1_ready0_cycles": valid_ready0_cycles,
    }


def verify_acc32(actual_path: Path, baseline_path: Path, expected_path: Path) -> dict[str, Any]:
    actual_text = actual_path.read_text(encoding="ascii").splitlines()
    baseline_text = baseline_path.read_text(encoding="ascii").splitlines()
    if actual_text != baseline_text:
        raise ValueError("candidate Acc32 differs byte-for-byte from v8 baseline")
    actual = np.asarray([int(value, 16) for value in actual_text], dtype=np.uint32).view(np.int32)
    with np.load(expected_path, allow_pickle=False) as handle:
        expected = handle["expected_acc32"]
    if expected.dtype != np.int32 or not np.array_equal(actual, expected):
        raise ValueError("candidate Acc32 differs from software integer reference")
    return {
        "scalars": int(actual.size),
        "candidate_sha256": sha256(actual_path),
        "baseline_v8_sha256": sha256(baseline_path),
        "expected_npz_sha256": sha256(expected_path),
        "mismatch": 0,
    }


def verify_log(path: Path, identity: dict[str, int]) -> int:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(PASS_PREFIX)
    ]
    if len(lines) != 1:
        raise ValueError("candidate Verilator log lacks one exact PASS line")
    fields = {
        key: value
        for token in lines[0].split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }
    heads = identity["heads"]
    expected = {
        "transaction_service": "0", "identity_service": "1",
        "seed": "20260810", "stage": str(identity["stage"]),
        "block": str(identity["block"]), "window": str(identity["window"]),
        "token": str(heads * heads * TOKENS),
        "result_service": str(heads * TOKENS * OUT_DIM),
        "final": str(heads * TOKENS * OUT_DIM),
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate PASS identity/count fields differ")
    try:
        return int(fields["cycles"])
    except (KeyError, ValueError) as error:
        raise ValueError("candidate PASS cycles field differs") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-actual", type=Path, required=True)
    parser.add_argument("--baseline-actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--verilator-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("schema") not in {
            "local5_h3_phase_template_patch_manifest_v1",
            "local5_phase_template_patch_manifest_v2",
        }
        or manifest.get("formal_g0") != "DENY"
        or manifest.get("archive_sha256") != sha256(paths["archive"])
        or manifest.get("source_trace_sha256") != sha256(paths["candidate_trace"])
    ):
        raise ValueError("template manifest binding differs")
    arrays = read_archive(paths["archive"])
    heads = int(arrays["heads"][0])
    identity = manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"sample", "stage", "block", "window", "heads"}
        or any(
            isinstance(identity[name], bool)
            or not isinstance(identity[name], int)
            or identity[name] < 0
            for name in identity
        )
        or identity["heads"] != heads
    ):
        raise ValueError("template manifest identity differs")
    expansion = verify_expansion(paths["candidate_trace"], arrays)
    verify_trace_bindings(
        paths["candidate_trace"], paths["identity_manifest"],
        paths["identity_receipt"],
    )
    candidate_ledgers = semantic_ledgers(paths["candidate_trace"])
    baseline_ledgers = semantic_ledgers(paths["baseline_trace"])
    comparable_keys = ("handshake", "boundary", "state", "core_all")
    if any(candidate_ledgers[key] != baseline_ledgers[key] for key in comparable_keys):
        raise ValueError("candidate cycle-free transaction/boundary/state ledger differs from baseline")
    payload = verify_payloads_and_weight_stalls(
        paths["candidate_trace"], paths["inputs"], paths["weights"],
        paths["candidate_actual"], heads,
    )
    validation_cycles = verify_log(paths["verilator_log"], identity)
    report = {
        "schema": "local5_phase_template_patch_verification_v2",
        "status": "PASS_PHASE_TEMPLATE_TILE_PATCH_NOT_G0",
        "evidence": "[rtl]+[独立软件展开验证]",
        "formal_g0": "DENY",
        "identity": identity,
        "expansion": expansion,
        "semantic_ledgers": candidate_ledgers,
        "payload_and_backpressure": payload,
        "acc32": verify_acc32(
            paths["candidate_actual"], paths["baseline_actual"], paths["expected"]
        ),
        "candidate_validation_cycles": validation_cycles,
        "archive": {
            "sha256": sha256(paths["archive"]),
            "file_bytes": paths["archive"].stat().st_size,
            "source_trace_bytes": paths["candidate_trace"].stat().st_size,
            "file_size_reduction": paths["candidate_trace"].stat().st_size
            / paths["archive"].stat().st_size,
            "base_event_reuse_factor": manifest["base_event_reuse_factor"],
            "template_rows": manifest["template_rows"],
            "expanded_rows": manifest["expanded_rows"],
        },
        "bindings": {name: sha256(path) for name, path in paths.items() if name != "output"},
        "boundary": [
            "单个参数化窗口；模板容量包含 typed tile patch",
            "压缩率是验证 archive 文件大小，不是片上存储或架构性能",
            "formal G0、full encoder 和 ASIC PPA 均未通过",
        ],
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "archive": report["archive"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
