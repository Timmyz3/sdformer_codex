#!/usr/bin/env python3
"""独立验证参数化 Local5 identity-service trace-v2、状态序列和 Acc32。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TOKENS = 450
LANES = 32
OUT_DIM = 32
TRACE_FIELDS = [
    "cycle", "event", "tile", "head", "source", "lane", "out", "delay",
    "index", "origin", "payload",
]
STATE_EVENTS = ("tx_state", "acc_state", "head_state")
STATE_REFERENCE_FIELDS = TRACE_FIELDS[:-1]
HEX_PAYLOAD = re.compile(r"^[0-9a-f]+$")
PASS_PREFIX = "PASS Local5 "


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def read_memh(path: Path, count: int) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    if len(rows) != count or any(len(row) != 1 or row not in "0123" for row in rows):
        raise ValueError(f"{path.name} is not an exact 2-bit delay table")
    return np.asarray([int(row, 16) for row in rows], dtype=np.uint8)


def integer_row(row: dict[str, str]) -> dict[str, int | str]:
    result: dict[str, int | str] = {
        "event": row["event"],
        "origin": row["origin"],
        "payload": row["payload"],
    }
    for name in TRACE_FIELDS:
        if name not in ("event", "origin", "payload"):
            try:
                result[name] = int(row[name])
            except ValueError as error:
                raise ValueError(f"trace {name} is not an integer") from error
    return result


def canonical_state_bytes(row: dict[str, int | str]) -> bytes:
    values = [str(row[name]) for name in STATE_REFERENCE_FIELDS]
    return ("\x1f".join(values) + "\n").encode("ascii")


def exact_metadata(
    row: dict[str, int | str],
    expected: tuple[int, int, int, int, int, int, int],
) -> bool:
    actual = tuple(int(row[name]) for name in (
        "tile", "head", "source", "lane", "out", "delay", "index",
    ))
    return actual == expected


def require_payload(row: dict[str, int | str], event: str) -> str:
    payload = str(row["payload"])
    if not HEX_PAYLOAD.fullmatch(payload):
        raise ValueError(f"{event} payload is not lowercase hexadecimal")
    return payload


def verify_state_reference(
    state_reference_path: Path,
    state_counts: Counter[str],
    state_digests: dict[str, hashlib._Hash],
    all_state_count: int,
    all_state_digest: hashlib._Hash,
    identity: dict[str, int],
) -> dict[str, Any]:
    reference = read_json(state_reference_path)
    if (
        reference.get("schema")
        not in {
            "local5_identity_service_h3_state_reference_v1",
            "local5_identity_service_state_reference_v2",
        }
        or reference.get("status")
        not in {
            "FROZEN_FROM_ACCEPTED_V7_H3_NOT_G0",
            "FROZEN_FROM_VERIFIED_IDENTITY_TRACE_NOT_G0",
        }
        or reference.get("formal_g0") != "DENY"
        or reference.get("canonical_fields") != STATE_REFERENCE_FIELDS
        or reference.get("identity") != identity
    ):
        raise ValueError("state reference contract differs")
    for name in STATE_EVENTS:
        expected = reference.get("states", {}).get(name, {})
        if (
            expected.get("count") != state_counts[name]
            or expected.get("ordered_sha256") != state_digests[name].hexdigest()
        ):
            raise ValueError(f"{name} exact count/order digest differs")
    if reference.get("all_states") != {
        "count": all_state_count,
        "ordered_sha256": all_state_digest.hexdigest(),
    }:
        raise ValueError("combined state event count/order digest differs")
    return {
        "state_reference_sha256": sha256(state_reference_path),
        "state_counts": {name: state_counts[name] for name in STATE_EVENTS},
        "state_ordered_sha256": {
            name: state_digests[name].hexdigest() for name in STATE_EVENTS
        },
        "all_state_count": all_state_count,
        "all_state_ordered_sha256": all_state_digest.hexdigest(),
    }


def verify_trace(
    trace_path: Path,
    package_dir: Path,
    state_reference_path: Path | None,
    expected_weight_hold_cycles: int,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    receipt_path = package_dir / "verification_receipt.json"
    manifest = read_json(manifest_path)
    receipt = read_json(receipt_path)
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "sample", "stage", "block", "window", "heads"
    }:
        raise ValueError("identity-service package identity schema differs")
    if any(
        isinstance(identity[name], bool)
        or not isinstance(identity[name], int)
        or identity[name] < 0
        for name in identity
    ):
        raise ValueError("identity-service package identity value differs")
    heads = int(identity["heads"])
    if not 1 <= heads <= 32:
        raise ValueError("identity-service package heads is outside contract")
    if not 0 <= expected_weight_hold_cycles <= 7:
        raise ValueError("expected weight hold cycles is outside contract")
    if (
        manifest.get("formal_g0") != "DENY"
        or receipt.get("status") != "PASS_INDEPENDENT_VERIFY_NOT_G0"
        or receipt.get("manifest_sha256") != sha256(manifest_path)
    ):
        raise ValueError("identity-service package is not independently verified")
    delays = {
        "relation": read_memh(package_dir / "relation_delay.memh", heads * TOKENS),
        "weight": read_memh(
            package_dir / "weight_delay.memh", heads * heads * LANES * OUT_DIM
        ),
        "final": read_memh(package_dir / "final_delay.memh", heads * TOKENS * OUT_DIM),
    }
    manifest_sha = sha256(manifest_path)
    receipt_sha = sha256(receipt_path)
    counts: Counter[str] = Counter()
    bindings: dict[str, str] = {}
    boundary_values: dict[str, list[tuple[int, int]]] = {
        name: [] for name in (
            "group_start", "tile_start", "head_start", "head_done", "tile_done",
            "group_done",
        )
    }
    state_counts: Counter[str] = Counter()
    state_digests = {name: hashlib.sha256() for name in STATE_EVENTS}
    all_state_digest = hashlib.sha256()
    all_state_count = 0
    relation_pending: dict[str, int | str] | None = None
    weight_pending: dict[str, int | str] | None = None
    final_pending: dict[str, int | str] | None = None
    relation_started = relation_completed = 0
    weight_started = weight_completed = 0
    final_started = final_completed = 0
    previous_cycle = -1
    weight_stall_cycles = 0
    weight_stall_pairs = 0

    with trace_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TRACE_FIELDS:
            raise ValueError("trace CSV header differs")
        for text_row in reader:
            if None in text_row or any(value is None for value in text_row.values()):
                raise ValueError("trace row has a different column count")
            row = integer_row(text_row)
            cycle = int(row["cycle"])
            event = str(row["event"])
            if cycle < previous_cycle:
                raise ValueError("trace cycle order regressed")
            previous_cycle = cycle
            counts[event] += 1
            if event in ("manifest_binding", "receipt_binding"):
                if (
                    cycle != 0
                    or event in bindings
                    or row["payload"] != "-"
                    or not exact_metadata(row, (-1, -1, -1, -1, -1, -1, -1))
                ):
                    raise ValueError("trace package binding is duplicated or malformed")
                bindings[event] = str(row["origin"])
                continue
            if event in boundary_values:
                if row["origin"] != "rtl_boundary" or row["payload"] != "-":
                    raise ValueError("boundary origin/payload differs")
                boundary_values[event].append((int(row["tile"]), int(row["head"])))
                continue
            if event in STATE_EVENTS:
                if row["origin"] != "rtl_internal_state" or row["payload"] != "-":
                    raise ValueError("state origin/payload differs")
                payload = canonical_state_bytes(row)
                state_digests[event].update(payload)
                all_state_digest.update(payload)
                state_counts[event] += 1
                all_state_count += 1
                continue
            if event == "weight_response_stall":
                if (
                    row["origin"] != "rtl_protocol_telemetry"
                    or weight_pending is None
                    or "available" not in weight_pending
                ):
                    raise ValueError("weight stall is outside one available response")
                expected = (
                    int(weight_pending["tile"]), int(weight_pending["head"]), -1,
                    int(weight_pending["lane"]), int(weight_pending["out"]),
                    int(weight_pending["delay"]), int(weight_pending["flat"]),
                )
                expected_cycle = int(weight_pending["available"]) + int(
                    weight_pending.get("stall_count", 0)
                )
                if (
                    cycle != expected_cycle
                    or not exact_metadata(row, expected)
                    or require_payload(row, event) != weight_pending["payload"]
                ):
                    raise ValueError("weight held-valid cycle is not contiguous/stable")
                weight_pending["stall_count"] = int(
                    weight_pending.get("stall_count", 0)
                ) + 1
                weight_stall_cycles += 1
                continue
            if row["origin"] != "rtl_handshake":
                raise ValueError(f"handshake origin differs for {event}")

            if event == "relation_accept":
                if relation_pending is not None or row["payload"] != "-":
                    raise ValueError("relation request overlapped or carried payload")
                index = relation_started
                tile = index // (heads * TOKENS)
                head = (index // TOKENS) % heads
                source = index % TOKENS
                flat = head * TOKENS + source
                delay = int(delays["relation"][flat])
                if not exact_metadata(row, (tile, head, source, -1, -1, delay, flat)):
                    raise ValueError("relation accept identity/index/delay differs")
                relation_pending = {
                    "cycle": cycle, "flat": flat, "delay": delay,
                    "tile": tile, "head": head, "source": source,
                }
                relation_started += 1
            elif event == "relation_response_available":
                if relation_pending is None or "available" in relation_pending:
                    raise ValueError("relation available without one request")
                expected = (
                    -1, int(relation_pending["head"]),
                    int(relation_pending["source"]), -1, -1,
                    int(relation_pending["delay"]), int(relation_pending["flat"]),
                )
                if (
                    cycle - int(relation_pending["cycle"])
                    != int(relation_pending["delay"]) + 1
                    or not exact_metadata(row, expected)
                ):
                    raise ValueError("relation response-available contract differs")
                relation_pending["available"] = cycle
                relation_pending["payload"] = require_payload(row, event)
            elif event == "relation_response_accept":
                if relation_pending is None or "available" not in relation_pending:
                    raise ValueError("relation response accepted before available")
                expected = (
                    -1, int(relation_pending["head"]),
                    int(relation_pending["source"]), -1, -1,
                    int(relation_pending["delay"]), int(relation_pending["flat"]),
                )
                if (
                    cycle < int(relation_pending["available"])
                    or not exact_metadata(row, expected)
                    or require_payload(row, event) != relation_pending["payload"]
                ):
                    raise ValueError("relation response-accept metadata/payload differs")
                relation_pending = None
                relation_completed += 1
            elif event == "weight_accept":
                if weight_pending is not None or row["payload"] != "-":
                    raise ValueError("weight request overlapped or carried payload")
                index = weight_started
                tile = index // (heads * LANES * OUT_DIM)
                head = (index // (LANES * OUT_DIM)) % heads
                lane = (index // OUT_DIM) % LANES
                out = index % OUT_DIM
                delay = int(delays["weight"][index])
                if not exact_metadata(row, (tile, head, -1, lane, out, delay, index)):
                    raise ValueError("weight accept identity/index/delay differs")
                weight_pending = {
                    "cycle": cycle, "flat": index, "delay": delay,
                    "tile": tile, "head": head, "lane": lane, "out": out,
                }
                weight_started += 1
            elif event == "weight_response_available":
                if weight_pending is None or "available" in weight_pending:
                    raise ValueError("weight available without one request")
                expected = (
                    int(weight_pending["tile"]), int(weight_pending["head"]), -1,
                    int(weight_pending["lane"]), int(weight_pending["out"]),
                    int(weight_pending["delay"]), int(weight_pending["flat"]),
                )
                if (
                    cycle - int(weight_pending["cycle"])
                    != int(weight_pending["delay"]) + 1
                    or not exact_metadata(row, expected)
                ):
                    raise ValueError("weight response-available contract differs")
                weight_pending["available"] = cycle
                weight_pending["payload"] = require_payload(row, event)
                weight_pending["stall_count"] = 0
            elif event == "weight_response_accept":
                if weight_pending is None or "available" not in weight_pending:
                    raise ValueError("weight response accepted before available")
                expected = (
                    int(weight_pending["tile"]), int(weight_pending["head"]), -1,
                    int(weight_pending["lane"]), int(weight_pending["out"]),
                    int(weight_pending["delay"]), int(weight_pending["flat"]),
                )
                if (
                    cycle < int(weight_pending["available"])
                    or not exact_metadata(row, expected)
                    or require_payload(row, event) != weight_pending["payload"]
                    or int(weight_pending.get("stall_count", 0))
                    != expected_weight_hold_cycles
                ):
                    raise ValueError("weight response-accept metadata/payload differs")
                if expected_weight_hold_cycles:
                    weight_stall_pairs += 1
                weight_pending = None
                weight_completed += 1
            elif event == "final_request":
                if final_pending is not None:
                    raise ValueError("final request overlapped")
                index = final_started
                tile = index // (TOKENS * OUT_DIM)
                source = (index // OUT_DIM) % TOKENS
                out = index % OUT_DIM
                delay = int(delays["final"][index])
                if not exact_metadata(row, (tile, -1, source, -1, out, delay, index)):
                    raise ValueError("final request identity/index/delay differs")
                final_pending = {
                    "cycle": cycle, "delay": delay, "flat": index,
                    "tile": tile, "source": source, "out": out,
                    "payload": require_payload(row, event),
                }
                final_started += 1
            elif event == "final_accept":
                if final_pending is None:
                    raise ValueError("final accepted without request")
                expected = (
                    int(final_pending["tile"]), -1, int(final_pending["source"]),
                    -1, int(final_pending["out"]), int(final_pending["delay"]),
                    int(final_pending["flat"]),
                )
                if (
                    cycle - int(final_pending["cycle"])
                    != int(final_pending["delay"]) + 1
                    or not exact_metadata(row, expected)
                    or require_payload(row, event) != final_pending["payload"]
                ):
                    raise ValueError("final response latency/metadata/payload differs")
                final_pending = None
                final_completed += 1
            else:
                raise ValueError(f"unexpected trace event {event}")

    if bindings != {
        "manifest_binding": manifest_sha,
        "receipt_binding": receipt_sha,
    }:
        raise ValueError("trace package SHA bindings differ")
    if any(value is not None for value in (relation_pending, weight_pending, final_pending)):
        raise ValueError("trace ended with a pending transaction")
    if (relation_started, relation_completed) != (heads * heads * TOKENS,) * 2:
        raise ValueError("relation transaction count differs")
    if (weight_started, weight_completed) != (heads * heads * LANES * OUT_DIM,) * 2:
        raise ValueError("weight transaction count differs")
    if (final_started, final_completed) != (heads * TOKENS * OUT_DIM,) * 2:
        raise ValueError("final transaction count differs")
    expected_boundaries = {
        "group_start": [(-1, -1)],
        "tile_start": [(tile, -1) for tile in range(heads)],
        "head_start": [(tile, head) for tile in range(heads) for head in range(heads)],
        "head_done": [(tile, head) for tile in range(heads) for head in range(heads)],
        "tile_done": [(tile, -1) for tile in range(heads)],
        "group_done": [(-1, -1)],
    }
    if boundary_values != expected_boundaries:
        raise ValueError("group/tile/head boundary sequence differs")
    if state_reference_path is None:
        state = {
            "status": "PASS_DERIVED_STATE_LEDGER_NO_FROZEN_REFERENCE",
            "state_counts": {name: state_counts[name] for name in STATE_EVENTS},
            "state_ordered_sha256": {
                name: state_digests[name].hexdigest() for name in STATE_EVENTS
            },
            "all_state_count": all_state_count,
            "all_state_ordered_sha256": all_state_digest.hexdigest(),
        }
    else:
        state = verify_state_reference(
            state_reference_path.resolve(), state_counts, state_digests,
            all_state_count, all_state_digest, identity,
        )
    return {
        "status": "PASS_IDENTITY_SERVICE_RTL_TRACE_V2_NOT_G0",
        "evidence": "[rtl]+[软件确定性服务合同]",
        "formal_g0": "DENY",
        "identity": identity,
        "trace_sha256": sha256(trace_path),
        "trace_rows": sum(counts.values()),
        "event_counts": dict(sorted(counts.items())),
        "last_cycle": previous_cycle,
        "manifest_sha256": manifest_sha,
        "verification_receipt_sha256": receipt_sha,
        "payload_stability": {
            "relation_pairs": relation_completed,
            "weight_pairs": weight_completed,
            "final_pairs": final_completed,
            "status": "PASS_EXACT_AVAILABLE_ACCEPT_PAYLOAD",
            "weight_hold_cycles_per_response": expected_weight_hold_cycles,
            "weight_held_valid_pairs": weight_stall_pairs,
            "weight_valid1_ready0_cycles": weight_stall_cycles,
        },
        "state_reference": state,
    }


def verify_acc32(actual_path: Path, expected_path: Path, heads: int) -> dict[str, Any]:
    rows = actual_path.read_text(encoding="ascii").splitlines()
    if len(rows) != heads * TOKENS * OUT_DIM or any(len(row) != 8 for row in rows):
        raise ValueError("actual Acc32 memh shape differs")
    actual_u32 = np.asarray([int(row, 16) for row in rows], dtype=np.uint32)
    actual = actual_u32.view(np.int32)
    with np.load(expected_path, allow_pickle=False) as archive:
        if set(archive.files) != {"schema_version", "expected_acc32"}:
            raise ValueError("software expected NPZ schema differs")
        expected = archive["expected_acc32"]
    if expected.dtype != np.int32 or expected.shape != actual.shape:
        raise ValueError("software expected Acc32 shape/dtype differs")
    difference = actual.astype(np.int64) - expected.astype(np.int64)
    mismatch = int(np.count_nonzero(difference))
    maximum = int(np.max(np.abs(difference), initial=0))
    if mismatch or maximum:
        raise ValueError("RTL Acc32 differs from software integer reference")
    return {
        "actual_acc32_sha256": sha256(actual_path),
        "expected_npz_sha256": sha256(expected_path),
        "scalars": int(actual.size),
        "mismatch": mismatch,
        "max_abs_error": maximum,
    }


def verify_log(path: Path, identity: dict[str, int]) -> int:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(PASS_PREFIX)
    ]
    if len(lines) != 1:
        raise ValueError("Verilator log lacks one exact identity-service PASS line")
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
        raise ValueError("Verilator PASS line identity/count fields differ")
    try:
        return int(fields["cycles"])
    except (KeyError, ValueError) as error:
        raise ValueError("Verilator PASS line cycles field differs") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--state-reference", type=Path)
    parser.add_argument("--expected-weight-hold-cycles", type=int, default=0)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--verilator-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_report = verify_trace(
        args.trace.resolve(), args.package_dir.resolve(),
        args.state_reference.resolve() if args.state_reference else None,
        args.expected_weight_hold_cycles,
    )
    identity = trace_report["identity"]
    report = {
        **trace_report,
        "acc32": verify_acc32(
            args.actual.resolve(), args.expected.resolve(), identity["heads"]
        ),
        "rtl_cycles": verify_log(args.verilator_log.resolve(), identity),
        "verilator_log_sha256": sha256(args.verilator_log.resolve()),
        "boundary": [
            "cycles are validation-environment latency, not architecture performance",
            "single parameterized Local5 window canary only",
            "no formal G0, full encoder, PPA, throughput, or energy claim",
        ],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"verification output exists: {output}")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
