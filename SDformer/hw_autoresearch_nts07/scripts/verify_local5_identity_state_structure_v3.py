#!/usr/bin/env python3
"""用解析结构 oracle 验证 Local5 identity trace 的 cycle-free state 全序。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


STATE_EVENTS = (b"tx_state", b"acc_state", b"head_state")
CANONICAL_FIELDS = (
    "event", "tile", "head", "source", "lane", "out", "delay", "index",
    "origin",
)
TOKENS = 450
HEAD_DIM = 32
OUT_DIM = 32
WEIGHTS_PER_JOB = HEAD_DIM * OUT_DIM
RESULTS_PER_JOB = TOKENS * OUT_DIM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def state_row(event: bytes, tile: int, head: int, state: int) -> tuple[bytes, ...]:
    return (
        event, str(tile).encode("ascii"), str(head).encode("ascii"), b"-1",
        b"-1", b"-1", b"-1", str(state).encode("ascii"),
        b"rtl_internal_state",
    )


def expected_tx_states(heads: int) -> Iterator[tuple[bytes, ...]]:
    yield state_row(b"tx_state", 0, -1, 0)
    for tile in range(heads):
        for _head in range(heads):
            yield state_row(b"tx_state", tile, -1, 1)
            yield state_row(b"tx_state", tile, -1, 2)
            yield state_row(b"tx_state", tile, -1, 3)
        for _result in range(RESULTS_PER_JOB):
            yield state_row(b"tx_state", tile, -1, 4)
            yield state_row(b"tx_state", tile, -1, 5)
            yield state_row(b"tx_state", tile, -1, 6)
        yield state_row(b"tx_state", tile, -1, 7)
        yield state_row(b"tx_state", tile, -1, 0)


def expected_acc_states(heads: int) -> Iterator[tuple[bytes, ...]]:
    yield state_row(b"acc_state", 0, -1, 0)
    for tile in range(heads):
        for _head in range(1, heads):
            for _result in range(RESULTS_PER_JOB):
                yield state_row(b"acc_state", tile, -1, 1)
                yield state_row(b"acc_state", tile, -1, 0)


def expected_head_states(heads: int) -> Iterator[tuple[bytes, ...]]:
    yield state_row(b"head_state", 0, 0, 0)
    for tile in range(heads):
        for head in range(heads):
            for _weight in range(WEIGHTS_PER_JOB):
                yield state_row(b"head_state", tile, head, 1)
                yield state_row(b"head_state", tile, head, 2)
            yield state_row(b"head_state", tile, head, 3)
            for _plane in range(2):
                yield state_row(b"head_state", tile, head, 4)
                for _token in range(TOKENS // 2):
                    yield state_row(b"head_state", tile, head, 5)
                    yield state_row(b"head_state", tile, head, 6)
            yield state_row(b"head_state", tile, head, 7)
            yield state_row(b"head_state", tile, head, 8)
            yield state_row(b"head_state", tile, head, 9)
            for _result in range(RESULTS_PER_JOB):
                yield state_row(b"head_state", tile, head, 10)
                yield state_row(b"head_state", tile, head, 11)
                yield state_row(b"head_state", tile, head, 12)
            yield state_row(b"head_state", tile, head, 13)
            yield state_row(b"head_state", tile, head, 14)
            yield state_row(b"head_state", tile, head, 0)


def expected_head_body_states(
    tile: int,
    head: int,
) -> Iterator[tuple[bytes, ...]]:
    for _weight in range(WEIGHTS_PER_JOB):
        yield state_row(b"head_state", tile, head, 1)
        yield state_row(b"head_state", tile, head, 2)
    yield state_row(b"head_state", tile, head, 3)
    for _plane in range(2):
        yield state_row(b"head_state", tile, head, 4)
        for _token in range(TOKENS // 2):
            yield state_row(b"head_state", tile, head, 5)
            yield state_row(b"head_state", tile, head, 6)
    yield state_row(b"head_state", tile, head, 7)
    yield state_row(b"head_state", tile, head, 8)
    yield state_row(b"head_state", tile, head, 9)
    yield state_row(b"head_state", tile, head, 10)
    yield state_row(b"head_state", tile, head, 11)
    yield state_row(b"head_state", tile, head, 12)
    if head == 0:
        for _result in range(1, RESULTS_PER_JOB):
            yield state_row(b"head_state", tile, head, 10)
            yield state_row(b"head_state", tile, head, 11)
            yield state_row(b"head_state", tile, head, 12)
        yield state_row(b"head_state", tile, head, 13)
        yield state_row(b"head_state", tile, head, 14)
    else:
        for _result in range(1, RESULTS_PER_JOB):
            yield state_row(b"acc_state", tile, -1, 1)
            yield state_row(b"head_state", tile, head, 10)
            yield state_row(b"acc_state", tile, -1, 0)
            yield state_row(b"head_state", tile, head, 11)
            yield state_row(b"head_state", tile, head, 12)
        yield state_row(b"acc_state", tile, -1, 1)
        yield state_row(b"head_state", tile, head, 13)
        yield state_row(b"acc_state", tile, -1, 0)
        yield state_row(b"head_state", tile, head, 14)


def expected_all_states(heads: int) -> Iterator[tuple[bytes, ...]]:
    yield state_row(b"tx_state", 0, -1, 0)
    yield state_row(b"acc_state", 0, -1, 0)
    yield state_row(b"head_state", 0, 0, 0)
    for tile in range(heads):
        for head in range(heads):
            yield state_row(b"tx_state", tile, -1, 1)
            yield state_row(b"tx_state", tile, -1, 2)
            yield from expected_head_body_states(tile, head)
            yield state_row(b"tx_state", tile, -1, 3)
            yield state_row(b"head_state", tile, head, 0)
        for _result in range(RESULTS_PER_JOB):
            yield state_row(b"tx_state", tile, -1, 4)
            yield state_row(b"tx_state", tile, -1, 5)
            yield state_row(b"tx_state", tile, -1, 6)
        yield state_row(b"tx_state", tile, -1, 7)
        yield state_row(b"tx_state", tile, -1, 0)


def expected_counts(heads: int) -> dict[str, int]:
    return {
        "tx_state": 3 * heads * heads + 43_202 * heads + 1,
        "acc_state": 28_800 * heads * heads - 28_800 * heads + 1,
        "head_state": 46_157 * heads * heads + 1,
    }


def canonical_bytes(row: tuple[bytes, ...]) -> bytes:
    return b"\x1f".join(row) + b"\n"


def verify_trace(
    trace: Path,
    identity: dict[str, int],
    cross_reference: Path | None,
    parent_complete_path: Path | None = None,
    legacy_state_verification_path: Path | None = None,
) -> dict[str, Any]:
    heads = identity["heads"]
    generators = {
        b"tx_state": expected_tx_states(heads),
        b"acc_state": expected_acc_states(heads),
        b"head_state": expected_head_states(heads),
    }
    global_generator = expected_all_states(heads)
    per_event = {event: hashlib.sha256() for event in STATE_EVENTS}
    combined = hashlib.sha256()
    counts = {event: 0 for event in STATE_EVENTS}
    previous_cycle = -1
    total_trace_rows = 0
    with trace.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
        if header != (
            b"cycle,event,tile,head,source,lane,out,delay,index,origin,payload"
        ):
            raise ValueError("trace header differs from the frozen 11-column contract")
        for line_number, line in enumerate(handle, start=2):
            total_trace_rows += 1
            fields = line.rstrip(b"\r\n").split(b",", 10)
            if len(fields) != 11:
                raise ValueError(f"trace row {line_number} does not contain 11 fields")
            event = fields[1]
            if event not in generators:
                continue
            try:
                cycle = int(fields[0])
            except ValueError as error:
                raise ValueError(f"trace row {line_number} has a non-integer cycle") from error
            if cycle < previous_cycle:
                raise ValueError(f"state cycle regressed at trace row {line_number}")
            previous_cycle = cycle
            observed = tuple(fields[1:10])
            try:
                expected_global = next(global_generator)
            except StopIteration as error:
                raise ValueError(
                    f"state stream has an extra row at trace row {line_number}"
                ) from error
            if observed != expected_global:
                raise ValueError(
                    f"global state order differs at trace row {line_number}: "
                    f"observed={observed!r}, expected={expected_global!r}"
                )
            try:
                expected = next(generators[event])
            except StopIteration as error:
                raise ValueError(
                    f"{event.decode()} has an extra row at trace row {line_number}"
                ) from error
            if observed != expected:
                raise ValueError(
                    f"{event.decode()} differs at occurrence {counts[event]}: "
                    f"observed={observed!r}, expected={expected!r}"
                )
            payload = canonical_bytes(observed)
            per_event[event].update(payload)
            combined.update(payload)
            counts[event] += 1
    for event, generator in generators.items():
        try:
            missing = next(generator)
        except StopIteration:
            continue
        raise ValueError(f"{event.decode()} ended early before expected row {missing!r}")
    try:
        missing_global = next(global_generator)
    except StopIteration:
        missing_global = None
    if missing_global is not None:
        raise ValueError(
            f"global state stream ended early before expected row {missing_global!r}"
        )

    observed_counts = {event.decode(): counts[event] for event in STATE_EVENTS}
    if observed_counts != expected_counts(heads):
        raise ValueError("analytical and observed state counts differ")
    structural = {
        "canonical_fields": list(CANONICAL_FIELDS),
        "state_counts": observed_counts,
        "state_cycle_free_ordered_sha256": {
            event.decode(): per_event[event].hexdigest() for event in STATE_EVENTS
        },
        "all_state_count": sum(observed_counts.values()),
        "all_state_cycle_free_ordered_sha256": combined.hexdigest(),
        "global_analytical_order_match": True,
        "last_state_cycle": previous_cycle,
    }
    trace_digest = sha256(trace)
    parent_binding = None
    if parent_complete_path is not None:
        parent = read_json(parent_complete_path)
        internal = parent.get("internal_bindings")
        if (
            parent.get("status")
            != "PASS_SEALED_H24_IDENTITY_PHASE_ARRAY_CANARY_NOT_G0"
            or parent.get("formal_g0") != "DENY"
            or parent.get("identity") != identity
            or not isinstance(internal, dict)
            or trace_digest not in internal.values()
        ):
            raise ValueError("parent complete does not bind trace SHA and identity")
        parent_binding = {
            "status": "PASS_PARENT_IDENTITY_AND_TRACE_BINDING",
            "path": str(parent_complete_path),
            "sha256": sha256(parent_complete_path),
        }
    legacy_crosscheck = None
    if legacy_state_verification_path is not None:
        legacy = read_json(legacy_state_verification_path)
        legacy_state = legacy.get("state_reference")
        if (
            legacy.get("status") != "PASS_IDENTITY_SERVICE_RTL_TRACE_V2_NOT_G0"
            or legacy.get("formal_g0") != "DENY"
            or legacy.get("trace_sha256") != trace_digest
            or not isinstance(legacy_state, dict)
            or legacy_state.get("state_reference_sha256")
            != "0b49ce3526ffeeaf393ea3b8265f87df2f789252b852370b1189e6eead781ea1"
            or legacy_state.get("state_counts") != observed_counts
            or legacy_state.get("all_state_count") != sum(observed_counts.values())
        ):
            raise ValueError("legacy frozen H3 state reference cross-check differs")
        legacy_crosscheck = {
            "status": "PASS_INDEPENDENT_FROZEN_H3_STATE_REFERENCE",
            "path": str(legacy_state_verification_path),
            "sha256": sha256(legacy_state_verification_path),
            "state_reference_sha256": legacy_state["state_reference_sha256"],
        }
    cross = None
    if cross_reference is not None:
        reference = read_json(cross_reference)
        reference_identity = reference.get("identity")
        reference_structural = reference.get("structural_oracle")
        if (
            reference.get("schema")
            != "local5_identity_state_structure_verification_v3"
            or reference.get("status")
            != "PASS_ANALYTICAL_CYCLE_FREE_STATE_STRUCTURE_NOT_G0"
            or reference.get("formal_g0") != "DENY"
            or reference_identity != identity
            or not isinstance(reference_structural, dict)
            or reference_structural.get("canonical_fields")
            != list(CANONICAL_FIELDS)
            or reference_structural.get("state_counts") != observed_counts
            or reference_structural.get("state_cycle_free_ordered_sha256")
            != structural["state_cycle_free_ordered_sha256"]
            or reference_structural.get("all_state_count")
            != structural["all_state_count"]
            or reference_structural.get("all_state_cycle_free_ordered_sha256")
            != structural["all_state_cycle_free_ordered_sha256"]
            or reference_structural.get("global_analytical_order_match") is not True
        ):
            raise ValueError("cross-run cycle-free state structure differs")
        cross = {
            "status": "PASS_CROSS_RUN_CYCLE_FREE_STATE_STRUCTURE",
            "reference_path": str(cross_reference),
            "reference_sha256": sha256(cross_reference),
        }
    return {
        "schema": "local5_identity_state_structure_verification_v3",
        "status": "PASS_ANALYTICAL_CYCLE_FREE_STATE_STRUCTURE_NOT_G0",
        "evidence": "[独立解析结构oracle]+[rtl-trace-derived]",
        "formal_g0": "DENY",
        "identity": identity,
        "trace": {
            "path": str(trace),
            "sha256": trace_digest,
            "rows_excluding_header": total_trace_rows,
        },
        "structural_oracle": structural,
        "cross_run": cross,
        "parent_binding": parent_binding,
        "legacy_reference_crosscheck": legacy_crosscheck,
        "boundary": [
            "期望 state 值序列仅由 H、T450、HEAD_DIM32、OUT_DIM32 和控制合同解析生成",
            "oracle 不读取 RTL state 行生成期望，不依赖 trace cycle 或 payload 数值",
            "逐行验证 cycle-free state 值、身份和跨 tx/acc/head 的全局交错顺序",
            "不声称 cycle-sensitive 时序 oracle；cycle 只检查单调性",
            "单个真实窗口；不是 formal G0、性能、功耗、面积或 full encoder 证据",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--cross-reference", type=Path)
    parser.add_argument("--parent-complete", type=Path)
    parser.add_argument("--legacy-state-verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = {
        "sample": args.sample,
        "stage": args.stage,
        "block": args.block,
        "window": args.window,
        "heads": args.heads,
    }
    result = verify_trace(
        args.trace.resolve(),
        identity,
        args.cross_reference.resolve() if args.cross_reference else None,
        args.parent_complete.resolve() if args.parent_complete else None,
        (
            args.legacy_state_verification.resolve()
            if args.legacy_state_verification else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "identity": identity,
        "state_counts": result["structural_oracle"]["state_counts"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
