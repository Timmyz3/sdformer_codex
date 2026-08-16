#!/usr/bin/env python3
"""验证 Local5 identity-service H3 RTL 原始事件和 Acc32 输出。"""

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
HEADS = 3
TRACE_FIELDS = [
    "cycle", "event", "tile", "head", "source", "lane", "out", "delay",
    "index", "origin",
]
PASS_PATTERN = re.compile(
    r"PASS Local5 .*transaction_service=0 identity_service=1 seed=20260810 "
    r"stage=0 block=0 window=249 cycles=(\d+) token=4050 .*"
    r"result_service=43200 .* final=43200 .*"
)


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
    result: dict[str, int | str] = {"event": row["event"], "origin": row["origin"]}
    for name in TRACE_FIELDS:
        if name not in ("event", "origin"):
            try:
                result[name] = int(row[name])
            except ValueError as error:
                raise ValueError(f"trace {name} is not an integer") from error
    return result


def verify_trace(
    trace_path: Path,
    package_dir: Path,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    receipt_path = package_dir / "verification_receipt.json"
    manifest = read_json(manifest_path)
    receipt = read_json(receipt_path)
    if (
        manifest.get("identity")
        != {"sample": 2, "stage": 0, "block": 0, "window": 249, "heads": HEADS}
        or manifest.get("formal_g0") != "DENY"
        or receipt.get("status") != "PASS_INDEPENDENT_VERIFY_NOT_G0"
        or receipt.get("manifest_sha256") != sha256(manifest_path)
    ):
        raise ValueError("identity-service package is not the frozen H3 contract")
    delays = {
        "relation": read_memh(package_dir / "relation_delay.memh", HEADS * TOKENS),
        "weight": read_memh(
            package_dir / "weight_delay.memh", HEADS * HEADS * LANES * OUT_DIM
        ),
        "final": read_memh(package_dir / "final_delay.memh", HEADS * TOKENS * OUT_DIM),
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
    relation_pending: dict[str, int] | None = None
    weight_pending: dict[str, int] | None = None
    final_pending: dict[str, int] | None = None
    relation_started = relation_completed = 0
    weight_started = weight_completed = 0
    final_started = final_completed = 0
    previous_cycle = -1

    with trace_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TRACE_FIELDS:
            raise ValueError("trace CSV header differs")
        for text_row in reader:
            row = integer_row(text_row)
            cycle = int(row["cycle"])
            event = str(row["event"])
            if cycle < previous_cycle:
                raise ValueError("trace cycle order regressed")
            previous_cycle = cycle
            counts[event] += 1
            if event in ("manifest_binding", "receipt_binding"):
                if cycle != 0 or event in bindings:
                    raise ValueError("trace package binding is duplicated or late")
                bindings[event] = str(row["origin"])
                continue
            if event in boundary_values:
                if row["origin"] != "rtl_boundary":
                    raise ValueError("boundary origin differs")
                boundary_values[event].append((int(row["tile"]), int(row["head"])))
                continue
            if event in ("tx_state", "acc_state", "head_state"):
                if row["origin"] != "rtl_internal_state":
                    raise ValueError("state origin differs")
                continue
            if row["origin"] != "rtl_handshake":
                raise ValueError(f"handshake origin differs for {event}")

            if event == "relation_accept":
                if relation_pending is not None:
                    raise ValueError("relation request overlapped")
                index = relation_started
                tile = index // (HEADS * TOKENS)
                head = (index // TOKENS) % HEADS
                source = index % TOKENS
                flat = head * TOKENS + source
                delay = int(delays["relation"][flat])
                if (
                    (row["tile"], row["head"], row["source"], row["index"], row["delay"])
                    != (tile, head, source, flat, delay)
                ):
                    raise ValueError("relation accept identity/index/delay differs")
                relation_pending = {"cycle": cycle, "flat": flat, "delay": delay, "head": head, "source": source}
                relation_started += 1
            elif event == "relation_response_available":
                if relation_pending is None or "available" in relation_pending:
                    raise ValueError("relation available without one request")
                if (
                    cycle - relation_pending["cycle"] != relation_pending["delay"] + 1
                    or (row["head"], row["source"], row["index"], row["delay"])
                       != (relation_pending["head"], relation_pending["source"],
                           relation_pending["flat"], relation_pending["delay"])
                ):
                    raise ValueError("relation response-available contract differs")
                relation_pending["available"] = cycle
            elif event == "relation_response_accept":
                if relation_pending is None or "available" not in relation_pending:
                    raise ValueError("relation response accepted before available")
                if cycle < relation_pending["available"]:
                    raise ValueError("relation response acceptance cycle differs")
                relation_pending = None
                relation_completed += 1
            elif event == "weight_accept":
                if weight_pending is not None:
                    raise ValueError("weight request overlapped")
                index = weight_started
                tile = index // (HEADS * LANES * OUT_DIM)
                head = (index // (LANES * OUT_DIM)) % HEADS
                lane = (index // OUT_DIM) % LANES
                out = index % OUT_DIM
                delay = int(delays["weight"][index])
                if (
                    (row["tile"], row["head"], row["lane"], row["out"], row["index"], row["delay"])
                    != (tile, head, lane, out, index, delay)
                ):
                    raise ValueError("weight accept identity/index/delay differs")
                weight_pending = {"cycle": cycle, "flat": index, "delay": delay, "tile": tile, "head": head, "lane": lane, "out": out}
                weight_started += 1
            elif event == "weight_response_available":
                if weight_pending is None or "available" in weight_pending:
                    raise ValueError("weight available without one request")
                expected = (
                    weight_pending["tile"], weight_pending["head"], weight_pending["lane"],
                    weight_pending["out"], weight_pending["flat"], weight_pending["delay"],
                )
                if (
                    cycle - weight_pending["cycle"] != weight_pending["delay"] + 1
                    or (row["tile"], row["head"], row["lane"], row["out"], row["index"], row["delay"])
                       != expected
                ):
                    raise ValueError("weight response-available contract differs")
                weight_pending["available"] = cycle
            elif event == "weight_response_accept":
                if weight_pending is None or "available" not in weight_pending:
                    raise ValueError("weight response accepted before available")
                if cycle < weight_pending["available"]:
                    raise ValueError("weight response acceptance cycle differs")
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
                if (
                    (row["tile"], row["source"], row["out"], row["index"], row["delay"])
                    != (tile, source, out, index, delay)
                ):
                    raise ValueError("final request identity/index/delay differs")
                final_pending = {"cycle": cycle, "delay": delay, "flat": index}
                final_started += 1
            elif event == "final_accept":
                if final_pending is None:
                    raise ValueError("final accepted without request")
                if (
                    cycle - final_pending["cycle"] != final_pending["delay"] + 1
                    or row["index"] != final_pending["flat"]
                    or row["delay"] != final_pending["delay"]
                ):
                    raise ValueError("final response latency/index differs")
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
    if (relation_started, relation_completed) != (HEADS * HEADS * TOKENS,) * 2:
        raise ValueError("relation transaction count differs")
    if (weight_started, weight_completed) != (HEADS * HEADS * LANES * OUT_DIM,) * 2:
        raise ValueError("weight transaction count differs")
    if (final_started, final_completed) != (HEADS * TOKENS * OUT_DIM,) * 2:
        raise ValueError("final transaction count differs")
    expected_boundaries = {
        "group_start": [(-1, -1)],
        "tile_start": [(tile, -1) for tile in range(HEADS)],
        "head_start": [(tile, head) for tile in range(HEADS) for head in range(HEADS)],
        "head_done": [(tile, head) for tile in range(HEADS) for head in range(HEADS)],
        "tile_done": [(tile, -1) for tile in range(HEADS)],
        "group_done": [(-1, -1)],
    }
    if boundary_values != expected_boundaries:
        raise ValueError("group/tile/head boundary sequence differs")
    for name in ("tx_state", "acc_state", "head_state"):
        if counts[name] == 0:
            raise ValueError(f"trace lacks {name} events")
    return {
        "status": "PASS_IDENTITY_SERVICE_RTL_TRACE_NOT_G0",
        "evidence": "[rtl]+[软件确定性服务合同]",
        "formal_g0": "DENY",
        "trace_sha256": sha256(trace_path),
        "trace_rows": sum(counts.values()),
        "event_counts": dict(sorted(counts.items())),
        "last_cycle": previous_cycle,
        "manifest_sha256": manifest_sha,
        "verification_receipt_sha256": receipt_sha,
    }


def verify_acc32(actual_path: Path, expected_path: Path) -> dict[str, Any]:
    rows = actual_path.read_text(encoding="ascii").splitlines()
    if len(rows) != HEADS * TOKENS * OUT_DIM or any(len(row) != 8 for row in rows):
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


def verify_log(path: Path) -> int:
    matches = PASS_PATTERN.findall(path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("Verilator log lacks one exact identity-service PASS line")
    return int(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--verilator-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        **verify_trace(args.trace.resolve(), args.package_dir.resolve()),
        "acc32": verify_acc32(args.actual.resolve(), args.expected.resolve()),
        "rtl_cycles": verify_log(args.verilator_log.resolve()),
        "verilator_log_sha256": sha256(args.verilator_log.resolve()),
        "boundary": [
            "cycles are validation-environment latency, not architecture performance",
            "no formal G0, full encoder, PPA, throughput, or energy claim",
        ],
    }
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
