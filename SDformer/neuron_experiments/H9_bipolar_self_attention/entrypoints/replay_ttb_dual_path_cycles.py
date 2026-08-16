"""Replay ordered TTB/Delta traces through finite dense/sparse queues."""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


KAPPAS = (2, 4, 8, 12, 16)
SPARSE_LANES = (2, 4, 8)
FIFO_DEPTHS = (4, 8, 16)
BACKEND_ROW_CYCLES = (1, 4, 8, 16)


def decode_trace(encoded: dict[str, Any]) -> np.ndarray:
    if encoded.get("codec") != "zlib_base64" or encoded.get("dtype") != "int16_le":
        raise ValueError(f"unsupported ordered trace encoding: {encoded}")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    array = np.frombuffer(raw, dtype="<i2")
    shape = tuple(int(item) for item in encoded["shape"])
    if array.size != math.prod(shape):
        raise ValueError(f"ordered trace size mismatch: {array.size} != {shape}")
    return array.reshape(shape)


def trace_spec(record: dict[str, Any], route: str) -> tuple[np.ndarray, int]:
    lanes = int(record["head_dim"])
    if route == "delta1":
        return decode_trace(record["delta_update_ordered_trace"]), lanes
    if route not in {"ttb4", "ttb8"}:
        raise ValueError(route)
    bundle = int(route[-1])
    return decode_trace(record[f"ttb_tok{bundle}_active_ordered_trace"]), 2 * bundle * lanes


def service_cycles(active: int, *, capacity: int, lanes: int, sparse: bool) -> int:
    if active <= 0:
        return 0
    useful = active if sparse else capacity
    return max(1, math.ceil(useful / lanes)) + (1 if sparse else 0)


def analytical_record(
    counts: np.ndarray,
    *,
    capacity: int,
    kappa: int,
    sparse_lanes: int,
    dense_lanes: int,
) -> dict[str, int]:
    flat = counts.reshape(-1).astype(np.int64, copy=False)
    rows = int(math.prod(counts.shape[:-1])) if counts.ndim >= 2 else 1
    empty = int(np.count_nonzero(flat == 0))
    sparse_mask = (flat > 0) & (flat <= kappa)
    sparse_jobs = int(np.count_nonzero(sparse_mask))
    dense_jobs = int(flat.size - empty - sparse_jobs)
    sparse_active_lanes = int(flat[sparse_mask].sum())
    sparse_work = int(
        np.ceil(flat[sparse_mask] / sparse_lanes).sum() + sparse_jobs
    )
    dense_work = dense_jobs * max(1, math.ceil(capacity / dense_lanes))
    e0_work = (flat.size - empty) * max(1, math.ceil(capacity / dense_lanes))
    count_bits = math.ceil(math.log2(capacity + 1))
    index_bits = max(1, math.ceil(math.log2(capacity)))
    route_metadata_bits = int(flat.size) * (count_bits + 2)
    e0_payload_bits = (flat.size - empty) * 2 * capacity
    dense_payload_bits = dense_jobs * 2 * capacity
    bitmap_sparse_bits = sparse_jobs * capacity + 2 * sparse_active_lanes
    index_sparse_bits = sparse_jobs * count_bits + sparse_active_lanes * (index_bits + 2)
    metadata_words64 = int(flat.size)
    e0_payload_words64 = (flat.size - empty) * math.ceil(2 * capacity / 64)
    dense_payload_words64 = dense_jobs * math.ceil(2 * capacity / 64)
    sparse_values = flat[sparse_mask]
    bitmap_sparse_words64 = int(
        sparse_jobs * math.ceil(capacity / 64)
        + np.ceil((2 * sparse_values) / 64).sum()
    )
    index_sparse_words64 = int(
        np.ceil((count_bits + sparse_values * (index_bits + 2)) / 64).sum()
    )
    result = {
        "arrivals": int(flat.size),
        "rows": rows,
        "empty": empty,
        "sparse_jobs": sparse_jobs,
        "dense_jobs": dense_jobs,
        "sparse_active_lanes": sparse_active_lanes,
        "sparse_work": sparse_work,
        "dense_work": dense_work,
        "e0_work": e0_work,
        "route_metadata_bits": route_metadata_bits,
        "e0_traffic_bits": route_metadata_bits + e0_payload_bits,
        "dual_bitmap_traffic_bits": route_metadata_bits + dense_payload_bits + bitmap_sparse_bits,
        "dual_index_traffic_bits": route_metadata_bits + dense_payload_bits + index_sparse_bits,
        "e0_transactions64": metadata_words64 + e0_payload_words64,
        "dual_bitmap_transactions64": metadata_words64 + dense_payload_words64 + bitmap_sparse_words64,
        "dual_index_transactions64": metadata_words64 + dense_payload_words64 + index_sparse_words64,
        "dual_lower_bound": max(int(flat.size), sparse_work, dense_work),
        "e0_lower_bound": max(int(flat.size), e0_work),
    }
    for backend_cycles in BACKEND_ROW_CYCLES:
        backend_work = rows * backend_cycles
        result[f"backend_work_b{backend_cycles}"] = backend_work
        result[f"dual_lower_bound_b{backend_cycles}"] = max(
            int(flat.size), sparse_work, dense_work, backend_work
        )
        result[f"e0_lower_bound_b{backend_cycles}"] = max(
            int(flat.size), e0_work, backend_work
        )
    return result


def simulate_dual_record(
    counts: np.ndarray,
    *,
    capacity: int,
    kappa: int,
    sparse_lanes: int,
    dense_lanes: int,
    fifo_depth: int,
) -> dict[str, int]:
    """Cycle replay with one metadata arrival/cycle and exact backpressure."""

    values = counts.reshape(-1)
    sparse_q: deque[int] = deque()
    dense_q: deque[int] = deque()
    sparse_remaining = 0
    dense_remaining = 0
    cursor = 0
    cycles = 0
    input_stalls = 0
    sparse_busy = 0
    dense_busy = 0
    max_sparse_fifo = 0
    max_dense_fifo = 0
    while cursor < values.size or sparse_q or dense_q or sparse_remaining or dense_remaining:
        if sparse_remaining == 0 and sparse_q:
            sparse_remaining = sparse_q.popleft()
        if dense_remaining == 0 and dense_q:
            dense_remaining = dense_q.popleft()

        if cursor < values.size:
            active = int(values[cursor])
            if active <= 0:
                cursor += 1
            else:
                sparse = active <= kappa
                queue = sparse_q if sparse else dense_q
                remaining = sparse_remaining if sparse else dense_remaining
                if remaining == 0 and not queue:
                    work = service_cycles(
                        active,
                        capacity=capacity,
                        lanes=sparse_lanes if sparse else dense_lanes,
                        sparse=sparse,
                    )
                    if sparse:
                        sparse_remaining = work
                    else:
                        dense_remaining = work
                    cursor += 1
                elif len(queue) < fifo_depth:
                    queue.append(service_cycles(
                        active,
                        capacity=capacity,
                        lanes=sparse_lanes if sparse else dense_lanes,
                        sparse=sparse,
                    ))
                    cursor += 1
                else:
                    input_stalls += 1

        max_sparse_fifo = max(max_sparse_fifo, len(sparse_q))
        max_dense_fifo = max(max_dense_fifo, len(dense_q))
        if sparse_remaining > 0:
            sparse_busy += 1
            sparse_remaining -= 1
        if dense_remaining > 0:
            dense_busy += 1
            dense_remaining -= 1
        cycles += 1
    return {
        "cycles": cycles,
        "input_stalls": input_stalls,
        "sparse_busy": sparse_busy,
        "dense_busy": dense_busy,
        "max_sparse_fifo": max_sparse_fifo,
        "max_dense_fifo": max_dense_fifo,
    }


def sum_dict(rows: list[dict[str, int]]) -> dict[str, int]:
    keys = {key for row in rows for key in row}
    return {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}


def aggregate_replay(rows: list[dict[str, int]]) -> dict[str, int]:
    totals = sum_dict(rows)
    for key in ("max_sparse_fifo", "max_dense_fifo"):
        totals[key] = max((int(row.get(key, 0)) for row in rows), default=0)
    return totals


def profile_records(profile: dict[str, Any]) -> list[dict[str, Any]]:
    if not profile.get("ordered_trace"):
        raise ValueError("profile was not collected with --ordered-trace")
    records = profile["summary"]["h60_records"]
    if not records:
        raise ValueError("profile has no H60 records")
    return records


def prepare_traces(
    records: list[dict[str, Any]], route: str
) -> list[tuple[np.ndarray, int, int]]:
    prepared = []
    for record in records:
        counts, capacity = trace_spec(record, route)
        prepared.append((counts, capacity, int(record["stage"])))
    return prepared


def analytical_sweep(
    prepared: list[tuple[np.ndarray, int, int]], route: str, dense_lanes: int
) -> list[dict[str, Any]]:
    rows = []
    for kappa in KAPPAS:
        for sparse_lanes in SPARSE_LANES:
            parts = []
            for counts, capacity, _stage in prepared:
                parts.append(analytical_record(
                    counts,
                    capacity=capacity,
                    kappa=kappa,
                    sparse_lanes=sparse_lanes,
                    dense_lanes=dense_lanes,
                ))
            totals = sum_dict(parts)
            totals["dual_lower_bound"] = sum(part["dual_lower_bound"] for part in parts)
            totals["e0_lower_bound"] = sum(part["e0_lower_bound"] for part in parts)
            for backend_cycles in BACKEND_ROW_CYCLES:
                totals[f"dual_lower_bound_b{backend_cycles}"] = sum(
                    part[f"dual_lower_bound_b{backend_cycles}"] for part in parts
                )
                totals[f"e0_lower_bound_b{backend_cycles}"] = sum(
                    part[f"e0_lower_bound_b{backend_cycles}"] for part in parts
                )
            rows.append({
                "route": route,
                "kappa": kappa,
                "sparse_lanes": sparse_lanes,
                "dense_lanes": dense_lanes,
                **totals,
                "lower_bound_reduction_vs_e0": (
                    1.0 - totals["dual_lower_bound"] / totals["e0_lower_bound"]
                    if totals["e0_lower_bound"] else 0.0
                ),
                "bitmap_traffic_reduction_vs_e0": (
                    1.0 - totals["dual_bitmap_traffic_bits"] / totals["e0_traffic_bits"]
                    if totals["e0_traffic_bits"] else 0.0
                ),
                "index_traffic_reduction_vs_e0": (
                    1.0 - totals["dual_index_traffic_bits"] / totals["e0_traffic_bits"]
                    if totals["e0_traffic_bits"] else 0.0
                ),
                "backend_lower_bound_reduction": {
                    str(backend_cycles): (
                        1.0
                        - totals[f"dual_lower_bound_b{backend_cycles}"]
                        / totals[f"e0_lower_bound_b{backend_cycles}"]
                        if totals[f"e0_lower_bound_b{backend_cycles}"] else 0.0
                    )
                    for backend_cycles in BACKEND_ROW_CYCLES
                },
                "bitmap_transaction_reduction_vs_e0": (
                    1.0 - totals["dual_bitmap_transactions64"] / totals["e0_transactions64"]
                    if totals["e0_transactions64"] else 0.0
                ),
                "index_transaction_reduction_vs_e0": (
                    1.0 - totals["dual_index_transactions64"] / totals["e0_transactions64"]
                    if totals["e0_transactions64"] else 0.0
                ),
            })
    return rows


def finite_replay(
    prepared: list[tuple[np.ndarray, int, int]],
    route: str,
    candidate: dict[str, Any],
    fifo_depth: int,
) -> dict[str, Any]:
    parts = []
    by_stage: dict[int, list[dict[str, int]]] = {}
    for counts, capacity, stage in prepared:
        result = simulate_dual_record(
            counts,
            capacity=capacity,
            kappa=int(candidate["kappa"]),
            sparse_lanes=int(candidate["sparse_lanes"]),
            dense_lanes=int(candidate["dense_lanes"]),
            fifo_depth=fifo_depth,
        )
        parts.append(result)
        by_stage.setdefault(stage, []).append(result)
    totals = aggregate_replay(parts)
    e0 = int(candidate["e0_lower_bound"])
    return {
        "route": route,
        "kappa": int(candidate["kappa"]),
        "sparse_lanes": int(candidate["sparse_lanes"]),
        "dense_lanes": int(candidate["dense_lanes"]),
        "fifo_depth": fifo_depth,
        **totals,
        "cycle_reduction_vs_e0_work_bound": 1.0 - totals["cycles"] / e0 if e0 else 0.0,
        "e0_traffic_bits": int(candidate["e0_traffic_bits"]),
        "dual_bitmap_traffic_bits": int(candidate["dual_bitmap_traffic_bits"]),
        "dual_index_traffic_bits": int(candidate["dual_index_traffic_bits"]),
        "bitmap_traffic_reduction_vs_e0": float(candidate["bitmap_traffic_reduction_vs_e0"]),
        "index_traffic_reduction_vs_e0": float(candidate["index_traffic_reduction_vs_e0"]),
        "e0_transactions64": int(candidate["e0_transactions64"]),
        "dual_bitmap_transactions64": int(candidate["dual_bitmap_transactions64"]),
        "dual_index_transactions64": int(candidate["dual_index_transactions64"]),
        "bitmap_transaction_reduction_vs_e0": float(candidate["bitmap_transaction_reduction_vs_e0"]),
        "index_transaction_reduction_vs_e0": float(candidate["index_transaction_reduction_vs_e0"]),
        "by_stage": [
            {"stage": stage, **aggregate_replay(stage_rows)}
            for stage, stage_rows in sorted(by_stage.items())
        ],
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# TTB/Exact-Delta finite FIFO cycle replay",
        "",
        "Ordered traces are replayed independently per attention invocation; queues drain at block/sample boundaries.",
        "Finite replay includes metadata arrival, dense/sparse service and FIFO backpressure. Analytical rows additionally sweep a 1/4/8/16-cycle per-row shared-backend work lower bound, but do not model join ordering/backend FIFO. SRAM bank timing, projection, decoder and NoC remain excluded, so reductions are row-kernel proxies rather than end-to-end speedup.",
        "",
        "| route | kappa | sparse lanes | dense lanes | FIFO | cycles | input stalls | max S/D FIFO | cycle reduction vs E0 work bound | bitmap/index traffic reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["finite_replay"]:
        lines.append(
            f"| {row['route']} | {row['kappa']} | {row['sparse_lanes']} | {row['dense_lanes']} | "
            f"{row['fifo_depth']} | {row['cycles']} | {row['input_stalls']} | "
            f"{row['max_sparse_fifo']}/{row['max_dense_fifo']} | "
            f"{row['cycle_reduction_vs_e0_work_bound']:.4%} | "
            f"{row['bitmap_traffic_reduction_vs_e0']:.4%}/{row['index_traffic_reduction_vs_e0']:.4%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-lanes", type=int, default=32)
    args = parser.parse_args()
    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    records = profile_records(profile)
    analytical = []
    finite = []
    for route in ("delta1", "ttb4", "ttb8"):
        prepared = prepare_traces(records, route)
        sweep = analytical_sweep(prepared, route, args.dense_lanes)
        analytical.extend(sweep)
        promoted = sorted(
            sweep,
            key=lambda row: (row["dual_lower_bound"], -row["kappa"], -row["sparse_lanes"]),
        )[:3]
        for candidate in promoted:
            for depth in FIFO_DEPTHS:
                finite.append(finite_replay(prepared, route, candidate, depth))
    result = {
        "schema_version": 1,
        "source_profile": str(args.profile_json),
        "scope": (
            "finite replay covers route/front-end only; analytical sweep includes shared-backend "
            "row-work lower bounds; excludes backend join/FIFO, SRAM, projection, decoder and NoC"
        ),
        "backend_sensitivity_scope": (
            "row-work lower bounds for 1/4/8/16 cycles per window-head; "
            "does not model join ordering or backend FIFO"
        ),
        "analytical_sweep": analytical,
        "finite_replay": finite,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
