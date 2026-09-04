#!/usr/bin/env python3
"""M504 exact single-port parent-scratch schedule audit on frozen H67 rows.

This audit changes only the parent-result scratch port schedule of the frozen
M473 row_tile=64, 8-bank, 128-B/cycle point.  It reproduces the exact subset
mapping and topological issue order, then simulates the two-entry M498 parent
response queue cycle by cycle.  A single-port macro accepts at most one read or
write per cycle; a same-address final-write/prefetch is forwarded without a
macro read.  No consume credit is used for prefetch readiness.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "contracts" / "m504_h67_single_port_parent_scratch_execution_contract_r3_20260827.json"
DEFAULT_OUT = ROOT / "results" / "m504_h67_single_port_parent_scratch_r3_20260827"
BYTES_PER_ROW = 9
ROWS_PER_PHASE = 3000
PARTITIONS = 432
SAMPLES = 10
OPERATORS = 4
PHASES = SAMPLES * OPERATORS * PARTITIONS
ROW_TILE = 64
BLOCK_BANKS = 8
OUTPUT_BLOCKS = 8
POPCOUNT = np.array([value.bit_count() for value in range(1 << 16)], dtype=np.uint8)

_ROWS_FD: int | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError(f"non-standard JSON token: {token}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def cleanroom_subset(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen official-Prosperity-compatible subset mapping from M473."""

    masks = np.asarray(masks, dtype=np.uint16)
    count = masks.size
    indices = np.arange(count, dtype=np.int32)
    residual = masks.copy()
    parent = np.full(count, -1, dtype=np.int16)
    for row, current in enumerate(masks):
        if POPCOUNT[int(current)] < 2:
            continue
        subset = np.bitwise_and(masks, current) == masks
        equal_later = (masks == current) & (indices >= row)
        candidate_indices = np.flatnonzero(subset & ~equal_later)
        if candidate_indices.size == 0:
            continue
        candidate_popcounts = POPCOUNT[masks[candidate_indices]]
        if int(candidate_popcounts.max(initial=0)) < 1:
            continue
        chosen = int(candidate_indices[int(np.argmax(candidate_popcounts))])
        parent[row] = chosen
        residual[row] = np.uint16(current ^ masks[chosen])
    return residual, parent


def simulate_single_port_task(
    masks: np.ndarray, policy: str = "work_conserving"
) -> dict[str, int]:
    """Cycle-exact M498 queue plus one-read-or-write scratch arbitration.

    ``deadline_lookahead`` is an executable one-descriptor policy, not a claim
    of global optimality.  It may hold a current final issue for one cycle to
    read the immediately next active row's unmet parent edge.  The returning
    word then moves behind the current FIFO head on the following consume/write
    edge.  All other behavior is work conserving.
    """

    require(policy in ("work_conserving", "deadline_lookahead"), "unknown policy")

    masks = np.asarray(masks, dtype=np.uint16)
    residual, parent = cleanroom_subset(masks)
    original_pc = POPCOUNT[masks].astype(np.int32)
    residual_pc = POPCOUNT[residual].astype(np.int32)
    order = np.lexsort((np.arange(masks.size, dtype=np.int32), original_pc))
    active_order = [int(row) for row in order if int(masks[row]) != 0]
    requirements = [int(parent[row]) for row in active_order if int(parent[row]) >= 0]
    requirement_consumers = [
        cursor for cursor, row in enumerate(active_order) if int(parent[row]) >= 0
    ]

    ideal_issues = int(
        residual_pc.sum()
        + np.count_nonzero((parent >= 0) & (residual == 0) & (masks != 0))
    )
    queue: list[int] = []
    pending: int | None = None
    next_requirement = 0
    cycles = 0
    issue_cycles = 0
    stall_cycles = 0
    macro_reads = 0
    macro_writes = 0
    forwarded_reads = 0
    concurrent_issue_reads = 0
    concurrent_issue_forwards = 0
    empty_queue_parent_stalls = 0
    full_queue_prefetch_blocks = 0
    producer_not_ready_prefetch_blocks = 0
    deadline_lookahead_holds = 0
    written = np.zeros(masks.size, dtype=np.bool_)

    row_cursor = 0
    beat = 0
    # Every accepted prefetch corresponds to exactly one consumer edge and the
    # FIFO order is the topological consumer order.  This is the earliest-edge
    # greedy schedule; delaying an accepted earliest read cannot make a later
    # in-order consumer ready sooner under the same two-entry FIFO.
    while row_cursor < len(active_order):
        row = active_order[row_cursor]
        parent_id = int(parent[row])
        work = int(residual_pc[row])
        if parent_id >= 0 and work == 0:
            work = 1
        require(work > 0, "active row has zero executable work")

        parent_ready = parent_id < 0 or (queue and queue[0] == parent_id)
        issue_possible = bool(parent_ready)
        final_if_issued = issue_possible and beat + 1 == work

        reserved = len(queue) + int(pending is not None)
        require(reserved <= 2, "parent response queue over-reserved")
        request_exists = next_requirement < len(requirements)
        requested_parent = requirements[next_requirement] if request_exists else -1
        requested_consumer = (
            requirement_consumers[next_requirement] if request_exists else -1
        )
        has_capacity = reserved < 2
        producer_ready = bool(request_exists and written[requested_parent])
        deadline_hold = bool(
            policy == "deadline_lookahead"
            and final_if_issued
            and request_exists
            and has_capacity
            and producer_ready
            and requested_parent != row
            and requested_consumer == row_cursor + 1
        )
        issue = bool(issue_possible and not deadline_hold)
        last = issue and beat + 1 == work
        write = bool(last)
        if deadline_hold:
            deadline_lookahead_holds += 1
        forward = bool(write and request_exists and has_capacity and requested_parent == row)
        read = bool((not write) and request_exists and has_capacity and written[requested_parent])
        if request_exists and not has_capacity:
            full_queue_prefetch_blocks += 1
        if (
            request_exists
            and has_capacity
            and not written[requested_parent]
            and not forward
        ):
            producer_not_ready_prefetch_blocks += 1

        consumed = bool(last and parent_id >= 0)
        if issue:
            issue_cycles += 1
            if read:
                concurrent_issue_reads += 1
            if forward:
                concurrent_issue_forwards += 1
        else:
            stall_cycles += 1
            if parent_id >= 0 and not queue:
                empty_queue_parent_stalls += 1

        # M498 edge order: optional head pop, prior synchronous response, then
        # same-cycle RAW forwarding.  A newly issued macro read becomes next
        # cycle's pending response and cannot feed this cycle's issue.
        if consumed:
            require(queue and queue[0] == parent_id, "consumed parent is not FIFO head")
            queue.pop(0)
        if pending is not None:
            require(len(queue) < 2, "return response overflow")
            queue.append(pending)
        if forward:
            require(len(queue) < 2, "forward response overflow")
            queue.append(requested_parent)
            next_requirement += 1
            forwarded_reads += 1
        if read:
            next_pending: int | None = requested_parent
            next_requirement += 1
            macro_reads += 1
        else:
            next_pending = None
        pending = next_pending

        if write:
            macro_writes += 1
            written[row] = True
        if issue:
            if last:
                row_cursor += 1
                beat = 0
            else:
                beat += 1
        cycles += 1
        require(cycles <= ideal_issues + len(requirements) + len(active_order) + 8, "single-port task failed to make bounded progress")

    # No speculative prefetch is allowed beyond the final real consumer.
    require(next_requirement == len(requirements), "not every parent edge was prefetched")
    require(pending is None, "orphan pending response after final row")
    require(len(queue) == 0, "parent queue did not drain")
    require(issue_cycles == ideal_issues, "single-port schedule changed arithmetic issue count")
    require(macro_reads + forwarded_reads == len(requirements), "parent edge accounting mismatch")
    require(macro_writes == len(active_order), "active-row write accounting mismatch")
    return {
        "row_count": int(masks.size),
        "active_rows": len(active_order),
        "parent_edges": len(requirements),
        "ideal_1r1w_issue_cycles": ideal_issues,
        "single_port_issue_window_cycles": cycles,
        "single_port_stall_cycles": stall_cycles,
        "macro_reads": macro_reads,
        "macro_writes": macro_writes,
        "forwarded_reads": forwarded_reads,
        "concurrent_issue_reads": concurrent_issue_reads,
        "concurrent_issue_forwards": concurrent_issue_forwards,
        "empty_queue_parent_stalls": empty_queue_parent_stalls,
        "full_queue_prefetch_blocks": full_queue_prefetch_blocks,
        "producer_not_ready_prefetch_blocks": producer_not_ready_prefetch_blocks,
        "deadline_lookahead_holds": deadline_lookahead_holds,
    }


def optimal_single_port_task_cycles(masks: np.ndarray) -> int:
    """Small-task BFS oracle used only for self-test, never full-run claims."""

    masks = np.asarray(masks, dtype=np.uint16)
    residual, parent = cleanroom_subset(masks)
    original_pc = POPCOUNT[masks].astype(np.int32)
    residual_pc = POPCOUNT[residual].astype(np.int32)
    order = np.lexsort((np.arange(masks.size, dtype=np.int32), original_pc))
    active_order = [int(row) for row in order if int(masks[row]) != 0]
    requirements = [int(parent[row]) for row in active_order if int(parent[row]) >= 0]
    producer_position = {row: cursor for cursor, row in enumerate(active_order)}
    # State: active-row cursor, beat, accepted-edge prefix, queue occupancy,
    # one synchronous response pending.  FIFO IDs and written rows are implied
    # by in-order edge acceptance and the fixed topological issue prefix.
    start = (0, 0, 0, 0, 0)
    distance = {start: 0}
    frontier = deque([start])
    while frontier:
        row_cursor, beat, accepted, queue_count, pending = frontier.popleft()
        cycles = distance[(row_cursor, beat, accepted, queue_count, pending)]
        if row_cursor == len(active_order):
            if accepted == len(requirements) and queue_count == 0 and pending == 0:
                return cycles
            continue
        row = active_order[row_cursor]
        parent_id = int(parent[row])
        work = int(residual_pc[row])
        if parent_id >= 0 and work == 0:
            work = 1
        issue_possible = parent_id < 0 or queue_count > 0
        issue_choices = (False, True) if issue_possible else (False,)
        for issue in issue_choices:
            last = bool(issue and beat + 1 == work)
            write = last
            consumed = bool(last and parent_id >= 0)
            reserved = queue_count + pending
            request_exists = accepted < len(requirements)
            requested_parent = requirements[accepted] if request_exists else -1
            producer_ready = bool(
                request_exists
                and producer_position[requested_parent] < row_cursor
            )
            can_forward = bool(
                write and request_exists and reserved < 2 and requested_parent == row
            )
            can_read = bool(
                not write and request_exists and reserved < 2 and producer_ready
            )
            prefetch_choices = (False, True) if (can_forward or can_read) else (False,)
            for prefetch in prefetch_choices:
                if not issue and not prefetch and not pending:
                    continue
                queue_next = queue_count - int(consumed) + pending
                accepted_next = accepted
                pending_next = 0
                if prefetch:
                    accepted_next += 1
                    if can_forward:
                        queue_next += 1
                    else:
                        pending_next = 1
                require(0 <= queue_next <= 2, "oracle queue bound")
                row_next = row_cursor + int(last)
                beat_next = 0 if last else beat + int(issue)
                state = (row_next, beat_next, accepted_next, queue_next, pending_next)
                if state not in distance:
                    distance[state] = cycles + 1
                    frontier.append(state)
    raise RuntimeError("oracle found no legal terminal schedule")


def policy_self_test() -> dict[str, Any]:
    cases = [
        np.asarray([1, 3, 5], dtype=np.uint16),
        np.asarray([1, 3, 7, 15], dtype=np.uint16),
        np.asarray([3, 3, 3, 3], dtype=np.uint16),
        np.asarray([1, 2, 3, 4, 5, 7, 15, 0], dtype=np.uint16),
    ]
    rng = np.random.default_rng(504)
    for _ in range(256):
        count = int(rng.integers(1, 6))
        cases.append(rng.integers(0, 8, size=count, dtype=np.uint16))
    gaps = []
    work_gaps = []
    for masks in cases:
        oracle = optimal_single_port_task_cycles(masks)
        work = simulate_single_port_task(masks, "work_conserving")["single_port_issue_window_cycles"]
        deadline = simulate_single_port_task(masks, "deadline_lookahead")["single_port_issue_window_cycles"]
        require(work >= oracle and deadline >= oracle, "policy beat BFS oracle")
        require(deadline <= work, "deadline policy regressed work-conserving policy")
        gaps.append(deadline - oracle)
        work_gaps.append(work - oracle)
    counterexample = np.asarray([1, 3, 5], dtype=np.uint16)
    require(optimal_single_port_task_cycles(counterexample) == 4, "counterexample oracle drift")
    require(simulate_single_port_task(counterexample, "work_conserving")["single_port_issue_window_cycles"] == 5, "counterexample work-conserving drift")
    require(simulate_single_port_task(counterexample, "deadline_lookahead")["single_port_issue_window_cycles"] == 4, "counterexample deadline drift")
    return {
        "cases": len(cases),
        "deadline_nonoptimal_cases": int(np.count_nonzero(np.asarray(gaps))),
        "deadline_max_gap_cycles": int(max(gaps, default=0)),
        "work_conserving_nonoptimal_cases": int(np.count_nonzero(np.asarray(work_gaps))),
        "work_conserving_max_gap_cycles": int(max(work_gaps, default=0)),
        "masks_1_3_5": {"oracle": 4, "work_conserving": 5, "deadline_lookahead": 4},
    }


def worker_init(rows_path: str) -> None:
    global _ROWS_FD
    _ROWS_FD = os.open(rows_path, os.O_RDONLY)


def read_phase(phase_index: int) -> np.ndarray:
    require(_ROWS_FD is not None, "worker rows file is not open")
    phase_bytes = ROWS_PER_PHASE * BYTES_PER_ROW
    raw = os.pread(_ROWS_FD, phase_bytes, phase_index * phase_bytes)
    require(len(raw) == phase_bytes, f"short phase read: {phase_index}")
    lines = raw.splitlines()
    require(len(lines) == ROWS_PER_PHASE, f"phase row mismatch: {phase_index}")
    words = np.fromiter((int(line, 16) for line in lines), dtype=np.uint32)
    return np.bitwise_and(words, np.uint32(0xFFFF)).astype(np.uint16)


def worker_phase(phase_index: int) -> tuple[int, dict[str, np.ndarray]]:
    masks = read_phase(phase_index)
    chunks = int(math.ceil(ROWS_PER_PHASE / ROW_TILE))
    fields = {
        key: np.zeros(chunks, dtype=np.int32)
        for key in (
            "row_count", "active_rows", "search_rows", "parent_edges",
            "ideal_1r1w_issue_cycles", "work_conserving_cycles",
            "work_conserving_stall_cycles", "deadline_lookahead_cycles",
            "deadline_lookahead_stall_cycles", "deadline_lookahead_holds",
            "deadline_macro_reads", "deadline_macro_writes",
            "deadline_forwarded_reads", "deadline_concurrent_issue_reads",
            "deadline_concurrent_issue_forwards", "deadline_empty_queue_parent_stalls",
            "deadline_full_queue_prefetch_blocks",
            "deadline_producer_not_ready_prefetch_blocks",
        )
    }
    for chunk, start in enumerate(range(0, ROWS_PER_PHASE, ROW_TILE)):
        tile_masks = masks[start:min(start + ROW_TILE, ROWS_PER_PHASE)]
        work = simulate_single_port_task(tile_masks, "work_conserving")
        deadline = simulate_single_port_task(tile_masks, "deadline_lookahead")
        for key in ("row_count", "active_rows", "parent_edges", "ideal_1r1w_issue_cycles"):
            require(work[key] == deadline[key], f"policy semantic count drift: {key}")
            fields[key][chunk] = work[key]
        fields["search_rows"][chunk] = int(np.count_nonzero(POPCOUNT[tile_masks] > 1))
        fields["work_conserving_cycles"][chunk] = work["single_port_issue_window_cycles"]
        fields["work_conserving_stall_cycles"][chunk] = work["single_port_stall_cycles"]
        fields["deadline_lookahead_cycles"][chunk] = deadline["single_port_issue_window_cycles"]
        fields["deadline_lookahead_stall_cycles"][chunk] = deadline["single_port_stall_cycles"]
        fields["deadline_lookahead_holds"][chunk] = deadline["deadline_lookahead_holds"]
        for source, destination in (
            ("macro_reads", "deadline_macro_reads"),
            ("macro_writes", "deadline_macro_writes"),
            ("forwarded_reads", "deadline_forwarded_reads"),
            ("concurrent_issue_reads", "deadline_concurrent_issue_reads"),
            ("concurrent_issue_forwards", "deadline_concurrent_issue_forwards"),
            ("empty_queue_parent_stalls", "deadline_empty_queue_parent_stalls"),
            ("full_queue_prefetch_blocks", "deadline_full_queue_prefetch_blocks"),
            ("producer_not_ready_prefetch_blocks", "deadline_producer_not_ready_prefetch_blocks"),
        ):
            fields[destination][chunk] = deadline[source]
    return phase_index, fields


def pipeline_cycles(preprocess: np.ndarray, work: np.ndarray, tail: int) -> int:
    preprocess = np.asarray(preprocess, dtype=np.int64)
    work = np.asarray(work, dtype=np.int64)
    require(preprocess.shape == work.shape and preprocess.size > 0, "pipeline shape mismatch")
    total = int(preprocess[0])
    if preprocess.size > 1:
        total += int(np.maximum(work[:-1], preprocess[1:]).sum())
        total += (preprocess.size - 1) * tail
    return total + int(work[-1]) + tail


def flatten_sample(array: np.ndarray, sample: int) -> np.ndarray:
    return np.asarray(array[sample]).reshape(-1).astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    args.out = args.out.resolve()
    contract = strict_json(args.contract)
    require(contract["schema"] == "m504_h67_single_port_parent_scratch_execution_contract_v3", "contract schema drift")
    require(1 <= args.workers <= int(contract["runtime"]["maximum_workers"]), "worker count outside contract")
    require(not args.out.exists(), f"refuse to overwrite existing output: {args.out}")
    self_test = policy_self_test()

    for item in contract["frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing frozen input: {path}")
        require(sha256_file(path) == item["sha256"], f"frozen SHA drift: {path}")

    rows_path = ROOT / contract["frozen_inputs"]["m410r2_rows"]["path"]
    m473 = strict_json(ROOT / contract["frozen_inputs"]["m473_result"]["path"])
    selected = m473["best_128Bps_feasible_point"]
    require(m473["status"] == "PASS_M473_CPU_DSE_NO_GO", "M473 status drift")
    require(int(selected["row_tile"]) == ROW_TILE, "M473 row tile drift")
    require(int(selected["resident_block_banks"]) == BLOCK_BANKS, "M473 bank drift")
    require(int(selected["bandwidth_bytes_per_cycle"]) == 128, "M473 bandwidth drift")
    require(selected["scratch_latency_mode"] == "fused_forwarded_1r1w", "M473 latency mode drift")

    chunks = int(math.ceil(ROWS_PER_PHASE / ROW_TILE))
    # Preserve the exact M473 task stream: sample, operator, row chunk, then
    # partition.  This is intentionally not operator/partition/chunk.
    shape = (SAMPLES, OPERATORS, chunks, PARTITIONS)
    field_names = (
        "row_count", "active_rows", "search_rows", "parent_edges",
        "ideal_1r1w_issue_cycles", "work_conserving_cycles",
        "work_conserving_stall_cycles", "deadline_lookahead_cycles",
        "deadline_lookahead_stall_cycles", "deadline_lookahead_holds",
        "deadline_macro_reads", "deadline_macro_writes", "deadline_forwarded_reads",
        "deadline_concurrent_issue_reads", "deadline_concurrent_issue_forwards",
        "deadline_empty_queue_parent_stalls", "deadline_full_queue_prefetch_blocks",
        "deadline_producer_not_ready_prefetch_blocks",
    )
    arrays = {key: np.zeros(shape, dtype=np.int32) for key in field_names}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(rows_path),),
    ) as pool:
        for phase, fields in pool.map(worker_phase, range(PHASES), chunksize=args.chunksize):
            sample = phase // (OPERATORS * PARTITIONS)
            operator = (phase // PARTITIONS) % OPERATORS
            partition = phase % PARTITIONS
            for key in field_names:
                arrays[key][sample, operator, :, partition] = fields[key]

    # Rebuild the frozen M473 task pipeline at its selected coordinate.  Only
    # product work is replaced by the exact single-port issue window.
    deadline_total_without_commit = 0
    work_conserving_total_without_commit = 0
    ideal_total_without_commit = 0
    tail = int(contract["cycle_model"]["tail_cycles_per_pass"])
    weight_dma = int(contract["cycle_model"]["eight_bank_weight_dma_cycles"])
    cam_lanes = int(contract["cycle_model"]["cam_compare_lanes"])
    for sample in range(SAMPLES):
        row_count = flatten_sample(arrays["row_count"], sample)
        active_rows = flatten_sample(arrays["active_rows"], sample)
        search_rows = flatten_sample(arrays["search_rows"], sample)
        capture = (row_count + 7) // 8
        frontend = capture + search_rows * ((row_count + cam_lanes - 1) // cam_lanes) + 17 * capture + 2
        preprocess = np.where(active_rows != 0, np.maximum(frontend, weight_dma), frontend)
        ideal_work = flatten_sample(arrays["ideal_1r1w_issue_cycles"], sample) * BLOCK_BANKS
        work_conserving_work = flatten_sample(arrays["work_conserving_cycles"], sample) * BLOCK_BANKS
        deadline_work = flatten_sample(arrays["deadline_lookahead_cycles"], sample) * BLOCK_BANKS
        ideal_total_without_commit += pipeline_cycles(preprocess, ideal_work, tail)
        work_conserving_total_without_commit += pipeline_cycles(preprocess, work_conserving_work, tail)
        deadline_total_without_commit += pipeline_cycles(preprocess, deadline_work, tail)
    commit_cycles = int(contract["cycle_model"]["commit_cycles_total"])
    ideal_total = ideal_total_without_commit + commit_cycles
    work_conserving_total = work_conserving_total_without_commit + commit_cycles
    deadline_total = deadline_total_without_commit + commit_cycles
    require(ideal_total == int(selected["product_cycles"]), "reconstructed M473 fused-cycle anchor mismatch")

    aggregate = {key: int(value.sum()) for key, value in arrays.items()}
    dp_area = float(contract["macro_model"]["dual_port_fallback_area_um2"])
    sp_area = float(contract["macro_model"]["single_port_generated_area_um2"])
    result = {
        "schema": "m504_h67_single_port_parent_scratch_result_v3",
        "date": contract["date"],
        "status": "PASS_M504_SINGLE_PORT_FASTKILL",
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators; exact cycle audit only",
        "identity": {
            "contract": {"path": str(args.contract.relative_to(ROOT)), "sha256": sha256_file(args.contract)},
            **contract["frozen_inputs"],
        },
        "population": {
            "samples": SAMPLES, "operators": OPERATORS, "partitions": PARTITIONS,
            "rows_per_phase": ROWS_PER_PHASE, "row_tile": ROW_TILE,
            "tasks": int(np.prod(shape[:-1]) * shape[-1]),
        },
        "small_task_bfs_oracle": self_test,
        "aggregate_one_output_block": aggregate,
        "cycle_comparison": {
            "m473_ideal_1r1w_cycles": ideal_total,
            "work_conserving_single_port_cycles": work_conserving_total,
            "deadline_lookahead_single_port_cycles": deadline_total,
            "deadline_lookahead_cycle_overhead_fraction": deadline_total / ideal_total - 1.0,
            "deadline_lookahead_retained_speedup_vs_same_coordinate_bit": int(selected["bit_cycles"]) / deadline_total,
            "deadline_lookahead_retained_speedup_vs_best_same_budget_m468_zero": int(selected["best_same_budget_m468_zero_cycles"]) / deadline_total,
            "deadline_lookahead_policy_is_global_optimum": False,
            "performance_admitted": False,
            "system_speedup": False,
        },
        "macro_comparison": {
            "dual_port_fallback_area_um2": dp_area,
            "dual_port_overdepth_proxy_area_um2": float(contract["macro_model"]["dual_port_overdepth_proxy_area_um2"]),
            "single_port_generated_area_um2": sp_area,
            "area_reduction_fraction_vs_exact_capacity_fallback": 1.0 - sp_area / dp_area,
            "area_reduction_ratio_vs_exact_capacity_fallback": dp_area / sp_area,
            "area_reduction_fraction_vs_overdepth_proxy": 1.0 - sp_area / float(contract["macro_model"]["dual_port_overdepth_proxy_area_um2"]),
            "evidence_boundary": "DP is foundry-QRT fallback; SP is nine instances of an existing generated 128x128 1RW macro. Neither is integrated post-layout area.",
        },
        "decision": {},
        "claim_boundary": contract["claim_boundary"],
    }
    gates = contract["materiality_gates"]
    result["decision"] = {
        "cycle_overhead_gate_pass": result["cycle_comparison"]["deadline_lookahead_cycle_overhead_fraction"] <= float(gates["maximum_cycle_overhead_fraction"]),
        "retained_speedup_gate_pass": result["cycle_comparison"]["deadline_lookahead_retained_speedup_vs_best_same_budget_m468_zero"] >= float(gates["minimum_speedup_vs_m468_zero"]),
        "exact_fallback_area_reduction_gate_pass": result["macro_comparison"]["area_reduction_fraction_vs_exact_capacity_fallback"] >= float(gates["minimum_area_reduction_vs_exact_capacity_fallback"]),
        "overdepth_proxy_area_reduction_gate_pass": result["macro_comparison"]["area_reduction_fraction_vs_overdepth_proxy"] >= float(gates["minimum_area_reduction_vs_overdepth_proxy"]),
    }
    result["decision"]["rtl_nomination"] = all(result["decision"].values())
    result["decision"]["verdict"] = "GO_M504_SINGLE_PORT_RTL" if result["decision"]["rtl_nomination"] else "NO_GO_M504_RTL"

    args.out.mkdir(parents=True, exist_ok=False)
    result_path = args.out / "m504_h67_single_port_parent_scratch_result_r3.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.out / "m504_operator_sample_summary_r3.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample", "operator", "ideal_cycles", "work_conserving_cycles",
                "deadline_lookahead_cycles", "work_conserving_stall_cycles",
                "deadline_lookahead_stall_cycles", "deadline_lookahead_holds",
                "parent_edges", "deadline_macro_reads", "deadline_forwarded_reads",
            ],
        )
        writer.writeheader()
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                writer.writerow({
                    "sample": sample, "operator": operator,
                    "ideal_cycles": int(arrays["ideal_1r1w_issue_cycles"][sample, operator].sum()),
                    "work_conserving_cycles": int(arrays["work_conserving_cycles"][sample, operator].sum()),
                    "deadline_lookahead_cycles": int(arrays["deadline_lookahead_cycles"][sample, operator].sum()),
                    "work_conserving_stall_cycles": int(arrays["work_conserving_stall_cycles"][sample, operator].sum()),
                    "deadline_lookahead_stall_cycles": int(arrays["deadline_lookahead_stall_cycles"][sample, operator].sum()),
                    "deadline_lookahead_holds": int(arrays["deadline_lookahead_holds"][sample, operator].sum()),
                    "parent_edges": int(arrays["parent_edges"][sample, operator].sum()),
                    "deadline_macro_reads": int(arrays["deadline_macro_reads"][sample, operator].sum()),
                    "deadline_forwarded_reads": int(arrays["deadline_forwarded_reads"][sample, operator].sum()),
                })
    manifest = args.out / "SHA256SUMS"
    manifest.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in sorted((result_path, csv_path))) + "\n",
        encoding="utf-8",
    )
    (args.out / "SHA256SUMS.seal.sha256").write_text(f"{sha256_file(manifest)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps(result["cycle_comparison"], sort_keys=True))
    print(json.dumps(result["macro_comparison"], sort_keys=True))
    print(json.dumps(result["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
