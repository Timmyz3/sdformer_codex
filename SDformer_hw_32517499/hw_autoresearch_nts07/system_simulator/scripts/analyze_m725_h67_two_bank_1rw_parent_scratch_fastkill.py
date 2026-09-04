#!/usr/bin/env python3
"""Stratified exact fast-kill for a two-bank 1RW H67 parent scratch.

M528 maps one 1152-bit logical parent row across nine 128-bit 1RW macros, so
those nine macros are bit slices and cannot provide cross-macro pseudo-dual
porting.  M725 instead evaluates the physically legal alternative: two
complete nine-macro banks, with a compile-time balanced XOR of the six-bit row
address selecting the bank.  One read and one write may proceed together only
when they select different banks.  Arithmetic, parent selection, issue order,
two-entry response queue, forwarding, dead-write elision and synchronous read
latency are otherwise identical to M505.

This first pass is deliberately stratified and reports local issue-window
cycles only.  It may nominate one mapping for a full same-ledger replay; it
cannot admit a Conv or system speedup.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
M504_PATH = ROOT / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
M505_PATH = ROOT / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M504 = load(M504_PATH, "m504_for_m725")
M505 = load(M505_PATH, "m505_for_m725")
DEFAULT_CONTRACT = ROOT / "contracts/m725_h67_two_bank_1rw_parent_scratch_fastkill_contract_r1_20260828.json"
DEFAULT_OUT = ROOT / "results/m725_h67_two_bank_1rw_parent_scratch_fastkill_r1_20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bank(row: int, xor_mask: int) -> int:
    return (int(row) & int(xor_mask)).bit_count() & 1


def simulate_two_bank(masks: np.ndarray, xor_mask: int) -> dict[str, int]:
    """M505 dead-write-only schedule with two balanced single-port banks."""

    M504.require(1 <= xor_mask < 64, "unbalanced/zero XOR bank mask")
    masks = np.asarray(masks, dtype=np.uint16)
    residual, parent = M504.cleanroom_subset(masks)
    original_pc = M504.POPCOUNT[masks].astype(np.int32)
    residual_pc = M504.POPCOUNT[residual].astype(np.int32)
    order = np.lexsort((np.arange(masks.size, dtype=np.int32), original_pc))
    active_order = [int(row) for row in order if int(masks[row]) != 0]
    requirements = [int(parent[row]) for row in active_order if int(parent[row]) >= 0]
    consumers = [cursor for cursor, row in enumerate(active_order)
                 if int(parent[row]) >= 0]
    use_count = np.bincount(np.asarray(requirements, dtype=np.int64),
                            minlength=masks.size).astype(np.int32)
    ideal = int(residual_pc.sum() + np.count_nonzero(
        (parent >= 0) & (residual == 0) & (masks != 0)))

    queue: list[int] = []
    pending: int | None = None
    next_requirement = 0
    row_cursor = 0
    beat_index = 0
    cycles = issues = stalls = reads = writes = forwards = holds = 0
    same_bank_conflicts = cross_bank_read_writes = 0
    written = np.zeros(masks.size, dtype=np.bool_)

    while row_cursor < len(active_order):
        row = active_order[row_cursor]
        parent_id = int(parent[row])
        work = int(residual_pc[row])
        if parent_id >= 0 and work == 0:
            work = 1
        M504.require(work > 0, "active row has no work")

        parent_ready = parent_id < 0 or bool(queue and queue[0] == parent_id)
        final_if_issued = bool(parent_ready and beat_index + 1 == work)
        reserved = len(queue) + int(pending is not None)
        M504.require(reserved <= 2, "response queue over-reserved")
        request_exists = next_requirement < len(requirements)
        requested_parent = requirements[next_requirement] if request_exists else -1
        requested_consumer = consumers[next_requirement] if request_exists else -1
        capacity = reserved < 2
        producer_ready = bool(request_exists and written[requested_parent])
        predicted_forward = bool(final_if_issued and request_exists and capacity
                                 and requested_parent == row)
        predicted_write = bool(final_if_issued and int(use_count[row]) > 0)
        predicted_same_bank = bool(predicted_write and request_exists and
                                   producer_ready and not predicted_forward and
                                   bank(row, xor_mask) == bank(requested_parent, xor_mask))
        deadline_hold = bool(predicted_same_bank and requested_consumer == row_cursor + 1)
        issue = bool(parent_ready and not deadline_hold)
        last = bool(issue and beat_index + 1 == work)
        forward = bool(last and request_exists and capacity and requested_parent == row)
        write = bool(last and int(use_count[row]) > 0)
        same_bank = bool(write and request_exists and producer_ready and not forward and
                         bank(row, xor_mask) == bank(requested_parent, xor_mask))
        read = bool(request_exists and capacity and producer_ready and not forward and
                    (not write or not same_bank))
        M504.require(not (read and write and same_bank), "same-bank 1RW collision")
        holds += int(deadline_hold)
        same_bank_conflicts += int(same_bank)
        cross_bank_read_writes += int(read and write)
        consumed = bool(last and parent_id >= 0)
        issues += int(issue)
        stalls += int(not issue)

        if consumed:
            M504.require(queue and queue[0] == parent_id, "FIFO parent mismatch")
            queue.pop(0)
        if pending is not None:
            M504.require(len(queue) < 2, "response overflow")
            queue.append(pending)
        if forward:
            M504.require(len(queue) < 2, "forward overflow")
            queue.append(requested_parent)
            next_requirement += 1
            forwards += 1
        if read:
            next_pending: int | None = requested_parent
            next_requirement += 1
            reads += 1
        else:
            next_pending = None
        pending = next_pending

        if last and write:
            writes += 1
            written[row] = True
        if issue:
            if last:
                row_cursor += 1
                beat_index = 0
            else:
                beat_index += 1
        cycles += 1
        M504.require(cycles <= ideal + len(requirements) + len(active_order) + 8,
                     "two-bank schedule failed bounded progress")

    M504.require(next_requirement == len(requirements), "unserved parent edge")
    M504.require(pending is None and not queue, "response queue did not drain")
    M504.require(issues == ideal, "arithmetic issue drift")
    M504.require(reads + forwards == len(requirements), "parent edge drift")
    M504.require(writes == int(np.count_nonzero(use_count)), "live write drift")
    return {
        "cycles": cycles, "issues": issues, "stalls": stalls,
        "reads": reads, "writes": writes, "forwards": forwards,
        "deadline_holds": holds, "same_bank_conflicts": same_bank_conflicts,
        "cross_bank_read_writes": cross_bank_read_writes,
        "parent_edges": len(requirements), "active_rows": len(active_order),
    }


def worker_init(rows_path: str) -> None:
    M504.worker_init(rows_path)


def worker_phase(item: tuple[int, tuple[int, ...]]) -> dict[str, Any]:
    phase, mappings = item
    masks = M504.read_phase(phase)
    totals = {str(mask): {key: 0 for key in (
        "cycles", "issues", "stalls", "reads", "writes", "forwards",
        "deadline_holds", "same_bank_conflicts", "cross_bank_read_writes",
        "parent_edges", "active_rows")} for mask in mappings}
    single_cycles = single_stalls = 0
    for start in range(0, M504.ROWS_PER_PHASE, M504.ROW_TILE):
        tile = masks[start:min(start + M504.ROW_TILE, M504.ROWS_PER_PHASE)]
        single = M505.simulate_liveness_task(tile, False)
        single_cycles += int(single["liveness_cycles"])
        single_stalls += int(single["liveness_stall_cycles"])
        for mask in mappings:
            row = simulate_two_bank(tile, mask)
            M504.require(row["cycles"] <= int(single["liveness_cycles"]),
                         "two-bank schedule regressed single bank")
            for key, value in row.items():
                totals[str(mask)][key] += int(value)
    return {"phase": phase, "single_cycles": single_cycles,
            "single_stalls": single_stalls, "mappings": totals}


def selftest(mappings: tuple[int, ...]) -> dict[str, Any]:
    rng = np.random.default_rng(725)
    strict = 0
    for _ in range(512):
        count = int(rng.integers(1, 65))
        masks = rng.integers(0, 1 << 16, size=count, dtype=np.uint16)
        single = M505.simulate_liveness_task(masks, False)
        for mapping in mappings:
            dual = simulate_two_bank(masks, mapping)
            M504.require(dual["issues"] == single["ideal_1r1w_issue_cycles"],
                         "selftest issue drift")
            M504.require(dual["parent_edges"] == single["parent_edges"],
                         "selftest edge drift")
            M504.require(dual["cycles"] <= single["liveness_cycles"],
                         "selftest cycle regression")
            strict += int(dual["cycles"] < single["liveness_cycles"])
    return {"cases": 512, "mappings_per_case": len(mappings),
            "zero_regressions": True, "strict_improvements": strict}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    contract = M504.strict_json(args.contract.resolve())
    M504.require(contract["schema"] == "m725_h67_two_bank_1rw_parent_scratch_fastkill_contract_v1",
                 "contract schema drift")
    M504.require(not args.out.exists(), "refuse output overwrite")
    M504.require(1 <= args.workers <= contract["runtime"]["maximum_workers"],
                 "worker count outside contract")
    for item in contract["frozen_inputs"].values():
        path = ROOT / item["path"]
        M504.require(path.is_file() and not path.is_symlink(), "missing input")
        M504.require(sha256(path) == item["sha256"], "input SHA drift")
    mappings = tuple(int(value) for value in contract["dse"]["xor_masks"])
    M504.require(all(1 <= value < 64 for value in mappings), "bad mapping")
    M504.require(len(set(mappings)) == len(mappings), "duplicate mapping")
    test = selftest(mappings)

    partitions = tuple(int(value) for value in contract["population"]["partitions_per_sample_operator"])
    phases = tuple(sample * (M504.OPERATORS * M504.PARTITIONS) +
                   operator * M504.PARTITIONS + partition
                   for sample in range(M504.SAMPLES)
                   for operator in range(M504.OPERATORS)
                   for partition in partitions)
    rows_path = ROOT / contract["frozen_inputs"]["row_ledger"]["path"]
    aggregate = {str(mask): {key: 0 for key in (
        "cycles", "issues", "stalls", "reads", "writes", "forwards",
        "deadline_holds", "same_bank_conflicts", "cross_bank_read_writes",
        "parent_edges", "active_rows")} for mask in mappings}
    single_cycles = single_stalls = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=worker_init, initargs=(str(rows_path),)) as pool:
        for result in pool.map(worker_phase, ((phase, mappings) for phase in phases), chunksize=1):
            single_cycles += result["single_cycles"]
            single_stalls += result["single_stalls"]
            for mask in mappings:
                for key, value in result["mappings"][str(mask)].items():
                    aggregate[str(mask)][key] += value

    rows = []
    for mask in mappings:
        row = aggregate[str(mask)]
        row.update({
            "xor_mask": mask,
            "single_bank_cycles": single_cycles,
            "single_bank_stalls": single_stalls,
            "local_issue_window_speedup": single_cycles / row["cycles"],
            "stall_reduction_fraction": 1.0 - row["stalls"] / max(single_stalls, 1),
        })
        rows.append(row)
    rows.sort(key=lambda value: (value["cycles"], value["xor_mask"]))
    best = rows[0]
    capacity = contract["capacity"]
    decision = {
        "balanced_two_bank_capacity_fits_240k": capacity["candidate_macro_rounded_bytes"] <= capacity["budget_bytes"],
        "local_issue_window_speedup_gate": best["local_issue_window_speedup"] >= contract["gates"]["minimum_local_issue_window_speedup"],
        "stall_reduction_gate": best["stall_reduction_fraction"] >= contract["gates"]["minimum_stall_reduction_fraction"],
    }
    decision["go_full_same_ledger_replay"] = all(decision.values())
    output = {
        "schema": "m725_h67_two_bank_1rw_parent_scratch_fastkill_result_v1",
        "status": "PASS_STRATIFIED_CPU_FASTKILL",
        "identity": {"contract_sha256": sha256(args.contract.resolve()),
                     "docs359_sha256": contract["frozen_inputs"]["docs359"]["sha256"]},
        "population": {"selected_phases": len(phases), "total_phases": M504.PHASES,
                       "samples": M504.SAMPLES, "operators": M504.OPERATORS,
                       "partitions": list(partitions), "row_chunks_per_phase": math.ceil(M504.ROWS_PER_PHASE / M504.ROW_TILE)},
        "selftest": test, "capacity": capacity, "mapping_rows": rows,
        "best_mapping": best, "decision": decision,
        "claim_boundary": contract["claim_boundary"],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    result_path = args.out / "m725_h67_two_bank_1rw_parent_scratch_fastkill_result_r1.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = args.out / "SHA256SUMS"
    manifest.write_text("{}  {}\n".format(sha256(result_path), result_path.name), encoding="utf-8")
    seal = args.out / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
    print(json.dumps({"best": best, "decision": decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
