#!/usr/bin/env python3
"""M505 exact liveness-aware 1RW parent-scratch audit on frozen H67 rows.

M505 retains M504's exact row order, arithmetic, two-entry ordered response
queue and synchronous single-port macro.  It adds a two-bit saturating
reference class per row: dead results are never written to parent scratch, and
a single-use result forwarded at producer completion is also not written.
The architectural output commit is unchanged and is outside parent scratch.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
M504_PATH = ROOT / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
_SPEC = importlib.util.spec_from_file_location("m504_frozen", M504_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load frozen M504 analyzer")
M504 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M504)

DEFAULT_CONTRACT = ROOT / "contracts/m505_h67_liveness_aware_single_port_parent_scratch_contract_r1_20260827.json"
DEFAULT_OUT = ROOT / "results/m505_h67_liveness_aware_single_port_parent_scratch_r1_20260827"


def simulate_liveness_task(
    masks: np.ndarray, enable_single_use_forward_elision: bool
) -> dict[str, int]:
    """Execute dead-write-only or combined liveness-aware scratch stores."""

    masks = np.asarray(masks, dtype=np.uint16)
    residual, parent = M504.cleanroom_subset(masks)
    original_pc = M504.POPCOUNT[masks].astype(np.int32)
    residual_pc = M504.POPCOUNT[residual].astype(np.int32)
    order = np.lexsort((np.arange(masks.size, dtype=np.int32), original_pc))
    active_order = [int(row) for row in order if int(masks[row]) != 0]
    requirements = [int(parent[row]) for row in active_order if int(parent[row]) >= 0]
    requirement_consumers = [
        cursor for cursor, row in enumerate(active_order) if int(parent[row]) >= 0
    ]
    use_count = np.bincount(
        np.asarray(requirements, dtype=np.int64), minlength=masks.size
    ).astype(np.int32)
    use_class = np.minimum(use_count, 2)
    ideal_issues = int(
        residual_pc.sum()
        + np.count_nonzero((parent >= 0) & (residual == 0) & (masks != 0))
    )

    queue: list[int] = []
    pending: int | None = None
    next_requirement = 0
    row_cursor = 0
    beat = 0
    cycles = 0
    issue_cycles = 0
    stall_cycles = 0
    macro_reads = 0
    macro_writes = 0
    forwarded_reads = 0
    concurrent_issue_reads = 0
    concurrent_issue_forwards = 0
    deadline_holds = 0
    dead_writes_elided = 0
    single_use_forwarded_writes_elided = 0
    written = np.zeros(masks.size, dtype=np.bool_)

    while row_cursor < len(active_order):
        row = active_order[row_cursor]
        parent_id = int(parent[row])
        work = int(residual_pc[row])
        if parent_id >= 0 and work == 0:
            work = 1
        M504.require(work > 0, "active row has zero executable work")

        parent_ready = parent_id < 0 or (queue and queue[0] == parent_id)
        issue_possible = bool(parent_ready)
        final_if_issued = bool(issue_possible and beat + 1 == work)
        reserved = len(queue) + int(pending is not None)
        M504.require(reserved <= 2, "parent response queue over-reserved")
        request_exists = next_requirement < len(requirements)
        requested_parent = requirements[next_requirement] if request_exists else -1
        requested_consumer = requirement_consumers[next_requirement] if request_exists else -1
        has_capacity = reserved < 2
        producer_ready = bool(request_exists and written[requested_parent])

        predicted_forward = bool(
            final_if_issued and request_exists and has_capacity and requested_parent == row
        )
        predicted_elided_uses = int(
            predicted_forward
            and enable_single_use_forward_elision
            and requested_consumer == row_cursor + 1
        )
        predicted_write = bool(
            final_if_issued and int(use_count[row]) > predicted_elided_uses
        )
        deadline_hold = bool(
            predicted_write
            and request_exists
            and has_capacity
            and producer_ready
            and requested_parent != row
            and requested_consumer == row_cursor + 1
        )
        issue = bool(issue_possible and not deadline_hold)
        last = bool(issue and beat + 1 == work)
        forward = bool(
            last and request_exists and has_capacity and requested_parent == row
        )
        elided_uses = int(
            forward
            and enable_single_use_forward_elision
            and requested_consumer == row_cursor + 1
        )
        write = bool(last and int(use_count[row]) > elided_uses)
        read = bool(
            (not write)
            and (not forward)
            and request_exists
            and has_capacity
            and written[requested_parent]
        )
        M504.require(not (read and write), "single-port read/write collision")

        if deadline_hold:
            deadline_holds += 1
        consumed = bool(last and parent_id >= 0)
        if issue:
            issue_cycles += 1
            concurrent_issue_reads += int(read)
            concurrent_issue_forwards += int(forward)
        else:
            stall_cycles += 1

        if consumed:
            M504.require(queue and queue[0] == parent_id, "consumed parent is not FIFO head")
            queue.pop(0)
        if pending is not None:
            M504.require(len(queue) < 2, "return response overflow")
            queue.append(pending)
        if forward:
            M504.require(len(queue) < 2, "forward response overflow")
            queue.append(requested_parent)
            next_requirement += 1
            forwarded_reads += 1
        if read:
            M504.require(written[requested_parent], "read before producer store")
            next_pending: int | None = requested_parent
            next_requirement += 1
            macro_reads += 1
        else:
            next_pending = None
        pending = next_pending

        if last:
            if write:
                macro_writes += 1
                written[row] = True
            elif int(use_class[row]) == 0:
                dead_writes_elided += 1
            elif enable_single_use_forward_elision:
                M504.require(
                    int(use_class[row]) == 1
                    and forward
                    and requested_consumer == row_cursor + 1,
                    "live store elided without sole-use immediate-next forwarding",
                )
                single_use_forwarded_writes_elided += 1
            else:
                M504.require(False, "dead-only mode elided a live parent store")
        if issue:
            if last:
                row_cursor += 1
                beat = 0
            else:
                beat += 1
        cycles += 1
        M504.require(
            cycles <= ideal_issues + len(requirements) + len(active_order) + 8,
            "liveness schedule failed bounded progress",
        )

    M504.require(next_requirement == len(requirements), "not every parent edge was served")
    M504.require(pending is None and not queue, "parent response queue did not drain")
    M504.require(issue_cycles == ideal_issues, "liveness changed arithmetic issue count")
    M504.require(macro_reads + forwarded_reads == len(requirements), "edge accounting mismatch")
    M504.require(
        macro_writes + dead_writes_elided + single_use_forwarded_writes_elided == len(active_order),
        "write-liveness accounting mismatch",
    )
    expected_dead = sum(1 for row in active_order if int(use_class[row]) == 0)
    M504.require(dead_writes_elided == expected_dead, "dead-write count mismatch")
    return {
        "row_count": int(masks.size),
        "active_rows": len(active_order),
        "parent_edges": len(requirements),
        "unique_live_parents": int(np.count_nonzero(use_count)),
        "single_use_parents": int(np.count_nonzero(use_count == 1)),
        "multi_use_parents": int(np.count_nonzero(use_count > 1)),
        "maximum_parent_refcount": int(use_count.max(initial=0)),
        "ideal_1r1w_issue_cycles": ideal_issues,
        "liveness_cycles": cycles,
        "liveness_stall_cycles": stall_cycles,
        "liveness_deadline_holds": deadline_holds,
        "macro_reads": macro_reads,
        "macro_writes": macro_writes,
        "forwarded_reads": forwarded_reads,
        "concurrent_issue_reads": concurrent_issue_reads,
        "concurrent_issue_forwards": concurrent_issue_forwards,
        "dead_writes_elided": dead_writes_elided,
        "single_use_forwarded_writes_elided": single_use_forwarded_writes_elided,
    }


def policy_self_test() -> dict[str, Any]:
    cases = [
        np.asarray([1, 3, 5], dtype=np.uint16),
        np.asarray([1, 3, 7, 15], dtype=np.uint16),
        np.asarray([3, 3, 3, 3], dtype=np.uint16),
        np.asarray([1, 2, 3, 4, 5, 7, 15, 0], dtype=np.uint16),
    ]
    rng = np.random.default_rng(505)
    for _ in range(1024):
        count = int(rng.integers(1, 9))
        cases.append(rng.integers(0, 16, size=count, dtype=np.uint16))
    strict_improvements = 0
    maximum_improvement = 0
    for masks in cases:
        old = M504.simulate_single_port_task(masks, "deadline_lookahead")
        dead = simulate_liveness_task(masks, False)
        new = simulate_liveness_task(masks, True)
        M504.require(new["ideal_1r1w_issue_cycles"] == old["ideal_1r1w_issue_cycles"], "self-test ideal drift")
        M504.require(new["parent_edges"] == old["parent_edges"], "self-test edge drift")
        M504.require(dead["liveness_cycles"] <= old["single_port_issue_window_cycles"], "dead-only policy regressed M504")
        M504.require(new["liveness_cycles"] <= old["single_port_issue_window_cycles"], "liveness policy regressed M504")
        M504.require(new["liveness_cycles"] <= dead["liveness_cycles"], "combined policy regressed dead-only")
        improvement = old["single_port_issue_window_cycles"] - new["liveness_cycles"]
        strict_improvements += int(improvement > 0)
        maximum_improvement = max(maximum_improvement, improvement)
    return {
        "cases": len(cases),
        "zero_regressions_vs_m504": True,
        "strict_improvement_cases": strict_improvements,
        "maximum_improvement_cycles": maximum_improvement,
    }


def worker_init(rows_path: str) -> None:
    M504.worker_init(rows_path)


def worker_phase(phase_index: int) -> tuple[int, dict[str, np.ndarray]]:
    masks = M504.read_phase(phase_index)
    chunks = int(math.ceil(M504.ROWS_PER_PHASE / M504.ROW_TILE))
    names = (
        "row_count", "active_rows", "parent_edges", "unique_live_parents",
        "single_use_parents", "multi_use_parents", "maximum_parent_refcount",
        "search_rows", "ideal_1r1w_issue_cycles", "m504_deadline_cycles",
        "m504_macro_reads", "m504_macro_writes",
        "dead_only_cycles", "dead_only_stall_cycles", "dead_only_deadline_holds",
        "dead_only_macro_reads", "dead_only_macro_writes", "dead_only_forwarded_reads",
        "liveness_cycles", "liveness_stall_cycles", "liveness_deadline_holds",
        "macro_reads", "macro_writes", "forwarded_reads",
        "concurrent_issue_reads", "concurrent_issue_forwards", "dead_writes_elided",
        "single_use_forwarded_writes_elided",
    )
    fields = {name: np.zeros(chunks, dtype=np.int32) for name in names}
    dead_only_map = {
        "dead_only_cycles": "liveness_cycles",
        "dead_only_stall_cycles": "liveness_stall_cycles",
        "dead_only_deadline_holds": "liveness_deadline_holds",
        "dead_only_macro_reads": "macro_reads",
        "dead_only_macro_writes": "macro_writes",
        "dead_only_forwarded_reads": "forwarded_reads",
    }
    for chunk, start in enumerate(range(0, M504.ROWS_PER_PHASE, M504.ROW_TILE)):
        tile = masks[start:min(start + M504.ROW_TILE, M504.ROWS_PER_PHASE)]
        old = M504.simulate_single_port_task(tile, "deadline_lookahead")
        dead = simulate_liveness_task(tile, False)
        new = simulate_liveness_task(tile, True)
        M504.require(new["ideal_1r1w_issue_cycles"] == old["ideal_1r1w_issue_cycles"], "phase ideal drift")
        M504.require(new["parent_edges"] == old["parent_edges"], "phase edge drift")
        M504.require(dead["liveness_cycles"] <= old["single_port_issue_window_cycles"], "phase dead-only regression")
        M504.require(new["liveness_cycles"] <= old["single_port_issue_window_cycles"], "phase cycle regression")
        for name in names:
            if name == "m504_deadline_cycles":
                fields[name][chunk] = old["single_port_issue_window_cycles"]
            elif name == "m504_macro_reads":
                fields[name][chunk] = old["macro_reads"]
            elif name == "m504_macro_writes":
                fields[name][chunk] = old["macro_writes"]
            elif name == "search_rows":
                fields[name][chunk] = int(np.count_nonzero(M504.POPCOUNT[tile] > 1))
            elif name.startswith("dead_only_"):
                fields[name][chunk] = dead[dead_only_map[name]]
            else:
                fields[name][chunk] = new[name]
    return phase_index, fields


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
    contract = M504.strict_json(args.contract)
    M504.require(contract["schema"] == "m505_h67_liveness_aware_single_port_parent_scratch_contract_v1", "contract schema drift")
    M504.require(1 <= args.workers <= int(contract["runtime"]["maximum_workers"]), "worker count outside contract")
    M504.require(not args.out.exists(), "refuse to overwrite output")
    self_test = policy_self_test()
    for item in contract["frozen_inputs"].values():
        path = ROOT / item["path"]
        M504.require(path.is_file(), f"missing frozen input: {path}")
        M504.require(M504.sha256_file(path) == item["sha256"], f"frozen SHA drift: {path}")

    m473 = M504.strict_json(ROOT / contract["frozen_inputs"]["m473_result"]["path"])
    m504_result = M504.strict_json(ROOT / contract["frozen_inputs"]["m504_result"]["path"])
    selected = m473["best_128Bps_feasible_point"]
    M504.require(int(selected["product_cycles"]) == int(contract["cycle_model"]["m473_anchor_cycles"]), "M473 anchor drift")
    M504.require(int(m504_result["cycle_comparison"]["deadline_lookahead_single_port_cycles"]) == int(contract["cycle_model"]["m504_anchor_cycles"]), "M504 anchor drift")

    chunks = int(math.ceil(M504.ROWS_PER_PHASE / M504.ROW_TILE))
    shape = (M504.SAMPLES, M504.OPERATORS, chunks, M504.PARTITIONS)
    field_names = (
        "row_count", "active_rows", "parent_edges", "unique_live_parents",
        "single_use_parents", "multi_use_parents", "maximum_parent_refcount",
        "search_rows", "ideal_1r1w_issue_cycles", "m504_deadline_cycles",
        "m504_macro_reads", "m504_macro_writes",
        "dead_only_cycles", "dead_only_stall_cycles", "dead_only_deadline_holds",
        "dead_only_macro_reads", "dead_only_macro_writes", "dead_only_forwarded_reads",
        "liveness_cycles", "liveness_stall_cycles", "liveness_deadline_holds",
        "macro_reads", "macro_writes", "forwarded_reads",
        "concurrent_issue_reads", "concurrent_issue_forwards", "dead_writes_elided",
        "single_use_forwarded_writes_elided",
    )
    arrays = {name: np.zeros(shape, dtype=np.int32) for name in field_names}
    context = mp.get_context("spawn")
    rows_path = ROOT / contract["frozen_inputs"]["m410r2_rows"]["path"]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(rows_path),),
    ) as pool:
        for phase, fields in pool.map(worker_phase, range(M504.PHASES), chunksize=args.chunksize):
            sample = phase // (M504.OPERATORS * M504.PARTITIONS)
            operator = (phase // M504.PARTITIONS) % M504.OPERATORS
            partition = phase % M504.PARTITIONS
            for name in field_names:
                arrays[name][sample, operator, :, partition] = fields[name]

    totals = {"ideal": 0, "m504": 0, "dead_only": 0, "m505": 0}
    tail = int(contract["cycle_model"]["tail_cycles_per_pass"])
    weight_dma = int(contract["cycle_model"]["eight_bank_weight_dma_cycles"])
    cam_lanes = int(contract["cycle_model"]["cam_compare_lanes"])
    for sample in range(M504.SAMPLES):
        row_count = flatten_sample(arrays["row_count"], sample)
        active_rows = flatten_sample(arrays["active_rows"], sample)
        search = flatten_sample(arrays["search_rows"], sample)
        capture = (row_count + 7) // 8
        frontend = capture + search * ((row_count + cam_lanes - 1) // cam_lanes) + 17 * capture + 2
        preprocess = np.where(active_rows != 0, np.maximum(frontend, weight_dma), frontend)
        for key, field in (
            ("ideal", "ideal_1r1w_issue_cycles"),
            ("m504", "m504_deadline_cycles"),
            ("dead_only", "dead_only_cycles"),
            ("m505", "liveness_cycles"),
        ):
            work = flatten_sample(arrays[field], sample) * M504.BLOCK_BANKS
            totals[key] += M504.pipeline_cycles(preprocess, work, tail)
    commit = int(contract["cycle_model"]["commit_cycles_total"])
    totals = {key: value + commit for key, value in totals.items()}
    M504.require(totals["ideal"] == int(contract["cycle_model"]["m473_anchor_cycles"]), "ideal cycle reconstruction drift")
    M504.require(totals["m504"] == int(contract["cycle_model"]["m504_anchor_cycles"]), "M504 cycle reconstruction drift")

    aggregate = {name: int(value.sum()) for name, value in arrays.items()}
    aggregate["maximum_parent_refcount"] = int(arrays["maximum_parent_refcount"].max(initial=0))
    old_accesses = int(m504_result["aggregate_one_output_block"]["deadline_macro_reads"]) + int(m504_result["aggregate_one_output_block"]["deadline_macro_writes"])
    M504.require(
        aggregate["m504_macro_reads"] + aggregate["m504_macro_writes"] == old_accesses,
        "M504 macro-access anchor drift",
    )
    new_accesses = aggregate["macro_reads"] + aggregate["macro_writes"]
    cycles = {
        "m473_ideal_1r1w_cycles": totals["ideal"],
        "m504_single_port_cycles": totals["m504"],
        "dead_write_only_single_port_cycles": totals["dead_only"],
        "m505_liveness_single_port_cycles": totals["m505"],
        "cycle_overhead_fraction_vs_m473": totals["m505"] / totals["ideal"] - 1.0,
        "speedup_vs_best_same_budget_m468_zero": int(selected["best_same_budget_m468_zero_cycles"]) / totals["m505"],
        "speedup_vs_same_coordinate_bit": int(selected["bit_cycles"]) / totals["m505"],
        "performance_admitted": False,
        "system_speedup": False,
    }
    liveness = {
        "dead_write_fraction_of_active_rows": aggregate["dead_writes_elided"] / aggregate["active_rows"],
        "single_use_forwarded_write_fraction_of_active_rows": aggregate["single_use_forwarded_writes_elided"] / aggregate["active_rows"],
        "scratch_access_reduction_fraction_vs_m504": 1.0 - new_accesses / old_accesses,
        "m504_macro_accesses": old_accesses,
        "m505_macro_accesses": new_accesses,
        "dead_only_macro_accesses": aggregate["dead_only_macro_reads"] + aggregate["dead_only_macro_writes"],
        "dead_only_scratch_access_reduction_fraction_vs_m504": 1.0 - (aggregate["dead_only_macro_reads"] + aggregate["dead_only_macro_writes"]) / old_accesses,
        "refcount_histogram_active_rows": {
            "zero": aggregate["dead_writes_elided"],
            "one": aggregate["single_use_parents"],
            "two_or_more": aggregate["multi_use_parents"],
            "maximum_exact": aggregate["maximum_parent_refcount"]
        },
        "metadata_bits_per_64_row_task": 128,
        "metadata_model": "two-bit saturating reference class per row, generated after exact parent matching and included in future matched RTL",
    }
    macro = {
        "single_port_generated_area_um2": float(contract["macro_model"]["single_port_generated_area_um2"]),
        "dual_port_fallback_area_um2": float(contract["macro_model"]["dual_port_fallback_area_um2"]),
        "dual_port_overdepth_proxy_area_um2": float(contract["macro_model"]["dual_port_overdepth_proxy_area_um2"]),
    }
    macro["area_reduction_fraction_vs_exact_capacity_fallback"] = 1.0 - macro["single_port_generated_area_um2"] / macro["dual_port_fallback_area_um2"]
    macro["area_reduction_fraction_vs_overdepth_proxy"] = 1.0 - macro["single_port_generated_area_um2"] / macro["dual_port_overdepth_proxy_area_um2"]
    gates = contract["materiality_gates"]
    decision = {
        "cycle_overhead_gate_pass": cycles["cycle_overhead_fraction_vs_m473"] <= float(gates["maximum_cycle_overhead_fraction"]),
        "retained_speedup_gate_pass": cycles["speedup_vs_best_same_budget_m468_zero"] >= float(gates["minimum_speedup_vs_m468_zero"]),
        "exact_fallback_area_gate_pass": macro["area_reduction_fraction_vs_exact_capacity_fallback"] >= float(gates["minimum_area_reduction_vs_exact_fallback"]),
        "overdepth_proxy_area_gate_pass": macro["area_reduction_fraction_vs_overdepth_proxy"] >= float(gates["minimum_area_reduction_vs_overdepth_proxy"]),
        "scratch_access_gate_pass": liveness["scratch_access_reduction_fraction_vs_m504"] >= float(gates["minimum_scratch_access_reduction_fraction"]),
    }
    decision["rtl_nomination"] = all(decision.values())
    decision["verdict"] = "GO_M505_PVRF_RTL" if decision["rtl_nomination"] else "NO_GO_M505_RTL"
    result = {
        "schema": "m505_h67_liveness_aware_single_port_parent_scratch_result_v1",
        "date": contract["date"],
        "status": "PASS_M505_LIVENESS_FASTKILL",
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators; exact cycle audit only",
        "identity": {"contract": {"path": str(args.contract.relative_to(ROOT)), "sha256": M504.sha256_file(args.contract)}, **contract["frozen_inputs"]},
        "population": {"samples": M504.SAMPLES, "operators": M504.OPERATORS, "partitions": M504.PARTITIONS, "rows_per_phase": M504.ROWS_PER_PHASE, "row_tile": M504.ROW_TILE, "tasks": int(np.prod(shape))},
        "self_test": self_test,
        "aggregate_one_output_block": aggregate,
        "cycle_comparison": cycles,
        "liveness_comparison": liveness,
        "macro_comparison": macro,
        "decision": decision,
        "claim_boundary": contract["claim_boundary"],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    result_path = args.out / "m505_h67_liveness_aware_single_port_parent_scratch_result_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.out / "m505_operator_sample_summary_r1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample", "operator", "active_rows", "parent_edges",
            "unique_live_parents", "refcount_zero", "refcount_one",
            "refcount_two_or_more", "maximum_parent_refcount",
            "dead_writes_elided", "single_use_forwarded_writes_elided",
            "m504_macro_reads", "m504_macro_writes",
            "dead_only_macro_reads", "dead_only_macro_writes",
            "m505_macro_reads", "m505_macro_writes", "m505_forwarded_reads",
            "m504_issue_cycles", "dead_only_issue_cycles", "m505_issue_cycles",
            "dead_only_stall_cycles", "m505_stall_cycles",
            "dead_only_deadline_holds", "m505_deadline_holds",
            "m504_pipeline_slice_cycles_no_commit",
            "dead_only_pipeline_slice_cycles_no_commit",
            "m505_pipeline_slice_cycles_no_commit",
        ])
        writer.writeheader()
        for sample in range(M504.SAMPLES):
            for operator in range(M504.OPERATORS):
                row_count = np.asarray(arrays["row_count"][sample, operator]).reshape(-1).astype(np.int64)
                active_rows = np.asarray(arrays["active_rows"][sample, operator]).reshape(-1).astype(np.int64)
                search = np.asarray(arrays["search_rows"][sample, operator]).reshape(-1).astype(np.int64)
                capture = (row_count + 7) // 8
                frontend = capture + search * ((row_count + cam_lanes - 1) // cam_lanes) + 17 * capture + 2
                preprocess = np.where(active_rows != 0, np.maximum(frontend, weight_dma), frontend)
                slice_cycles = {}
                for key, field in (
                    ("m504", "m504_deadline_cycles"),
                    ("dead_only", "dead_only_cycles"),
                    ("m505", "liveness_cycles"),
                ):
                    work = np.asarray(arrays[field][sample, operator]).reshape(-1).astype(np.int64) * M504.BLOCK_BANKS
                    slice_cycles[key] = M504.pipeline_cycles(preprocess, work, tail)
                writer.writerow({
                    "sample": sample,
                    "operator": operator,
                    "active_rows": int(arrays["active_rows"][sample, operator].sum()),
                    "parent_edges": int(arrays["parent_edges"][sample, operator].sum()),
                    "unique_live_parents": int(arrays["unique_live_parents"][sample, operator].sum()),
                    "refcount_zero": int(arrays["dead_writes_elided"][sample, operator].sum()),
                    "refcount_one": int(arrays["single_use_parents"][sample, operator].sum()),
                    "refcount_two_or_more": int(arrays["multi_use_parents"][sample, operator].sum()),
                    "maximum_parent_refcount": int(arrays["maximum_parent_refcount"][sample, operator].max(initial=0)),
                    "dead_writes_elided": int(arrays["dead_writes_elided"][sample, operator].sum()),
                    "single_use_forwarded_writes_elided": int(arrays["single_use_forwarded_writes_elided"][sample, operator].sum()),
                    "m504_macro_reads": int(arrays["m504_macro_reads"][sample, operator].sum()),
                    "m504_macro_writes": int(arrays["m504_macro_writes"][sample, operator].sum()),
                    "dead_only_macro_reads": int(arrays["dead_only_macro_reads"][sample, operator].sum()),
                    "dead_only_macro_writes": int(arrays["dead_only_macro_writes"][sample, operator].sum()),
                    "m505_macro_reads": int(arrays["macro_reads"][sample, operator].sum()),
                    "m505_macro_writes": int(arrays["macro_writes"][sample, operator].sum()),
                    "m505_forwarded_reads": int(arrays["forwarded_reads"][sample, operator].sum()),
                    "m504_issue_cycles": int(arrays["m504_deadline_cycles"][sample, operator].sum()),
                    "dead_only_issue_cycles": int(arrays["dead_only_cycles"][sample, operator].sum()),
                    "m505_issue_cycles": int(arrays["liveness_cycles"][sample, operator].sum()),
                    "dead_only_stall_cycles": int(arrays["dead_only_stall_cycles"][sample, operator].sum()),
                    "m505_stall_cycles": int(arrays["liveness_stall_cycles"][sample, operator].sum()),
                    "dead_only_deadline_holds": int(arrays["dead_only_deadline_holds"][sample, operator].sum()),
                    "m505_deadline_holds": int(arrays["liveness_deadline_holds"][sample, operator].sum()),
                    "m504_pipeline_slice_cycles_no_commit": slice_cycles["m504"],
                    "dead_only_pipeline_slice_cycles_no_commit": slice_cycles["dead_only"],
                    "m505_pipeline_slice_cycles_no_commit": slice_cycles["m505"],
                })
    manifest = args.out / "SHA256SUMS"
    manifest.write_text("\n".join(f"{M504.sha256_file(path)}  {path.name}" for path in sorted((result_path, csv_path))) + "\n", encoding="utf-8")
    (args.out / "SHA256SUMS.seal.sha256").write_text(f"{M504.sha256_file(manifest)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps(cycles, sort_keys=True))
    print(json.dumps(liveness, sort_keys=True))
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
