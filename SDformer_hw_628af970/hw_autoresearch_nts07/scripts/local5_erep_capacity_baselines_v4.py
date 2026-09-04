#!/usr/bin/env python3
"""Trace-derived C4/C5 exact relation-memo baselines for Local5 EREP v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

if __package__:
    from .local5_erep_command_schedule_v4 import (
        CommandKind,
        CommandResource,
        HeadCommandWork,
        WindowCommandWork,
    )
else:
    from local5_erep_command_schedule_v4 import (
        CommandKind,
        CommandResource,
        HeadCommandWork,
        WindowCommandWork,
    )


RELATION_RECORD_BITS = 112
C4_PAYLOAD_CAPACITY_BITS = 561_600
C4_PAYLOAD_CAPACITY_RECORDS = C4_PAYLOAD_CAPACITY_BITS // RELATION_RECORD_BITS
C4_PAYLOAD_STATIC_UNUSED_BITS = C4_PAYLOAD_CAPACITY_BITS % RELATION_RECORD_BITS
C4_TAG_BITS_PER_ENTRY = 32
C4_VALID_BITS_PER_ENTRY = 1
C5_STAGE3_WORST_CASE_RECORDS = 10_800
C5_STAGE3_WORST_CASE_PAYLOAD_BITS = 1_209_600

ORACLE_TIE_BREAK = (
    "maximize trace-derived saved cycles, then minimize admitted payload records, "
    "then choose the lexicographically smallest tuple of input-head indices"
)


@dataclass(frozen=True)
class TraceHeadCost:
    input_head: int
    records: int
    direct_cycles_by_tile: tuple[int, ...]
    fill_cycles: int
    execute_cycles_by_tile: tuple[int, ...]

    @property
    def direct_cycles(self) -> int:
        return sum(self.direct_cycles_by_tile)

    @property
    def resident_cycles(self) -> int:
        return self.fill_cycles + sum(self.execute_cycles_by_tile)

    @property
    def saved_cycles(self) -> int:
        return self.direct_cycles - self.resident_cycles


@dataclass(frozen=True)
class MemoResult:
    baseline: Literal["c4_oracle", "c4_first_fit", "c5_full"]
    admission: tuple[bool, ...]
    admitted_heads: tuple[int, ...]
    cycles: int
    common_prepare_cycles: int
    common_drain_cycles: int
    direct_path_cycles: int
    fill_path_cycles: int
    execute_path_cycles: int
    saved_cycles_vs_c0: int
    epoch_record_reads: int
    epoch_record_writes: int
    payload_capacity_bits: int
    payload_capacity_records: int
    payload_used_bits: int
    payload_used_records: int
    unused_bits: int
    tag_bits: int
    valid_bits: int
    metadata_bits: int
    metadata_in_payload_capacity: bool = False

    @property
    def allocated_state_bits(self) -> int:
        return self.payload_capacity_bits + self.metadata_bits

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["allocated_state_bits"] = self.allocated_state_bits
        return result


def _strict_capacity_bits(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("capacity_bits must be a non-negative integer")
    return value


def trace_head_cost(head: HeadCommandWork) -> TraceHeadCost:
    if not isinstance(head, HeadCommandWork):
        raise ValueError("head must be HeadCommandWork")
    relation_reads = head.fill.commands_for_pair(
        CommandResource.RELATION_WORKSPACE_1RW,
        CommandKind.RELATION_READ,
    )
    slot_writes = head.fill.commands_for_pair(
        CommandResource.EPOCH_SLOT_1RW,
        CommandKind.EPOCH_RECORD_WRITE,
    )
    if len(relation_reads) != head.epoch_records or len(slot_writes) != head.epoch_records:
        raise ValueError("fill trace does not carry the frozen epoch record count")
    for execute in head.execute_by_tile:
        reads = execute.commands_for_pair(
            CommandResource.EPOCH_SLOT_1RW,
            CommandKind.EPOCH_RECORD_READ,
        )
        if len(reads) != head.epoch_records:
            raise ValueError("execute trace does not carry the frozen epoch record count")
    return TraceHeadCost(
        input_head=head.input_head,
        records=head.epoch_records,
        direct_cycles_by_tile=tuple(trace.duration for trace in head.direct_by_tile),
        fill_cycles=head.fill.duration,
        execute_cycles_by_tile=tuple(trace.duration for trace in head.execute_by_tile),
    )


def trace_head_costs(window: WindowCommandWork) -> tuple[TraceHeadCost, ...]:
    if not isinstance(window, WindowCommandWork):
        raise ValueError("window must be WindowCommandWork")
    costs = tuple(trace_head_cost(head) for head in window.heads)
    if tuple(cost.input_head for cost in costs) != tuple(range(len(costs))):
        raise ValueError("trace heads must be in strict input-head order")
    if any(
        len(cost.direct_cycles_by_tile) != window.output_tile_count
        or len(cost.execute_cycles_by_tile) != window.output_tile_count
        for cost in costs
    ):
        raise ValueError("head cost tile count disagrees with the window")
    return costs


def _best_saved_by_exact_records(
    costs: tuple[TraceHeadCost, ...], capacity_records: int
) -> dict[int, int]:
    best = {0: 0}
    for cost in costs:
        updated = dict(best)
        for used_records, saved_cycles in best.items():
            candidate_records = used_records + cost.records
            if candidate_records > capacity_records:
                continue
            candidate_saved = saved_cycles + cost.saved_cycles
            if candidate_saved > updated.get(candidate_records, candidate_saved - 1):
                updated[candidate_records] = candidate_saved
        best = updated
    return best


def offline_optimal_admission(
    window: WindowCommandWork,
    *,
    capacity_bits: int = C4_PAYLOAD_CAPACITY_BITS,
) -> tuple[bool, ...]:
    costs = trace_head_costs(window)
    capacity_records = _strict_capacity_bits(capacity_bits) // RELATION_RECORD_BITS
    all_best = _best_saved_by_exact_records(costs, capacity_records)
    best_saved = max(all_best.values())
    best_records = min(
        records for records, saved in all_best.items() if saved == best_saved
    )
    suffix = [dict() for _ in range(len(costs) + 1)]
    suffix[-1] = {0: 0}
    for index in range(len(costs) - 1, -1, -1):
        suffix[index] = _best_saved_by_exact_records(costs[index:], best_records)

    selected: set[int] = set()
    remaining_records = best_records
    remaining_saved = best_saved
    for index, cost in enumerate(costs):
        if remaining_records == 0 and remaining_saved == 0:
            break
        next_records = remaining_records - cost.records
        next_saved = remaining_saved - cost.saved_cycles
        if next_records >= 0 and suffix[index + 1].get(next_records) == next_saved:
            selected.add(index)
            remaining_records = next_records
            remaining_saved = next_saved
    if remaining_records != 0 or remaining_saved != 0:
        raise AssertionError("failed to reconstruct deterministic trace knapsack")
    return tuple(index in selected for index in range(len(costs)))


def head_order_first_fit_admission(
    window: WindowCommandWork,
    *,
    capacity_bits: int = C4_PAYLOAD_CAPACITY_BITS,
) -> tuple[bool, ...]:
    costs = trace_head_costs(window)
    capacity_records = _strict_capacity_bits(capacity_bits) // RELATION_RECORD_BITS
    used_records = 0
    admission: list[bool] = []
    for cost in costs:
        # C4 is an implementable capacity baseline, not a trace-profit oracle.
        # It admits every complete nonempty head in input-head order while the
        # next whole entry fits, without inspecting the future cycle saving.
        fits = cost.records > 0 and used_records + cost.records <= capacity_records
        admission.append(fits)
        if fits:
            used_records += cost.records
    return tuple(admission)


def evaluate_admission(
    window: WindowCommandWork,
    admission: tuple[bool, ...],
    *,
    baseline: Literal["c4_oracle", "c4_first_fit", "c5_full"],
    capacity_bits: int,
) -> MemoResult:
    costs = trace_head_costs(window)
    if not isinstance(admission, tuple) or len(admission) != len(costs):
        raise ValueError("admission must be a bool tuple matching all heads")
    if any(not isinstance(value, bool) for value in admission):
        raise ValueError("admission entries must be bool")
    capacity_bits = _strict_capacity_bits(capacity_bits)
    capacity_records = capacity_bits // RELATION_RECORD_BITS
    used_records = sum(
        cost.records for cost, admitted in zip(costs, admission, strict=True) if admitted
    )
    if used_records > capacity_records:
        raise ValueError("admission exceeds payload capacity")

    prepare_cycles = sum(trace.duration for trace in window.prepare_by_tile)
    drain_cycles = sum(trace.duration for trace in window.drain_by_tile)
    c0_direct_cycles = sum(cost.direct_cycles for cost in costs)
    direct_cycles = sum(
        cost.direct_cycles
        for cost, admitted in zip(costs, admission, strict=True)
        if not admitted
    )
    fill_cycles = sum(
        cost.fill_cycles
        for cost, admitted in zip(costs, admission, strict=True)
        if admitted
    )
    execute_cycles = sum(
        sum(cost.execute_cycles_by_tile)
        for cost, admitted in zip(costs, admission, strict=True)
        if admitted
    )
    cycles = prepare_cycles + drain_cycles + direct_cycles + fill_cycles + execute_cycles
    c0_cycles = prepare_cycles + drain_cycles + c0_direct_cycles
    admitted_count = sum(admission)
    return MemoResult(
        baseline=baseline,
        admission=admission,
        admitted_heads=tuple(
            cost.input_head
            for cost, admitted in zip(costs, admission, strict=True)
            if admitted
        ),
        cycles=cycles,
        common_prepare_cycles=prepare_cycles,
        common_drain_cycles=drain_cycles,
        direct_path_cycles=direct_cycles,
        fill_path_cycles=fill_cycles,
        execute_path_cycles=execute_cycles,
        saved_cycles_vs_c0=c0_cycles - cycles,
        epoch_record_reads=sum(
            cost.records * window.output_tile_count
            for cost, admitted in zip(costs, admission, strict=True)
            if admitted
        ),
        epoch_record_writes=used_records,
        payload_capacity_bits=capacity_bits,
        payload_capacity_records=capacity_records,
        payload_used_bits=used_records * RELATION_RECORD_BITS,
        payload_used_records=used_records,
        unused_bits=capacity_bits - used_records * RELATION_RECORD_BITS,
        tag_bits=admitted_count * C4_TAG_BITS_PER_ENTRY,
        valid_bits=admitted_count * C4_VALID_BITS_PER_ENTRY,
        metadata_bits=admitted_count
        * (C4_TAG_BITS_PER_ENTRY + C4_VALID_BITS_PER_ENTRY),
    )


def evaluate_c4_oracle(window: WindowCommandWork) -> MemoResult:
    return evaluate_admission(
        window,
        offline_optimal_admission(window),
        baseline="c4_oracle",
        capacity_bits=C4_PAYLOAD_CAPACITY_BITS,
    )


def evaluate_c4_first_fit(window: WindowCommandWork) -> MemoResult:
    return evaluate_admission(
        window,
        head_order_first_fit_admission(window),
        baseline="c4_first_fit",
        capacity_bits=C4_PAYLOAD_CAPACITY_BITS,
    )


def evaluate_c5_full(window: WindowCommandWork) -> MemoResult:
    costs = trace_head_costs(window)
    if sum(cost.records for cost in costs) > C5_STAGE3_WORST_CASE_RECORDS:
        raise ValueError("window exceeds the frozen stage-3 full-memo record bound")
    return evaluate_admission(
        window,
        (True,) * len(costs),
        baseline="c5_full",
        capacity_bits=C5_STAGE3_WORST_CASE_PAYLOAD_BITS,
    )


simulate_c4_oracle = evaluate_c4_oracle
simulate_c4_first_fit = evaluate_c4_first_fit
simulate_c5_full = evaluate_c5_full
