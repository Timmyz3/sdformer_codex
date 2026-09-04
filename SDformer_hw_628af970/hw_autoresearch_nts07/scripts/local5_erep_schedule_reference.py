#!/usr/bin/env python3
"""Local5 EREP v1 的有界双 slot 周期参考模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TaskTiming:
    front_cycles: int
    execute_cycles: int

    def __post_init__(self) -> None:
        if self.front_cycles <= 0 or self.execute_cycles <= 0:
            raise ValueError("front/execute cycle必须为正")


@dataclass(frozen=True)
class PipelineTiming:
    cycles: int
    tasks: int
    producer_busy_cycles: int
    consumer_busy_cycles: int
    producer_slot_stall_cycles: int
    consumer_wait_cycles: int


@dataclass(frozen=True)
class CandidateTiming:
    direct_serial_cycles: int
    reuse_only_cycles: int
    overlap_only_cycles: int
    erep_cycles: int
    output_tiles: int
    stripe_width: int

    def speedups(self) -> dict[str, float]:
        baseline = float(self.direct_serial_cycles)
        return {
            "reuse_only": baseline / self.reuse_only_cycles,
            "overlap_only": baseline / self.overlap_only_cycles,
            "erep": baseline / self.erep_cycles,
        }


def simulate_two_slot_pipeline(tasks: Iterable[TaskTiming]) -> PipelineTiming:
    """In-order producer/consumer pipeline with exactly two epoch slots."""
    rows = tuple(tasks)
    if not rows:
        return PipelineTiming(0, 0, 0, 0, 0, 0)

    fill_done: list[int] = []
    consume_done: list[int] = []
    producer_stall = 0
    consumer_wait = 0

    for index, task in enumerate(rows):
        previous_fill = fill_done[-1] if fill_done else 0
        slot_release = consume_done[index - 2] if index >= 2 else 0
        fill_start = max(previous_fill, slot_release)
        producer_stall += max(0, fill_start - previous_fill)
        fill_done.append(fill_start + task.front_cycles)

        previous_consume = consume_done[-1] if consume_done else 0
        consume_start = max(previous_consume, fill_done[-1])
        consumer_wait += max(0, consume_start - previous_consume)
        consume_done.append(consume_start + task.execute_cycles)

    return PipelineTiming(
        cycles=consume_done[-1],
        tasks=len(rows),
        producer_busy_cycles=sum(row.front_cycles for row in rows),
        consumer_busy_cycles=sum(row.execute_cycles for row in rows),
        producer_slot_stall_cycles=producer_stall,
        consumer_wait_cycles=consumer_wait,
    )


def _validate_window(
    front_cycles: Sequence[int], execute_cycles: Sequence[int], output_tiles: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    front = tuple(int(value) for value in front_cycles)
    execute = tuple(int(value) for value in execute_cycles)
    if not front or len(front) != len(execute):
        raise ValueError("joint window必须有等长非空front/execute数组")
    if any(value <= 0 for value in front + execute):
        raise ValueError("周期必须为正")
    if output_tiles <= 0:
        raise ValueError("output_tiles必须为正")
    return front, execute


def evaluate_window(
    front_cycles: Sequence[int],
    execute_cycles: Sequence[int],
    *,
    output_tiles: int,
    stripe_width: int = 2,
    drain_cycles_per_tile: int = 0,
) -> CandidateTiming:
    """Compare equal-boundary Direct/reuse/overlap/EREP schedules."""
    front, execute = _validate_window(
        front_cycles, execute_cycles, output_tiles
    )
    if stripe_width <= 0:
        raise ValueError("stripe_width必须为正")
    if drain_cycles_per_tile < 0:
        raise ValueError("drain_cycles_per_tile不得为负")

    common_drain = output_tiles * drain_cycles_per_tile
    per_tile_serial = sum(
        front_value + execute_value
        for front_value, execute_value in zip(front, execute)
    )
    direct = output_tiles * per_tile_serial + common_drain

    overlap_tasks = [
        TaskTiming(front_value, execute_value)
        for _ in range(output_tiles)
        for front_value, execute_value in zip(front, execute)
    ]
    overlap = simulate_two_slot_pipeline(overlap_tasks).cycles + common_drain

    reuse = 0
    erep = 0
    remaining = output_tiles
    while remaining:
        width = min(stripe_width, remaining)
        reuse += sum(
            front_value + width * execute_value
            for front_value, execute_value in zip(front, execute)
        )
        erep_tasks = [
            TaskTiming(front_value, width * execute_value)
            for front_value, execute_value in zip(front, execute)
        ]
        erep += simulate_two_slot_pipeline(erep_tasks).cycles
        remaining -= width

    reuse += common_drain
    erep += common_drain
    return CandidateTiming(
        direct_serial_cycles=direct,
        reuse_only_cycles=reuse,
        overlap_only_cycles=overlap,
        erep_cycles=erep,
        output_tiles=output_tiles,
        stripe_width=stripe_width,
    )

