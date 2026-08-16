#!/usr/bin/env python3
"""Local5 EREP v2 candidate lifecycle and discrete-event reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HeadWork:
    epoch_records: int
    epoch_fill_cycles: int
    direct_cycles_by_tile: tuple[int, ...]
    execute_cycles_by_tile: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.epoch_records <= 450:
            raise ValueError("epoch_records必须在0..450")
        if self.epoch_fill_cycles <= 0:
            raise ValueError("epoch_fill_cycles必须为正")
        if (
            not self.direct_cycles_by_tile
            or len(self.direct_cycles_by_tile) != len(self.execute_cycles_by_tile)
            or any(value <= 0 for value in self.direct_cycles_by_tile)
            or any(value <= 0 for value in self.execute_cycles_by_tile)
        ):
            raise ValueError("direct/execute tile周期必须等长非空且为正")
        if self.epoch_fill_cycles < self.epoch_records + 1:
            raise ValueError("epoch fill必须容纳逐record 1RW write和atomic seal")
        if any(value < self.epoch_records + 1 for value in self.execute_cycles_by_tile):
            raise ValueError("epoch execute必须容纳逐record 1RW read和end")


@dataclass(frozen=True)
class WindowWork:
    identity: str
    heads: tuple[HeadWork, ...]
    prepare_cycles_by_tile: tuple[int, ...]
    drain_cycles_by_tile: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.identity or not self.heads:
            raise ValueError("window identity/heads不得为空")
        output_tiles = len(self.heads[0].direct_cycles_by_tile)
        if (
            output_tiles <= 0
            or len(self.prepare_cycles_by_tile) != output_tiles
            or len(self.drain_cycles_by_tile) != output_tiles
            or any(value < 0 for value in self.prepare_cycles_by_tile)
            or any(value < 0 for value in self.drain_cycles_by_tile)
            or any(
                len(head.direct_cycles_by_tile) != output_tiles for head in self.heads
            )
        ):
            raise ValueError("window tile shape/prepare/drain合同失效")

    @property
    def output_tiles(self) -> int:
        return len(self.prepare_cycles_by_tile)


@dataclass(frozen=True)
class Event:
    resource: str
    kind: str
    start: int
    end: int
    head: int | None = None
    tile: int | None = None
    slot: int | None = None
    context: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event必须使用非空半开周期区间")


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    cycles: int
    events: tuple[Event, ...]
    acc_contexts: int
    epoch_slots: int
    epoch_record_writes: int
    epoch_record_reads: int


def _append(
    events: list[Event],
    *,
    resource: str,
    kind: str,
    start: int,
    duration: int,
    head: int | None = None,
    tile: int | None = None,
    slot: int | None = None,
    context: int | None = None,
) -> int:
    if duration <= 0:
        return start
    end = start + duration
    events.append(
        Event(resource, kind, start, end, head, tile, slot, context)
    )
    return end


def _assert_non_overlapping(events: Iterable[Event], resource: str) -> None:
    rows = sorted(
        (event for event in events if event.resource == resource),
        key=lambda event: (event.start, event.end),
    )
    for left, right in zip(rows, rows[1:]):
        if left.end > right.start:
            raise AssertionError(
                f"resource overlap {resource}: {left.end}>{right.start}"
            )


def _validate_result(result: CandidateResult) -> CandidateResult:
    resources = {event.resource for event in result.events}
    for resource in resources:
        _assert_non_overlapping(result.events, resource)
    if result.cycles != max((event.end for event in result.events), default=0):
        raise AssertionError("candidate cycles与ledger末尾不一致")
    return result


def simulate_direct_serial(window: WindowWork) -> CandidateResult:
    events: list[Event] = []
    now = 0
    for tile in range(window.output_tiles):
        now = _append(
            events,
            resource="acc_context_0",
            kind="prepare",
            start=now,
            duration=window.prepare_cycles_by_tile[tile],
            tile=tile,
            context=0,
        )
        for head, work in enumerate(window.heads):
            execute_start = now
            now = _append(
                events,
                resource="direct_lane",
                kind="direct_relation_and_execute",
                start=now,
                duration=work.direct_cycles_by_tile[tile],
                head=head,
                tile=tile,
                context=0,
            )
            events.append(
                Event(
                    "acc_context_0",
                    "accumulate",
                    execute_start,
                    now,
                    head=head,
                    tile=tile,
                    context=0,
                )
            )
        now = _append(
            events,
            resource="acc_context_0",
            kind="drain",
            start=now,
            duration=window.drain_cycles_by_tile[tile],
            tile=tile,
            context=0,
        )
    return _validate_result(
        CandidateResult("c0_direct_serial", now, tuple(events), 1, 0, 0, 0)
    )


def _simulate_epoch_sequence(
    window: WindowWork,
    *,
    tiles: tuple[int, ...],
    overlap: bool,
    start_cycle: int,
    slot_count: int,
    context_base: int,
) -> tuple[int, list[Event], int, int]:
    if not tiles or slot_count not in {1, 2}:
        raise ValueError("epoch sequence tiles/slot_count非法")
    events: list[Event] = []
    fill_done: list[int] = []
    consume_done: list[int] = []
    writes = 0
    reads = 0

    for head, work in enumerate(window.heads):
        slot = head % slot_count
        previous_fill = fill_done[-1] if fill_done else start_cycle
        slot_release = (
            consume_done[head - slot_count]
            if head >= slot_count
            else start_cycle
        )
        if overlap:
            fill_start = max(previous_fill, slot_release)
        else:
            fill_start = max(previous_fill, consume_done[-1] if consume_done else start_cycle)
        fill_end = _append(
            events,
            resource=f"epoch_slot_{slot}",
            kind="fill_and_atomic_seal",
            start=fill_start,
            duration=work.epoch_fill_cycles,
            head=head,
            slot=slot,
        )
        fill_done.append(fill_end)
        writes += work.epoch_records

        consume_start = max(consume_done[-1] if consume_done else start_cycle, fill_end)
        cursor = consume_start
        for context_offset, tile in enumerate(tiles):
            execute_start = cursor
            cursor = _append(
                events,
                resource="direct_1rw_execution_lane",
                kind="epoch_read_builder_execute",
                start=cursor,
                duration=work.execute_cycles_by_tile[tile],
                head=head,
                tile=tile,
                slot=slot,
                context=context_base + context_offset,
            )
            events.append(
                Event(
                    resource=f"acc_context_{context_base + context_offset}",
                    kind="accumulate",
                    start=execute_start,
                    end=cursor,
                    head=head,
                    tile=tile,
                    context=context_base + context_offset,
                )
            )
            reads += work.epoch_records
        consume_done.append(cursor)
        events.append(
            Event(
                resource=f"epoch_slot_{slot}",
                kind="sealed_hold_consume",
                start=consume_start,
                end=cursor,
                head=head,
                slot=slot,
            )
        )

    return consume_done[-1], events, writes, reads


def simulate_reuse_only_s2(window: WindowWork) -> CandidateResult:
    events: list[Event] = []
    now = 0
    writes = 0
    reads = 0
    for stripe_start in range(0, window.output_tiles, 2):
        tiles = tuple(range(stripe_start, min(stripe_start + 2, window.output_tiles)))
        for context, tile in enumerate(tiles):
            now = _append(
                events,
                resource=f"acc_context_{context}",
                kind="prepare",
                start=now,
                duration=window.prepare_cycles_by_tile[tile],
                tile=tile,
                context=context,
            )
        now, rows, row_writes, row_reads = _simulate_epoch_sequence(
            window,
            tiles=tiles,
            overlap=False,
            start_cycle=now,
            slot_count=1,
            context_base=0,
        )
        events.extend(rows)
        writes += row_writes
        reads += row_reads
        for context, tile in enumerate(tiles):
            now = _append(
                events,
                resource=f"acc_context_{context}",
                kind="drain",
                start=now,
                duration=window.drain_cycles_by_tile[tile],
                tile=tile,
                context=context,
            )
    return _validate_result(
        CandidateResult("c1_reuse_only_s2", now, tuple(events), 2, 1, writes, reads)
    )


def simulate_overlap_only(window: WindowWork) -> CandidateResult:
    events: list[Event] = []
    now = 0
    writes = 0
    reads = 0
    for tile in range(window.output_tiles):
        now = _append(
            events,
            resource="acc_context_0",
            kind="prepare",
            start=now,
            duration=window.prepare_cycles_by_tile[tile],
            tile=tile,
            context=0,
        )
        now, rows, row_writes, row_reads = _simulate_epoch_sequence(
            window,
            tiles=(tile,),
            overlap=True,
            start_cycle=now,
            slot_count=2,
            context_base=0,
        )
        events.extend(rows)
        writes += row_writes
        reads += row_reads
        now = _append(
            events,
            resource="acc_context_0",
            kind="drain",
            start=now,
            duration=window.drain_cycles_by_tile[tile],
            tile=tile,
            context=0,
        )
    return _validate_result(
        CandidateResult("c2_overlap_only", now, tuple(events), 1, 2, writes, reads)
    )


def simulate_erep_s2(window: WindowWork) -> CandidateResult:
    events: list[Event] = []
    now = 0
    writes = 0
    reads = 0
    for stripe_start in range(0, window.output_tiles, 2):
        tiles = tuple(range(stripe_start, min(stripe_start + 2, window.output_tiles)))
        for context, tile in enumerate(tiles):
            now = _append(
                events,
                resource=f"acc_context_{context}",
                kind="prepare",
                start=now,
                duration=window.prepare_cycles_by_tile[tile],
                tile=tile,
                context=context,
            )
        now, rows, row_writes, row_reads = _simulate_epoch_sequence(
            window,
            tiles=tiles,
            overlap=True,
            start_cycle=now,
            slot_count=2,
            context_base=0,
        )
        events.extend(rows)
        writes += row_writes
        reads += row_reads
        # v2 conservatively serializes both drains and forbids next-stripe fill.
        for context, tile in enumerate(tiles):
            now = _append(
                events,
                resource=f"acc_context_{context}",
                kind="drain",
                start=now,
                duration=window.drain_cycles_by_tile[tile],
                tile=tile,
                context=context,
            )
    return _validate_result(
        CandidateResult("c3_erep_s2", now, tuple(events), 2, 2, writes, reads)
    )


def evaluate_window(window: WindowWork) -> dict[str, CandidateResult]:
    return {
        "c0_direct_serial": simulate_direct_serial(window),
        "c1_reuse_only_s2": simulate_reuse_only_s2(window),
        "c2_overlap_only": simulate_overlap_only(window),
        "c3_erep_s2": simulate_erep_s2(window),
    }
