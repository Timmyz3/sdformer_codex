#!/usr/bin/env python3
"""EREP v3 command-trace consumer and complete resource-lifecycle scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RelativeCommand:
    cycle: int
    resource: str
    kind: str
    identity: str

    def __post_init__(self) -> None:
        if self.cycle < 0 or not self.resource or not self.kind or not self.identity:
            raise ValueError("relative command fields are invalid")


@dataclass(frozen=True)
class PhaseTrace:
    duration: int
    commands: tuple[RelativeCommand, ...]

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("phase duration must be positive")
        seen: set[tuple[str, int]] = set()
        for command in self.commands:
            if command.cycle >= self.duration:
                raise ValueError("command lies outside phase")
            key = (command.resource, command.cycle)
            if key in seen:
                raise ValueError(f"same-cycle single-port collision: {key}")
            seen.add(key)
        self._validate_fifo2()

    def _validate_fifo2(self) -> None:
        occupancy = 0
        for cycle in range(self.duration):
            enqueues = sum(
                command.resource == "fifo2_enq" and command.cycle == cycle
                for command in self.commands
            )
            dequeues = sum(
                command.resource == "fifo2_deq" and command.cycle == cycle
                for command in self.commands
            )
            if enqueues > 1 or dequeues > 1:
                raise ValueError("FIFO2 has more than one enqueue/dequeue in a cycle")
            # A same-cycle enqueue can be observed by the dequeue in this model.
            occupancy += enqueues
            if dequeues > occupancy:
                raise ValueError("FIFO2 underflow")
            occupancy -= dequeues
            if occupancy > 2:
                raise ValueError("FIFO2 overflow")
        if occupancy != 0:
            raise ValueError("FIFO2 phase must retire all descriptors")

    def count_kind(self, kind: str) -> int:
        return sum(command.kind == kind for command in self.commands)


@dataclass(frozen=True)
class HeadCommandWork:
    epoch_records: int
    fill: PhaseTrace
    direct_by_tile: tuple[PhaseTrace, ...]
    execute_by_tile: tuple[PhaseTrace, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.epoch_records <= 450:
            raise ValueError("epoch_records must be in 0..450")
        if not self.direct_by_tile or len(self.direct_by_tile) != len(
            self.execute_by_tile
        ):
            raise ValueError("direct/execute tile traces must be non-empty and equal")
        if self.fill.count_kind("epoch_record_write") != self.epoch_records:
            raise ValueError("fill record-write count mismatch")
        write_ids = sorted(
            command.identity
            for command in self.fill.commands
            if command.kind == "epoch_record_write"
        )
        for execute in self.execute_by_tile:
            if execute.count_kind("epoch_record_read") != self.epoch_records:
                raise ValueError("execute record-read count mismatch")
            if execute.count_kind("fifo_enqueue") != self.epoch_records:
                raise ValueError("execute FIFO enqueue count mismatch")
            if execute.count_kind("fifo_dequeue") != self.epoch_records:
                raise ValueError("execute FIFO dequeue count mismatch")
            read_ids = sorted(
                command.identity
                for command in execute.commands
                if command.kind == "epoch_record_read"
            )
            if write_ids != read_ids:
                raise ValueError("epoch write/read record identity mismatch")


@dataclass(frozen=True)
class WindowCommandWork:
    identity: str
    heads: tuple[HeadCommandWork, ...]
    prepare_by_tile: tuple[PhaseTrace, ...]
    drain_by_tile: tuple[PhaseTrace, ...]

    def __post_init__(self) -> None:
        if not self.identity or not self.heads:
            raise ValueError("window identity/heads must be non-empty")
        tiles = len(self.heads[0].direct_by_tile)
        if (
            len(self.prepare_by_tile) != tiles
            or len(self.drain_by_tile) != tiles
            or any(len(head.direct_by_tile) != tiles for head in self.heads)
        ):
            raise ValueError("window tile shapes disagree")

    @property
    def output_tiles(self) -> int:
        return len(self.prepare_by_tile)


@dataclass(frozen=True)
class Event:
    resource: str
    kind: str
    start: int
    end: int
    identity: str

    def __post_init__(self) -> None:
        if not self.resource or not self.kind or not self.identity:
            raise ValueError("event identity is incomplete")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event interval must be non-empty half-open [start,end)")


@dataclass(frozen=True)
class ScheduleResult:
    candidate: str
    cycles: int
    events: tuple[Event, ...]
    epoch_record_writes: int
    epoch_record_reads: int
    acc_contexts: int
    epoch_slots: int


def _mapped_resource(resource: str, *, slot: int | None, context: int | None) -> str:
    if resource == "epoch_slot_1rw":
        if slot is None:
            raise ValueError("epoch command lacks slot")
        return f"epoch_slot_{slot}_1rw"
    if resource.startswith("acc_bank_"):
        if context is None:
            raise ValueError("accumulator command lacks context")
        return f"context_{context}_{resource}"
    return resource


def _place_phase(
    events: list[Event],
    trace: PhaseTrace,
    *,
    start: int,
    owner_resource: str,
    owner_kind: str,
    identity: str,
    slot: int | None = None,
    context: int | None = None,
) -> int:
    end = start + trace.duration
    events.append(Event(owner_resource, owner_kind, start, end, identity))
    for command in trace.commands:
        resource = _mapped_resource(command.resource, slot=slot, context=context)
        absolute = start + command.cycle
        events.append(
            Event(resource, command.kind, absolute, absolute + 1, command.identity)
        )
    return end


def _state_event(
    events: list[Event], resource: str, kind: str, start: int, end: int, identity: str
) -> None:
    if end > start:
        events.append(Event(resource, kind, start, end, identity))


def _validate_schedule(result: ScheduleResult) -> ScheduleResult:
    resources = {event.resource for event in result.events}
    for resource in resources:
        rows = sorted(
            (event for event in result.events if event.resource == resource),
            key=lambda event: (event.start, event.end, event.kind),
        )
        for left, right in zip(rows, rows[1:]):
            if left.end > right.start:
                raise AssertionError(
                    f"resource overlap {resource}: {left.kind} {left.end}>{right.start}"
                )
    if result.cycles != max((event.end for event in result.events), default=0):
        raise AssertionError("schedule cycle does not equal ledger tail")
    for slot in range(result.epoch_slots):
        prefix = f"slot_{slot}_task_"
        task_ids = sorted(
            {
                event.identity
                for event in result.events
                if event.resource == f"slot_{slot}_owner"
                and event.identity.startswith(prefix)
            }
        )
        for task_id in task_ids:
            rows = sorted(
                (
                    event for event in result.events
                    if event.resource == f"slot_{slot}_owner"
                    and event.identity == task_id
                ),
                key=lambda event: event.start,
            )
            if not rows or rows[0].kind != "FILL" or rows[-1].kind != "CONSUME":
                raise AssertionError("slot lifecycle lacks FILL/CONSUME")
            for left, right in zip(rows, rows[1:]):
                if left.end != right.start:
                    raise AssertionError("slot lifecycle has an ownership gap")
    return result


def simulate_direct(window: WindowCommandWork) -> ScheduleResult:
    events: list[Event] = []
    now = 0
    for tile in range(window.output_tiles):
        context_start = now
        now = _place_phase(
            events,
            window.prepare_by_tile[tile],
            start=now,
            owner_resource="context_0_prepare_port",
            owner_kind="PREPARE",
            identity=f"tile_{tile}_prepare",
            context=0,
        )
        for head, work in enumerate(window.heads):
            now = _place_phase(
                events,
                work.direct_by_tile[tile],
                start=now,
                owner_resource="direct_serial_lane",
                owner_kind="DIRECT",
                identity=f"tile_{tile}_head_{head}_direct",
                context=0,
            )
        now = _place_phase(
            events,
            window.drain_by_tile[tile],
            start=now,
            owner_resource="common_vector_drain",
            owner_kind="DRAIN",
            identity=f"tile_{tile}_drain",
            context=0,
        )
        events.append(
            Event(
                "context_0_owner",
                "OWNED",
                context_start,
                now,
                f"tile_{tile}_context",
            )
        )
    return _validate_schedule(ScheduleResult("c0_direct", now, tuple(events), 0, 0, 1, 0))


def _epoch_sequence(
    window: WindowCommandWork,
    *,
    tiles: tuple[int, ...],
    start_cycle: int,
    overlap: bool,
    slot_count: int,
) -> tuple[int, list[Event], int, int]:
    events: list[Event] = []
    fill_done: list[int] = []
    consume_done: list[int] = []
    writes = 0
    reads = 0
    for head, work in enumerate(window.heads):
        slot = head % slot_count
        previous_fill = fill_done[-1] if fill_done else start_cycle
        slot_release = consume_done[head - slot_count] if head >= slot_count else start_cycle
        previous_consume = consume_done[-1] if consume_done else start_cycle
        fill_start = max(
            previous_fill,
            slot_release,
            previous_consume if not overlap else start_cycle,
        )
        task_id = f"slot_{slot}_task_{tiles[0]}_{head}"
        fill_end = _place_phase(
            events,
            work.fill,
            start=fill_start,
            owner_resource="common_relation_producer",
            owner_kind="FILL",
            identity=f"stripe_{tiles[0]}_head_{head}_fill",
            slot=slot,
        )
        _state_event(events, f"slot_{slot}_owner", "FILL", fill_start, fill_end, task_id)
        fill_done.append(fill_end)
        writes += work.epoch_records

        consume_start = max(previous_consume, fill_end)
        _state_event(
            events,
            f"slot_{slot}_owner",
            "SEALED",
            fill_end,
            consume_start,
            task_id,
        )
        cursor = consume_start
        for context, tile in enumerate(tiles):
            cursor = _place_phase(
                events,
                work.execute_by_tile[tile],
                start=cursor,
                owner_resource="common_direct_1rw_execution_lane",
                owner_kind="EXECUTE",
                identity=f"stripe_{tiles[0]}_head_{head}_tile_{tile}_execute",
                slot=slot,
                context=context,
            )
            reads += work.epoch_records
        _state_event(
            events,
            f"slot_{slot}_owner",
            "CONSUME",
            consume_start,
            cursor,
            task_id,
        )
        consume_done.append(cursor)
    return consume_done[-1], events, writes, reads


def _simulate_epoch_candidate(
    window: WindowCommandWork,
    *,
    candidate: str,
    stripe_width: int,
    overlap: bool,
    slot_count: int,
) -> ScheduleResult:
    events: list[Event] = []
    now = 0
    writes = 0
    reads = 0
    for stripe_start in range(0, window.output_tiles, stripe_width):
        tiles = tuple(
            range(stripe_start, min(stripe_start + stripe_width, window.output_tiles))
        )
        context_starts: list[int] = []
        for context, tile in enumerate(tiles):
            context_starts.append(now)
            now = _place_phase(
                events,
                window.prepare_by_tile[tile],
                start=now,
                owner_resource=f"context_{context}_prepare_port",
                owner_kind="PREPARE",
                identity=f"tile_{tile}_prepare",
                context=context,
            )
        now, rows, row_writes, row_reads = _epoch_sequence(
            window,
            tiles=tiles,
            start_cycle=now,
            overlap=overlap,
            slot_count=slot_count,
        )
        events.extend(rows)
        writes += row_writes
        reads += row_reads
        context_ends: list[int] = []
        for context, tile in enumerate(tiles):
            now = _place_phase(
                events,
                window.drain_by_tile[tile],
                start=now,
                owner_resource="common_vector_drain",
                owner_kind="DRAIN",
                identity=f"tile_{tile}_drain",
                context=context,
            )
            context_ends.append(now)
        for context, tile in enumerate(tiles):
            events.append(
                Event(
                    f"context_{context}_owner",
                    "OWNED",
                    context_starts[context],
                    context_ends[context],
                    f"tile_{tile}_context",
                )
            )
    return _validate_schedule(
        ScheduleResult(
            candidate,
            now,
            tuple(events),
            writes,
            reads,
            stripe_width,
            slot_count,
        )
    )


def evaluate_window(window: WindowCommandWork) -> dict[str, ScheduleResult]:
    return {
        "c0_direct_serial": simulate_direct(window),
        "c1_reuse_only_s2": _simulate_epoch_candidate(
            window,
            candidate="c1_reuse_only_s2",
            stripe_width=2,
            overlap=False,
            slot_count=1,
        ),
        "c2_overlap_only": _simulate_epoch_candidate(
            window,
            candidate="c2_overlap_only",
            stripe_width=1,
            overlap=True,
            slot_count=2,
        ),
        "c3_erep_s2": _simulate_epoch_candidate(
            window,
            candidate="c3_erep_s2",
            stripe_width=2,
            overlap=True,
            slot_count=2,
        ),
    }

