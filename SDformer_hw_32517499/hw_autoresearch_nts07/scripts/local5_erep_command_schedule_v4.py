#!/usr/bin/env python3
"""EREP v4 fail-closed command schema and resource-lifecycle scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandResource(str, Enum):
    RELATION_WORKSPACE_1RW = "relation_workspace_1rw"
    EPOCH_SLOT_1RW = "epoch_slot_1rw"
    FIFO2_ENQ = "fifo2_enq"
    FIFO2_DEQ = "fifo2_deq"
    ACC_BANK_0_1RW = "acc_bank_0_1rw"
    ACC_BANK_1_1RW = "acc_bank_1_1rw"
    ACC_BANK_2_1RW = "acc_bank_2_1rw"
    ACC_BANK_3_1RW = "acc_bank_3_1rw"
    ACC_BANK_4_1RW = "acc_bank_4_1rw"
    CONTEXT_PREPARE_1RW = "context_prepare_1rw"
    DRAIN_READ_1RW = "drain_read_1rw"


class CommandKind(str, Enum):
    RELATION_READ = "relation_read"
    EPOCH_RECORD_WRITE = "epoch_record_write"
    EPOCH_RECORD_READ = "epoch_record_read"
    FIFO_ENQUEUE = "fifo_enqueue"
    FIFO_DEQUEUE = "fifo_dequeue"
    ACC_WRITE = "acc_write"
    CONTEXT_PREPARE = "context_prepare"
    DRAIN_READ = "drain_read"


ACC_BANK_RESOURCES = frozenset(
    {
        CommandResource.ACC_BANK_0_1RW,
        CommandResource.ACC_BANK_1_1RW,
        CommandResource.ACC_BANK_2_1RW,
        CommandResource.ACC_BANK_3_1RW,
        CommandResource.ACC_BANK_4_1RW,
    }
)

LEGAL_COMMAND_PAIRS = frozenset(
    {
        (CommandResource.RELATION_WORKSPACE_1RW, CommandKind.RELATION_READ),
        (CommandResource.EPOCH_SLOT_1RW, CommandKind.EPOCH_RECORD_WRITE),
        (CommandResource.EPOCH_SLOT_1RW, CommandKind.EPOCH_RECORD_READ),
        (CommandResource.FIFO2_ENQ, CommandKind.FIFO_ENQUEUE),
        (CommandResource.FIFO2_DEQ, CommandKind.FIFO_DEQUEUE),
        *((resource, CommandKind.ACC_WRITE) for resource in ACC_BANK_RESOURCES),
        (CommandResource.CONTEXT_PREPARE_1RW, CommandKind.CONTEXT_PREPARE),
        (CommandResource.DRAIN_READ_1RW, CommandKind.DRAIN_READ),
    }
)


def _strict_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-bool integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _command_resource(value: object) -> CommandResource:
    if not isinstance(value, str):
        raise ValueError("command resource must be a frozen string enum value")
    try:
        return CommandResource(value)
    except ValueError as exc:
        raise ValueError(f"unknown command resource: {value!r}") from exc


def _command_kind(value: object) -> CommandKind:
    if not isinstance(value, str):
        raise ValueError("command kind must be a frozen string enum value")
    try:
        return CommandKind(value)
    except ValueError as exc:
        raise ValueError(f"unknown command kind: {value!r}") from exc


@dataclass(frozen=True)
class RelativeCommand:
    cycle: int
    resource: CommandResource
    kind: CommandKind
    identity: str

    def __post_init__(self) -> None:
        _strict_int("relative command cycle", self.cycle, minimum=0)
        resource = _command_resource(self.resource)
        kind = _command_kind(self.kind)
        _nonempty_string("relative command identity", self.identity)
        if (resource, kind) not in LEGAL_COMMAND_PAIRS:
            raise ValueError(
                f"illegal command resource/kind pair: {resource.value}/{kind.value}"
            )
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class PhaseTrace:
    duration: int
    commands: tuple[RelativeCommand, ...]

    def __post_init__(self) -> None:
        _strict_int("phase duration", self.duration, minimum=1)
        if not isinstance(self.commands, tuple) or any(
            not isinstance(command, RelativeCommand) for command in self.commands
        ):
            raise ValueError("phase commands must be a tuple of RelativeCommand")

        seen: set[tuple[CommandResource, int]] = set()
        for command in self.commands:
            if command.cycle >= self.duration:
                raise ValueError("command lies outside phase")
            if (command.resource, command.kind) not in LEGAL_COMMAND_PAIRS:
                raise ValueError("phase contains an illegal command pair")
            key = (command.resource, command.cycle)
            if key in seen:
                raise ValueError(f"same-cycle single-port collision: {key}")
            seen.add(key)
        self._validate_fifo2()

    def _validate_fifo2(self) -> None:
        occupancy = 0
        for cycle in range(self.duration):
            enqueues = sum(
                command.resource is CommandResource.FIFO2_ENQ
                and command.cycle == cycle
                for command in self.commands
            )
            dequeues = sum(
                command.resource is CommandResource.FIFO2_DEQ
                and command.cycle == cycle
                for command in self.commands
            )
            if enqueues > 1 or dequeues > 1:
                raise ValueError("FIFO2 has more than one enqueue/dequeue in a cycle")
            # Enqueue is visible to dequeue in the same cycle through the frozen bypass.
            occupancy += enqueues
            if dequeues > occupancy:
                raise ValueError("FIFO2 underflow")
            occupancy -= dequeues
            if occupancy > 2:
                raise ValueError("FIFO2 overflow")
        if occupancy != 0:
            raise ValueError("FIFO2 phase must retire all descriptors")

    def count_kind(self, kind: CommandKind | str) -> int:
        normalized = _command_kind(kind)
        return sum(command.kind is normalized for command in self.commands)

    def commands_for_pair(
        self,
        resource: CommandResource | str,
        kind: CommandKind | str,
    ) -> tuple[RelativeCommand, ...]:
        normalized_resource = _command_resource(resource)
        normalized_kind = _command_kind(kind)
        if (normalized_resource, normalized_kind) not in LEGAL_COMMAND_PAIRS:
            raise ValueError("requested command resource/kind pair is illegal")
        return tuple(
            command
            for command in self.commands
            if command.resource is normalized_resource and command.kind is normalized_kind
        )


@dataclass(frozen=True)
class HeadCommandWork:
    input_head: int
    epoch_records: int
    fill: PhaseTrace
    direct_by_tile: tuple[PhaseTrace, ...]
    execute_by_tile: tuple[PhaseTrace, ...]

    def __post_init__(self) -> None:
        _strict_int("input_head", self.input_head, minimum=0)
        _strict_int("epoch_records", self.epoch_records, minimum=0, maximum=450)
        if not isinstance(self.fill, PhaseTrace):
            raise ValueError("fill must be a PhaseTrace")
        if not isinstance(self.direct_by_tile, tuple) or not isinstance(
            self.execute_by_tile, tuple
        ):
            raise ValueError("direct/execute traces must be tuples")
        if not self.direct_by_tile or len(self.direct_by_tile) != len(
            self.execute_by_tile
        ):
            raise ValueError("direct/execute tile traces must be non-empty and equal")
        if any(
            not isinstance(trace, PhaseTrace)
            for trace in self.direct_by_tile + self.execute_by_tile
        ):
            raise ValueError("direct/execute entries must be PhaseTrace values")

        writes = self.fill.commands_for_pair(
            CommandResource.EPOCH_SLOT_1RW,
            CommandKind.EPOCH_RECORD_WRITE,
        )
        relation_reads = self.fill.commands_for_pair(
            CommandResource.RELATION_WORKSPACE_1RW,
            CommandKind.RELATION_READ,
        )
        if len(writes) != self.epoch_records:
            raise ValueError("fill epoch-slot record-write count mismatch")
        write_ids = sorted(command.identity for command in writes)
        if len(relation_reads) != self.epoch_records:
            raise ValueError("fill relation-workspace record-read count mismatch")
        if write_ids != sorted(command.identity for command in relation_reads):
            raise ValueError("relation-read/epoch-write record identity mismatch")

        for execute in self.execute_by_tile:
            reads = execute.commands_for_pair(
                CommandResource.EPOCH_SLOT_1RW,
                CommandKind.EPOCH_RECORD_READ,
            )
            enqueues = execute.commands_for_pair(
                CommandResource.FIFO2_ENQ,
                CommandKind.FIFO_ENQUEUE,
            )
            dequeues = execute.commands_for_pair(
                CommandResource.FIFO2_DEQ,
                CommandKind.FIFO_DEQUEUE,
            )
            if len(reads) != self.epoch_records:
                raise ValueError("execute epoch-slot record-read count mismatch")
            if len(enqueues) != self.epoch_records:
                raise ValueError("execute FIFO enqueue count mismatch")
            if len(dequeues) != self.epoch_records:
                raise ValueError("execute FIFO dequeue count mismatch")
            read_ids = sorted(command.identity for command in reads)
            if write_ids != read_ids:
                raise ValueError("epoch write/read record identity mismatch")
            if read_ids != sorted(command.identity for command in enqueues):
                raise ValueError("epoch-read/FIFO-enqueue record identity mismatch")
            if read_ids != sorted(command.identity for command in dequeues):
                raise ValueError("epoch-read/FIFO-dequeue record identity mismatch")


@dataclass(frozen=True)
class WindowCommandWork:
    identity: str
    heads: tuple[HeadCommandWork, ...]
    output_tiles: tuple[int, ...]
    prepare_by_tile: tuple[PhaseTrace, ...]
    drain_by_tile: tuple[PhaseTrace, ...]

    def __post_init__(self) -> None:
        _nonempty_string("window identity", self.identity)
        if not isinstance(self.heads, tuple) or not self.heads or any(
            not isinstance(head, HeadCommandWork) for head in self.heads
        ):
            raise ValueError("window heads must be a non-empty HeadCommandWork tuple")
        if not isinstance(self.output_tiles, tuple) or not self.output_tiles:
            raise ValueError("output_tiles must be a non-empty tuple")

        output_tile_count = len(self.output_tiles)
        for output_tile in self.output_tiles:
            _strict_int(
                "output tile",
                output_tile,
                minimum=0,
                maximum=output_tile_count - 1,
            )
        if self.output_tiles != tuple(range(output_tile_count)):
            raise ValueError("output tile identity/order must be exactly 0..O-1")

        observed_heads = tuple(head.input_head for head in self.heads)
        if observed_heads != tuple(range(len(self.heads))):
            raise ValueError("input head identity/order must be exactly 0..H-1")
        if not isinstance(self.prepare_by_tile, tuple) or not isinstance(
            self.drain_by_tile, tuple
        ):
            raise ValueError("prepare/drain traces must be tuples")
        if any(
            not isinstance(trace, PhaseTrace)
            for trace in self.prepare_by_tile + self.drain_by_tile
        ):
            raise ValueError("prepare/drain entries must be PhaseTrace values")
        if (
            len(self.prepare_by_tile) != output_tile_count
            or len(self.drain_by_tile) != output_tile_count
            or any(len(head.direct_by_tile) != output_tile_count for head in self.heads)
        ):
            raise ValueError("window tile shapes disagree")

    @property
    def output_tile_count(self) -> int:
        return len(self.output_tiles)


@dataclass(frozen=True)
class Event:
    resource: str
    kind: str
    start: int
    end: int
    identity: str

    def __post_init__(self) -> None:
        _nonempty_string("event resource", self.resource)
        _nonempty_string("event kind", self.kind)
        _nonempty_string("event identity", self.identity)
        _strict_int("event start", self.start, minimum=0)
        _strict_int("event end", self.end, minimum=1)
        if self.end <= self.start:
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

    def __post_init__(self) -> None:
        _nonempty_string("schedule candidate", self.candidate)
        _strict_int("schedule cycles", self.cycles, minimum=0)
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, Event) for event in self.events
        ):
            raise ValueError("schedule events must be an Event tuple")
        _strict_int("epoch_record_writes", self.epoch_record_writes, minimum=0)
        _strict_int("epoch_record_reads", self.epoch_record_reads, minimum=0)
        _strict_int("acc_contexts", self.acc_contexts, minimum=1, maximum=2)
        _strict_int("epoch_slots", self.epoch_slots, minimum=0, maximum=2)


def _mapped_resource(
    resource: CommandResource,
    *,
    slot: int | None,
    context: int | None,
) -> str:
    if resource is CommandResource.EPOCH_SLOT_1RW:
        if slot is None:
            raise ValueError("epoch command lacks slot")
        _strict_int("epoch slot", slot, minimum=0, maximum=1)
        return f"epoch_slot_{slot}_1rw"
    if resource in ACC_BANK_RESOURCES:
        if context is None:
            raise ValueError("accumulator command lacks context")
        _strict_int("accumulator context", context, minimum=0, maximum=1)
        return f"context_{context}_{resource.value}"
    return resource.value


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
    _strict_int("phase start", start, minimum=0)
    end = start + trace.duration
    events.append(Event(owner_resource, owner_kind, start, end, identity))
    for command in trace.commands:
        resource = _mapped_resource(command.resource, slot=slot, context=context)
        absolute = start + command.cycle
        events.append(
            Event(resource, command.kind.value, absolute, absolute + 1, command.identity)
        )
    return end


def _state_event(
    events: list[Event],
    resource: str,
    kind: str,
    start: int,
    end: int,
    identity: str,
) -> None:
    _strict_int("state start", start, minimum=0)
    _strict_int("state end", end, minimum=0)
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

    writes = sum(event.kind == CommandKind.EPOCH_RECORD_WRITE.value for event in result.events)
    reads = sum(event.kind == CommandKind.EPOCH_RECORD_READ.value for event in result.events)
    if writes != result.epoch_record_writes or reads != result.epoch_record_reads:
        raise AssertionError("epoch record counters disagree with command ledger")

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
                    event
                    for event in result.events
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
    if not isinstance(window, WindowCommandWork):
        raise ValueError("window must be WindowCommandWork")
    events: list[Event] = []
    now = 0
    for output_tile in window.output_tiles:
        context_start = now
        now = _place_phase(
            events,
            window.prepare_by_tile[output_tile],
            start=now,
            owner_resource="context_0_prepare_port",
            owner_kind="PREPARE",
            identity=f"tile_{output_tile}_prepare",
            context=0,
        )
        for work in window.heads:
            now = _place_phase(
                events,
                work.direct_by_tile[output_tile],
                start=now,
                owner_resource="direct_serial_lane",
                owner_kind="DIRECT",
                identity=f"tile_{output_tile}_head_{work.input_head}_direct",
                context=0,
            )
        now = _place_phase(
            events,
            window.drain_by_tile[output_tile],
            start=now,
            owner_resource="common_vector_drain",
            owner_kind="DRAIN",
            identity=f"tile_{output_tile}_drain",
            context=0,
        )
        events.append(
            Event(
                "context_0_owner",
                "OWNED",
                context_start,
                now,
                f"tile_{output_tile}_context",
            )
        )
    return _validate_schedule(
        ScheduleResult("c0_direct_serial", now, tuple(events), 0, 0, 1, 0)
    )


def _epoch_sequence(
    window: WindowCommandWork,
    *,
    tiles: tuple[int, ...],
    start_cycle: int,
    overlap: bool,
    slot_count: int,
) -> tuple[int, list[Event], int, int]:
    _strict_int("epoch start_cycle", start_cycle, minimum=0)
    _strict_int("slot_count", slot_count, minimum=1, maximum=2)
    if not isinstance(overlap, bool):
        raise ValueError("overlap must be bool")
    if not isinstance(tiles, tuple) or not tiles:
        raise ValueError("epoch tiles must be a non-empty tuple")
    for tile in tiles:
        _strict_int("epoch output tile", tile, minimum=0, maximum=window.output_tile_count - 1)
        if tile not in window.output_tiles:
            raise ValueError("epoch output tile is not declared by the window")

    events: list[Event] = []
    fill_done: list[int] = []
    consume_done: list[int] = []
    writes = 0
    reads = 0
    for work in window.heads:
        head = work.input_head
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
        for context, output_tile in enumerate(tiles):
            cursor = _place_phase(
                events,
                work.execute_by_tile[output_tile],
                start=cursor,
                owner_resource="common_direct_1rw_execution_lane",
                owner_kind="EXECUTE",
                identity=(
                    f"stripe_{tiles[0]}_head_{head}_tile_{output_tile}_execute"
                ),
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
    _nonempty_string("candidate", candidate)
    _strict_int("stripe_width", stripe_width, minimum=1, maximum=2)
    _strict_int("slot_count", slot_count, minimum=1, maximum=2)
    if not isinstance(overlap, bool):
        raise ValueError("overlap must be bool")

    events: list[Event] = []
    now = 0
    writes = 0
    reads = 0
    for stripe_start in range(0, window.output_tile_count, stripe_width):
        tiles = window.output_tiles[stripe_start : stripe_start + stripe_width]
        context_starts: list[int] = []
        for context, output_tile in enumerate(tiles):
            context_starts.append(now)
            now = _place_phase(
                events,
                window.prepare_by_tile[output_tile],
                start=now,
                owner_resource=f"context_{context}_prepare_port",
                owner_kind="PREPARE",
                identity=f"tile_{output_tile}_prepare",
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
        for context, output_tile in enumerate(tiles):
            now = _place_phase(
                events,
                window.drain_by_tile[output_tile],
                start=now,
                owner_resource="common_vector_drain",
                owner_kind="DRAIN",
                identity=f"tile_{output_tile}_drain",
                context=context,
            )
            context_ends.append(now)
        for context, output_tile in enumerate(tiles):
            events.append(
                Event(
                    f"context_{context}_owner",
                    "OWNED",
                    context_starts[context],
                    context_ends[context],
                    f"tile_{output_tile}_context",
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
    if not isinstance(window, WindowCommandWork):
        raise ValueError("window must be WindowCommandWork")
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
