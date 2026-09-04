#!/usr/bin/env python3
"""Independent replay contract for Local5 EREP head, window and command ledgers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

if __package__:
    from .local5_erep_capacity_baselines_v4 import evaluate_c4_oracle
    from .local5_erep_command_schedule_v4 import (
        ACC_BANK_RESOURCES,
        CommandKind,
        CommandResource,
        HeadCommandWork,
        PhaseTrace,
        RelativeCommand,
        ScheduleResult,
        WindowCommandWork,
        evaluate_window,
    )
else:
    from local5_erep_capacity_baselines_v4 import evaluate_c4_oracle
    from local5_erep_command_schedule_v4 import (
        ACC_BANK_RESOURCES,
        CommandKind,
        CommandResource,
        HeadCommandWork,
        PhaseTrace,
        RelativeCommand,
        ScheduleResult,
        WindowCommandWork,
        evaluate_window,
    )


HEAD_LEDGER_SCHEMA = "local5_erep_head_phase_ledger_v4"
WINDOW_LEDGER_SCHEMA = "local5_erep_window_schedule_ledger_v4"
COMMAND_LEDGER_SCHEMA = "local5_erep_g0_command_ledger_v4"
SELECTION_PLAN_SHA256 = "4e8732210a64cfcb553e7f4eee3657be70cc975a38839527e4792668d6deaf6b"
PROJECTION_CONTRACT_SHA256 = "c2bf6f406345d1bcc0f8a883318f59dc63116a96c96cd4138af83ce495ce9669"
STAGE_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
STAGE_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
STAGE_WEIGHTS = {0: 440, 1: 120, 2: 30, 3: 10}
BLOCK_ORDER = tuple(
    (stage, block)
    for stage in range(4)
    for block in range(STAGE_BLOCKS[stage])
)
CANDIDATE_KEYS = {
    "c0": "c0_direct_serial",
    "c1": "c1_reuse_only_s2",
    "c2": "c2_overlap_only",
    "c3": "c3_erep_s2",
}

ACC_RESOURCES = tuple(sorted(resource.value for resource in ACC_BANK_RESOURCES))
ROLE_RESOURCES = {
    "prepare": (CommandResource.CONTEXT_PREPARE_1RW.value,),
    "drain": (CommandResource.DRAIN_READ_1RW.value,),
    "fill": (
        CommandResource.RELATION_WORKSPACE_1RW.value,
        CommandResource.EPOCH_SLOT_1RW.value,
    ),
    "direct": ACC_RESOURCES,
    "execute": (
        CommandResource.EPOCH_SLOT_1RW.value,
        CommandResource.FIFO2_ENQ.value,
        CommandResource.FIFO2_DEQ.value,
        *ACC_RESOURCES,
    ),
}
PHASE_FIELDS = {"duration", "resource_events", "phase_event_sha256"}
RESOURCE_EVENT_FIELDS = {"cycle", "identity"}
WINDOW_FIELDS = {
    "sample", "stage", "block", "window", "weight", "heads",
    "output_tiles", "prepare_by_tile", "drain_by_tile",
    "acc32_miter_sha256", "acc32_mismatch_count",
}
HEAD_FIELDS = {
    "sample", "stage", "block", "window", "input_head", "epoch_records",
    "rtl_trace_sha256", "fill", "direct_by_tile", "execute_by_tile",
}
HEAD_LEDGER_FIELDS = {
    "schema", "evidence_level", "selection_plan_sha256",
    "formal_manifest_sha256", "projection_contract_sha256",
    "rtl_trace_archive_file", "rtl_trace_archive_sha256",
    "acc32_miter_archive_file", "acc32_miter_archive_sha256", "windows", "heads",
}


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_uint(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} non-boolean integer")
    return value


def strict_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _kind_for_resource(role: str, resource: str) -> CommandKind:
    if resource == CommandResource.RELATION_WORKSPACE_1RW.value:
        return CommandKind.RELATION_READ
    if resource == CommandResource.EPOCH_SLOT_1RW.value:
        return (
            CommandKind.EPOCH_RECORD_WRITE
            if role == "fill"
            else CommandKind.EPOCH_RECORD_READ
        )
    if resource == CommandResource.FIFO2_ENQ.value:
        return CommandKind.FIFO_ENQUEUE
    if resource == CommandResource.FIFO2_DEQ.value:
        return CommandKind.FIFO_DEQUEUE
    if resource in ACC_RESOURCES:
        return CommandKind.ACC_WRITE
    if resource == CommandResource.CONTEXT_PREPARE_1RW.value:
        return CommandKind.CONTEXT_PREPARE
    if resource == CommandResource.DRAIN_READ_1RW.value:
        return CommandKind.DRAIN_READ
    raise ValueError(f"resource {resource!r} is not legal for role {role!r}")


def encode_phase(trace: PhaseTrace, role: str) -> dict[str, Any]:
    if not isinstance(trace, PhaseTrace) or role not in ROLE_RESOURCES:
        raise ValueError("encode_phase requires a PhaseTrace and frozen phase role")
    grouped: dict[str, list[dict[str, Any]]] = {
        resource: [] for resource in ROLE_RESOURCES[role]
    }
    for command in trace.commands:
        resource = command.resource.value
        if resource not in grouped or command.kind is not _kind_for_resource(role, resource):
            raise ValueError(f"phase command is illegal for {role}: {resource}/{command.kind}")
        grouped[resource].append(
            {"cycle": command.cycle, "identity": command.identity}
        )
    for events in grouped.values():
        events.sort(key=lambda event: (event["cycle"], event["identity"]))
    body = {"duration": trace.duration, "resource_events": grouped}
    return {**body, "phase_event_sha256": canonical_sha(body)}


def decode_phase(
    value: Mapping[str, Any], role: str, *, epoch_records: int | None = None
) -> PhaseTrace:
    if role not in ROLE_RESOURCES:
        raise ValueError(f"unknown phase role {role!r}")
    if not isinstance(value, Mapping) or set(value) != PHASE_FIELDS:
        raise ValueError(f"{role} phase has a non-frozen field set")
    duration = strict_uint(value["duration"], f"{role} duration", positive=True)
    resource_events = value["resource_events"]
    if not isinstance(resource_events, Mapping) or set(resource_events) != set(
        ROLE_RESOURCES[role]
    ):
        raise ValueError(f"{role} resource-cycle set is not frozen")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for resource in ROLE_RESOURCES[role]:
        events = resource_events[resource]
        if not isinstance(events, list):
            raise ValueError(f"{role}/{resource} events must be a list")
        checked = []
        for index, event in enumerate(events):
            if not isinstance(event, Mapping) or set(event) != RESOURCE_EVENT_FIELDS:
                raise ValueError(f"{role}/{resource} event {index} field set is not frozen")
            cycle = strict_uint(event["cycle"], f"{role}/{resource} event cycle")
            identity = event["identity"]
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{role}/{resource} event identity must be nonempty")
            checked.append({"cycle": cycle, "identity": identity})
        if checked != sorted(
            checked, key=lambda event: (event["cycle"], event["identity"])
        ):
            raise ValueError(f"{role}/{resource} events are not in canonical order")
        cycles = [event["cycle"] for event in checked]
        identities = [event["identity"] for event in checked]
        if len(cycles) != len(set(cycles)) or any(cycle >= duration for cycle in cycles):
            raise ValueError(f"{role}/{resource} cycles are not unique in-phase order")
        if len(identities) != len(set(identities)):
            raise ValueError(f"{role}/{resource} identities are not unique")
        normalized[resource] = checked
    body = {"duration": duration, "resource_events": normalized}
    if strict_sha(value["phase_event_sha256"], f"{role} phase SHA") != canonical_sha(body):
        raise ValueError(f"{role} phase event digest mismatch")

    if role in {"fill", "execute"}:
        records = strict_uint(epoch_records, f"{role} epoch records")
        record_resources = (
            (
                CommandResource.RELATION_WORKSPACE_1RW.value,
                CommandResource.EPOCH_SLOT_1RW.value,
            )
            if role == "fill"
            else (
                CommandResource.EPOCH_SLOT_1RW.value,
                CommandResource.FIFO2_ENQ.value,
                CommandResource.FIFO2_DEQ.value,
            )
        )
        if any(len(normalized[resource]) != records for resource in record_resources):
            raise ValueError(f"{role} record resource counts disagree with epoch records")
        identity_sequences = [
            [event["identity"] for event in normalized[resource]]
            for resource in record_resources
        ]
        if (
            any(sequence != identity_sequences[0] for sequence in identity_sequences[1:])
            or len(set(identity_sequences[0])) != records
        ):
            raise ValueError(f"{role} record identity/cycle order mismatch")

    commands: list[RelativeCommand] = []
    for resource in ROLE_RESOURCES[role]:
        kind = _kind_for_resource(role, resource)
        for event in normalized[resource]:
            commands.append(
                RelativeCommand(event["cycle"], resource, kind, event["identity"])
            )
    return PhaseTrace(duration, tuple(commands))


def encode_window_fixture(
    window: WindowCommandWork,
    *,
    sample: int,
    stage: int,
    block: int,
    selected_window: int,
    weight: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Encode a WindowCommandWork for unit/integration fixtures, never admission."""
    window_row = {
        "sample": sample,
        "stage": stage,
        "block": block,
        "window": selected_window,
        "weight": weight,
        "heads": len(window.heads),
        "output_tiles": list(window.output_tiles),
        "prepare_by_tile": [encode_phase(trace, "prepare") for trace in window.prepare_by_tile],
        "drain_by_tile": [encode_phase(trace, "drain") for trace in window.drain_by_tile],
        "acc32_miter_sha256": canonical_sha(
            ["fixture-window-miter", sample, stage, block, selected_window]
        ),
        "acc32_mismatch_count": 0,
    }
    head_rows = []
    for head in window.heads:
        fill_phase = encode_phase(head.fill, "fill")
        direct_phases = [
            encode_phase(trace, "direct") for trace in head.direct_by_tile
        ]
        execute_phases = [
            encode_phase(trace, "execute") for trace in head.execute_by_tile
        ]
        trace_body = {
            "fill": fill_phase,
            "direct_by_tile": direct_phases,
            "execute_by_tile": execute_phases,
        }
        head_rows.append(
            {
                "sample": sample,
                "stage": stage,
                "block": block,
                "window": selected_window,
                "input_head": head.input_head,
                "epoch_records": head.epoch_records,
                "rtl_trace_sha256": canonical_sha(trace_body),
                **trace_body,
            }
        )
    return window_row, head_rows


def decode_window(
    window_row: Mapping[str, Any], head_rows: Sequence[Mapping[str, Any]]
) -> WindowCommandWork:
    if not isinstance(window_row, Mapping) or set(window_row) != WINDOW_FIELDS:
        raise ValueError("window phase row has a non-frozen field set")
    coordinate = tuple(
        strict_uint(window_row[field], f"window {field}")
        for field in ("sample", "stage", "block", "window")
    )
    heads = strict_uint(window_row["heads"], "window heads", positive=True)
    strict_uint(window_row["weight"], "window weight", positive=True)
    strict_sha(window_row["acc32_miter_sha256"], "window Acc32 miter SHA")
    if strict_uint(window_row["acc32_mismatch_count"], "window Acc32 mismatch") != 0:
        raise ValueError("formal window phase row has nonzero Acc32 mismatch")
    output_tiles = window_row["output_tiles"]
    if (
        not isinstance(output_tiles, list)
        or any(type(value) is not int for value in output_tiles)
        or output_tiles != list(range(heads))
    ):
        raise ValueError("window output tiles must be exact 0..H-1")
    prepare_rows = window_row["prepare_by_tile"]
    drain_rows = window_row["drain_by_tile"]
    if (
        not isinstance(prepare_rows, list)
        or not isinstance(drain_rows, list)
        or len(prepare_rows) != heads
        or len(drain_rows) != heads
    ):
        raise ValueError("window common phase arrays must each contain H entries")
    if len(head_rows) != heads:
        raise ValueError("window does not contain exactly H input-head phase rows")

    decoded_heads = []
    for index, row in enumerate(head_rows):
        if not isinstance(row, Mapping) or set(row) != HEAD_FIELDS:
            raise ValueError(f"head phase row {index} has a non-frozen field set")
        observed_coordinate = tuple(
            strict_uint(row[field], f"head {index} {field}")
            for field in ("sample", "stage", "block", "window")
        )
        if observed_coordinate != coordinate:
            raise ValueError(f"head phase row {index} has a foreign window coordinate")
        input_head = strict_uint(row["input_head"], f"head {index} input head")
        if input_head != index:
            raise ValueError("input-head phase rows must be exact 0..H-1 order")
        records = strict_uint(row["epoch_records"], f"head {index} records")
        if records > 450:
            raise ValueError("head epoch records exceed T450")
        strict_sha(row["rtl_trace_sha256"], f"head {index} RTL trace SHA")
        direct_rows = row["direct_by_tile"]
        execute_rows = row["execute_by_tile"]
        if (
            not isinstance(direct_rows, list)
            or not isinstance(execute_rows, list)
            or len(direct_rows) != heads
            or len(execute_rows) != heads
        ):
            raise ValueError("head direct/execute phase arrays must each contain H entries")
        decoded_heads.append(
            HeadCommandWork(
                input_head,
                records,
                decode_phase(row["fill"], "fill", epoch_records=records),
                tuple(decode_phase(value, "direct") for value in direct_rows),
                tuple(
                    decode_phase(value, "execute", epoch_records=records)
                    for value in execute_rows
                ),
            )
        )
    return WindowCommandWork(
        identity="/".join(str(value) for value in coordinate),
        heads=tuple(decoded_heads),
        output_tiles=tuple(output_tiles),
        prepare_by_tile=tuple(decode_phase(value, "prepare") for value in prepare_rows),
        drain_by_tile=tuple(decode_phase(value, "drain") for value in drain_rows),
    )


def _schedule_record(result: ScheduleResult) -> dict[str, Any]:
    events = [
        {
            "resource": event.resource,
            "kind": event.kind,
            "start": event.start,
            "end": event.end,
            "identity": event.identity,
        }
        for event in result.events
    ]
    resource_counts = dict(sorted(Counter(event.resource for event in result.events).items()))
    return {
        "cycles": result.cycles,
        "event_count": len(events),
        "event_ledger_sha256": canonical_sha(events),
        "resource_event_counts": resource_counts,
        "epoch_record_writes": result.epoch_record_writes,
        "epoch_record_reads": result.epoch_record_reads,
        "acc_contexts": result.acc_contexts,
        "epoch_slots": result.epoch_slots,
    }


def replay_window(
    window_row: Mapping[str, Any], head_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    window = decode_window(window_row, head_rows)
    schedules = evaluate_window(window)
    c4 = evaluate_c4_oracle(window)
    candidate_records = {
        short: _schedule_record(schedules[name])
        for short, name in CANDIDATE_KEYS.items()
    }
    candidate_records["c4"] = {
        "cycles": c4.cycles,
        "decision_sha256": canonical_sha(c4.to_dict()),
        "admission": list(c4.admission),
        "admitted_heads": list(c4.admitted_heads),
        "payload_used_records": c4.payload_used_records,
        "payload_capacity_records": c4.payload_capacity_records,
        "metadata_bits": c4.metadata_bits,
    }
    coordinate = {
        field: strict_uint(window_row[field], f"replay {field}")
        for field in ("sample", "stage", "block", "window")
    }
    head_phase_sha = canonical_sha(
        {"window": dict(window_row), "heads": [dict(row) for row in head_rows]}
    )
    schedule_body = {
        **coordinate,
        "head_phase_sha256": head_phase_sha,
        "candidates": candidate_records,
        "tail_cycles": {
            candidate: candidate_records[candidate]["cycles"]
            for candidate in ("c0", "c1", "c2", "c3", "c4")
        },
        "resource_conflict_count": 0,
    }
    schedule_row = {
        **schedule_body,
        "window_schedule_sha256": canonical_sha(schedule_body),
    }
    command_body = {
        **coordinate,
        **{
            candidate: candidate_records[candidate]["cycles"]
            for candidate in ("c0", "c1", "c2", "c3", "c4")
        },
        "window_schedule_sha256": schedule_row["window_schedule_sha256"],
    }
    command_row = {
        **command_body,
        "command_ledger_sha256": canonical_sha(command_body),
    }
    return schedule_row, command_row


def replay_ledger_document(
    ledger: Mapping[str, Any], *, formal: bool = False,
    plan_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(ledger, Mapping) or set(ledger) != HEAD_LEDGER_FIELDS:
        raise ValueError("head phase ledger has a non-frozen field set")
    if ledger["schema"] != HEAD_LEDGER_SCHEMA:
        raise ValueError("head phase ledger schema is invalid")
    strict_sha(ledger["selection_plan_sha256"], "head ledger selection SHA")
    strict_sha(ledger["formal_manifest_sha256"], "head ledger manifest SHA")
    strict_sha(ledger["projection_contract_sha256"], "head ledger projection SHA")
    strict_sha(ledger["rtl_trace_archive_sha256"], "RTL trace archive SHA")
    strict_sha(ledger["acc32_miter_archive_sha256"], "Acc32 miter archive SHA")
    for field in ("rtl_trace_archive_file", "acc32_miter_archive_file"):
        if (
            not isinstance(ledger[field], str)
            or not ledger[field]
            or "/" in ledger[field]
            or "\\" in ledger[field]
        ):
            raise ValueError(f"{field} must be an admission-local basename")
    if formal and (
        ledger["evidence_level"] != "formal_t450_rtl_phase_ledger"
        or ledger["selection_plan_sha256"] != SELECTION_PLAN_SHA256
        or ledger["projection_contract_sha256"] != PROJECTION_CONTRACT_SHA256
        or ledger["rtl_trace_archive_file"] != "rtl_trace_archive.npz"
        or ledger["acc32_miter_archive_file"] != "acc32_miter_archive.npz"
    ):
        raise ValueError("formal head phase ledger is not bound to frozen inputs")
    if not isinstance(ledger["evidence_level"], str) or not ledger["evidence_level"]:
        raise ValueError("head phase ledger evidence level must be nonempty")
    windows = ledger["windows"]
    heads = ledger["heads"]
    if not isinstance(windows, list) or not isinstance(heads, list):
        raise ValueError("head phase ledger windows/heads must be lists")
    if formal and (len(windows) != 1200 or len(heads) != 13_800):
        raise ValueError("formal head phase ledger must contain 1200 windows/13800 heads")
    if formal and (not isinstance(plan_records, Sequence) or len(plan_records) != 1200):
        raise ValueError("formal replay requires exactly 1200 frozen plan records")

    grouped: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for index, row in enumerate(heads):
        if not isinstance(row, Mapping):
            raise ValueError(f"head row {index} is not an object")
        key = tuple(
            strict_uint(row.get(field), f"head {index} {field}")
            for field in ("sample", "stage", "block", "window")
        )
        grouped.setdefault(key, []).append(row)

    schedule_rows = []
    command_rows = []
    consumed_keys = set()
    for index, window_row in enumerate(windows):
        if not isinstance(window_row, Mapping):
            raise ValueError(f"window row {index} is not an object")
        key = tuple(
            strict_uint(window_row.get(field), f"window {index} {field}")
            for field in ("sample", "stage", "block", "window")
        )
        if key in consumed_keys:
            raise ValueError("head phase ledger contains duplicate window coordinates")
        consumed_keys.add(key)
        if formal:
            plan = plan_records[index]
            expected_stage, expected_block = BLOCK_ORDER[index % len(BLOCK_ORDER)]
            expected = (
                index // len(BLOCK_ORDER),
                expected_stage,
                expected_block,
                plan.get("window"),
            )
            if key != expected:
                raise ValueError("formal windows do not match canonical selection-plan order")
            stage = key[1]
            if (
                window_row.get("heads") != STAGE_HEADS[stage]
                or window_row.get("weight") != STAGE_WEIGHTS[stage]
            ):
                raise ValueError("formal window H/weight does not match stage topology")
        schedule, command = replay_window(window_row, grouped.get(key, []))
        schedule_rows.append(schedule)
        command_rows.append(command)
    if set(grouped) != consumed_keys:
        raise ValueError("head phase ledger has orphan or missing window/head groups")

    window_document = {
        "schema": WINDOW_LEDGER_SCHEMA,
        "head_phase_ledger_canonical_sha256": canonical_sha(ledger),
        "rows": schedule_rows,
    }
    command_document = {
        "schema": COMMAND_LEDGER_SCHEMA,
        "window_schedule_ledger_canonical_sha256": canonical_sha(window_document),
        "rows": command_rows,
    }
    return window_document, command_document


def validate_replayed_ledgers(
    head_ledger: Mapping[str, Any],
    window_ledger: Mapping[str, Any],
    command_ledger: Mapping[str, Any],
    *,
    formal: bool = False,
    plan_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    expected_window, expected_command = replay_ledger_document(
        head_ledger, formal=formal, plan_records=plan_records
    )
    if canonical_sha(window_ledger) != canonical_sha(expected_window):
        raise ValueError("window schedule ledger differs from independent phase replay")
    if canonical_sha(command_ledger) != canonical_sha(expected_command):
        raise ValueError("command ledger differs from independent window replay")
    return expected_command["rows"]
