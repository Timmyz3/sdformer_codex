#!/usr/bin/env python3
"""Normalize passive RTL trace rows into auditable Local5 EREP v4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local5_erep_command_schedule_v4 import (  # noqa: E402
    CommandKind,
    CommandResource,
    HeadCommandWork,
    PhaseTrace,
    RelativeCommand,
    WindowCommandWork,
    simulate_direct,
)
from local5_erep_identity_service_v4 import (  # noqa: E402
    DEFAULT_SEED,
    SCHEMA as IDENTITY_SCHEMA,
    IdentityService,
)


RAW_SCHEMA = "local5_erep_raw_trace_v4"
OUTPUT_SCHEMA = "local5_erep_calibration_evidence_v4"
TRACE_PREFIX = "EREP_V4 "
REQUIRED_BOUNDARIES = {
    "direct_online": (
        "PREPARE_BEGIN",
        "FILL_BEGIN",
        "EXECUTE_BEGIN",
        "DRAIN_BEGIN",
        "COMPUTE_DONE",
    ),
    "tcfm5_1rw": (
        "PREPARE_BEGIN",
        "EXECUTE_BEGIN",
        "DRAIN_BEGIN",
        "COMPUTE_DONE",
    ),
}
EXPECTED_SYNTH_BOUNDARIES = {
    "direct_online": {
        "PREPARE_BEGIN": 0,
        "FILL_BEGIN": 2,
        "EXECUTE_BEGIN": 21,
        "DRAIN_BEGIN": 61,
        "COMPUTE_DONE": 63,
    },
    "tcfm5_1rw": {
        "PREPARE_BEGIN": 0,
        "EXECUTE_BEGIN": 1,
        "DRAIN_BEGIN": 6,
        "COMPUTE_DONE": 9,
    },
}
EXPECTED_DIRECT_RELATION_DONE_CYCLE = 49

COMMON_FIELDS = frozenset(
    {"schema", "candidate", "event", "cycle", "window", "phase", "time", "scope"}
)
HANDSHAKE_FIELDS = frozenset({"resource", "kind", "valid", "ready", "fire"})


def _fields(*names: str, handshake: bool = True) -> frozenset[str]:
    return COMMON_FIELDS | (HANDSHAKE_FIELDS if handshake else frozenset()) | frozenset(names)


EVENT_FIELDS = {
    ("direct_online", "weight_accept"): _fields("lane", "out", "data", "last"),
    ("direct_online", "context_prepare"): _fields("identity"),
    ("direct_online", "cycle_snapshot"): _fields(
        "fifo_occupancy", "projection_busy", "projection_done", "relation_active",
        "relation_done", "protocol_error"
    ),
    ("direct_online", "relation_accept"): _fields(
        "source_id", "plane", "y", "x", "candidate_valid", "active_mask", "k", "gates"
    ),
    ("direct_online", "relation_read"): _fields(
        "source_id", "plane", "y", "x", "last"
    ),
    ("direct_online", "fifo_enqueue"): _fields(
        "source_id", "plane", "y", "x", "k", "gates", "mask", "last",
        "occupancy_pre", "occupancy_post"
    ),
    ("direct_online", "fifo_dequeue"): _fields(
        "source_id", "plane", "y", "x", "occupancy_pre", "occupancy_post"
    ),
    ("direct_online", "term_accept"): _fields(
        "source_id", "plane", "y", "x", "lane", "gate", "destination_mask",
        "last", "source_last", "delta"
    ),
    ("direct_online", "acc_update_accept"): _fields(
        "bank", "source_id", "lane", "gate", "address", "delta"
    ),
    ("direct_online", "acc_physical_command"): _fields(
        "bank", "address", "data"
    ),
    ("direct_online", "drain_read_accept"): _fields(
        "source_id", "plane", "y", "x", "out"
    ),
    ("direct_online", "drain_read_response"): _fields(
        "source_id", "out", "data"
    ),
    ("tcfm5_1rw", "weight_accept"): _fields("lane", "out", "data", "last"),
    ("tcfm5_1rw", "context_prepare"): _fields("identity"),
    ("tcfm5_1rw", "cycle_snapshot"): _fields(
        "state", "run_busy", "run_done", "protocol_error"
    ),
    ("tcfm5_1rw", "term_accept"): _fields(
        "commit", "source_id", "plane", "y", "x", "lane", "gate",
        "destination_mask", "last", "delta"
    ),
    ("tcfm5_1rw", "acc_update_accept"): _fields(
        "bank", "source_id", "lane", "gate", "address", "delta"
    ),
    ("tcfm5_1rw", "acc_physical_command"): _fields(
        "bank", "address", "data"
    ),
    ("tcfm5_1rw", "vector_read_accept"): _fields(
        "source_id", "plane", "y", "x"
    ),
    ("tcfm5_1rw", "vector_read_response"): _fields(
        "source_id", "data"
    ),
    ("tcfm5_1rw", "serializer_input"): _fields(
        "source_id", "plane", "y", "x", "data", "last"
    ),
    ("tcfm5_1rw", "serializer_output"): _fields(
        "source_id", "plane", "y", "x", "out", "data", "last"
    ),
}
STALL_EVENT_FIELDS = {
    ("direct_online", "descriptor"): _fields(
        "source_id", "plane", "y", "x", "lane", "gate", "destination_mask",
        "last", "source_last"
    ),
    ("direct_online", "execute_lane"): _fields(
        "source_id", "plane", "y", "x", "lane", "gate", "destination_mask",
        "last", "source_last"
    ),
    ("tcfm5_1rw", "execute_lane"): _fields(
        "commit", "source_id", "plane", "y", "x", "lane", "gate",
        "destination_mask", "last"
    ),
    ("tcfm5_1rw", "vector_serializer"): _fields(
        "source_id", "plane", "y", "x", "out", "data", "last"
    ),
}

PHASE_FIELDS = COMMON_FIELDS | frozenset({"kind"})
TERMINAL_RE = re.compile(
    r"PASS Local5 EREP calibration v4 direct_terms=(\d+) direct_updates=(\d+) "
    r"tcfm5_terms=(\d+) tcfm5_updates=(\d+) serializer_outputs=(\d+)"
)
FINISH_RES = (
    re.compile(r".+: \$finish called at \d+ \(1ps\)"),
    re.compile(r"- .+:\d+: Verilog \$finish"),
)
SYNTH_HEIGHT = 3
SYNTH_WIDTH = 3
SYNTH_PLANES = 1
SYNTH_HEAD_DIM = 4
SYNTH_OUT_DIM = 2
SYNTH_ACC_W = 32
SYNTH_CYCLE_TIME = 2000
SYNTH_DIRECT_K = (0x3, 0x2, 0xC, 0x9, 0x4, 0x7, 0x7, 0x7, 0x7)
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)

EVENT_SEMANTICS = {
    ("direct_online", "weight_accept"): ("weight_service", "weight", {0}),
    ("direct_online", "context_prepare"): (
        "context_prepare_1rw", "context_prepare", {0}
    ),
    ("direct_online", "cycle_snapshot"): ("pipeline", "state", {0, 1, 2, 3, 4}),
    ("direct_online", "relation_accept"): (
        "relation_workspace_1rw", "relation_write", {1}
    ),
    ("direct_online", "relation_read"): (
        "relation_workspace_1rw", "relation_read", {2}
    ),
    ("direct_online", "fifo_enqueue"): ("fifo2_enq", "fifo_enqueue", {2}),
    ("direct_online", "fifo_dequeue"): ("fifo2_deq", "fifo_dequeue", {2}),
    ("direct_online", "term_accept"): ("execute_lane", "term", {2}),
    ("direct_online", "acc_update_accept"): (None, "acc_write", {2}),
    ("direct_online", "acc_physical_command"): (None, None, {2, 3, 4}),
    ("direct_online", "drain_read_accept"): (
        "drain_read_1rw", "drain_read", {3, 4}
    ),
    ("direct_online", "drain_read_response"): (
        "drain_read_response", "final", {4}
    ),
    ("tcfm5_1rw", "weight_accept"): ("weight_service", "weight", {0}),
    ("tcfm5_1rw", "context_prepare"): (
        "context_prepare_1rw", "context_prepare", {0}
    ),
    ("tcfm5_1rw", "cycle_snapshot"): ("pipeline", "state", {0, 2, 3, 4}),
    ("tcfm5_1rw", "term_accept"): ("execute_lane", "term", {2}),
    ("tcfm5_1rw", "acc_update_accept"): (None, "acc_write", {2}),
    ("tcfm5_1rw", "acc_physical_command"): (None, None, {2, 3, 4}),
    ("tcfm5_1rw", "vector_read_accept"): (
        "drain_read_1rw", "drain_read", {3, 4}
    ),
    ("tcfm5_1rw", "vector_read_response"): (
        "drain_read_response", "final_vector", {4}
    ),
    ("tcfm5_1rw", "serializer_input"): (
        "vector_serializer", "vector_accept", {4}
    ),
    ("tcfm5_1rw", "serializer_output"): (
        "vector_serializer", "final", {4}
    ),
}
STALL_SEMANTICS = {
    ("direct_online", "execute_lane"): 2,
    ("tcfm5_1rw", "execute_lane"): 2,
    ("tcfm5_1rw", "vector_serializer"): 4,
}
EXPECTED_STALL_COUNTS = Counter(
    {
        ("direct_online", "execute_lane"): 4,
        ("tcfm5_1rw", "execute_lane"): 1,
        ("tcfm5_1rw", "vector_serializer"): 18,
    }
)
EXPECTED_STALL_RUN_LENGTHS = {
    ("direct_online", "execute_lane"): [1, 1, 1, 1],
    ("tcfm5_1rw", "execute_lane"): [1],
    ("tcfm5_1rw", "vector_serializer"): [2] * 9,
}


def parse_trace(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    terminal: tuple[int, int, int, int, int] | None = None
    finish_seen = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if finish_seen and line:
            raise ValueError(f"line {line_number}: output follows simulator finish")
        if not line.startswith(TRACE_PREFIX):
            match = TERMINAL_RE.fullmatch(line)
            if match:
                if terminal is not None or not rows:
                    raise ValueError("trace has duplicate or premature terminal PASS")
                terminal = tuple(int(value) for value in match.groups())
                continue
            if any(pattern.fullmatch(line) for pattern in FINISH_RES):
                if terminal is None or finish_seen:
                    raise ValueError("trace has misplaced or duplicate simulator finish")
                finish_seen = True
                continue
            if not line:
                continue
            raise ValueError(f"line {line_number}: unknown non-trace output {line!r}")
        if terminal is not None:
            raise ValueError(f"line {line_number}: trace row follows terminal PASS")
        fields: dict[str, str] = {}
        for token in shlex.split(line[len(TRACE_PREFIX) :]):
            if "=" not in token:
                raise ValueError(f"line {line_number}: malformed token {token!r}")
            key, value = token.split("=", 1)
            if key in fields:
                raise ValueError(f"line {line_number}: duplicate key {key!r}")
            fields[key] = value
        if fields.get("schema") != RAW_SCHEMA:
            raise ValueError(f"line {line_number}: raw schema mismatch")
        key = (fields.get("candidate", ""), fields.get("event", ""))
        if fields.get("event") == "phase_boundary":
            allowed = PHASE_FIELDS | frozenset({"relation_records"})
            if not PHASE_FIELDS <= set(fields) or not set(fields) <= allowed:
                raise ValueError(f"line {line_number}: phase boundary field schema mismatch")
        elif fields.get("event") == "stall_observation":
            expected = STALL_EVENT_FIELDS.get(
                (fields.get("candidate", ""), fields.get("resource", ""))
            )
            if expected is None or set(fields) != expected:
                raise ValueError(
                    f"line {line_number}: exact stall field schema mismatch"
                )
        else:
            expected = EVENT_FIELDS.get(key)
            if expected is None or set(fields) != expected:
                raise ValueError(
                    f"line {line_number}: exact event field schema mismatch for {key}"
                )
        fields["_line"] = str(line_number)
        rows.append(fields)
    if not rows or terminal is None or not finish_seen:
        raise ValueError("trace lacks EREP rows, terminal PASS, or simulator finish")
    counts = (
        len(rows_for(rows, "direct_online", "term_accept")),
        len(rows_for(rows, "direct_online", "acc_update_accept")),
        len(rows_for(rows, "tcfm5_1rw", "term_accept")),
        len(rows_for(rows, "tcfm5_1rw", "acc_update_accept")),
        len(rows_for(rows, "tcfm5_1rw", "serializer_output")),
    )
    if terminal != counts:
        raise ValueError(f"terminal PASS counts disagree with trace: {terminal} != {counts}")
    return rows


def strict_int(row: dict[str, str], key: str, *, minimum: int | None = None) -> int:
    try:
        value = int(row[key], 10)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"line {row.get('_line')}: {key} is not a decimal integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"line {row.get('_line')}: {key} must be >= {minimum}")
    return value


def bounded_int(
    row: dict[str, str], key: str, *, minimum: int, maximum: int
) -> int:
    value = strict_int(row, key)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"line {row.get('_line')}: {key}={value} is outside "
            f"[{minimum}, {maximum}]"
        )
    return value


def bounded_hex(row: dict[str, str], key: str, *, width: int) -> int:
    value = parse_hex(row[key], key)
    if value >= (1 << width):
        raise ValueError(
            f"line {row.get('_line')}: {key} exceeds its {width}-bit RTL field"
        )
    return value


def validate_payload_ranges(row: dict[str, str]) -> None:
    """Check every emitted payload against this fixed synthetic RTL instance."""
    event = row["event"]
    candidate = row["candidate"]

    if event not in {"phase_boundary", "cycle_snapshot"}:
        for field in ("valid", "ready", "fire"):
            if field in row:
                bounded_int(row, field, minimum=0, maximum=1)
    elif event == "cycle_snapshot":
        handshake_width = 5 if candidate == "direct_online" else 4
        valid = bounded_hex(row, "valid", width=handshake_width)
        ready = bounded_hex(row, "ready", width=handshake_width)
        fire = bounded_hex(row, "fire", width=handshake_width)
        if fire != (valid & ready):
            raise ValueError(
                f"line {row.get('_line')}: packed snapshot fire != valid & ready"
            )

    for field in (
        "last",
        "source_last",
        "commit",
        "projection_busy",
        "projection_done",
        "relation_active",
        "relation_done",
        "protocol_error",
        "run_busy",
        "run_done",
    ):
        if field in row:
            bounded_int(row, field, minimum=0, maximum=1)

    for field, maximum in (
        ("source_id", SYNTH_HEIGHT * SYNTH_WIDTH * SYNTH_PLANES - 1),
        ("plane", SYNTH_PLANES - 1),
        ("y", SYNTH_HEIGHT - 1),
        ("x", SYNTH_WIDTH - 1),
        ("lane", SYNTH_HEAD_DIM - 1),
        ("out", SYNTH_OUT_DIM - 1),
        ("bank", 4),
        ("address", SYNTH_PLANES * SYNTH_HEIGHT * ((SYNTH_WIDTH + 4) // 5) - 1),
    ):
        if field in row:
            bounded_int(row, field, minimum=0, maximum=maximum)

    if all(field in row for field in ("source_id", "plane", "y", "x")):
        source = strict_int(row, "source_id")
        expected = (
            strict_int(row, "plane") * SYNTH_HEIGHT * SYNTH_WIDTH
            + strict_int(row, "y") * SYNTH_WIDTH
            + strict_int(row, "x")
        )
        if source != expected:
            raise ValueError(f"line {row.get('_line')}: source/coordinate identity mismatch")

    if "gate" in row:
        gate = bounded_int(row, "gate", minimum=0, maximum=(1 << 9) - 1)
        if event in {"term_accept", "stall_observation"} and row.get("resource") == "execute_lane" and gate == 0:
            raise ValueError(f"line {row.get('_line')}: executable gate must be nonzero")
    for field in ("candidate_valid", "active_mask", "mask", "destination_mask"):
        if field in row:
            bounded_hex(row, field, width=5)
    if "k" in row:
        bounded_hex(row, "k", width=SYNTH_HEAD_DIM)
    if "gates" in row:
        bounded_hex(row, "gates", width=5 * 9)
    if "delta" in row:
        bounded_hex(row, "delta", width=SYNTH_OUT_DIM * SYNTH_ACC_W)
    if "data" in row:
        if event in {"acc_physical_command", "vector_read_response", "serializer_input"}:
            bounded_hex(row, "data", width=SYNTH_OUT_DIM * SYNTH_ACC_W)
        elif event == "weight_accept":
            bounded_int(row, "data", minimum=-(1 << 7), maximum=(1 << 7) - 1)
        else:
            bounded_int(
                row,
                "data",
                minimum=-(1 << (SYNTH_ACC_W - 1)),
                maximum=(1 << (SYNTH_ACC_W - 1)) - 1,
            )
    for field in ("fifo_occupancy", "occupancy_pre", "occupancy_post"):
        if field in row:
            bounded_int(row, field, minimum=0, maximum=2)
    if "state" in row:
        bounded_int(row, "state", minimum=0, maximum=4)
    if "identity" in row and row["identity"] != "window_0":
        raise ValueError(f"line {row.get('_line')}: context identity mismatch")


def rows_for(
    rows: Iterable[dict[str, str]],
    candidate: str,
    event: str | None = None,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["candidate"] == candidate]
    if event is not None:
        selected = [row for row in selected if row["event"] == event]
    return selected


def validate_event_semantics(row: dict[str, str], phase: int) -> None:
    candidate = row["candidate"]
    event = row["event"]
    resource = row.get("resource")
    kind = row.get("kind")
    if event == "phase_boundary":
        return
    if event == "stall_observation":
        expected_phase = STALL_SEMANTICS.get((candidate, resource or ""))
        if expected_phase is None or kind != "backpressure" or phase != expected_phase:
            raise ValueError("stall resource/kind/phase semantic mismatch")
    else:
        semantics = EVENT_SEMANTICS.get((candidate, event))
        if semantics is None:
            raise ValueError(f"missing semantic contract for {(candidate, event)}")
        expected_resource, expected_kind, phases = semantics
        if expected_resource is not None and resource != expected_resource:
            raise ValueError(f"{candidate}:{event} resource semantic mismatch")
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"{candidate}:{event} kind semantic mismatch")
        if phase not in phases:
            raise ValueError(f"{candidate}:{event} phase semantic mismatch")
        if event in {"acc_update_accept", "acc_physical_command"}:
            bank = strict_int(row, "bank", minimum=0)
            if not 0 <= bank < 5 or resource != f"acc_bank_{bank}_1rw":
                raise ValueError(f"{candidate}:{event} bank/resource mismatch")
        if event == "acc_physical_command":
            if kind not in {"physical_read", "physical_write"}:
                raise ValueError(f"{candidate}: invalid physical command kind")
            if kind == "physical_write" and phase != 2:
                raise ValueError(f"{candidate}: physical write outside compute phase")

    if candidate == "direct_online":
        suffix = "tb_qfit_local5_erep_calibration_v4.u_direct_monitor.monitor_direct"
    elif event in {"serializer_input", "serializer_output"} or resource == "vector_serializer":
        suffix = "tb_qfit_local5_erep_calibration_v4.u_serializer_monitor"
    else:
        suffix = "tb_qfit_local5_erep_calibration_v4.u_tcfm5_monitor.monitor_tcfm5"
    if row["scope"] not in {suffix, f"TOP.{suffix}"}:
        raise ValueError(f"{candidate}:{event} monitor scope mismatch")


def validate_common_event_values(rows: list[dict[str, str]]) -> dict[str, Any]:
    snapshots: dict[str, list[int]] = defaultdict(list)
    snapshot_times: dict[tuple[str, int], int] = {}
    all_times: list[int] = []
    for row in rows:
        cycle = strict_int(row, "cycle", minimum=0)
        event_time = strict_int(row, "time", minimum=0)
        all_times.append(event_time)
        phase = strict_int(row, "phase", minimum=0)
        if phase > 4 or not row["scope"]:
            raise ValueError("event phase/scope is outside the synthetic contract")
        validate_event_semantics(row, phase)
        validate_payload_ranges(row)
        window = strict_int(row, "window")
        if row["event"] == "weight_accept":
            if window != -1 or phase != 0:
                raise ValueError("weight preload must be outside window -1 in phase 0")
        elif window != 0:
            raise ValueError("synthetic calibration supports exactly window 0")

        if row["event"] == "cycle_snapshot":
            snapshots[row["candidate"]].append(cycle)
            snapshot_key = (row["candidate"], cycle)
            if snapshot_key in snapshot_times:
                raise ValueError(f"duplicate cycle snapshot {snapshot_key}")
            snapshot_times[snapshot_key] = event_time
            if row.get("protocol_error") != "0":
                raise ValueError("cycle snapshot reports protocol_error")
        elif row["event"] == "stall_observation":
            if (row["valid"], row["ready"], row["fire"]) != ("1", "0", "0"):
                raise ValueError("stall observation has invalid handshake values")
        elif row["event"] != "phase_boundary":
            if (row["valid"], row["ready"], row["fire"]) != ("1", "1", "1"):
                raise ValueError("accepted transaction has invalid handshake values")
    for candidate, cycles in snapshots.items():
        if cycles != list(range(cycles[-1] + 1)):
            raise ValueError(f"{candidate}: cycle snapshots are not contiguous from zero")
    if set(snapshots) != {"direct_online", "tcfm5_1rw"}:
        raise ValueError("cycle snapshot candidate coverage is incomplete")
    if all_times != sorted(all_times):
        raise ValueError("trace event time is not globally monotonic")
    scope_cycle_epochs: dict[str, int] = {}
    for row in rows:
        if row["event"] in {"phase_boundary", "weight_accept"}:
            continue
        epoch = strict_int(row, "time") - SYNTH_CYCLE_TIME * strict_int(row, "cycle")
        scope = row["scope"]
        if scope in scope_cycle_epochs and scope_cycle_epochs[scope] != epoch:
            raise ValueError(
                f"line {row.get('_line')}: event time is not affine to its monitor cycle"
            )
        scope_cycle_epochs[scope] = epoch
    if len(scope_cycle_epochs) != 3:
        raise ValueError("synthetic trace lacks one of the three monitor time domains")
    return {
        "candidate_windows": 1,
        "cycle_snapshots_contiguous": True,
        "accepted_handshakes_exact": True,
        "stall_handshakes_exact": True,
        "protocol_error_count": 0,
        "event_time_globally_monotonic": True,
        "event_time_matches_monitor_cycle": True,
        "monitor_time_domain_count": len(scope_cycle_epochs),
        "synthetic_payload_widths_and_ranges_exact": True,
    }


def validate_phase_boundaries(rows: list[dict[str, str]], candidate: str) -> dict[str, int]:
    required = REQUIRED_BOUNDARIES.get(candidate)
    if required is None:
        raise ValueError(f"{candidate}: no phase-boundary contract")
    boundaries: dict[str, int] = {}
    phase_by_kind = {
        "PREPARE_BEGIN": 0,
        "FILL_BEGIN": 1,
        "EXECUTE_BEGIN": 2,
        "DRAIN_BEGIN": 3,
        "COMPUTE_DONE": 4,
    }
    for row in rows_for(rows, candidate, "phase_boundary"):
        kind = row.get("kind")
        if kind in boundaries:
            raise ValueError(f"{candidate}: duplicate phase boundary {kind}")
        if kind is None:
            raise ValueError(f"{candidate}: phase boundary lacks kind")
        if kind not in required or strict_int(row, "phase") != phase_by_kind[kind]:
            raise ValueError(f"{candidate}: unknown or mis-phased boundary {kind}")
        scope_suffix = (
            "tb_qfit_local5_erep_calibration_v4.u_direct_monitor.monitor_direct"
            if candidate == "direct_online"
            else "tb_qfit_local5_erep_calibration_v4.u_tcfm5_monitor.monitor_tcfm5"
        )
        if row["scope"] not in {scope_suffix, f"TOP.{scope_suffix}"}:
            raise ValueError(f"{candidate}: phase boundary monitor scope mismatch")
        has_relation_records = "relation_records" in row
        if has_relation_records != (candidate == "direct_online" and kind == "EXECUTE_BEGIN"):
            raise ValueError(f"{candidate}: relation_records boundary field is misplaced")
        boundaries[kind] = strict_int(row, "cycle", minimum=0)
    missing = [kind for kind in required if kind not in boundaries]
    if missing:
        raise ValueError(f"{candidate}: missing phase boundaries {missing}")
    if set(boundaries) != set(required):
        raise ValueError(f"{candidate}: phase boundary set is not exact")
    if candidate == "direct_online":
        execute = next(
            row
            for row in rows_for(rows, candidate, "phase_boundary")
            if row["kind"] == "EXECUTE_BEGIN"
        )
        if strict_int(execute, "relation_records") != len(
            rows_for(rows, candidate, "relation_accept")
        ):
            raise ValueError("Direct EXECUTE_BEGIN relation count mismatch")
    ordered = [boundaries[kind] for kind in required]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError(f"{candidate}: phase boundaries are not strictly ordered")
    if boundaries != EXPECTED_SYNTH_BOUNDARIES[candidate]:
        raise ValueError(
            f"{candidate}: boundaries differ from frozen synthetic fixture golden"
        )
    return boundaries


def full_boundary_cycles(
    rows: list[dict[str, str]], candidate: str, terminal_events: set[str]
) -> int:
    terminals = [row for row in rows_for(rows, candidate) if row["event"] in terminal_events]
    if not terminals:
        raise ValueError(f"{candidate}: no terminal readout event")
    terminal_time = max(strict_int(row, "time", minimum=0) for row in terminals)
    snapshots = [
        row
        for row in rows_for(rows, candidate, "cycle_snapshot")
        if strict_int(row, "time", minimum=0) <= terminal_time
    ]
    if not snapshots:
        raise ValueError(f"{candidate}: cannot map terminal time to candidate cycle")
    return max(strict_int(row, "cycle", minimum=0) for row in snapshots) + 1


def validate_snapshot_state_contract(
    rows: list[dict[str, str]],
    direct_boundaries: dict[str, int],
    tcfm5_boundaries: dict[str, int],
) -> dict[str, Any]:
    direct_relation_done_cycle = EXPECTED_DIRECT_RELATION_DONE_CYCLE
    for row in rows_for(rows, "direct_online", "cycle_snapshot"):
        cycle = strict_int(row, "cycle")
        if cycle < direct_boundaries["FILL_BEGIN"]:
            expected_phase = 0
        elif cycle < direct_boundaries["EXECUTE_BEGIN"]:
            expected_phase = 1
        elif cycle < direct_boundaries["DRAIN_BEGIN"]:
            expected_phase = 2
        elif cycle <= direct_boundaries["COMPUTE_DONE"]:
            expected_phase = 3
        else:
            expected_phase = 4
        expected = {
            "phase": expected_phase,
            "projection_busy": int(
                0 < cycle < direct_boundaries["COMPUTE_DONE"]
            ),
            "projection_done": int(cycle >= direct_boundaries["COMPUTE_DONE"]),
            "relation_active": int(0 < cycle < direct_relation_done_cycle),
            "relation_done": int(cycle >= direct_relation_done_cycle),
        }
        for field, value in expected.items():
            if strict_int(row, field) != value:
                raise ValueError(
                    f"Direct snapshot {cycle} has invalid frozen {field} state"
                )

    execute = tcfm5_boundaries["EXECUTE_BEGIN"]
    drain = tcfm5_boundaries["DRAIN_BEGIN"]
    done = tcfm5_boundaries["COMPUTE_DONE"]
    for row in rows_for(rows, "tcfm5_1rw", "cycle_snapshot"):
        cycle = strict_int(row, "cycle")
        if cycle < execute:
            expected_state = 0
        elif cycle < drain:
            expected_state = 2
        elif cycle < done:
            expected_state = 3
        else:
            expected_state = 4
        if cycle <= execute:
            expected_phase = 0
        elif cycle <= drain:
            expected_phase = 2
        elif cycle <= done:
            expected_phase = 3
        else:
            expected_phase = 4
        expected = {
            "phase": expected_phase,
            "state": expected_state,
            "run_busy": int(execute <= cycle < done),
            "run_done": int(cycle >= done),
        }
        for field, value in expected.items():
            if strict_int(row, field) != value:
                raise ValueError(
                    f"TCFM5 snapshot {cycle} has invalid frozen {field} state"
                )
    return {
        "packed_fire_equals_valid_and_ready": True,
        "direct_pipeline_state_exact": True,
        "tcfm5_pipeline_state_exact": True,
        "state_expectations_use_frozen_fixture_boundaries": True,
    }


def validate_snapshot_event_ledger(
    rows: list[dict[str, str]],
    direct_boundaries: dict[str, int],
) -> dict[str, Any]:
    mappings = {
        "direct_online": {
            0: "relation_accept",
            1: "fifo_enqueue",
            2: "term_accept",
            4: "drain_read_accept",
        },
        "tcfm5_1rw": {
            0: "term_accept",
            3: "vector_read_accept",
        },
    }
    matched_event_counts: dict[str, dict[str, int]] = {}
    for candidate, bit_events in mappings.items():
        snapshots = {
            strict_int(row, "cycle"): row
            for row in rows_for(rows, candidate, "cycle_snapshot")
        }
        event_cycles: dict[int, set[int]] = {
            bit: {
                strict_int(row, "cycle")
                for row in rows_for(rows, candidate, event)
            }
            for bit, event in bit_events.items()
        }
        for bit, event in bit_events.items():
            if len(event_cycles[bit]) != len(rows_for(rows, candidate, event)):
                raise ValueError(f"{candidate}: duplicate same-cycle {event} transaction")
            if not event_cycles[bit] <= set(snapshots):
                raise ValueError(f"{candidate}: {event} is outside snapshot coverage")
        matched_event_counts[candidate] = {
            event: len(event_cycles[bit]) for bit, event in bit_events.items()
        }
        for cycle, snapshot in snapshots.items():
            expected_fire = sum(
                1 << bit for bit, cycles in event_cycles.items() if cycle in cycles
            )
            if candidate == "direct_online" and cycle == direct_boundaries["DRAIN_BEGIN"] - 1:
                expected_fire |= 1 << 3
            observed_fire = parse_hex(snapshot["fire"], "snapshot fire")
            if observed_fire != expected_fire:
                raise ValueError(
                    f"{candidate}: snapshot cycle {cycle} fire does not match discrete event ledger"
                )
            if candidate == "tcfm5_1rw" and observed_fire & 0b0110:
                raise ValueError("TCFM5 synthetic fixture unexpectedly used scalar-read/close fire")
    return {
        "packed_fire_matches_discrete_event_ledger": True,
        "direct_close_fire_matches_frozen_drain_boundary": True,
        "tcfm5_unused_scalar_read_and_close_fire_zero": True,
        "matched_event_counts": matched_event_counts,
    }


def validate_relation_and_fifo(rows: list[dict[str, str]]) -> dict[str, Any]:
    accepted = rows_for(rows, "direct_online", "relation_accept")
    source_ids = [strict_int(row, "source_id", minimum=0) for row in accepted]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Direct accepted relation records contain duplicate source IDs")
    if sorted(source_ids) != list(range(len(source_ids))):
        raise ValueError("Direct accepted relation records are not exact contiguous source IDs")

    enqueues = rows_for(rows, "direct_online", "fifo_enqueue")
    dequeues = rows_for(rows, "direct_online", "fifo_dequeue")
    enqueue_ids = [strict_int(row, "source_id", minimum=0) for row in enqueues]
    dequeue_ids = [strict_int(row, "source_id", minimum=0) for row in dequeues]
    accepted_by_id = {strict_int(row, "source_id"): row for row in accepted}
    for row in accepted:
        source = strict_int(row, "source_id")
        plane = strict_int(row, "plane")
        y = strict_int(row, "y")
        x = strict_int(row, "x")
        if source != plane * SYNTH_HEIGHT * SYNTH_WIDTH + y * SYNTH_WIDTH + x:
            raise ValueError("accepted relation source identity/coordinate mismatch")
        candidate_valid = parse_hex(row["candidate_valid"], "candidate_valid")
        active_mask = parse_hex(row["active_mask"], "active_mask")
        if active_mask == 0 or active_mask & ~candidate_valid:
            raise ValueError("accepted relation active mask is not a nonempty candidate subset")
        if (
            candidate_valid != 1
            or active_mask != 1
            or parse_hex(row["k"], "relation K") != SYNTH_DIRECT_K[source]
            or parse_hex(row["gates"], "relation gates") != source + 1
        ):
            raise ValueError("accepted relation differs from the frozen synthetic fixture")
    relation_reads = rows_for(rows, "direct_online", "relation_read")
    relation_read_ids = [strict_int(row, "source_id") for row in relation_reads]
    if sorted(relation_read_ids) != source_ids:
        raise ValueError("relation read coverage differs from accepted relation coverage")
    if enqueue_ids != relation_read_ids or dequeue_ids != relation_read_ids:
        raise ValueError("FIFO2 does not preserve exact relation-read order")
    for enqueue, dequeue, relation_read in zip(
        enqueues, dequeues, relation_reads, strict=True
    ):
        source = strict_int(enqueue, "source_id")
        accepted_row = accepted_by_id[source]
        for field in ("plane", "y", "x"):
            if enqueue[field] != accepted_row[field] or dequeue[field] != accepted_row[field]:
                raise ValueError(f"FIFO2 changed descriptor coordinate {field}")
            if relation_read[field] != accepted_row[field]:
                raise ValueError(f"relation read changed descriptor coordinate {field}")
        for left, right in (("k", "k"), ("gates", "gates"), ("mask", "active_mask")):
            if enqueue[left] != accepted_row[right]:
                raise ValueError(f"FIFO2 changed descriptor payload {left}")
        if enqueue["last"] != relation_read["last"]:
            raise ValueError("FIFO2 changed relation last marker")

    if [strict_int(row, "last") for row in relation_reads] != [0] * (
        len(relation_reads) - 1
    ) + [1]:
        raise ValueError("relation read last marker is not exact")

    expected_terms: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    for enqueue in enqueues:
        source = strict_int(enqueue, "source_id")
        plane = strict_int(enqueue, "plane")
        y = strict_int(enqueue, "y")
        x = strict_int(enqueue, "x")
        k = parse_hex(enqueue["k"], "FIFO K")
        packed_gates = parse_hex(enqueue["gates"], "FIFO gates")
        mask = parse_hex(enqueue["mask"], "FIFO mask")
        if k >> SYNTH_HEAD_DIM or packed_gates >> (5 * 9) or mask >> 5:
            raise ValueError("FIFO payload exceeds synthetic field width")
        unique_gates: list[tuple[int, int]] = []
        for role in range(5):
            gate = (packed_gates >> (role * 9)) & ((1 << 9) - 1)
            if not ((mask >> role) & 1) or gate == 0:
                continue
            for index, (known_gate, known_mask) in enumerate(unique_gates):
                if known_gate == gate:
                    unique_gates[index] = (known_gate, known_mask | (1 << role))
                    break
            else:
                unique_gates.append((gate, 1 << role))
        active_lanes = [lane for lane in range(SYNTH_HEAD_DIM) if (k >> lane) & 1]
        term_count = len(active_lanes) * len(unique_gates)
        term_index = 0
        for lane in active_lanes:
            for gate, destination_mask in unique_gates:
                term_index += 1
                term_last = int(term_index == term_count)
                expected_terms.append(
                    (
                        source,
                        plane,
                        y,
                        x,
                        lane,
                        gate,
                        destination_mask,
                        term_last,
                        term_last * strict_int(enqueue, "last"),
                    )
                )

    observed_terms = [
        (
            strict_int(row, "source_id"),
            strict_int(row, "plane"),
            strict_int(row, "y"),
            strict_int(row, "x"),
            strict_int(row, "lane"),
            strict_int(row, "gate"),
            parse_hex(row["destination_mask"], "term destination mask"),
            strict_int(row, "last"),
            strict_int(row, "source_last"),
        )
        for row in rows_for(rows, "direct_online", "term_accept")
    ]
    if observed_terms != expected_terms:
        raise ValueError("Direct relation/FIFO payload does not reconstruct exact term stream")
    occupancies = []
    for row in rows_for(rows, "direct_online"):
        for key in ("fifo_occupancy", "occupancy_pre", "occupancy_post"):
            if key in row:
                occupancies.append(strict_int(row, key, minimum=0))
    if not occupancies or max(occupancies) > 2:
        raise ValueError("FIFO2 occupancy evidence is absent or exceeds two")
    return {
        "accepted_relation_records": len(accepted),
        "accepted_relation_source_ids": source_ids,
        "fifo_enqueues": len(enqueues),
        "fifo_dequeues": len(dequeues),
        "fifo_max_occupancy": max(occupancies),
        "fifo_identity_multiset_match": True,
        "fifo_order_and_payload_match": True,
        "relation_payload_to_term_stream_match": True,
        "reconstructed_term_count": len(expected_terms),
    }


def validate_bank_commands(rows: list[dict[str, str]], candidate: str) -> dict[str, Any]:
    logical = rows_for(rows, candidate, "acc_update_accept")
    physical = rows_for(rows, candidate, "acc_physical_command")
    seen: set[tuple[int, int]] = set()
    for row in physical:
        key = (strict_int(row, "bank", minimum=0), strict_int(row, "cycle", minimum=0))
        if key in seen:
            raise ValueError(f"{candidate}: same-cycle physical 1RW collision at {key}")
        seen.add(key)
    logical_by_bank = Counter(strict_int(row, "bank", minimum=0) for row in logical)
    physical_by_bank = Counter(strict_int(row, "bank", minimum=0) for row in physical)
    read_by_bank = Counter(
        strict_int(row, "bank", minimum=0)
        for row in physical
        if row.get("kind") == "physical_read"
    )
    write_by_bank = Counter(
        strict_int(row, "bank", minimum=0)
        for row in physical
        if row.get("kind") == "physical_write"
    )
    return {
        "logical_acc_updates": len(logical),
        "logical_updates_by_bank": {str(bank): logical_by_bank[bank] for bank in range(5)},
        "physical_1rw_commands": len(physical),
        "physical_commands_by_bank": {str(bank): physical_by_bank[bank] for bank in range(5)},
        "physical_reads_by_bank": {str(bank): read_by_bank[bank] for bank in range(5)},
        "physical_writes_by_bank": {str(bank): write_by_bank[bank] for bank in range(5)},
        "same_cycle_single_port_collisions": 0,
    }


def parse_hex(value: str, name: str) -> int:
    if re.fullmatch(r"[0-9a-fA-F]+", value) is None:
        raise ValueError(f"{name} is not an unsigned hexadecimal value")
    return int(value, 16)


def signed_word(value: int, width: int = SYNTH_ACC_W) -> int:
    mask = (1 << width) - 1
    value &= mask
    return value - (1 << width) if value & (1 << (width - 1)) else value


def pack_vector(values: Iterable[int]) -> int:
    packed = 0
    mask = (1 << SYNTH_ACC_W) - 1
    for index, value in enumerate(values):
        packed |= (value & mask) << (index * SYNTH_ACC_W)
    return packed


def unpack_vector(value: int) -> tuple[int, ...]:
    mask = (1 << SYNTH_ACC_W) - 1
    return tuple(
        signed_word((value >> (index * SYNTH_ACC_W)) & mask)
        for index in range(SYNTH_OUT_DIM)
    )


def destination_geometry(plane: int, y: int, x: int, role: int) -> tuple[int, int, int, int, int]:
    dy = y + ROLE_DY[role]
    dx = x + ROLE_DX[role]
    if not (0 <= plane < SYNTH_PLANES and 0 <= dy < SYNTH_HEIGHT and 0 <= dx < SYNTH_WIDTH):
        raise ValueError("term destination mask selects an out-of-range role")
    source_id = plane * SYNTH_HEIGHT * SYNTH_WIDTH + dy * SYNTH_WIDTH + dx
    bank = (dx + 2 * dy) % 5
    x_groups = (SYNTH_WIDTH + 4) // 5
    address = plane * SYNTH_HEIGHT * x_groups + dy * x_groups + dx // 5
    return source_id, plane, dy, dx, bank, address


def _weights(rows: list[dict[str, str]], candidate: str) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    selected = rows_for(rows, candidate, "weight_accept")
    for row in selected:
        key = (strict_int(row, "lane"), strict_int(row, "out"))
        if key in result:
            raise ValueError(f"{candidate}: duplicate weight {key}")
        result[key] = strict_int(row, "data")
    expected = {
        (lane, out)
        for lane in range(SYNTH_HEAD_DIM)
        for out in range(SYNTH_OUT_DIM)
    }
    if set(result) != expected:
        raise ValueError(f"{candidate}: incomplete weight ledger")
    if [strict_int(row, "last") for row in selected] != [0] * (len(selected) - 1) + [1]:
        raise ValueError(f"{candidate}: weight last marker is not exact")
    return result


def validate_term_to_final_numeric_ledger(
    rows: list[dict[str, str]], candidate: str
) -> dict[str, Any]:
    weights = _weights(rows, candidate)
    terms = rows_for(rows, candidate, "term_accept")
    logical = rows_for(rows, candidate, "acc_update_accept")
    expected_updates: list[tuple[int, int, int, int, int, int]] = []
    accumulators: dict[int, list[int]] = defaultdict(lambda: [0] * SYNTH_OUT_DIM)
    for row in terms:
        if candidate == "tcfm5_1rw" and strict_int(row, "commit") != 1:
            raise ValueError("TCFM5 accepted term is not an atomic commit")
        source = strict_int(row, "source_id")
        plane = strict_int(row, "plane")
        y = strict_int(row, "y")
        x = strict_int(row, "x")
        if source != plane * SYNTH_HEIGHT * SYNTH_WIDTH + y * SYNTH_WIDTH + x:
            raise ValueError(f"{candidate}: term source identity/coordinate mismatch")
        lane = strict_int(row, "lane")
        gate = strict_int(row, "gate")
        if not 0 <= lane < SYNTH_HEAD_DIM or gate <= 0:
            raise ValueError(f"{candidate}: invalid term lane/gate")
        delta_values = tuple(gate * weights[(lane, out)] for out in range(SYNTH_OUT_DIM))
        delta = pack_vector(delta_values)
        if parse_hex(row["delta"], "term delta") != delta:
            raise ValueError(f"{candidate}: term delta is not gate times weight")
        mask = parse_hex(row["destination_mask"], "destination mask")
        if mask == 0 or mask >> 5:
            raise ValueError(f"{candidate}: invalid destination mask")
        for role in range(5):
            if not (mask >> role) & 1:
                continue
            destination, _, _, _, bank, address = destination_geometry(plane, y, x, role)
            expected_updates.append(
                (strict_int(row, "cycle"), bank, address, source, lane, gate)
            )
            for out, value in enumerate(delta_values):
                accumulators[destination][out] += value

    if candidate == "tcfm5_1rw" and [strict_int(row, "last") for row in terms] != [
        0
    ] * (len(terms) - 1) + [1]:
        raise ValueError("TCFM5 term last marker is not exact")

    observed_updates = [
        (
            strict_int(row, "cycle"), strict_int(row, "bank"),
            strict_int(row, "address"), strict_int(row, "source_id"),
            strict_int(row, "lane"), strict_int(row, "gate"),
        )
        for row in logical
    ]
    if Counter(observed_updates) != Counter(expected_updates):
        raise ValueError(f"{candidate}: term-to-logical-update expansion mismatch")
    for row in logical:
        source = strict_int(row, "source_id")
        matching = [
            term for term in terms
            if strict_int(term, "cycle") == strict_int(row, "cycle")
            and strict_int(term, "source_id") == source
            and strict_int(term, "lane") == strict_int(row, "lane")
            and strict_int(term, "gate") == strict_int(row, "gate")
        ]
        if len(matching) != 1 or row["delta"] != matching[0]["delta"]:
            raise ValueError(f"{candidate}: logical update lost term value identity")

    final_stream_order_exact = False
    final_last_marker_exact: bool | None = None
    final_last_marker_applicable = False
    read_accept_response_identity_exact = False
    if candidate == "direct_online":
        accepts = rows_for(rows, candidate, "drain_read_accept")
        finals = rows_for(rows, candidate, "drain_read_response")
        expected_identities = [
            (source, out)
            for source in range(SYNTH_HEIGHT * SYNTH_WIDTH * SYNTH_PLANES)
            for out in range(SYNTH_OUT_DIM)
        ]
        accept_identities = [
            (strict_int(row, "source_id"), strict_int(row, "out"))
            for row in accepts
        ]
        final_identities = [
            (strict_int(row, "source_id"), strict_int(row, "out"))
            for row in finals
        ]
        if accept_identities != expected_identities or final_identities != expected_identities:
            raise ValueError("Direct final readout is not exact source-major/out-major order")
        for accept, response in zip(accepts, finals, strict=True):
            source = strict_int(accept, "source_id")
            expected_plane = source // (SYNTH_HEIGHT * SYNTH_WIDTH)
            spatial = source % (SYNTH_HEIGHT * SYNTH_WIDTH)
            if (
                strict_int(response, "source_id") != strict_int(accept, "source_id")
                or strict_int(response, "out") != strict_int(accept, "out")
                or strict_int(response, "cycle") != strict_int(accept, "cycle") + 2
                or strict_int(accept, "plane") != expected_plane
                or strict_int(accept, "y") != spatial // SYNTH_WIDTH
                or strict_int(accept, "x") != spatial % SYNTH_WIDTH
            ):
                raise ValueError("Direct drain response lost accepted read identity/latency")
        final_stream_order_exact = True
        read_accept_response_identity_exact = True
    else:
        finals = rows_for(rows, candidate, "serializer_output")
        vector_accepts = rows_for(rows, candidate, "vector_read_accept")
        vectors = rows_for(rows, candidate, "vector_read_response")
        serializer_inputs = rows_for(rows, candidate, "serializer_input")
        expected_sources = list(
            range(SYNTH_HEIGHT * SYNTH_WIDTH * SYNTH_PLANES)
        )
        for name, selected in (
            ("vector read accept", vector_accepts),
            ("vector read response", vectors),
            ("serializer input", serializer_inputs),
        ):
            if [strict_int(row, "source_id") for row in selected] != expected_sources:
                raise ValueError(f"TCFM5 {name} is not exact source order")
        for accept, response in zip(vector_accepts, vectors, strict=True):
            if strict_int(response, "source_id") != strict_int(accept, "source_id"):
                raise ValueError("TCFM5 vector response lost accepted read identity")
        for row in vector_accepts + serializer_inputs:
            source = strict_int(row, "source_id")
            expected_plane = source // (SYNTH_HEIGHT * SYNTH_WIDTH)
            spatial = source % (SYNTH_HEIGHT * SYNTH_WIDTH)
            if (
                strict_int(row, "plane") != expected_plane
                or strict_int(row, "y") != spatial // SYNTH_WIDTH
                or strict_int(row, "x") != spatial % SYNTH_WIDTH
            ):
                raise ValueError("TCFM5 final stream source/coordinate identity mismatch")
        if [strict_int(row, "last") for row in serializer_inputs] != [0] * (
            len(serializer_inputs) - 1
        ) + [1]:
            raise ValueError("TCFM5 serializer input last marker is not exact")
        expected_final_identities = [
            (source, out)
            for source in expected_sources
            for out in range(SYNTH_OUT_DIM)
        ]
        if [
            (strict_int(row, "source_id"), strict_int(row, "out"))
            for row in finals
        ] != expected_final_identities:
            raise ValueError("TCFM5 serializer output is not source-major/out-major order")
        if [strict_int(row, "last") for row in finals] != [0] * (
            len(finals) - 1
        ) + [1]:
            raise ValueError("TCFM5 serializer output last marker is not exact")
        vector_by_source = {strict_int(row, "source_id"): row for row in vectors}
        input_by_source = {strict_int(row, "source_id"): row for row in serializer_inputs}
        if len(vector_by_source) != SYNTH_HEIGHT * SYNTH_WIDTH or set(vector_by_source) != set(input_by_source):
            raise ValueError("TCFM5 vector read/serializer input coverage mismatch")
        for source, row in vector_by_source.items():
            expected = pack_vector(accumulators[source])
            if parse_hex(row["data"], "vector read data") != expected:
                raise ValueError("TCFM5 vector read response differs from term accumulation")
            if input_by_source[source]["data"] != row["data"]:
                raise ValueError("TCFM5 serializer input differs from vector read response")
        final_stream_order_exact = True
        final_last_marker_exact = True
        final_last_marker_applicable = True
        read_accept_response_identity_exact = True

    observed_final: dict[tuple[int, int], int] = {}
    for row in finals:
        key = (strict_int(row, "source_id"), strict_int(row, "out"))
        if key in observed_final:
            raise ValueError(f"{candidate}: duplicate final scalar {key}")
        observed_final[key] = strict_int(row, "data")
    expected_final = {
        (source, out): accumulators[source][out]
        for source in range(SYNTH_HEIGHT * SYNTH_WIDTH * SYNTH_PLANES)
        for out in range(SYNTH_OUT_DIM)
    }
    if observed_final != expected_final:
        raise ValueError(f"{candidate}: final Acc32 values differ from term ledger")
    return {
        "term_count": len(terms),
        "logical_update_count": len(logical),
        "term_delta_matches_gate_times_weight": True,
        "term_to_logical_update_expansion_match": True,
        "final_stream_order_exact": final_stream_order_exact,
        "final_last_marker_exact": final_last_marker_exact,
        "final_last_marker_applicable": final_last_marker_applicable,
        "read_accept_response_identity_exact": read_accept_response_identity_exact,
        "final_acc32_mismatch_count": 0,
    }


def validate_physical_rmw_ledger(rows: list[dict[str, str]], candidate: str) -> dict[str, Any]:
    logical = rows_for(rows, candidate, "acc_update_accept")
    commands = rows_for(rows, candidate, "acc_physical_command")
    logical_by_key: dict[tuple[int, int, int], int] = {}
    for row in logical:
        key = (strict_int(row, "cycle"), strict_int(row, "bank"), strict_int(row, "address"))
        if key in logical_by_key:
            raise ValueError(f"{candidate}: duplicate same-cycle logical bank update")
        logical_by_key[key] = parse_hex(row["delta"], "logical delta")
    memories: list[dict[int, int]] = [dict() for _ in range(5)]
    pending: list[tuple[int, int] | None] = [None] * 5
    compute_commands = [row for row in commands if strict_int(row, "phase") == 2]
    for row in sorted(compute_commands, key=lambda value: (strict_int(value, "cycle"), strict_int(value, "bank"))):
        cycle = strict_int(row, "cycle")
        bank = strict_int(row, "bank")
        address = strict_int(row, "address")
        data = parse_hex(row["data"], "physical command data")
        key = (cycle, bank, address)
        if key in logical_by_key:
            delta = logical_by_key.pop(key)
            if pending[bank] is not None:
                raise ValueError(f"{candidate}: bank accepted update with pending RMW")
            if address in memories[bank]:
                if row["kind"] != "physical_read" or data != delta:
                    raise ValueError(f"{candidate}: RMW read command/delta mismatch")
                pending[bank] = (address, delta)
            else:
                if row["kind"] != "physical_write" or data != delta:
                    raise ValueError(f"{candidate}: first-touch physical write mismatch")
                memories[bank][address] = delta
        else:
            entry = pending[bank]
            if entry is None or row["kind"] != "physical_write" or entry[0] != address:
                raise ValueError(f"{candidate}: physical command has no logical/RMW cause")
            old = unpack_vector(memories[bank][address])
            delta_values = unpack_vector(entry[1])
            expected = pack_vector(a + b for a, b in zip(old, delta_values, strict=True))
            if data != expected:
                raise ValueError(f"{candidate}: RMW writeback data mismatch")
            memories[bank][address] = expected
            pending[bank] = None
    if logical_by_key or any(entry is not None for entry in pending):
        raise ValueError(f"{candidate}: incomplete logical-to-physical RMW ledger")
    if any(row["kind"] != "physical_read" for row in commands if strict_int(row, "phase") >= 3):
        raise ValueError(f"{candidate}: non-read command observed during drain/readout")
    return {
        "compute_physical_commands": len(compute_commands),
        "logical_update_to_rmw_match": True,
        "rmw_write_data_mismatch_count": 0,
        "readout_commands_are_reads": True,
    }


def summarize_backpressure(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts: Counter[tuple[str, str]] = Counter(
        (row["candidate"], row["resource"])
        for row in rows
        if row["event"] == "stall_observation"
    )
    if counts != EXPECTED_STALL_COUNTS:
        raise ValueError(
            f"synthetic fixture stall ledger mismatch: {counts} != {EXPECTED_STALL_COUNTS}"
        )

    accepted_event = {
        ("direct_online", "execute_lane"): "term_accept",
        ("tcfm5_1rw", "execute_lane"): "term_accept",
        ("tcfm5_1rw", "vector_serializer"): "serializer_output",
    }
    payload_fields = {
        ("direct_online", "execute_lane"): (
            "source_id", "plane", "y", "x", "lane", "gate",
            "destination_mask", "last", "source_last",
        ),
        ("tcfm5_1rw", "execute_lane"): (
            "source_id", "plane", "y", "x", "lane", "gate",
            "destination_mask", "last",
        ),
        ("tcfm5_1rw", "vector_serializer"): (
            "source_id", "plane", "y", "x", "out", "data", "last",
        ),
    }
    run_lengths_by_resource: dict[str, list[int]] = {}
    unique_stalled_cycles: dict[str, list[int]] = {}
    unique_accepted_identities: dict[str, list[dict[str, Any]]] = {}
    consumed_accepts: set[tuple[str, str, int, tuple[str, ...]]] = set()
    for key in EXPECTED_STALL_COUNTS:
        candidate, resource = key
        stalls = sorted(
            (
                row
                for row in rows_for(rows, candidate, "stall_observation")
                if row["resource"] == resource
            ),
            key=lambda row: strict_int(row, "cycle"),
        )
        stall_cycles = [strict_int(row, "cycle") for row in stalls]
        if len(stall_cycles) != len(set(stall_cycles)):
            raise ValueError(f"{candidate}:{resource} has duplicate stalled cycles")
        accepts = sorted(
            rows_for(rows, candidate, accepted_event[key]),
            key=lambda row: strict_int(row, "cycle"),
        )
        key_name = f"{candidate}:{resource}"
        unique_stalled_cycles[key_name] = stall_cycles
        accepted_identities: list[dict[str, Any]] = []
        resource_run_lengths: list[int] = []
        index = 0
        while index < len(stalls):
            first = stalls[index]
            fields = payload_fields[key]
            payload = tuple(first[field] for field in fields)
            first_cycle = strict_int(first, "cycle")
            last_cycle = first_cycle
            end = index + 1
            while end < len(stalls):
                next_cycle = strict_int(stalls[end], "cycle")
                next_payload = tuple(stalls[end][field] for field in fields)
                if next_cycle != last_cycle + 1 or next_payload != payload:
                    break
                last_cycle = next_cycle
                end += 1
            matching = [
                row
                for row in accepts
                if strict_int(row, "cycle") == last_cycle + 1
                and tuple(row[field] for field in fields) == payload
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"{candidate}:{resource} stall run does not lead to exact acceptance"
                )
            accept_identity = (
                candidate,
                resource,
                strict_int(matching[0], "cycle"),
                tuple(matching[0][field] for field in fields),
            )
            if accept_identity in consumed_accepts:
                raise ValueError(
                    f"{candidate}:{resource} acceptance was consumed by multiple stall runs"
                )
            consumed_accepts.add(accept_identity)
            if key == ("tcfm5_1rw", "execute_lane"):
                if any(strict_int(row, "commit") != 0 for row in stalls[index:end]):
                    raise ValueError("TCFM5 stalled term unexpectedly committed")
                if strict_int(matching[0], "commit") != 1:
                    raise ValueError("TCFM5 term was not committed on acceptance")
            resource_run_lengths.append(end - index)
            accepted_identities.append(
                {
                    "cycle": strict_int(matching[0], "cycle"),
                    "payload": {
                        field: matching[0][field]
                        for field in fields
                    },
                }
            )
            index = end
        if resource_run_lengths != EXPECTED_STALL_RUN_LENGTHS[key]:
            raise ValueError(
                f"{candidate}:{resource} stall run shape mismatch: "
                f"{resource_run_lengths} != {EXPECTED_STALL_RUN_LENGTHS[key]}"
            )
        run_lengths_by_resource[key_name] = resource_run_lengths
        unique_accepted_identities[key_name] = accepted_identities
    return {
        "counts": {
            f"{candidate}:{resource}": count
            for (candidate, resource), count in sorted(counts.items())
        },
        "exact_expected_counts": {
            f"{candidate}:{resource}": count
            for (candidate, resource), count in sorted(EXPECTED_STALL_COUNTS.items())
        },
        "stall_to_next_accept_identity_checked": True,
        "unique_stalled_cycles": unique_stalled_cycles,
        "unique_accepted_identities": unique_accepted_identities,
        "acceptance_consumed_at_most_once": True,
        "expected_run_lengths_by_resource": {
            f"{candidate}:{resource}": lengths
            for (candidate, resource), lengths in EXPECTED_STALL_RUN_LENGTHS.items()
        },
        "run_lengths_by_resource": run_lengths_by_resource,
        "consecutive_multi_cycle_run_count": sum(
            length >= 2
            for lengths in run_lengths_by_resource.values()
            for length in lengths
        ),
    }


def relative_command(row: dict[str, str], cycle: int) -> RelativeCommand:
    event = row["event"]
    source_id = row.get("source_id", "na")
    if event == "relation_read":
        return RelativeCommand(
            cycle,
            CommandResource.RELATION_WORKSPACE_1RW,
            CommandKind.RELATION_READ,
            f"source_{source_id}",
        )
    if event == "fifo_enqueue":
        return RelativeCommand(
            cycle,
            CommandResource.FIFO2_ENQ,
            CommandKind.FIFO_ENQUEUE,
            f"source_{source_id}",
        )
    if event == "fifo_dequeue":
        return RelativeCommand(
            cycle,
            CommandResource.FIFO2_DEQ,
            CommandKind.FIFO_DEQUEUE,
            f"source_{source_id}",
        )
    if event == "acc_update_accept":
        bank = strict_int(row, "bank", minimum=0)
        resources = (
            CommandResource.ACC_BANK_0_1RW,
            CommandResource.ACC_BANK_1_1RW,
            CommandResource.ACC_BANK_2_1RW,
            CommandResource.ACC_BANK_3_1RW,
            CommandResource.ACC_BANK_4_1RW,
        )
        return RelativeCommand(
            cycle,
            resources[bank],
            CommandKind.ACC_WRITE,
            (
                f"source_{source_id}_lane_{row.get('lane', 'na')}"
                f"_bank_{bank}_addr_{row.get('address', 'na')}"
            ),
        )
    if event in ("drain_read_accept", "vector_read_accept"):
        return RelativeCommand(
            cycle,
            CommandResource.DRAIN_READ_1RW,
            CommandKind.DRAIN_READ,
            f"source_{source_id}_out_{row.get('out', 'vector')}",
        )
    raise ValueError(f"cannot map event {event!r} to a strict RelativeCommand")


def build_direct_c0_schedule(
    rows: list[dict[str, str]], boundaries: dict[str, int], full_cycles: int
) -> dict[str, Any]:
    relation_records = len(rows_for(rows, "direct_online", "relation_accept"))
    terms = rows_for(rows, "direct_online", "term_accept")
    read_accepts = rows_for(rows, "direct_online", "drain_read_accept")
    read_responses = rows_for(rows, "direct_online", "drain_read_response")
    if not terms or len(read_accepts) != len(read_responses):
        raise ValueError("Direct tail-rule reconstruction requires terms and paired readout")

    predicted = {
        "PREPARE_BEGIN": 0,
        "FILL_BEGIN": 2,
        "EXECUTE_BEGIN": 2 + 2 * relation_records + 1,
        "DRAIN_BEGIN": max(strict_int(row, "cycle") for row in terms) + 2,
    }
    predicted["COMPUTE_DONE"] = predicted["DRAIN_BEGIN"] + 2
    if boundaries != predicted:
        raise ValueError(f"Direct phase rule mismatch: measured={boundaries} predicted={predicted}")
    expected_accept_cycles = [
        predicted["COMPUTE_DONE"] + 3 * index for index in range(len(read_accepts))
    ]
    expected_response_cycles = [cycle + 2 for cycle in expected_accept_cycles]
    if [strict_int(row, "cycle") for row in read_accepts] != expected_accept_cycles:
        raise ValueError("Direct read-accept service interval differs from frozen 1RW rule")
    if [strict_int(row, "cycle") for row in read_responses] != expected_response_cycles:
        raise ValueError("Direct read-response latency differs from frozen two-cycle rule")
    predicted_full_cycles = predicted["COMPUTE_DONE"] + 3 * len(read_accepts)
    if predicted_full_cycles != full_cycles:
        raise ValueError(
            f"C0 measured-tail rule reconstruction mismatch: reconstructed={predicted_full_cycles} "
            f"measured={full_cycles}"
        )

    direct_start = predicted["FILL_BEGIN"]
    compute_done = predicted["COMPUTE_DONE"]

    direct_events = {
        "relation_read",
        "fifo_enqueue",
        "fifo_dequeue",
        "acc_update_accept",
    }
    direct_commands = tuple(
        relative_command(row, strict_int(row, "cycle") - direct_start)
        for row in sorted(
            (
                row
                for row in rows_for(rows, "direct_online")
                if row["event"] in direct_events
                and direct_start <= strict_int(row, "cycle") < compute_done
            ),
            key=lambda row: (
                strict_int(row, "cycle"),
                row.get("resource", ""),
                row.get("source_id", ""),
            ),
        )
    )
    drain_commands = tuple(
        relative_command(row, strict_int(row, "cycle") - compute_done)
        for row in sorted(
            (
                row
                for row in rows_for(rows, "direct_online")
                if row["event"] == "drain_read_accept"
                and compute_done <= strict_int(row, "cycle") < full_cycles
            ),
            key=lambda row: (strict_int(row, "cycle"), row.get("source_id", "")),
        )
    )

    prepare = PhaseTrace(
        predicted["FILL_BEGIN"],
        (
            RelativeCommand(
                0,
                CommandResource.CONTEXT_PREPARE_1RW,
                CommandKind.CONTEXT_PREPARE,
                "window_0_context_prepare",
            ),
        ),
    )
    direct = PhaseTrace(compute_done - direct_start, direct_commands)
    drain = PhaseTrace(full_cycles - compute_done, drain_commands)
    empty = PhaseTrace(1, ())
    head = HeadCommandWork(
        input_head=0,
        epoch_records=0,
        fill=empty,
        direct_by_tile=(direct,),
        execute_by_tile=(empty,),
    )
    window = WindowCommandWork(
        identity="synthetic_direct_window_0",
        heads=(head,),
        output_tiles=(0,),
        prepare_by_tile=(prepare,),
        drain_by_tile=(drain,),
    )
    schedule = simulate_direct(window)
    if schedule.cycles != full_cycles:
        raise ValueError(
            f"C0 boundary reconstruction mismatch: schedule={schedule.cycles} measured={full_cycles}"
        )
    serialized_events = [asdict(event) for event in schedule.events]
    digest = hashlib.sha256(
        json.dumps(
            serialized_events, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "candidate": schedule.candidate,
        "cycles": schedule.cycles,
        "measured_full_boundary_cycles": full_cycles,
        "measured_execute_tail_rule_reconstruction_cycles": predicted_full_cycles,
        "prediction_class": "measured_execute_tail_plus_frozen_drain_readout_rules",
        "boundary_match": True,
        "phase_rule": {
            "prepare": "2 cycles from accepted context start to relation fill",
            "fill": "2*accepted_relation_records+1 cycles",
            "execute_end": "measured last accepted term + 2 cycles",
            "compute_drain": "2 cycles",
            "readout": "one scalar accept every 3 cycles with 2-cycle response latency",
        },
        "phase_boundaries_predicted": predicted,
        "phase_durations": {
            "prepare": predicted["FILL_BEGIN"],
            "fill": predicted["EXECUTE_BEGIN"] - predicted["FILL_BEGIN"],
            "execute": predicted["DRAIN_BEGIN"] - predicted["EXECUTE_BEGIN"],
            "compute_drain": predicted["COMPUTE_DONE"] - predicted["DRAIN_BEGIN"],
            "readout": predicted_full_cycles - predicted["COMPUTE_DONE"],
        },
        "prepare_duration": prepare.duration,
        "direct_duration": direct.duration,
        "drain_duration": drain.duration,
        "relative_command_count": len(direct_commands) + len(drain_commands) + 1,
        "event_count": len(schedule.events),
        "event_ledger_sha256": digest,
        "events": serialized_events,
    }


def make_identity(
    kind: str,
    row: dict[str, str],
    *,
    sample: str,
    stage: int,
    block: int,
    input_head: int,
    output_tile: int,
) -> dict[str, Any]:
    window = max(strict_int(row, "window"), 0)
    base: dict[str, Any] = {
        "sample": sample,
        "stage": stage,
        "block": block,
        "window": window,
    }
    if kind == "relation":
        return {
            **base,
            "input_head": input_head,
            "source_id": strict_int(row, "source_id", minimum=0),
        }
    if kind == "weight":
        return {
            **base,
            "input_head": input_head,
            "output_tile": output_tile,
            "lane": strict_int(row, "lane", minimum=0),
            "out": strict_int(row, "out", minimum=0),
        }
    if kind == "final":
        return {
            **base,
            "output_tile": output_tile,
            "source_id": strict_int(row, "source_id", minimum=0),
            "out": strict_int(row, "out", minimum=0),
        }
    raise ValueError(f"unsupported identity kind {kind}")


def build_identity_ledger(
    rows: list[dict[str, str]],
    candidate: str,
    service: IdentityService,
    *,
    sample: str,
    stage: int,
    block: int,
    input_head: int,
    output_tile: int,
) -> tuple[list[Any], dict[str, Any]]:
    selected: list[tuple[str, dict[str, str]]] = []
    if candidate == "direct_online":
        selected.extend(("relation", row) for row in rows_for(rows, candidate, "relation_read"))
        selected.extend(("final", row) for row in rows_for(rows, candidate, "drain_read_response"))
    else:
        selected.extend(("final", row) for row in rows_for(rows, candidate, "serializer_output"))
    selected.extend(("weight", row) for row in rows_for(rows, candidate, "weight_accept"))
    selected.sort(key=lambda item: (strict_int(item[1], "time"), item[0], item[1].get("out", "")))
    transactions = [
        service.transaction(
            kind,
            make_identity(
                kind,
                row,
                sample=sample,
                stage=stage,
                block=block,
                input_head=input_head,
                output_tile=output_tile,
            ),
        )
        for kind, row in selected
    ]
    digests = service.ledger_digests(transactions)
    return transactions, {
        **digests.as_dict(),
        "service_delays": [transaction.as_dict() for transaction in transactions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = parse_trace(args.trace)
    common_values = validate_common_event_values(rows)
    candidates = sorted({row["candidate"] for row in rows})
    if candidates != ["direct_online", "tcfm5_1rw"]:
        raise ValueError(f"unexpected candidate set: {candidates}")

    direct_boundaries = validate_phase_boundaries(rows, "direct_online")
    tcfm5_boundaries = validate_phase_boundaries(rows, "tcfm5_1rw")
    snapshot_state = validate_snapshot_state_contract(
        rows, direct_boundaries, tcfm5_boundaries
    )
    snapshot_events = validate_snapshot_event_ledger(rows, direct_boundaries)
    direct_full = full_boundary_cycles(rows, "direct_online", {"drain_read_response"})
    tcfm5_full = full_boundary_cycles(rows, "tcfm5_1rw", {"serializer_output"})
    relation_fifo = validate_relation_and_fifo(rows)
    direct_banks = validate_bank_commands(rows, "direct_online")
    tcfm5_banks = validate_bank_commands(rows, "tcfm5_1rw")
    direct_numeric = validate_term_to_final_numeric_ledger(rows, "direct_online")
    tcfm5_numeric = validate_term_to_final_numeric_ledger(rows, "tcfm5_1rw")
    direct_rmw = validate_physical_rmw_ledger(rows, "direct_online")
    tcfm5_rmw = validate_physical_rmw_ledger(rows, "tcfm5_1rw")
    backpressure = summarize_backpressure(rows)
    c0 = build_direct_c0_schedule(rows, direct_boundaries, direct_full)

    service = IdentityService(seed=DEFAULT_SEED)
    direct_transactions, direct_ledger = build_identity_ledger(
        rows,
        "direct_online",
        service,
        sample="synthetic_fixture_v4",
        stage=0,
        block=0,
        input_head=0,
        output_tile=0,
    )
    tcfm5_transactions, tcfm5_ledger = build_identity_ledger(
        rows,
        "tcfm5_1rw",
        service,
        sample="synthetic_fixture_v4",
        stage=0,
        block=0,
        input_head=0,
        output_tile=0,
    )
    comparison = service.compare_candidates(direct_transactions, tcfm5_transactions)

    event_counts: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        event_counts[candidate] = dict(
            sorted(Counter(row["event"] for row in rows_for(rows, candidate)).items())
        )

    output = {
        "schema": OUTPUT_SCHEMA,
        "status": "PASS_SYNTHETIC_RTL_CALIBRATION_ONLY",
        "evidence": "[synthetic-RTL]+[rtl校准]",
        "evidence_boundary": "synthetic RTL calibration only; formal/T450 labeling is impossible in this entry point",
        "raw_trace": {
            "path": str(args.trace.resolve()),
            "sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            "row_count": len(rows),
            "event_counts": event_counts,
        },
        "metadata": {
            "sample": "synthetic_fixture_v4",
            "stage": 0,
            "block": 0,
            "input_head": 0,
            "output_tile": 0,
        },
        "phase_boundaries": {
            "direct_online": direct_boundaries,
            "tcfm5_1rw": tcfm5_boundaries,
        },
        "common_event_contract": common_values,
        "snapshot_state_contract": snapshot_state,
        "snapshot_event_contract": snapshot_events,
        "measured_boundaries": {
            "direct_online": {
                "compute_cycles": direct_boundaries["COMPUTE_DONE"] + 1,
                "full_with_readout_cycles": direct_full,
            },
            "tcfm5_1rw": {
                "compute_cycles": tcfm5_boundaries["COMPUTE_DONE"] + 1,
                "full_with_serializer_cycles": tcfm5_full,
            },
        },
        "relation_fifo2": relation_fifo,
        "bank_commands": {
            "direct_online": direct_banks,
            "tcfm5_1rw": tcfm5_banks,
        },
        "numeric_ledgers": {
            "direct_online": direct_numeric,
            "tcfm5_1rw": tcfm5_numeric,
        },
        "physical_rmw_ledgers": {
            "direct_online": direct_rmw,
            "tcfm5_1rw": tcfm5_rmw,
        },
        "backpressure_coverage": backpressure,
        "c0_measured_tail_schedule_reconstruction": c0,
        "identity_service": {
            "schema": IDENTITY_SCHEMA,
            "seed": DEFAULT_SEED,
            "direct_online": direct_ledger,
            "tcfm5_1rw": tcfm5_ledger,
            "candidate_comparison": comparison.as_dict(),
        },
        "formal_adapter_status": "DENY_SEPARATE_HASH_BOUND_ORDERED_MANIFEST_ADAPTER_REQUIRED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS Local5 EREP calibration trace v4 "
        f"rows={len(rows)} direct_full={direct_full} "
        f"tcfm5_full={tcfm5_full} c0_digest={c0['event_ledger_sha256']}"
    )


if __name__ == "__main__":
    main()
