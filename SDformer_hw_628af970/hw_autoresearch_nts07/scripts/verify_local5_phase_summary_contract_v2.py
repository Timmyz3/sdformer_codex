#!/usr/bin/env python3
"""Fail-closed static preflight and reference library for Local5 phase summary v2.

The rolling digests in this module are ordered error-detection checks. They are
not SHA and are not collision-resistant commitments. SHA256 is used separately
when a completed summary file is sealed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "local5_ordered_summary_v2"
CROSS_SUMMARY_SCHEMA = "local5_cross_acc_summary_v2"
TCFM5_SUMMARY_SCHEMA = "local5_tcfm5_summary_v2"
SUMMARY_ORIGINS = {
    SUMMARY_SCHEMA: "RTL_DIRECT",
    CROSS_SUMMARY_SCHEMA: "RTL_LOWER_PORT",
    TCFM5_SUMMARY_SCHEMA: "RTL_LOWER_BANKS",
}
PHASE_SCHEMA = "local5_phase_interval_ledger_v2"
STATE_ROLE_SCHEMA = "local5_phase_state_roles_v2"
STATIC_REPORT_SCHEMA = "local5_phase_summary_static_preflight_v2"
DOMAIN_TAG = "LOCAL5_PHASE_SUMMARY_V2"
SCHEMA_VERSION = 2
FRAME_SERIALIZATION = (
    "domain_u16le_ascii_schema_u16le_resource_u16le_instance_u16le_utf8_"
    "sequence_u64le_cycle_u64le_payload_len_u16le_payload_10xu64le"
)
ORIGIN = "RTL_DIRECT"

DIGEST_NAME = "FNV1A64_AND_DJB2XOR64_V1"
FNV1A64_SEED = 0xCBF29CE484222325
FNV1A64_PRIME = 0x00000100000001B3
DJB2XOR64_SEED = 0x00001505D3C4B2A1
MASK64 = (1 << 64) - 1
PAYLOAD_U64_COUNT = 10
PAYLOAD_LEN = PAYLOAD_U64_COUNT * 8

RESOURCE_CODES = {
    "RELATION_REQ_ACCEPT": 0,
    "RELATION_RSP_ACCEPT": 1,
    "WEIGHT_REQ_ACCEPT": 2,
    "WEIGHT_RSP_ACCEPT": 3,
    "FINAL_ACCEPT": 4,
    "CROSS_ACC_COMMAND": 5,
    "TCFM5_TERM_COMMIT": 6,
}

RESOURCE_FIELDS = {
    "RELATION_REQ_ACCEPT": (
        "tile", "head", "source", "reserved0", "reserved1",
        "reserved2", "reserved3", "reserved4", "reserved5", "reserved6",
    ),
    "RELATION_RSP_ACCEPT": (
        "tile", "head", "source", "reserved0", "reserved1",
        "reserved2", "reserved3", "reserved4", "reserved5", "reserved6",
    ),
    "WEIGHT_REQ_ACCEPT": (
        "tile", "head", "lane", "out", "reserved0", "reserved1",
        "reserved2", "reserved3", "reserved4", "reserved5",
    ),
    "WEIGHT_RSP_ACCEPT": (
        "tile", "head", "lane", "out", "reserved0", "reserved1",
        "reserved2", "reserved3", "reserved4", "reserved5",
    ),
    "FINAL_ACCEPT": (
        "tile", "source", "out", "reserved0", "reserved1",
        "reserved2", "reserved3", "reserved4", "reserved5", "reserved6",
    ),
    "CROSS_ACC_COMMAND": (
        "rw", "addr", "write_data", "reserved0", "reserved1",
        "reserved2", "reserved3", "reserved4", "reserved5", "reserved6",
    ),
    "TCFM5_TERM_COMMIT": (
        "source", "lane", "expected_mask", "actual_mask", "bank_addr0",
        "bank_addr1", "bank_addr2", "bank_addr3", "bank_addr4", "reserved0",
    ),
}

EXPECTED_MODULE = "qfit_local5_tagged_t450_job_engine"
EXPECTED_RTL_SOURCE = "rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
EXPECTED_ENUM_TYPE = "state_t"
EXPECTED_CONFIGURATION = {
    "use_relation_memo": False,
    "vector_result_mode": False,
    "service_mode": "identity_derived",
    "coverage": "direct_baseline_only",
}
EXPECTED_STATE_STATUS = "FROZEN_DIRECT_BASELINE_ENUM_NOT_G0"
ALLOWED_STATE_ROLES = {
    "NONE",
    "HEAD_WEIGHT",
    "HEAD_FRONTEND",
    "HEAD_READOUT",
    "HEAD_RELEASE",
    "HEAD_ERROR",
}

EVIDENCE_LIMIT_BYTES = 512 * 1024 * 1024
IDENTITY_KEYS = ("sample", "stage", "block", "window", "heads")


class ContractError(ValueError):
    """Raised when fail-closed contract validation rejects an input."""


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    return value


def _bounded_unsigned(value: Any, bits: int, label: str) -> int:
    number = _require_int(value, label)
    if not 0 <= number < (1 << bits):
        raise ContractError(f"{label} does not fit u{bits}")
    return number


def unsigned_bit_pattern(value: int, width: int) -> int:
    """Return a signed or unsigned value as its exact width bit pattern."""

    width = _require_int(width, "bit width")
    value = _require_int(value, "bit-pattern value")
    if not 1 <= width <= 64:
        raise ContractError("bit width must be in 1..64")
    minimum = -(1 << (width - 1))
    maximum = (1 << width) - 1
    if not minimum <= value <= maximum:
        raise ContractError(f"value does not fit a signed/unsigned {width}-bit field")
    return value & ((1 << width) - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkloadCounts:
    heads: int
    phase: int
    relation_req: int
    relation_rsp: int
    weight_req: int
    weight_rsp: int
    final: int
    aligned_total: int
    acc32: int
    cross_total: int
    cross_read: int
    cross_write: int


def workload_counts(heads: int) -> WorkloadCounts:
    heads = _require_int(heads, "heads")
    if heads <= 0:
        raise ContractError("heads must be positive")
    h2 = heads * heads
    relation = 450 * h2
    weight = 1024 * h2
    final = 14_400 * heads
    cross_half = 14_400 * h2
    return WorkloadCounts(
        heads=heads,
        phase=1 + 2 * heads + 5 * h2,
        relation_req=relation,
        relation_rsp=relation,
        weight_req=weight,
        weight_rsp=weight,
        final=final,
        aligned_total=2 * relation + 2 * weight + final,
        acc32=final,
        cross_total=2 * cross_half,
        cross_read=cross_half,
        cross_write=cross_half,
    )


@dataclass(frozen=True)
class SummaryFrame:
    domain_tag: str
    schema_version: int
    resource_code: int
    instance_path: str
    sequence: int
    cycle: int
    payload: tuple[int, ...]


def encode_u64_payload(values: Iterable[int]) -> bytes:
    frozen_values = tuple(values)
    if len(frozen_values) != PAYLOAD_U64_COUNT:
        raise ContractError("payload must contain exactly ten u64 fields")
    encoded = bytearray()
    for index, value in enumerate(frozen_values):
        encoded.extend(struct.pack("<Q", _bounded_unsigned(value, 64, f"payload[{index}]")))
    return bytes(encoded)


def encode_summary_frame(
    domain_tag: str,
    schema_version: int,
    resource_code: int,
    instance_path: str,
    sequence: int,
    cycle: int,
    payload: Iterable[int],
) -> bytes:
    try:
        domain = domain_tag.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ContractError("domain tag must be ASCII") from exc
    try:
        instance = instance_path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError("instance path is not valid UTF-8") from exc
    if len(domain) > 0xFFFF or len(instance) > 0xFFFF:
        raise ContractError("domain or instance path exceeds u16 length")
    payload_bytes = encode_u64_payload(payload)
    if len(payload_bytes) != PAYLOAD_LEN:
        raise ContractError("payload length is not the frozen 80 bytes")
    return b"".join(
        (
            struct.pack("<H", len(domain)),
            domain,
            struct.pack(
                "<HHH",
                _bounded_unsigned(schema_version, 16, "schema version"),
                _bounded_unsigned(resource_code, 16, "resource code"),
                len(instance),
            ),
            instance,
            struct.pack(
                "<QQH",
                _bounded_unsigned(sequence, 64, "sequence"),
                _bounded_unsigned(cycle, 64, "cycle"),
                len(payload_bytes),
            ),
            payload_bytes,
        )
    )


def parse_summary_frame(data: bytes, expected_payload_fields: int | None = PAYLOAD_U64_COUNT) -> SummaryFrame:
    if not isinstance(data, bytes):
        raise ContractError("frame must be bytes")
    offset = 0

    def take(size: int, label: str) -> bytes:
        nonlocal offset
        if offset + size > len(data):
            raise ContractError(f"truncated frame at {label}")
        value = data[offset : offset + size]
        offset += size
        return value

    domain_len = struct.unpack("<H", take(2, "domain length"))[0]
    try:
        domain_tag = take(domain_len, "domain").decode("ascii")
    except UnicodeDecodeError as exc:
        raise ContractError("frame domain is not ASCII") from exc
    schema_version, resource_code, instance_len = struct.unpack(
        "<HHH", take(6, "schema/resource/instance lengths")
    )
    instance_bytes = take(instance_len, "instance")
    try:
        instance_path = instance_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("frame instance is not UTF-8") from exc
    if instance_path.encode("utf-8") != instance_bytes:
        raise ContractError("frame instance is not canonical UTF-8")
    sequence, cycle, payload_len = struct.unpack("<QQH", take(18, "event header"))
    if payload_len != PAYLOAD_LEN:
        raise ContractError("payload length is not the frozen 80 bytes")
    payload_bytes = take(payload_len, "payload")
    if offset != len(data):
        raise ContractError("trailing bytes after frame")
    field_count = payload_len // 8
    if expected_payload_fields is not None and field_count != expected_payload_fields:
        raise ContractError("payload field count does not match the frozen projection")
    payload = tuple(
        struct.unpack_from("<Q", payload_bytes, index * 8)[0]
        for index in range(field_count)
    )
    return SummaryFrame(
        domain_tag,
        schema_version,
        resource_code,
        instance_path,
        sequence,
        cycle,
        payload,
    )


def rolling64_fnv1a(data: bytes, seed: int = FNV1A64_SEED) -> int:
    state = _bounded_unsigned(seed, 64, "FNV seed")
    for byte in data:
        state = ((state ^ byte) * FNV1A64_PRIME) & MASK64
    return state


def rolling64_djb2xor(data: bytes, seed: int = DJB2XOR64_SEED) -> int:
    state = _bounded_unsigned(seed, 64, "DJB2XOR seed")
    for byte in data:
        state = ((((state << 5) & MASK64) + state) ^ byte) & MASK64
    return state


@dataclass
class Rolling64Pair:
    first: int = FNV1A64_SEED
    second: int = DJB2XOR64_SEED

    def update(self, data: bytes) -> None:
        self.first = rolling64_fnv1a(data, self.first)
        self.second = rolling64_djb2xor(data, self.second)

    def values(self) -> tuple[int, int]:
        return self.first, self.second


@dataclass(frozen=True)
class SummaryEvent:
    sequence: int
    cycle: int
    payload: tuple[int, ...]


@dataclass(frozen=True)
class SummaryResource:
    name: str
    code: int
    instance_path: str
    field_names: tuple[str, ...]
    count: int
    digest0: int
    digest1: int
    first_anchor: tuple[int, ...] | None
    last_anchor: tuple[int, ...] | None


@dataclass(frozen=True)
class CrossProtocolLedger:
    count: int
    read_count: int
    write_count: int
    digest0: int | None
    digest1: int | None


@dataclass(frozen=True)
class Tcfm5TermLedger:
    term_count: int
    update_count: int
    mismatch_count: int


@dataclass(frozen=True)
class OrderedSummary:
    path: Path
    schema: str
    origin: str
    domain_tag: str
    schema_version: int
    digest_name: str
    resource_order: tuple[str, ...]
    resources: Mapping[str, SummaryResource]
    end_cycle: int
    file_sha256: str
    monitor_instance: str | None = None
    observer_instance: str | None = None
    target_instance: str | None = None
    cross_protocol_ledger: CrossProtocolLedger | None = None
    tcfm5_term_ledger: Tcfm5TermLedger | None = None


def summarize_events(
    resource_name: str,
    instance_path: str,
    events: Iterable[SummaryEvent],
    *,
    field_names: Sequence[str],
    domain_tag: str = DOMAIN_TAG,
    schema_version: int = SCHEMA_VERSION,
) -> SummaryResource:
    if resource_name not in RESOURCE_CODES:
        raise ContractError(f"unknown resource {resource_name}")
    names = tuple(field_names)
    if names != RESOURCE_FIELDS[resource_name]:
        raise ContractError("summary field names are not the frozen ten-field projection")
    rolling = Rolling64Pair()
    first_anchor: tuple[int, ...] | None = None
    last_anchor: tuple[int, ...] | None = None
    previous_cycle = -1
    count = 0
    for event in events:
        if event.sequence != count:
            raise ContractError("summary event sequence is missing, duplicated, or reordered")
        if event.cycle < previous_cycle:
            raise ContractError("summary event cycles are reordered")
        if len(event.payload) != len(names):
            raise ContractError("summary payload field count is not frozen")
        frame = encode_summary_frame(
            domain_tag,
            schema_version,
            RESOURCE_CODES[resource_name],
            instance_path,
            event.sequence,
            event.cycle,
            event.payload,
        )
        rolling.update(frame)
        anchor = (event.sequence, event.cycle, *event.payload)
        if first_anchor is None:
            first_anchor = anchor
        last_anchor = anchor
        previous_cycle = event.cycle
        count += 1
    digest0, digest1 = rolling.values()
    return SummaryResource(
        resource_name,
        RESOURCE_CODES[resource_name],
        instance_path,
        names,
        count,
        digest0,
        digest1,
        first_anchor,
        last_anchor,
    )


def _parse_decimal(text: str, label: str, *, allow_negative: bool = False) -> int:
    if not re.fullmatch(r"-?[0-9]+" if allow_negative else r"[0-9]+", text):
        raise ContractError(f"{label} is not canonical decimal")
    value = int(text, 10)
    if not allow_negative and value < 0:
        raise ContractError(f"{label} is negative")
    return value


def _parse_digest(text: str, label: str) -> int:
    if not re.fullmatch(r"[0-9a-fA-F]{16}", text):
        raise ContractError(f"{label} is not a 16-digit rolling64 value")
    return int(text, 16)


def parse_ordered_summary(path: Path) -> OrderedSummary:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or any(field != field.strip() for field in row):
                raise ContractError(f"summary line {line_number} is blank or non-canonical")
            rows.append(row)
    if not rows:
        raise ContractError("summary is empty")

    singleton: dict[str, list[str]] = {}
    resource_rows: dict[str, tuple[int, str]] = {}
    field_rows: dict[str, tuple[str, ...]] = {}
    anchors: dict[str, dict[str, tuple[int, ...]]] = {}
    summaries: dict[str, tuple[int, int, int]] = {}
    resource_order: list[str] = []
    field_order: list[str] = []
    summary_order: list[str] = []
    cross_protocol_ledger: CrossProtocolLedger | None = None
    tcfm5_term_ledger: Tcfm5TermLedger | None = None
    ended = False
    known_singletons = {
        "SCHEMA",
        "ORIGIN",
        "MONITOR_INSTANCE",
        "OBSERVER_INSTANCE",
        "TARGET_INSTANCE",
        "H",
        "DIGEST",
        "BYTE_ORDER",
        "SERIALIZATION",
        "PAYLOAD_U64_COUNT",
        "RESOURCE_CODE",
        "SAME_CYCLE_ORDER",
        "EMPTY_STREAM",
    }
    for line_number, row in enumerate(rows, start=1):
        kind = row[0]
        if ended:
            raise ContractError("summary contains records after END")
        if kind in known_singletons:
            if kind in singleton or len(row) < 2:
                raise ContractError(f"summary {kind} record is missing or duplicated")
            singleton[kind] = row[1:]
        elif kind == "R":
            if len(row) not in (3, 4):
                raise ContractError("summary R record has an invalid width")
            name = row[1]
            if name not in RESOURCE_CODES or name in resource_rows:
                raise ContractError("summary resource is unknown or duplicated")
            if len(row) == 3:
                code, instance = RESOURCE_CODES[name], row[2]
            else:
                code = _parse_decimal(row[2], f"{name} resource code")
                instance = row[3]
            if code != RESOURCE_CODES[name] or not instance:
                raise ContractError("summary resource code or instance is invalid")
            instance.encode("utf-8")
            resource_rows[name] = (code, instance)
            resource_order.append(name)
        elif kind == "F":
            if len(row) < 3 or row[1] in field_rows:
                raise ContractError("summary F record is missing fields or duplicated")
            name = row[1]
            names = tuple(row[2:])
            if name not in RESOURCE_CODES:
                raise ContractError("summary F record names an unknown resource")
            if any(not re.fullmatch(r"[a-z][a-z0-9_]*", item) for item in names):
                raise ContractError("summary field name is not canonical")
            if len(names) != len(list(dict.fromkeys(names))):
                raise ContractError("summary field names are duplicated")
            field_rows[name] = names
            field_order.append(name)
        elif kind == "A":
            if len(row) < 5 or row[2] not in ("FIRST", "LAST"):
                raise ContractError("summary anchor record is invalid")
            name, anchor_kind = row[1], row[2]
            if name not in RESOURCE_CODES or anchor_kind in anchors.setdefault(name, {}):
                raise ContractError("summary anchor resource is unknown or duplicated")
            values = tuple(_parse_decimal(item, f"{name} anchor") for item in row[3:])
            anchors[name][anchor_kind] = values
        elif kind == "S":
            if len(row) != 5 or row[1] in summaries:
                raise ContractError("summary S record has an invalid width or is duplicated")
            name = row[1]
            if name not in RESOURCE_CODES:
                raise ContractError("summary S record names an unknown resource")
            summaries[name] = (
                _parse_decimal(row[2], f"{name} count"),
                _parse_digest(row[3], f"{name} digest0"),
                _parse_digest(row[4], f"{name} digest1"),
            )
            summary_order.append(name)
        elif kind == "P":
            if len(row) < 3:
                raise ContractError(f"summary {kind} ledger declaration is invalid")
        elif kind == "L":
            if len(row) < 2:
                raise ContractError("summary L ledger record is invalid")
            if row[1] == "CROSS_ACC_PROTOCOL_LEDGER":
                if cross_protocol_ledger is not None or len(row) not in (5, 7):
                    raise ContractError("cross protocol ledger is duplicated or has invalid width")
                values = [_parse_decimal(item, "cross protocol ledger") for item in row[2:5]]
                digest0: int | None = None
                digest1: int | None = None
                if len(row) == 7:
                    digest0 = _parse_digest(row[5], "cross protocol digest0")
                    digest1 = _parse_digest(row[6], "cross protocol digest1")
                cross_protocol_ledger = CrossProtocolLedger(
                    values[0], values[1], values[2], digest0, digest1
                )
            elif row[1] == "TCFM5_TERM_LEDGER":
                if tcfm5_term_ledger is not None or len(row) != 5:
                    raise ContractError("TCFM5 term ledger is duplicated or has invalid width")
                values = [_parse_decimal(item, "TCFM5 term ledger") for item in row[2:5]]
                tcfm5_term_ledger = Tcfm5TermLedger(*values)
            else:
                raise ContractError("summary L record names an unknown ledger")
        elif kind == "END":
            if len(row) != 3:
                raise ContractError("summary END record is invalid")
            singleton["END"] = [row[1], row[2]]
            ended = True
        else:
            raise ContractError(f"summary line {line_number} has unknown record {kind}")

    required = {"SCHEMA", "ORIGIN", "DIGEST", "PAYLOAD_U64_COUNT", "END"}
    missing = sorted(required.difference(singleton))
    if missing:
        raise ContractError("summary metadata is incomplete: " + ",".join(missing))
    if len(singleton["SCHEMA"]) != 1 or singleton["SCHEMA"][0] not in SUMMARY_ORIGINS:
        raise ContractError("summary schema is not admitted")
    schema = singleton["SCHEMA"][0]
    origin = SUMMARY_ORIGINS[schema]
    if singleton["ORIGIN"] != [origin] or singleton["END"][1] != origin:
        raise ContractError("summary schema or origin is not admitted")
    if singleton["PAYLOAD_U64_COUNT"] != [str(PAYLOAD_U64_COUNT)]:
        raise ContractError("summary payload is not exactly ten u64 fields")
    digest_row = singleton["DIGEST"]
    expected_digest_width = 4 if schema == SUMMARY_SCHEMA else 3
    if len(digest_row) != expected_digest_width or digest_row[0] != DIGEST_NAME:
        raise ContractError("summary does not name the frozen dual rolling64 digest")
    if _parse_digest(digest_row[1], "digest seed0") != FNV1A64_SEED or _parse_digest(
        digest_row[2], "digest seed1"
    ) != DJB2XOR64_SEED:
        raise ContractError("summary rolling64 seeds are not frozen")
    if schema == SUMMARY_SCHEMA and _parse_digest(digest_row[3], "FNV prime") != FNV1A64_PRIME:
        raise ContractError("summary FNV prime is not frozen")
    if not ended:
        raise ContractError("summary END record is missing")
    monitor_instance: str | None = None
    observer_instance: str | None = None
    target_instance: str | None = None
    if schema == SUMMARY_SCHEMA:
        main_required = {
            "MONITOR_INSTANCE",
            "H",
            "BYTE_ORDER",
            "SERIALIZATION",
            "SAME_CYCLE_ORDER",
            "EMPTY_STREAM",
        }
        missing_main = sorted(main_required.difference(singleton))
        if missing_main:
            raise ContractError("main summary metadata is incomplete: " + ",".join(missing_main))
        if singleton["BYTE_ORDER"] != ["LITTLE_ENDIAN"] or singleton["SERIALIZATION"] != [FRAME_SERIALIZATION]:
            raise ContractError("main summary byte/frame serialization is not frozen")
        if singleton["EMPTY_STREAM"] != ["raw_seed_without_event_frame"]:
            raise ContractError("main summary empty-stream state is not the raw seed")
        if tuple(resource_order) != tuple(RESOURCE_CODES):
            raise ContractError("main summary does not contain all seven resources in code order")
        if field_order != resource_order or summary_order != resource_order:
            raise ContractError("main summary R/F/S resource order or cardinality is inconsistent")
        monitor_instance = singleton["MONITOR_INSTANCE"][0]
        if cross_protocol_ledger is None or cross_protocol_ledger.digest0 is None:
            raise ContractError("main summary lacks the cross protocol digest ledger")
        if tcfm5_term_ledger is None:
            raise ContractError("main summary lacks the TCFM5 term ledger")
    else:
        lower_required = {"OBSERVER_INSTANCE", "TARGET_INSTANCE", "RESOURCE_CODE"}
        missing_lower = sorted(lower_required.difference(singleton))
        if missing_lower:
            raise ContractError("lower summary metadata is incomplete: " + ",".join(missing_lower))
        observer_instance = singleton["OBSERVER_INSTANCE"][0]
        target_instance = singleton["TARGET_INSTANCE"][0]
        resource_code_row = singleton["RESOURCE_CODE"]
        if len(resource_code_row) != 2 or resource_code_row[0] not in RESOURCE_CODES:
            raise ContractError("lower summary resource code declaration is invalid")
        resource_name = resource_code_row[0]
        resource_code = _parse_decimal(resource_code_row[1], "lower resource code")
        expected_resource = "CROSS_ACC_COMMAND" if schema == CROSS_SUMMARY_SCHEMA else "TCFM5_TERM_COMMIT"
        if resource_name != expected_resource or resource_code != RESOURCE_CODES[resource_name]:
            raise ContractError("lower summary schema/resource code binding is inconsistent")
        if resource_order or field_order:
            raise ContractError("lower summary must use RESOURCE_CODE instead of main R/F records")
        resource_order.append(resource_name)
        resource_rows[resource_name] = (resource_code, target_instance)
        field_rows[resource_name] = RESOURCE_FIELDS[resource_name]
        if summary_order != resource_order:
            raise ContractError("lower summary S resource order or cardinality is inconsistent")
        if schema == CROSS_SUMMARY_SCHEMA:
            if cross_protocol_ledger is None or cross_protocol_ledger.digest0 is not None:
                raise ContractError("cross lower summary ledger is missing or has main-only digests")
            if tcfm5_term_ledger is not None:
                raise ContractError("cross lower summary contains a TCFM5 ledger")
        else:
            if tcfm5_term_ledger is None or cross_protocol_ledger is not None:
                raise ContractError("TCFM5 lower summary ledger set is invalid")

    resources: dict[str, SummaryResource] = {}
    for name in resource_order:
        code, instance = resource_rows[name]
        fields = field_rows[name]
        if fields != RESOURCE_FIELDS[name]:
            raise ContractError("summary F record differs from the frozen ten-field projection")
        count, digest0, digest1 = summaries[name]
        resource_anchors = anchors.get(name, {})
        first = resource_anchors.get("FIRST")
        last = resource_anchors.get("LAST")
        if count == 0:
            if first is not None or last is not None:
                raise ContractError("empty summary resource must not have anchors")
        else:
            if first is None or last is None:
                raise ContractError("non-empty summary resource lacks first/last anchors")
            expected_width = 2 + len(fields)
            if len(first) != expected_width or len(last) != expected_width:
                raise ContractError("summary anchor width does not match its payload schema")
            if first[0] != 0 or last[0] != count - 1:
                raise ContractError("summary first/last anchor sequences do not match count")
        resources[name] = SummaryResource(
            name,
            code,
            instance,
            fields,
            count,
            digest0,
            digest1,
            first,
            last,
        )
    if cross_protocol_ledger is not None:
        cross_resource = resources.get("CROSS_ACC_COMMAND")
        if cross_resource is None or cross_protocol_ledger.count != cross_resource.count:
            raise ContractError("cross protocol ledger count differs from summary count")
        if cross_protocol_ledger.read_count + cross_protocol_ledger.write_count != cross_protocol_ledger.count:
            raise ContractError("cross protocol read/write counts do not sum to total")
    if tcfm5_term_ledger is not None:
        tcfm_resource = resources.get("TCFM5_TERM_COMMIT")
        if tcfm_resource is None or tcfm5_term_ledger.term_count != tcfm_resource.count:
            raise ContractError("TCFM5 ledger term count differs from summary count")
        if not 0 <= tcfm5_term_ledger.update_count <= 5 * tcfm5_term_ledger.term_count:
            raise ContractError("TCFM5 ledger update count is outside 0..5*term_count")
        if tcfm5_term_ledger.mismatch_count != 0:
            raise ContractError("TCFM5 topology mismatch count is nonzero")
    return OrderedSummary(
        path=path,
        schema=schema,
        origin=origin,
        domain_tag=DOMAIN_TAG,
        schema_version=SCHEMA_VERSION,
        digest_name=DIGEST_NAME,
        resource_order=tuple(resource_order),
        resources=resources,
        end_cycle=_parse_decimal(singleton["END"][0], "summary end cycle"),
        file_sha256=sha256_file(path),
        monitor_instance=monitor_instance,
        observer_instance=observer_instance,
        target_instance=target_instance,
        cross_protocol_ledger=cross_protocol_ledger,
        tcfm5_term_ledger=tcfm5_term_ledger,
    )


def verify_summary_resource(
    summary: OrderedSummary,
    resource_name: str,
    events: Iterable[SummaryEvent],
    *,
    expected_instance_path: str | None = None,
) -> SummaryResource:
    if resource_name not in summary.resources:
        raise ContractError(f"summary lacks resource {resource_name}")
    observed = summary.resources[resource_name]
    if expected_instance_path is not None and observed.instance_path != expected_instance_path:
        raise ContractError("summary resource instance identity was rebound")
    expected = summarize_events(
        resource_name,
        observed.instance_path,
        events,
        field_names=observed.field_names,
        domain_tag=summary.domain_tag,
        schema_version=summary.schema_version,
    )
    if expected != observed:
        raise ContractError(f"summary count/digest/anchors mismatch for {resource_name}")
    return observed


ALIGNED_TRACE_RESOURCE = {
    "relation_accept": "RELATION_REQ_ACCEPT",
    "relation_response_accept": "RELATION_RSP_ACCEPT",
    "weight_accept": "WEIGHT_REQ_ACCEPT",
    "weight_response_accept": "WEIGHT_RSP_ACCEPT",
    "final_accept": "FINAL_ACCEPT",
}
ALIGNED_RESOURCES = tuple(ALIGNED_TRACE_RESOURCE.values())
IDENTITY_TRACE_COLUMNS = (
    "cycle",
    "event",
    "tile",
    "head",
    "source",
    "lane",
    "out",
    "delay",
    "index",
    "origin",
    "payload",
)


@dataclass
class _StreamingSummaryAccumulator:
    resource_name: str
    instance_path: str
    rolling: Rolling64Pair
    count: int = 0
    previous_cycle: int = -1
    first_anchor: tuple[int, ...] | None = None
    last_anchor: tuple[int, ...] | None = None

    def add(self, cycle: int, payload: tuple[int, ...]) -> None:
        if cycle < self.previous_cycle:
            raise ContractError(f"identity trace reordered {self.resource_name} cycles")
        if len(payload) != PAYLOAD_U64_COUNT:
            raise ContractError("identity trace projection does not contain ten u64 fields")
        frame = encode_summary_frame(
            DOMAIN_TAG,
            SCHEMA_VERSION,
            RESOURCE_CODES[self.resource_name],
            self.instance_path,
            self.count,
            cycle,
            payload,
        )
        self.rolling.update(frame)
        anchor = (self.count, cycle, *payload)
        if self.first_anchor is None:
            self.first_anchor = anchor
        self.last_anchor = anchor
        self.previous_cycle = cycle
        self.count += 1

    def finish(self) -> SummaryResource:
        digest0, digest1 = self.rolling.values()
        return SummaryResource(
            self.resource_name,
            RESOURCE_CODES[self.resource_name],
            self.instance_path,
            RESOURCE_FIELDS[self.resource_name],
            self.count,
            digest0,
            digest1,
            self.first_anchor,
            self.last_anchor,
        )


@dataclass(frozen=True)
class IdentityTraceAlignedAudit:
    path: str
    rows_read: int
    resources: Mapping[str, SummaryResource]


def _trace_int(row: Mapping[str, str], name: str, line_number: int) -> int:
    value = row.get(name)
    if value is None or not re.fullmatch(r"-?[0-9]+", value):
        raise ContractError(f"identity trace line {line_number} field {name} is invalid")
    return int(value, 10)


def stream_identity_trace_aligned_resources(
    path: Path,
    resource_instances: Mapping[str, str],
    *,
    heads: int | None = None,
) -> IdentityTraceAlignedAudit:
    if tuple(resource_instances.keys()) != ALIGNED_RESOURCES:
        raise ContractError("identity trace resource instance map is not the frozen aligned order")
    if any(not isinstance(value, str) or not value for value in resource_instances.values()):
        raise ContractError("identity trace resource instance map contains an invalid path")
    accumulators = {
        name: _StreamingSummaryAccumulator(name, resource_instances[name], Rolling64Pair())
        for name in ALIGNED_RESOURCES
    }
    previous_cycle = -1
    active_tile: int | None = None
    rows_read = 0
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != IDENTITY_TRACE_COLUMNS:
            raise ContractError("identity trace columns are not the frozen v1 trace columns")
        for line_number, row in enumerate(reader, start=2):
            rows_read += 1
            if None in row:
                raise ContractError(f"identity trace line {line_number} has extra columns")
            cycle = _trace_int(row, "cycle", line_number)
            if cycle < 0 or cycle < previous_cycle:
                raise ContractError("identity trace cycles are negative or reordered")
            previous_cycle = cycle
            event = row["event"]
            if event == "tile_start":
                tile = _trace_int(row, "tile", line_number)
                if tile < 0 or active_tile is not None:
                    raise ContractError("identity trace tile_start overlaps or has invalid identity")
                active_tile = tile
            if event in ALIGNED_TRACE_RESOURCE:
                if row["origin"] != "rtl_handshake":
                    raise ContractError("aligned identity trace event origin is not rtl_handshake")
                tile = _trace_int(row, "tile", line_number)
                head = _trace_int(row, "head", line_number)
                source = _trace_int(row, "source", line_number)
                lane = _trace_int(row, "lane", line_number)
                out = _trace_int(row, "out", line_number)
                resource = ALIGNED_TRACE_RESOURCE[event]
                if event == "relation_response_accept":
                    if active_tile is None:
                        raise ContractError("relation response has no active tile identity")
                    if tile not in (-1, active_tile):
                        raise ContractError("relation response tile differs from active tile")
                    tile = active_tile
                if event in ("relation_accept", "relation_response_accept"):
                    if tile < 0 or head < 0 or source < 0:
                        raise ContractError("relation accepted identity is incomplete")
                    payload = (tile, head, source, 0, 0, 0, 0, 0, 0, 0)
                elif event in ("weight_accept", "weight_response_accept"):
                    if tile < 0 or head < 0 or lane < 0 or out < 0:
                        raise ContractError("weight accepted identity is incomplete")
                    payload = (tile, head, lane, out, 0, 0, 0, 0, 0, 0)
                else:
                    if tile < 0 or source < 0 or out < 0:
                        raise ContractError("final accepted identity is incomplete")
                    payload = (tile, source, out, 0, 0, 0, 0, 0, 0, 0)
                accumulators[resource].add(cycle, payload)
            if event == "tile_done":
                tile = _trace_int(row, "tile", line_number)
                if active_tile is None or tile != active_tile:
                    raise ContractError("identity trace tile_done does not close the active tile")
                active_tile = None
    if active_tile is not None:
        raise ContractError("identity trace ends with an open tile identity")
    resources = {name: accumulators[name].finish() for name in ALIGNED_RESOURCES}
    if heads is not None:
        expected = workload_counts(heads)
        expected_counts = {
            "RELATION_REQ_ACCEPT": expected.relation_req,
            "RELATION_RSP_ACCEPT": expected.relation_rsp,
            "WEIGHT_REQ_ACCEPT": expected.weight_req,
            "WEIGHT_RSP_ACCEPT": expected.weight_rsp,
            "FINAL_ACCEPT": expected.final,
        }
        for name in ALIGNED_RESOURCES:
            if resources[name].count != expected_counts[name]:
                raise ContractError(f"identity trace {name} count differs from the H closed form")
    return IdentityTraceAlignedAudit(str(path), rows_read, resources)


def verify_main_summary_against_identity_trace(
    summary: OrderedSummary,
    identity_trace_path: Path,
    *,
    heads: int,
) -> IdentityTraceAlignedAudit:
    if summary.schema != SUMMARY_SCHEMA:
        raise ContractError("identity trace alignment requires the main ordered summary")
    resource_instances = {
        name: summary.resources[name].instance_path
        for name in ALIGNED_RESOURCES
        if name in summary.resources
    }
    if tuple(resource_instances) != ALIGNED_RESOURCES:
        raise ContractError("main summary lacks the five aligned accepted resources")
    audit = stream_identity_trace_aligned_resources(
        identity_trace_path,
        resource_instances,
        heads=heads,
    )
    for name in ALIGNED_RESOURCES:
        if audit.resources[name] != summary.resources[name]:
            raise ContractError(f"main summary differs from streamed identity trace for {name}")
    return audit


def compare_summary_resources(
    primary: OrderedSummary,
    secondary: OrderedSummary,
    resource_names: Sequence[str],
) -> None:
    for name in resource_names:
        if name not in primary.resources or name not in secondary.resources:
            raise ContractError(f"summary pair lacks resource {name}")
        if primary.resources[name] != secondary.resources[name]:
            raise ContractError(f"summary pair differs for common projection {name}")


def validate_observer_summary_binding(
    summary: OrderedSummary,
    *,
    expected_schema: str,
    expected_target_instance: str | None = None,
) -> None:
    if expected_schema not in (CROSS_SUMMARY_SCHEMA, TCFM5_SUMMARY_SCHEMA):
        raise ContractError("expected observer schema is not a lower summary schema")
    if summary.schema != expected_schema:
        raise ContractError("observer summary schema differs from the expected resource")
    observer = summary.observer_instance
    target = summary.target_instance
    if not observer or not target or "." not in observer:
        raise ContractError("observer/target instance identity is incomplete")
    observer_parent = observer.rsplit(".", 1)[0]
    if observer_parent != target:
        raise ContractError("observer parent does not equal TARGET_INSTANCE")
    if expected_target_instance is not None and target != expected_target_instance:
        raise ContractError("observer TARGET_INSTANCE differs from the run identity")
    only_resource = summary.resources[summary.resource_order[0]]
    if only_resource.instance_path != target:
        raise ContractError("observer frame instance is not TARGET_INSTANCE")


def parse_single_observer_summary_glob(
    pattern: str | Path,
    *,
    expected_schema: str,
    expected_target_instance: str | None = None,
) -> OrderedSummary:
    matches = sorted(Path(item) for item in glob.glob(str(pattern)))
    if len(matches) != 1:
        raise ContractError(
            f"observer summary glob cardinality must be exactly one, observed {len(matches)}"
        )
    path = matches[0]
    if path.is_symlink() or not path.is_file():
        raise ContractError("observer summary glob resolved to a symlink or non-file")
    summary = parse_ordered_summary(path)
    validate_observer_summary_binding(
        summary,
        expected_schema=expected_schema,
        expected_target_instance=expected_target_instance,
    )
    return summary


@dataclass(frozen=True)
class PhaseIdentity:
    stage: int
    block: int
    window: int


@dataclass(frozen=True)
class PhaseInterval:
    sequence: int
    stage: int
    block: int
    window: int
    tile: int
    head: int
    role: str
    start_cycle: int
    end_cycle: int
    duration: int
    origin: str


@dataclass(frozen=True)
class PhaseLedger:
    path: Path
    heads: int
    intervals: tuple[PhaseInterval, ...]
    end_cycle: int


def parse_phase_interval_ledger(path: Path) -> PhaseLedger:
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    expected_columns = [
        "COLUMNS",
        "record",
        "sequence",
        "stage",
        "block",
        "window",
        "tile",
        "head",
        "role",
        "start_cycle",
        "end_cycle",
        "duration",
        "origin",
    ]
    if len(rows) < 5:
        raise ContractError("phase ledger is incomplete")
    if rows[0] != ["SCHEMA", PHASE_SCHEMA] or rows[1] != ["ORIGIN", ORIGIN]:
        raise ContractError("phase ledger schema or origin is not admitted")
    if len(rows[2]) != 2 or rows[2][0] != "H":
        raise ContractError("phase ledger H record is missing")
    heads = _parse_decimal(rows[2][1], "phase ledger H")
    if heads <= 0 or rows[3] != expected_columns:
        raise ContractError("phase ledger H or columns are invalid")
    intervals: list[PhaseInterval] = []
    end_cycle: int | None = None
    for row in rows[4:]:
        if end_cycle is not None:
            raise ContractError("phase ledger contains records after END")
        if row and row[0] == "P":
            if len(row) != 12:
                raise ContractError("phase interval width is invalid")
            values = [
                _parse_decimal(row[index], f"phase field {index}", allow_negative=index in (5, 6))
                for index in (1, 2, 3, 4, 5, 6, 8, 9, 10)
            ]
            role = row[7]
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", role) or row[11] != ORIGIN:
                raise ContractError("phase role or origin is invalid")
            interval = PhaseInterval(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                role,
                values[6],
                values[7],
                values[8],
                row[11],
            )
            if interval.sequence != len(intervals):
                raise ContractError("phase sequence is missing, duplicated, or reordered")
            if interval.start_cycle < 0 or interval.end_cycle < interval.start_cycle:
                raise ContractError("phase interval start/end is invalid")
            if interval.duration != interval.end_cycle - interval.start_cycle + 1:
                raise ContractError("phase interval duration is inconsistent")
            intervals.append(interval)
        elif row and row[0] == "END":
            if len(row) != 4 or row[3] != ORIGIN:
                raise ContractError("phase ledger END record is invalid")
            end_cycle = _parse_decimal(row[1], "phase end cycle")
            declared_count = _parse_decimal(row[2], "phase declared count")
            if declared_count != len(intervals):
                raise ContractError("phase END count does not match records")
        else:
            raise ContractError("phase ledger contains an unknown record")
    if end_cycle is None:
        raise ContractError("phase ledger END record is missing")
    if intervals and end_cycle < max(interval.end_cycle for interval in intervals):
        raise ContractError("phase ledger END cycle precedes an interval")
    return PhaseLedger(path, heads, tuple(intervals), end_cycle)


def head_phase_roles_from_state_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    states = contract.get("states")
    if not isinstance(states, list):
        raise ContractError("state contract lacks states")
    roles: list[str] = []
    previous: str | None = None
    for row in states:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str):
            raise ContractError("state role row is invalid")
        role = row["role"]
        if role in ("NONE", "HEAD_ERROR"):
            previous = role
            continue
        if not role.startswith("HEAD_"):
            raise ContractError("state contract contains a non-head phase role")
        if role != previous:
            if role in roles:
                raise ContractError("state phase role is split into non-contiguous enum ranges")
            roles.append(role)
        previous = role
    if len(roles) != 4:
        raise ContractError("direct baseline must derive exactly four ordered head phase roles")
    return tuple(roles)


def validate_phase_interval_ledger(
    ledger: PhaseLedger,
    expected_identity: PhaseIdentity,
    head_roles: Sequence[str],
) -> None:
    if ledger.heads <= 0 or len(head_roles) != 4 or len(list(dict.fromkeys(head_roles))) != 4:
        raise ContractError("phase ledger head role contract is invalid")
    expected_count = workload_counts(ledger.heads).phase
    if len(ledger.intervals) != expected_count:
        raise ContractError("phase ledger cardinality does not match H")
    index = 0
    previous_tile_tx: PhaseInterval | None = None
    tile_transactions: list[PhaseInterval] = []
    for tile in range(ledger.heads):
        previous_head_tx: PhaseInterval | None = None
        head_transactions: list[PhaseInterval] = []
        for head in range(ledger.heads):
            role_intervals = list(ledger.intervals[index : index + len(head_roles)])
            index += len(head_roles)
            if [item.role for item in role_intervals] != list(head_roles):
                raise ContractError("phase head roles are reordered, duplicated, or missing")
            for item in role_intervals:
                if (item.stage, item.block, item.window) != (
                    expected_identity.stage,
                    expected_identity.block,
                    expected_identity.window,
                ):
                    raise ContractError("phase interval identity is rebound")
                if (item.tile, item.head) != (tile, head):
                    raise ContractError("phase interval tile/head identity is out of order")
            for left, right in zip(role_intervals, role_intervals[1:]):
                if right.start_cycle != left.end_cycle + 1:
                    raise ContractError("adjacent head phase intervals are not contiguous")
            head_tx = ledger.intervals[index]
            index += 1
            if head_tx.role != "HEAD_TRANSACTION" or (head_tx.tile, head_tx.head) != (tile, head):
                raise ContractError("head transaction interval is out of order")
            if (head_tx.stage, head_tx.block, head_tx.window) != (
                expected_identity.stage,
                expected_identity.block,
                expected_identity.window,
            ):
                raise ContractError("head transaction identity is rebound")
            if head_tx.start_cycle > role_intervals[0].start_cycle or head_tx.end_cycle < role_intervals[-1].end_cycle:
                raise ContractError("head transaction does not enclose its phase roles")
            if previous_head_tx is not None and head_tx.start_cycle <= previous_head_tx.end_cycle:
                raise ContractError("head transaction intervals overlap or are reordered")
            previous_head_tx = head_tx
            head_transactions.append(head_tx)
        drain = ledger.intervals[index]
        tile_tx = ledger.intervals[index + 1]
        index += 2
        for item, role in ((drain, "TILE_DRAIN"), (tile_tx, "TILE_TRANSACTION")):
            if item.role != role or (item.tile, item.head) != (tile, -1):
                raise ContractError("tile interval role or identity is out of order")
            if (item.stage, item.block, item.window) != (
                expected_identity.stage,
                expected_identity.block,
                expected_identity.window,
            ):
                raise ContractError("tile interval identity is rebound")
        if drain.start_cycle <= head_transactions[-1].end_cycle:
            raise ContractError("tile drain does not follow all head transactions")
        if tile_tx.start_cycle > head_transactions[0].start_cycle or tile_tx.end_cycle < drain.end_cycle:
            raise ContractError("tile transaction does not enclose head and drain intervals")
        if previous_tile_tx is not None and tile_tx.start_cycle <= previous_tile_tx.end_cycle:
            raise ContractError("tile transactions overlap or are reordered")
        previous_tile_tx = tile_tx
        tile_transactions.append(tile_tx)
    group_tx = ledger.intervals[index]
    index += 1
    if index != len(ledger.intervals):
        raise ContractError("phase ledger has trailing intervals")
    if group_tx.role != "GROUP_TRANSACTION" or (group_tx.tile, group_tx.head) != (-1, -1):
        raise ContractError("group transaction is missing or out of order")
    if (group_tx.stage, group_tx.block, group_tx.window) != (
        expected_identity.stage,
        expected_identity.block,
        expected_identity.window,
    ):
        raise ContractError("group transaction identity is rebound")
    if group_tx.start_cycle > tile_transactions[0].start_cycle or group_tx.end_cycle < tile_transactions[-1].end_cycle:
        raise ContractError("group transaction does not enclose all tile transactions")


@dataclass(frozen=True)
class IdentityTracePhaseAudit:
    path: str
    rows_read: int
    intervals_compared: int


def stream_compare_phase_ledger_to_identity_trace(
    ledger: PhaseLedger,
    identity_trace_path: Path,
    state_contract: Mapping[str, Any],
    expected_identity: PhaseIdentity,
) -> IdentityTracePhaseAudit:
    head_roles = head_phase_roles_from_state_contract(state_contract)
    validate_phase_interval_ledger(ledger, expected_identity, head_roles)
    state_rows = state_contract.get("states")
    if not isinstance(state_rows, list):
        raise ContractError("state contract lacks states for identity trace reconstruction")
    state_to_role: dict[int, str | None] = {}
    for row in state_rows:
        if not isinstance(row, dict):
            raise ContractError("state contract row is invalid")
        value = _require_int(row.get("value"), "state value")
        role = row.get("role")
        if role == "NONE":
            state_to_role[value] = None
        elif isinstance(role, str) and role.startswith("HEAD_"):
            state_to_role[value] = role
        else:
            raise ContractError("state contract role is invalid for trace reconstruction")

    interval_index = 0
    previous_cycle = -1
    rows_read = 0
    group_start: int | None = None
    tile_start: int | None = None
    active_tile: int | None = None
    head_start: int | None = None
    active_head: int | None = None
    active_head_tile: int | None = None
    active_role: str | None = None
    active_role_start: int | None = None
    active_role_tile: int | None = None
    active_role_head: int | None = None
    drain_start: int | None = None
    drain_tile: int | None = None
    pending_head_done: tuple[int, int, int] | None = None

    def emit(tile: int, head: int, role: str, start: int, end: int) -> None:
        nonlocal interval_index
        if interval_index >= len(ledger.intervals):
            raise ContractError("identity trace derives an extra phase interval")
        observed = ledger.intervals[interval_index]
        expected = (tile, head, role, start, end, end - start + 1)
        actual = (
            observed.tile,
            observed.head,
            observed.role,
            observed.start_cycle,
            observed.end_cycle,
            observed.duration,
        )
        if actual != expected:
            raise ContractError(
                f"phase interval {interval_index} differs from streamed identity trace boundary"
            )
        interval_index += 1

    def flush_head_done() -> None:
        nonlocal pending_head_done, head_start, active_head_tile, active_head
        if pending_head_done is None:
            return
        done_tile, done_head, done_cycle = pending_head_done
        if (
            head_start is None
            or active_head_tile is None
            or active_head is None
            or done_tile != active_head_tile
            or done_head != active_head
            or active_role is not None
        ):
            raise ContractError("identity trace head_done does not close the active head")
        emit(active_head_tile, active_head, "HEAD_TRANSACTION", head_start, done_cycle)
        head_start = None
        active_head_tile = None
        active_head = None
        pending_head_done = None

    with identity_trace_path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != IDENTITY_TRACE_COLUMNS:
            raise ContractError("identity trace columns are not the frozen v1 trace columns")
        for line_number, row in enumerate(reader, start=2):
            rows_read += 1
            cycle = _trace_int(row, "cycle", line_number)
            if cycle < 0 or cycle < previous_cycle:
                raise ContractError("identity trace cycles are negative or reordered")
            if previous_cycle >= 0 and cycle != previous_cycle:
                flush_head_done()
            previous_cycle = cycle
            event = row["event"]
            tile = _trace_int(row, "tile", line_number)
            head = _trace_int(row, "head", line_number)
            if event == "group_start":
                if group_start is not None:
                    raise ContractError("identity trace has overlapping group transactions")
                group_start = cycle
            elif event == "tile_start":
                if tile < 0 or tile_start is not None:
                    raise ContractError("identity trace has overlapping or invalid tile transactions")
                tile_start = cycle
                active_tile = tile
            elif event == "head_start":
                flush_head_done()
                if active_tile is None or tile != active_tile or head < 0 or head_start is not None:
                    raise ContractError("identity trace has invalid head transaction start")
                head_start = cycle
                active_head_tile = tile
                active_head = head
            elif event == "head_state":
                state = _trace_int(row, "index", line_number)
                if state not in state_to_role:
                    raise ContractError("identity trace head_state is outside the frozen enum")
                role = state_to_role[state]
                if role != active_role:
                    if active_role is not None:
                        if None in (
                            active_role_start,
                            active_role_tile,
                            active_role_head,
                        ):
                            raise ContractError("identity trace active head role identity is incomplete")
                        emit(
                            int(active_role_tile),
                            int(active_role_head),
                            active_role,
                            int(active_role_start),
                            cycle - 1,
                        )
                    active_role = role
                    if role is not None:
                        role_tile = active_tile if tile < 0 else tile
                        role_head = active_head if head < 0 else head
                        if role_tile is None or role_head is None:
                            raise ContractError("identity trace head role has no active identity")
                        if role_tile != active_tile or role_head != active_head:
                            raise ContractError("identity trace head role identity is rebound")
                        active_role_start = cycle
                        active_role_tile = role_tile
                        active_role_head = role_head
            elif event == "head_done":
                if pending_head_done is not None:
                    raise ContractError("identity trace has multiple head_done events in one cycle")
                pending_head_done = (tile, head, cycle)
            elif event == "tx_state":
                state = _trace_int(row, "index", line_number)
                in_drain = 4 <= state <= 6
                if in_drain and drain_start is None:
                    if active_tile is None:
                        raise ContractError("identity trace drain has no active tile")
                    drain_start = cycle
                    drain_tile = active_tile
                elif not in_drain and drain_start is not None:
                    if drain_tile is None:
                        raise ContractError("identity trace drain tile identity is missing")
                    emit(drain_tile, -1, "TILE_DRAIN", drain_start, cycle - 1)
                    drain_start = None
                    drain_tile = None
            elif event == "tile_done":
                if tile_start is None or active_tile is None or tile != active_tile:
                    raise ContractError("identity trace tile_done does not close the active tile")
                if drain_start is not None:
                    if drain_tile != active_tile:
                        raise ContractError("identity trace drain/tile identity differs")
                    # The trace emits tile_done before tx_state=7 in the same
                    # cycle. The drain state therefore ends one cycle earlier.
                    emit(active_tile, -1, "TILE_DRAIN", drain_start, cycle - 1)
                    drain_start = None
                    drain_tile = None
                emit(active_tile, -1, "TILE_TRANSACTION", tile_start, cycle)
                tile_start = None
                active_tile = None
            elif event == "group_done":
                if group_start is None or any(
                    value is not None
                    for value in (tile_start, head_start, active_role, drain_start)
                ):
                    raise ContractError("identity trace group_done has open child intervals")
                emit(-1, -1, "GROUP_TRANSACTION", group_start, cycle)
                group_start = None
    flush_head_done()
    if any(
        value is not None
        for value in (
            group_start,
            tile_start,
            head_start,
            active_role,
            drain_start,
            pending_head_done,
        )
    ):
        raise ContractError("identity trace ends with an open phase interval")
    if interval_index != len(ledger.intervals):
        raise ContractError("identity trace deleted one or more phase intervals")
    return IdentityTracePhaseAudit(str(identity_trace_path), rows_read, interval_index)


@dataclass(frozen=True)
class CrossAccCommand:
    sequence: int
    cycle: int
    rw: int
    addr: int
    write_data: int


@dataclass(frozen=True)
class CrossAccAudit:
    total: int
    reads: int
    writes: int
    output_tiles: int
    heads: int
    addresses_per_tile: int


AddressOrderFactory = Callable[[int], Sequence[int]]


def cross_acc_scalar_address(
    plane: int,
    y: int,
    x: int,
    out: int,
    *,
    height: int = 15,
    width: int = 15,
    time_planes: int = 2,
    out_dim: int = 32,
) -> int:
    if not (0 <= plane < time_planes and 0 <= y < height and 0 <= x < width and 0 <= out < out_dim):
        raise ContractError("cross-Acc scalar identity is out of range")
    return (((plane * height + y) * width + x) * out_dim) + out


def _default_address_order(addresses_per_tile: int) -> AddressOrderFactory:
    canonical = tuple(range(addresses_per_tile))
    return lambda _tile: canonical


def _validated_address_order(
    factory: AddressOrderFactory,
    tile: int,
    addresses_per_tile: int,
) -> tuple[int, ...]:
    values = tuple(factory(tile))
    if len(values) != addresses_per_tile:
        raise ContractError("runtime address order has the wrong cardinality")
    for index, value in enumerate(values):
        _bounded_unsigned(value, 64, f"address_order[{tile}][{index}]")
    if len(values) != len(list(dict.fromkeys(values))):
        raise ContractError("runtime address order aliases an address")
    return values


def summarize_cross_protocol_order(
    target_instance: str,
    rw_addr_events: Iterable[tuple[int, int]],
) -> CrossProtocolLedger:
    try:
        target_bytes = target_instance.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError("cross protocol target instance is not UTF-8") from exc
    if not target_bytes or len(target_bytes) > 0xFFFF:
        raise ContractError("cross protocol target instance length is invalid")
    rolling = Rolling64Pair()
    for value in (
        DOMAIN_TAG.encode("ascii"),
        b"local5_cross_acc_protocol_ledger_v2",
        b"CROSS_ACC_PROTOCOL_LEDGER",
        target_bytes,
    ):
        rolling.update(struct.pack("<H", len(value)) + value)
    count = 0
    reads = 0
    writes = 0
    for rw, addr in rw_addr_events:
        rw = _bounded_unsigned(rw, 64, "cross protocol rw")
        addr = _bounded_unsigned(addr, 64, "cross protocol address")
        if rw not in (0, 1):
            raise ContractError("cross protocol rw is not read/write")
        rolling.update(struct.pack("<QQQ", count, rw, addr))
        if rw == 0:
            reads += 1
        else:
            writes += 1
        count += 1
    digest0, digest1 = rolling.values()
    return CrossProtocolLedger(count, reads, writes, digest0, digest1)


def expected_cross_protocol_ledger(
    target_instance: str,
    *,
    heads: int,
    output_tiles: int | None = None,
    addresses_per_tile: int = 14_400,
    address_order_for_tile: AddressOrderFactory | None = None,
) -> CrossProtocolLedger:
    heads = _require_int(heads, "cross protocol heads")
    output_tiles = heads if output_tiles is None else _require_int(output_tiles, "output tiles")
    addresses_per_tile = _require_int(addresses_per_tile, "addresses per tile")
    if heads <= 0 or output_tiles <= 0 or addresses_per_tile <= 0:
        raise ContractError("cross protocol dimensions must be positive")
    factory = address_order_for_tile or _default_address_order(addresses_per_tile)

    def ordered_events() -> Iterable[tuple[int, int]]:
        for tile in range(output_tiles):
            order = _validated_address_order(factory, tile, addresses_per_tile)
            for address in order:
                yield 1, address
            for _head in range(1, heads):
                for address in order:
                    yield 0, address
                    yield 1, address
            for address in order:
                yield 0, address

    ledger = summarize_cross_protocol_order(target_instance, ordered_events())
    expected_half = heads * output_tiles * addresses_per_tile
    if (
        ledger.count != 2 * expected_half
        or ledger.read_count != expected_half
        or ledger.write_count != expected_half
    ):
        raise ContractError("independent cross protocol ledger violates its closed form")
    return ledger


def verify_cross_summary_pair(
    main: OrderedSummary,
    lower: OrderedSummary,
    *,
    heads: int,
    output_tiles: int | None = None,
    addresses_per_tile: int = 14_400,
    address_order_for_tile: AddressOrderFactory | None = None,
) -> CrossProtocolLedger:
    if main.schema != SUMMARY_SCHEMA:
        raise ContractError("cross summary pair lacks the main ordered summary")
    validate_observer_summary_binding(
        lower,
        expected_schema=CROSS_SUMMARY_SCHEMA,
        expected_target_instance=main.resources["CROSS_ACC_COMMAND"].instance_path,
    )
    compare_summary_resources(main, lower, ["CROSS_ACC_COMMAND"])
    main_ledger = main.cross_protocol_ledger
    lower_ledger = lower.cross_protocol_ledger
    if main_ledger is None or lower_ledger is None:
        raise ContractError("cross summary pair lacks protocol ledgers")
    if (
        main_ledger.count,
        main_ledger.read_count,
        main_ledger.write_count,
    ) != (
        lower_ledger.count,
        lower_ledger.read_count,
        lower_ledger.write_count,
    ):
        raise ContractError("main/lower cross protocol ledger counts differ")
    expected = expected_cross_protocol_ledger(
        main.resources["CROSS_ACC_COMMAND"].instance_path,
        heads=heads,
        output_tiles=output_tiles,
        addresses_per_tile=addresses_per_tile,
        address_order_for_tile=address_order_for_tile,
    )
    if main_ledger != expected:
        raise ContractError("main cross protocol count/order digest differs from the independent oracle")
    return expected


def verify_tcfm5_summary_pair(main: OrderedSummary, lower: OrderedSummary) -> Tcfm5TermLedger:
    if main.schema != SUMMARY_SCHEMA:
        raise ContractError("TCFM5 summary pair lacks the main ordered summary")
    validate_observer_summary_binding(
        lower,
        expected_schema=TCFM5_SUMMARY_SCHEMA,
        expected_target_instance=main.resources["TCFM5_TERM_COMMIT"].instance_path,
    )
    compare_summary_resources(main, lower, ["TCFM5_TERM_COMMIT"])
    main_ledger = main.tcfm5_term_ledger
    lower_ledger = lower.tcfm5_term_ledger
    if main_ledger is None or lower_ledger is None:
        raise ContractError("TCFM5 summary pair lacks term ledgers")
    if main_ledger != lower_ledger:
        raise ContractError("main/lower TCFM5 term/update/mismatch ledgers differ")
    return main_ledger


def verify_cross_acc_protocol(
    commands: Iterable[CrossAccCommand],
    *,
    heads: int,
    output_tiles: int | None = None,
    addresses_per_tile: int = 14_400,
    address_order_for_tile: AddressOrderFactory | None = None,
) -> CrossAccAudit:
    heads = _require_int(heads, "cross-Acc heads")
    output_tiles = heads if output_tiles is None else _require_int(output_tiles, "output tiles")
    addresses_per_tile = _require_int(addresses_per_tile, "addresses per tile")
    if heads <= 0 or output_tiles <= 0 or addresses_per_tile <= 0:
        raise ContractError("cross-Acc dimensions must be positive")
    factory = address_order_for_tile or _default_address_order(addresses_per_tile)
    iterator = iter(commands)
    count = 0
    reads = 0
    writes = 0
    previous_cycle = -1

    def consume(expected_rw: int, expected_addr: int, phase: str) -> None:
        nonlocal count, reads, writes, previous_cycle
        try:
            command = next(iterator)
        except StopIteration as exc:
            raise ContractError(f"cross-Acc command deleted during {phase}") from exc
        if command.sequence != count:
            raise ContractError("cross-Acc sequence is missing, duplicated, or reordered")
        if command.cycle < previous_cycle:
            raise ContractError("cross-Acc accepted cycles are reordered")
        if command.rw not in (0, 1) or command.rw != expected_rw or command.addr != expected_addr:
            raise ContractError(f"cross-Acc address phase/order mismatch during {phase}")
        _bounded_unsigned(command.write_data, 64, "cross-Acc write_data")
        if command.rw == 0:
            reads += 1
            if command.write_data != 0:
                raise ContractError("cross-Acc read projection must zero write_data")
        else:
            writes += 1
        previous_cycle = command.cycle
        count += 1

    for tile in range(output_tiles):
        order = _validated_address_order(factory, tile, addresses_per_tile)
        for address in order:
            consume(1, address, "first-head write")
        for head in range(1, heads):
            for address in order:
                consume(0, address, f"head {head} read")
                consume(1, address, f"head {head} write")
        for address in order:
            consume(0, address, "final drain read")
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ContractError("cross-Acc command stream has duplicate or trailing events")
    expected_half = heads * output_tiles * addresses_per_tile
    if count != 2 * expected_half or reads != expected_half or writes != expected_half:
        raise ContractError("cross-Acc total/read/write counts do not match the closed form")
    return CrossAccAudit(count, reads, writes, output_tiles, heads, addresses_per_tile)


ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)
ROLE_BANK_OFFSET = (0, 2, 3, 1, 4)


@dataclass(frozen=True)
class Tcfm5Topology:
    expected_mask: int
    bank_addresses: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Tcfm5Projection:
    sequence: int
    cycle: int
    source: int
    lane: int
    expected_mask: int
    actual_mask: int
    bank_addresses: tuple[int, int, int, int, int]


def expected_tcfm5_topology(
    plane: int,
    y: int,
    x: int,
    destination_mask: int,
    *,
    height: int = 15,
    width: int = 15,
    time_planes: int = 2,
    strict_boundary_mask: bool = True,
) -> Tcfm5Topology:
    if not (height > 0 and width > 0 and time_planes > 0):
        raise ContractError("TCFM5 dimensions must be positive")
    if not (0 <= plane < time_planes and 0 <= y < height and 0 <= x < width):
        raise ContractError("TCFM5 source identity is out of range")
    destination_mask = _bounded_unsigned(destination_mask, 5, "TCFM5 destination mask")
    x_groups = (width + 4) // 5
    plane_bank_depth = height * x_groups
    addresses = [0, 0, 0, 0, 0]
    source_color = (x + 2 * y) % 5
    expected_mask = 0
    for role, (dy, dx, bank_offset) in enumerate(zip(ROLE_DY, ROLE_DX, ROLE_BANK_OFFSET)):
        candidate_y, candidate_x = y + dy, x + dx
        valid = 0 <= candidate_y < height and 0 <= candidate_x < width
        if not valid:
            if strict_boundary_mask and ((destination_mask >> role) & 1):
                raise ContractError("TCFM5 destination mask selects an out-of-bound role")
            role_y, role_x = y, x
        else:
            role_y, role_x = candidate_y, candidate_x
        bank = (source_color + bank_offset) % 5
        addresses[bank] = plane * plane_bank_depth + role_y * x_groups + role_x // 5
        if valid and ((destination_mask >> role) & 1):
            expected_mask |= 1 << bank
    return Tcfm5Topology(expected_mask, tuple(addresses))


def verify_tcfm5_projection(
    event: Tcfm5Projection,
    *,
    plane: int,
    y: int,
    x: int,
    destination_mask: int,
    height: int = 15,
    width: int = 15,
    time_planes: int = 2,
    lanes: int = 32,
) -> Tcfm5Topology:
    topology = expected_tcfm5_topology(
        plane,
        y,
        x,
        destination_mask,
        height=height,
        width=width,
        time_planes=time_planes,
    )
    expected_source = plane * height * width + y * width + x
    if event.source != expected_source or not 0 <= event.lane < lanes:
        raise ContractError("TCFM5 source/lane runtime identity is rebound")
    if event.expected_mask != topology.expected_mask or event.actual_mask != topology.expected_mask:
        raise ContractError("TCFM5 expected/actual mask differs from the independent topology")
    if event.bank_addresses != topology.bank_addresses:
        raise ContractError("TCFM5 bank address projection differs from the independent topology")
    return topology


@dataclass(frozen=True)
class EvidencePayloadAudit:
    root: str
    bytes: int
    regular_files: int
    unique_inodes: int
    excluded_top_dirs: tuple[str, ...]
    limit_bytes: int


def audit_evidence_payload(
    root: Path,
    *,
    limit_bytes: int = EVIDENCE_LIMIT_BYTES,
    excluded_top_dirs: Sequence[str] = ("build", "source"),
) -> EvidencePayloadAudit:
    if root.is_symlink():
        raise ContractError("evidence payload root must be a real directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ContractError("evidence payload root must be a real directory")
    limit_bytes = _require_int(limit_bytes, "evidence byte limit")
    if limit_bytes < 0:
        raise ContractError("evidence byte limit is negative")
    excluded_names = tuple(excluded_top_dirs)
    if any(not name or "/" in name or name in (".", "..") for name in excluded_names):
        raise ContractError("excluded top directory name is invalid")
    seen_inodes: dict[tuple[int, int], Path] = {}
    total = 0
    regular_files = 0

    def visit(directory: Path, excluded: bool) -> None:
        nonlocal total, regular_files
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            entry_path = Path(entry.path)
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ContractError(f"evidence payload rejects symlink: {entry_path}")
            if stat.S_ISDIR(entry_stat.st_mode):
                child_excluded = excluded or (directory == root and entry.name in excluded_names)
                visit(entry_path, child_excluded)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ContractError(f"evidence payload rejects non-regular file: {entry_path}")
            if excluded:
                continue
            regular_files += 1
            inode = (entry_stat.st_dev, entry_stat.st_ino)
            if inode in seen_inodes:
                continue
            seen_inodes[inode] = entry_path
            total += entry_stat.st_size
            if total > limit_bytes:
                raise ContractError("evidence payload exceeds the 512 MiB logical st_size gate")

    visit(root, False)
    return EvidencePayloadAudit(
        str(root),
        total,
        regular_files,
        len(seen_inodes),
        excluded_names,
        limit_bytes,
    )


def _canonical_identity(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or tuple(value.keys()) != IDENTITY_KEYS:
        raise ContractError(f"{label} identity keys or order are not frozen")
    return {key: _require_int(value[key], f"{label}.{key}") for key in IDENTITY_KEYS}


def _digest_set_from_denylist(denylist: Any) -> list[str]:
    if denylist is None:
        return []
    if isinstance(denylist, dict):
        entries = denylist.get("entries")
    else:
        entries = denylist
    if not isinstance(entries, list):
        raise ContractError("denylist entries are not a list")
    output: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            digest = entry
        elif isinstance(entry, dict):
            digest = entry.get("package_digest") or entry.get("complete_digest")
        else:
            raise ContractError("denylist entry is invalid")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError("denylist digest is not canonical SHA256")
        output.append(digest)
    return output


def verify_receipt_admission(
    receipt: Mapping[str, Any],
    *,
    allowed_schemas: Sequence[str],
    allowed_statuses: Sequence[str],
    required_receipts: Mapping[str, Sequence[str]],
    expected_identity: Mapping[str, int] | None = None,
    package_digest: str,
    denylist: Any = None,
) -> dict[str, Any]:
    # Positive admission is intentionally complete before the denylist lookup.
    if receipt.get("schema") not in allowed_schemas:
        raise ContractError("receipt schema is not positively admitted")
    if receipt.get("status") not in allowed_statuses:
        raise ContractError("receipt status is not positively admitted")
    requested = _canonical_identity(receipt.get("requested_identity"), "requested")
    actual = _canonical_identity(receipt.get("actual_identity"), "actual")
    if requested != actual or receipt.get("identity_status") != "MATCH":
        raise ContractError("receipt requested/actual identity is not an exact MATCH")
    if expected_identity is not None:
        expected = _canonical_identity(dict(expected_identity), "expected")
        if actual != expected:
            raise ContractError("receipt actual identity differs from the expected run identity")
    receipts = receipt.get("receipts")
    if not isinstance(receipts, dict):
        raise ContractError("receipt evidence receipt map is missing")
    for name, admitted_statuses in required_receipts.items():
        row = receipts.get(name)
        if not isinstance(row, dict) or row.get("status") not in admitted_statuses:
            raise ContractError(f"required receipt {name} is not positively admitted")
    if not re.fullmatch(r"[0-9a-f]{64}", package_digest):
        raise ContractError("package digest is not canonical SHA256")
    if package_digest in _digest_set_from_denylist(denylist):
        raise ContractError("otherwise-admitted package digest is denylisted")
    return {
        "schema": receipt["schema"],
        "status": receipt["status"],
        "identity": actual,
        "receipt_names": list(required_receipts),
        "package_digest": package_digest,
        "denylist_status": "CLEAR",
    }


@dataclass(frozen=True)
class SvEnum:
    module: str
    enum_type: str
    width: int
    states: tuple[tuple[str, int], ...]


def _strip_sv_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _parse_sv_integer(text: str) -> int:
    token = text.strip().replace("_", "")
    if re.fullmatch(r"[0-9]+", token):
        return int(token, 10)
    match = re.fullmatch(r"(?:[0-9]+)?'[sS]?([bBoOdDhH])([0-9a-fA-F]+)", token)
    if not match:
        raise ContractError(f"enum value is not a static integer literal: {text.strip()}")
    radix = {"b": 2, "o": 8, "d": 10, "h": 16}[match.group(1).lower()]
    return int(match.group(2), radix)


def parse_sv_enum(path: Path, module_name: str, enum_type: str) -> SvEnum:
    text = _strip_sv_comments(path.read_text(encoding="ascii"))
    module_match = re.search(rf"\bmodule\s+{re.escape(module_name)}\b", text)
    if module_match is None:
        raise ContractError(f"RTL lacks module {module_name}")
    module_end = re.search(r"\bendmodule\b", text[module_match.end() :])
    if module_end is None:
        raise ContractError("RTL target module is not closed")
    module_text = text[module_match.start() : module_match.end() + module_end.end()]
    pattern = re.compile(
        rf"typedef\s+enum\s+logic\s*\[\s*([0-9]+)\s*:\s*([0-9]+)\s*\]"
        rf"\s*\{{(.*?)\}}\s*{re.escape(enum_type)}\s*;",
        re.DOTALL,
    )
    matches = list(pattern.finditer(module_text))
    if len(matches) != 1:
        raise ContractError("RTL target enum is missing or ambiguous")
    match = matches[0]
    width = abs(int(match.group(1)) - int(match.group(2))) + 1
    states: list[tuple[str, int]] = []
    previous = -1
    for raw_item in match.group(3).split(","):
        item = raw_item.strip()
        enum_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(.+))?", item)
        if enum_match is None:
            raise ContractError(f"RTL enum item is not statically parseable: {item}")
        symbol, expression = enum_match.group(1), enum_match.group(2)
        value = previous + 1 if expression is None else _parse_sv_integer(expression)
        if value < 0 or value >= (1 << width):
            raise ContractError("RTL enum value exceeds its declared width")
        if any(existing_symbol == symbol for existing_symbol, _ in states):
            raise ContractError("RTL enum symbol is duplicated")
        if any(existing_value == value for _, existing_value in states):
            raise ContractError("RTL enum value is duplicated")
        states.append((symbol, value))
        previous = value
    if not states:
        raise ContractError("RTL enum has no states")
    return SvEnum(module_name, enum_type, width, tuple(states))


def load_and_verify_state_role_contract(
    repo_root: Path,
    contract_path: Path | None = None,
) -> tuple[dict[str, Any], SvEnum, dict[str, Any]]:
    repo_root = repo_root.resolve()
    contract_path = contract_path or repo_root / "contracts/local5_phase_state_roles_v2.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("state role contract is absent or invalid JSON") from exc
    if not isinstance(contract, dict):
        raise ContractError("state role contract top level is not an object")
    if (
        contract.get("schema") != STATE_ROLE_SCHEMA
        or contract.get("status") != EXPECTED_STATE_STATUS
        or contract.get("formal_g0") != "DENY"
        or contract.get("module") != EXPECTED_MODULE
        or contract.get("configuration") != EXPECTED_CONFIGURATION
        or contract.get("rtl_source") != EXPECTED_RTL_SOURCE
        or contract.get("enum_type") != EXPECTED_ENUM_TYPE
    ):
        raise ContractError("state role module/configuration/boundary is not the trusted direct baseline")
    rtl_path = (repo_root / EXPECTED_RTL_SOURCE).resolve()
    try:
        rtl_path.relative_to(repo_root)
    except ValueError as exc:
        raise ContractError("state role RTL source escapes the repository") from exc
    if not rtl_path.is_file() or rtl_path.is_symlink():
        raise ContractError("state role RTL source is absent or a symlink")
    actual_sha = sha256_file(rtl_path)
    frozen_sha = contract.get("rtl_source_sha256")
    if not isinstance(frozen_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", frozen_sha):
        raise ContractError("state role RTL SHA256 is not canonical")
    if actual_sha != frozen_sha:
        raise ContractError("state role RTL SHA256 differs from the frozen contract")
    parsed = parse_sv_enum(rtl_path, EXPECTED_MODULE, EXPECTED_ENUM_TYPE)
    enum_width = contract.get("enum_width")
    if _require_int(enum_width, "contract enum width") != parsed.width:
        raise ContractError("RTL enum width differs from the JSON contract")
    state_rows = contract.get("states")
    if not isinstance(state_rows, list) or not state_rows:
        raise ContractError("state role contract states are absent")
    json_states: list[tuple[str, int]] = []
    for index, row in enumerate(state_rows):
        if not isinstance(row, dict) or tuple(row.keys()) != ("symbol", "value", "role"):
            raise ContractError(f"state role row {index} schema is not frozen")
        symbol, value, role = row["symbol"], row["value"], row["role"]
        if not isinstance(symbol, str) or not re.fullmatch(r"ST_[A-Z0-9_]+", symbol):
            raise ContractError("state role symbol is invalid")
        value = _require_int(value, f"state {symbol} value")
        if role not in ALLOWED_STATE_ROLES:
            raise ContractError("state role is outside the admitted role vocabulary")
        json_states.append((symbol, value))
    if tuple(json_states) != parsed.states:
        raise ContractError("ordered RTL enum symbol/value list differs from JSON")
    head_roles = head_phase_roles_from_state_contract(contract)
    report = {
        "module": parsed.module,
        "configuration": EXPECTED_CONFIGURATION,
        "rtl_source": EXPECTED_RTL_SOURCE,
        "rtl_source_sha256": actual_sha,
        "enum_type": parsed.enum_type,
        "enum_width": parsed.width,
        "state_count": len(parsed.states),
        "head_phase_roles": list(head_roles),
    }
    return contract, parsed, report


def audit_monitor_static_contract(path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file() or path.is_symlink():
        return {"path": str(path), "status": "BLOCKED", "blockers": ["monitor source absent or symlinked"]}
    source = path.read_text(encoding="ascii")
    module_match = re.search(r"\bmodule\s+local5_phase_summary_monitor_v2\b", source)
    if module_match is None:
        blockers.append("monitor module declaration is missing")
        header = ""
    else:
        header_end = source.find(");", module_match.end())
        header = source[module_match.start() : header_end + 2] if header_end >= 0 else ""
        if not header:
            blockers.append("monitor port list is not closed")
        elif re.search(r"\boutput\b", header):
            blockers.append("passive monitor declares an output")
    required_inputs = (
        "memory_command_valid",
        "memory_command_write",
        "memory_command_addr",
        "memory_command_write_data",
        "tcfm_term_commit",
        "tcfm_term_source_plane",
        "tcfm_term_source_y",
        "tcfm_term_source_x",
        "tcfm_term_lane",
        "tcfm_term_destination_mask",
    )
    for signal in required_inputs:
        if not re.search(rf"\binput\b[^;]*\b{re.escape(signal)}\b", header, re.DOTALL):
            blockers.append(f"monitor common-projection input missing: {signal}")
    if DIGEST_NAME not in source:
        blockers.append("monitor does not name the frozen FNV1A64/DJB2XOR64 digest")
    if FRAME_SERIALIZATION not in source:
        blockers.append("monitor does not declare the frozen frame serialization")
    for fragment in (
        '"SCHEMA,local5_ordered_summary_v2\\n"',
        '"ORIGIN,RTL_DIRECT\\n"',
        'domain_tag = "LOCAL5_PHASE_SUMMARY_V2"',
        "hash_u16(resource, 16'd2)",
        "hash_u16(resource, 16'(resource))",
        "hash_u16(resource, 16'd80)",
    ):
        if fragment not in source:
            blockers.append(f"monitor frame implementation lacks {fragment}")

    def self_updates(variable: str) -> list[str]:
        pattern = re.compile(
            rf"{re.escape(variable)}(?:\s*\[[^\]]+\])?\s*(?:=|<=)\s*(.*?);",
            re.DOTALL,
        )
        return [
            re.sub(r"\s+", "", match.group(1)).replace("digest0_q", "digest_q").replace("digest1_q", "digest_q")
            for match in pattern.finditer(source)
            if variable in match.group(1)
        ]

    update0 = self_updates("digest0_q")
    update1 = self_updates("digest1_q")
    if not update0 or not update1:
        blockers.append("monitor rolling64 self-update assignments are not statically visible")
    elif update0 == update1:
        blockers.append("monitor rolling64 update functions are not different")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
    }


def audit_lower_observer_sources(repo_root: Path) -> dict[str, Any]:
    specifications = (
        (
            "verif_qfit/local5_cross_acc_summary_monitor_v2.sv",
            "local5_cross_acc_summary_monitor_v2",
            CROSS_SUMMARY_SCHEMA,
            "RTL_LOWER_PORT",
            "localparam int RESOURCE_CODE = 5",
            ("command_valid", "command_write", "command_addr", "command_write_data"),
        ),
        (
            "verif_qfit/local5_tcfm5_summary_monitor_v2.sv",
            "local5_tcfm5_summary_monitor_v2",
            TCFM5_SUMMARY_SCHEMA,
            "RTL_LOWER_BANKS",
            "localparam int RESOURCE_CODE = 6",
            ("actual_bank_mask", "actual_bank_addr_flat"),
        ),
    )
    blockers: list[str] = []
    files: dict[str, Any] = {}
    for relative, module, schema, origin, code_fragment, actual_inputs in specifications:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            blockers.append(f"lower observer source absent or symlinked: {relative}")
            continue
        source = path.read_text(encoding="ascii")
        module_match = re.search(rf"\bmodule\s+{re.escape(module)}\b", source)
        header_end = source.find(");", module_match.end()) if module_match else -1
        header = source[module_match.start() : header_end + 2] if module_match and header_end >= 0 else ""
        required_fragments = (
            schema,
            origin,
            DIGEST_NAME,
            "64'hcbf29ce484222325",
            "64'h00001505d3c4b2a1",
            code_fragment,
            "hash_u16(16'd80)",
            'domain_tag = "LOCAL5_PHASE_SUMMARY_V2"',
            "hash_u16(16'd2)",
            "OBSERVER_INSTANCE",
            "TARGET_INSTANCE",
            "PAYLOAD_U64_COUNT,10",
        )
        if not header or re.search(r"\boutput\b", header):
            blockers.append(f"lower observer is not statically passive: {module}")
        for signal in actual_inputs:
            if not re.search(rf"\binput\b[^;]*\b{re.escape(signal)}\b", header, re.DOTALL):
                blockers.append(f"lower observer {module} lacks actual input {signal}")
        for fragment in required_fragments:
            if fragment not in source:
                blockers.append(f"lower observer {module} lacks {fragment}")
        files[module] = {"path": str(path), "sha256": sha256_file(path)}
    bind_path = repo_root / "verif_qfit/bind_local5_phase_summary_monitors_v2.sv"
    if not bind_path.is_file() or bind_path.is_symlink():
        blockers.append("phase summary bind source is absent or symlinked")
    else:
        bind_source = bind_path.read_text(encoding="ascii")
        for target in ("qfit_single_port_acc_memory", "qfit_tcfm5_projection_top"):
            if len(re.findall(rf"\bbind\s+{re.escape(target)}\b", bind_source)) != 1:
                blockers.append(f"bind source does not contain exactly one type bind for {target}")
        files["bind"] = {"path": str(bind_path), "sha256": sha256_file(bind_path)}
    return {"status": "PASS" if not blockers else "BLOCKED", "files": files, "blockers": blockers}


def run_static_preflight(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    blockers: list[str] = []
    checks: dict[str, Any] = {}
    try:
        _contract, _enum, state_report = load_and_verify_state_role_contract(repo_root)
        checks["state_role_contract"] = {"status": "PASS", **state_report}
    except ContractError as exc:
        blockers.append(f"state_role_contract: {exc}")
        checks["state_role_contract"] = {"status": "BLOCKED", "reason": str(exc)}
    expected_fixed = {
        3: {
            "phase": 52,
            "relation_req": 4050,
            "relation_rsp": 4050,
            "weight_req": 9216,
            "weight_rsp": 9216,
            "final": 43_200,
            "acc32": 43_200,
            "cross_total": 259_200,
            "cross_read": 129_600,
            "cross_write": 129_600,
        },
        24: {
            "phase": 2929,
            "relation_req": 259_200,
            "relation_rsp": 259_200,
            "weight_req": 589_824,
            "weight_rsp": 589_824,
            "final": 345_600,
            "aligned_total": 2_043_648,
            "acc32": 345_600,
            "cross_total": 16_588_800,
            "cross_read": 8_294_400,
            "cross_write": 8_294_400,
        },
    }
    count_reports: dict[str, Any] = {}
    for heads, expected in expected_fixed.items():
        actual = asdict(workload_counts(heads))
        mismatches = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
        if mismatches:
            blockers.append(f"H{heads}_closed_form: frozen values differ")
        count_reports[f"H{heads}"] = {"status": "PASS" if not mismatches else "BLOCKED", **actual}
    checks["closed_forms"] = count_reports
    monitor_report = audit_monitor_static_contract(repo_root / "verif_qfit/local5_phase_summary_monitor_v2.sv")
    checks["monitor_static"] = monitor_report
    blockers.extend(f"monitor_static: {item}" for item in monitor_report["blockers"])
    observer_report = audit_lower_observer_sources(repo_root)
    checks["lower_observer_static"] = observer_report
    blockers.extend(f"lower_observer_static: {item}" for item in observer_report["blockers"])
    return {
        "schema": STATIC_REPORT_SCHEMA,
        "status": "STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION" if not blockers else "STATIC_PREFLIGHT_BLOCKED",
        "formal_g0": "DENY",
        "repo_root": str(repo_root),
        "checks": checks,
        "blockers": blockers,
        "boundary": [
            "No RTL compile, simulation, H24 run, or GPU work was performed.",
            "Runtime identity, observer cardinality, receipts, summaries, and payload size remain run-package gates.",
            "The two rolling64 digests are ordered non-cryptographic checks; file sealing uses SHA256.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = run_static_preflight(args.root)
    rendered = json.dumps(report, ensure_ascii=True, indent=2) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(rendered, encoding="ascii")
    print(rendered, end="")
    return 0 if report["status"] == "STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
