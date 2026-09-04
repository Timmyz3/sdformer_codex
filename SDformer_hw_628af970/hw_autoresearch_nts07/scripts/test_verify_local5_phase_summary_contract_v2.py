#!/usr/bin/env python3
"""Unit tests for the Local5 phase summary v2 static verifier."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import verify_local5_phase_summary_contract_v2 as verifier


def make_event(sequence: int, cycle: int, *values: int) -> verifier.SummaryEvent:
    payload = tuple(values) + (0,) * (verifier.PAYLOAD_U64_COUNT - len(values))
    return verifier.SummaryEvent(sequence, cycle, payload)


def make_resource(
    name: str,
    instance: str,
    events: list[verifier.SummaryEvent],
) -> verifier.SummaryResource:
    return verifier.summarize_events(
        name,
        instance,
        events,
        field_names=verifier.RESOURCE_FIELDS[name],
    )


def anchor_line(resource: verifier.SummaryResource, kind: str) -> str | None:
    anchor = resource.first_anchor if kind == "FIRST" else resource.last_anchor
    if anchor is None:
        return None
    return ",".join(("A", resource.name, kind, *(str(value) for value in anchor)))


def write_main_summary(
    path: Path,
    overrides: dict[str, verifier.SummaryResource] | None = None,
) -> dict[str, verifier.SummaryResource]:
    overrides = overrides or {}
    resources: dict[str, verifier.SummaryResource] = {}
    for name in verifier.RESOURCE_CODES:
        resources[name] = overrides.get(name) or make_resource(name, "tb.u_main", [])
    lines = [
        f"SCHEMA,{verifier.SUMMARY_SCHEMA}",
        "ORIGIN,RTL_DIRECT",
        "MONITOR_INSTANCE,tb.u_main.u_monitor",
        "H,3",
        (
            f"DIGEST,{verifier.DIGEST_NAME},{verifier.FNV1A64_SEED:016x},"
            f"{verifier.DJB2XOR64_SEED:016x},{verifier.FNV1A64_PRIME:016x}"
        ),
        "BYTE_ORDER,LITTLE_ENDIAN",
        f"SERIALIZATION,{verifier.FRAME_SERIALIZATION}",
        f"PAYLOAD_U64_COUNT,{verifier.PAYLOAD_U64_COUNT}",
        "SAME_CYCLE_ORDER,RELATION_REQ,RELATION_RSP,WEIGHT_REQ,WEIGHT_RSP,FINAL,CROSS_ACC,TCFM5",
        "EMPTY_STREAM,raw_seed_without_event_frame",
    ]
    for resource in resources.values():
        lines.append(f"R,{resource.name},{resource.instance_path}")
    for resource in resources.values():
        lines.append(f"F,{resource.name},{','.join(resource.field_names)}")
    lines.append("P,CROSS_ACC_PROTOCOL_LEDGER,sequence_u64le,rw_u64le,addr_u64le")
    for resource in resources.values():
        for kind in ("FIRST", "LAST"):
            line = anchor_line(resource, kind)
            if line is not None:
                lines.append(line)
        lines.append(
            f"S,{resource.name},{resource.count},{resource.digest0:016x},{resource.digest1:016x}"
        )
    cross = resources["CROSS_ACC_COMMAND"]
    if cross.count == 0:
        protocol_events: list[tuple[int, int]] = []
    elif cross.count == 1 and cross.first_anchor is not None:
        protocol_events = [(cross.first_anchor[2], cross.first_anchor[3])]
    elif cross.count == 2 and cross.first_anchor is not None and cross.last_anchor is not None:
        protocol_events = [
            (cross.first_anchor[2], cross.first_anchor[3]),
            (cross.last_anchor[2], cross.last_anchor[3]),
        ]
    else:
        raise AssertionError("test helper only supports zero, one, or two cross events")
    protocol = verifier.summarize_cross_protocol_order(cross.instance_path, protocol_events)
    tcfm = resources["TCFM5_TERM_COMMIT"]
    tcfm_updates = 0
    if tcfm.count == 1 and tcfm.first_anchor is not None:
        tcfm_updates = tcfm.first_anchor[5].bit_count()
    lines.extend(
        (
            (
                f"L,CROSS_ACC_PROTOCOL_LEDGER,{protocol.count},{protocol.read_count},"
                f"{protocol.write_count},{protocol.digest0:016x},{protocol.digest1:016x}"
            ),
            f"L,TCFM5_TERM_LEDGER,{tcfm.count},{tcfm_updates},0",
            "END,100,RTL_DIRECT",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return resources


def write_lower_summary(
    path: Path,
    schema: str,
    resource: verifier.SummaryResource,
    *,
    observer: str | None = None,
    target: str | None = None,
) -> None:
    origin = verifier.SUMMARY_ORIGINS[schema]
    target = target or resource.instance_path
    observer = observer or f"{target}.u_observer"
    lines = [
        f"SCHEMA,{schema}",
        f"ORIGIN,{origin}",
        f"OBSERVER_INSTANCE,{observer}",
        f"TARGET_INSTANCE,{target}",
        (
            f"DIGEST,{verifier.DIGEST_NAME},{verifier.FNV1A64_SEED:016x},"
            f"{verifier.DJB2XOR64_SEED:016x}"
        ),
        f"RESOURCE_CODE,{resource.name},{resource.code}",
        f"PAYLOAD_U64_COUNT,{verifier.PAYLOAD_U64_COUNT}",
    ]
    for kind in ("FIRST", "LAST"):
        line = anchor_line(resource, kind)
        if line is not None:
            lines.append(line)
    lines.append(
        f"S,{resource.name},{resource.count},{resource.digest0:016x},{resource.digest1:016x}"
    )
    if schema == verifier.CROSS_SUMMARY_SCHEMA:
        reads = 0
        writes = 0
        anchors = [] if resource.count == 0 else [resource.first_anchor]
        if resource.count == 2:
            anchors.append(resource.last_anchor)
        for anchor in anchors:
            if anchor is None:
                raise AssertionError("non-empty test summary lacks an anchor")
            if anchor[2] == 0:
                reads += 1
            else:
                writes += 1
        lines.append(f"L,CROSS_ACC_PROTOCOL_LEDGER,{resource.count},{reads},{writes}")
    else:
        updates = resource.first_anchor[5].bit_count() if resource.count == 1 and resource.first_anchor else 0
        lines.append(f"L,TCFM5_TERM_LEDGER,{resource.count},{updates},0")
    lines.append(f"END,100,{origin}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_phase_ledger(path: Path, *, disorder: bool = False) -> None:
    roles = ("HEAD_WEIGHT", "HEAD_FRONTEND", "HEAD_READOUT", "HEAD_RELEASE")
    rows: list[list[object]] = []
    for index, role in enumerate(roles):
        cycle = 10 + index
        rows.append(["P", index, 2, 3, 4, 0, 0, role, cycle, cycle, 1, "RTL_DIRECT"])
    rows.extend(
        (
            ["P", 4, 2, 3, 4, 0, 0, "HEAD_TRANSACTION", 9, 14, 6, "RTL_DIRECT"],
            ["P", 5, 2, 3, 4, 0, -1, "TILE_DRAIN", 15, 15, 1, "RTL_DIRECT"],
            ["P", 6, 2, 3, 4, 0, -1, "TILE_TRANSACTION", 8, 16, 9, "RTL_DIRECT"],
            ["P", 7, 2, 3, 4, -1, -1, "GROUP_TRANSACTION", 7, 17, 11, "RTL_DIRECT"],
        )
    )
    if disorder:
        rows[0], rows[1] = rows[1], rows[0]
        rows[0][1], rows[1][1] = 0, 1
    lines = [
        ["SCHEMA", verifier.PHASE_SCHEMA],
        ["ORIGIN", "RTL_DIRECT"],
        ["H", 1],
        [
            "COLUMNS", "record", "sequence", "stage", "block", "window",
            "tile", "head", "role", "start_cycle", "end_cycle", "duration", "origin",
        ],
        *rows,
        ["END", 17, 8, "RTL_DIRECT"],
    ]
    with path.open("w", newline="", encoding="ascii") as handle:
        csv.writer(handle, lineterminator="\n").writerows(lines)


def write_phase_identity_trace(
    path: Path,
    *,
    head_done_head: int = 0,
    tile_done_before_exit_state: bool = False,
) -> None:
    def row(
        cycle: int,
        event: str,
        tile: int = -1,
        head: int = -1,
        index: int = -1,
    ) -> list[object]:
        return [
            cycle, event, tile, head, -1, -1, -1, -1, index,
            "rtl_internal_state", "-",
        ]

    rows = [
        row(7, "group_start"),
        row(8, "tile_start", 0),
        row(9, "head_start", 0, 0),
        row(10, "head_state", 0, 0, 1),
        row(11, "head_state", 0, 0, 3),
        row(12, "head_state", 0, 0, 10),
        row(13, "head_state", 0, 0, 13),
        row(14, "head_state", 0, 0, 0),
        row(14, "head_done", 0, head_done_head),
        row(15, "tx_state", 0, -1, 4),
    ]
    if tile_done_before_exit_state:
        rows.extend((
            row(16, "tile_done", 0),
            row(16, "tx_state", 0, -1, 7),
        ))
    else:
        rows.extend((
            row(16, "tx_state", 0, -1, 0),
            row(16, "tile_done", 0),
        ))
    rows.append(row(17, "group_done"))
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(verifier.IDENTITY_TRACE_COLUMNS)
        writer.writerows(rows)


def expected_cross_commands(heads: int, tiles: int, addresses: int) -> list[verifier.CrossAccCommand]:
    commands: list[verifier.CrossAccCommand] = []
    for _tile in range(tiles):
        for address in range(addresses):
            commands.append(verifier.CrossAccCommand(len(commands), len(commands), 1, address, address + 1))
        for _head in range(1, heads):
            for address in range(addresses):
                commands.append(verifier.CrossAccCommand(len(commands), len(commands), 0, address, 0))
                commands.append(verifier.CrossAccCommand(len(commands), len(commands), 1, address, address + 2))
        for address in range(addresses):
            commands.append(verifier.CrossAccCommand(len(commands), len(commands), 0, address, 0))
    return commands


class ClosedFormTests(unittest.TestCase):
    def test_h3_and_h24_fixed_values(self) -> None:
        h3 = verifier.workload_counts(3)
        self.assertEqual(
            (h3.phase, h3.relation_req, h3.weight_req, h3.final, h3.acc32),
            (52, 4050, 9216, 43_200, 43_200),
        )
        self.assertEqual((h3.cross_total, h3.cross_read, h3.cross_write), (259_200, 129_600, 129_600))
        h24 = verifier.workload_counts(24)
        self.assertEqual(
            (
                h24.phase, h24.relation_req, h24.weight_req, h24.final,
                h24.aligned_total, h24.acc32,
            ),
            (2929, 259_200, 589_824, 345_600, 2_043_648, 345_600),
        )
        self.assertEqual(
            (h24.cross_total, h24.cross_read, h24.cross_write),
            (16_588_800, 8_294_400, 8_294_400),
        )


class FrameAndSummaryTests(unittest.TestCase):
    def test_frozen_frame_and_known_digests(self) -> None:
        payload = tuple(range(1, 11))
        frame = verifier.encode_summary_frame(
            verifier.DOMAIN_TAG, 2, 5, "tb.u_mem", 7, 9, payload
        )
        parsed = verifier.parse_summary_frame(frame)
        self.assertEqual(parsed.payload, payload)
        self.assertEqual(parsed.instance_path, "tb.u_mem")
        self.assertEqual(len(frame), 2 + 23 + 6 + 8 + 18 + 80)
        self.assertEqual(verifier.rolling64_fnv1a(frame), 0x69D9B45A6A9EF5B3)
        self.assertEqual(verifier.rolling64_djb2xor(frame), 0x3963E6D98062E31B)

    def test_endian_and_field_width_fail_closed(self) -> None:
        payload = tuple(range(10))
        frame = verifier.encode_summary_frame("D", 2, 1, "i", 0, 0x0102030405060708, payload)
        cycle_offset = 2 + 1 + 6 + 1 + 8
        tampered = bytearray(frame)
        tampered[cycle_offset : cycle_offset + 8] = struct.pack(">Q", 0x0102030405060708)
        self.assertNotEqual(verifier.parse_summary_frame(bytes(tampered)).cycle, 0x0102030405060708)
        with self.assertRaises(verifier.ContractError):
            verifier.encode_summary_frame("D", 2, 1, "i", 1 << 64, 0, payload)
        with self.assertRaises(verifier.ContractError):
            verifier.encode_summary_frame("D", 2, 1, "i", 0, 0, payload[:-1])
        tampered = bytearray(frame)
        tampered[-82:-80] = struct.pack("<H", 72)
        with self.assertRaises(verifier.ContractError):
            verifier.parse_summary_frame(bytes(tampered))

    def test_summary_swap_duplicate_delete_and_rebind(self) -> None:
        original = [make_event(0, 10, 1, 2, 3), make_event(1, 11, 4, 5, 6)]
        resource = make_resource("RELATION_REQ_ACCEPT", "tb.u_main", original)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "main.csv"
            write_main_summary(path, {resource.name: resource})
            summary = verifier.parse_ordered_summary(path)
            verifier.verify_summary_resource(summary, resource.name, original, expected_instance_path="tb.u_main")
            mutations = (
                [make_event(0, 10, 4, 5, 6), make_event(1, 11, 1, 2, 3)],
                [make_event(0, 10, 1, 2, 3), make_event(1, 11, 1, 2, 3)],
                [make_event(0, 10, 1, 2, 3)],
            )
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    with self.assertRaises(verifier.ContractError):
                        verifier.verify_summary_resource(summary, resource.name, mutation)
            with self.assertRaises(verifier.ContractError):
                verifier.verify_summary_resource(summary, resource.name, original, expected_instance_path="tb.rebound")

        req = make_resource("RELATION_REQ_ACCEPT", "tb.u", original)
        rsp = make_resource("RELATION_RSP_ACCEPT", "tb.u", original)
        self.assertNotEqual((req.digest0, req.digest1), (rsp.digest0, rsp.digest1))

    def test_big_endian_digest_is_rejected(self) -> None:
        events = [make_event(0, 0x0102030405060708, 1, 2, 3)]
        resource = make_resource("RELATION_REQ_ACCEPT", "tb.u_main", events)
        frame = verifier.encode_summary_frame(
            verifier.DOMAIN_TAG, 2, 0, "tb.u_main", 0, events[0].cycle, events[0].payload
        )
        cycle_offset = 2 + len(verifier.DOMAIN_TAG) + 6 + len("tb.u_main") + 8
        wrong = bytearray(frame)
        wrong[cycle_offset : cycle_offset + 8] = struct.pack(">Q", events[0].cycle)
        wrong0 = verifier.rolling64_fnv1a(bytes(wrong))
        wrong1 = verifier.rolling64_djb2xor(bytes(wrong))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "main.csv"
            write_main_summary(path, {resource.name: resource})
            text = path.read_text(encoding="ascii")
            text = text.replace(f"{resource.digest0:016x},{resource.digest1:016x}", f"{wrong0:016x},{wrong1:016x}")
            path.write_text(text, encoding="ascii")
            summary = verifier.parse_ordered_summary(path)
            with self.assertRaises(verifier.ContractError):
                verifier.verify_summary_resource(summary, resource.name, events)

    def test_lower_summary_glob_cardinality_parent_and_common_projection(self) -> None:
        events = [make_event(0, 4, 1, 0, 0x55), make_event(1, 5, 0, 0, 0)]
        target = "tb.u_executor.u_cross_head_accumulator"
        cross = make_resource("CROSS_ACC_COMMAND", target, events)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main_path = root / "main.csv"
            write_main_summary(main_path, {cross.name: cross})
            main = verifier.parse_ordered_summary(main_path)
            lower_path = root / "cross.one.csv"
            write_lower_summary(lower_path, verifier.CROSS_SUMMARY_SCHEMA, cross)
            lower = verifier.parse_single_observer_summary_glob(
                root / "cross.*.csv",
                expected_schema=verifier.CROSS_SUMMARY_SCHEMA,
                expected_target_instance=target,
            )
            verifier.compare_summary_resources(main, lower, ["CROSS_ACC_COMMAND"])
            verifier.verify_cross_summary_pair(
                main, lower, heads=1, output_tiles=1, addresses_per_tile=1
            )
            shutil.copyfile(lower_path, root / "cross.two.csv")
            with self.assertRaises(verifier.ContractError):
                verifier.parse_single_observer_summary_glob(
                    root / "cross.*.csv", expected_schema=verifier.CROSS_SUMMARY_SCHEMA
                )
            bad = root / "bad.csv"
            write_lower_summary(
                bad,
                verifier.CROSS_SUMMARY_SCHEMA,
                cross,
                observer="tb.other.u_observer",
                target=target,
            )
            bad_summary = verifier.parse_ordered_summary(bad)
            with self.assertRaises(verifier.ContractError):
                verifier.validate_observer_summary_binding(
                    bad_summary, expected_schema=verifier.CROSS_SUMMARY_SCHEMA
                )

    def test_cross_ledger_delete_tamper_and_phase_swap(self) -> None:
        target = "tb.u_executor.u_cross_head_accumulator"
        events = [make_event(0, 4, 1, 0, 7), make_event(1, 5, 0, 0, 0)]
        cross = make_resource("CROSS_ACC_COMMAND", target, events)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main_path = root / "main.csv"
            lower_path = root / "lower.csv"
            write_main_summary(main_path, {cross.name: cross})
            write_lower_summary(lower_path, verifier.CROSS_SUMMARY_SCHEMA, cross)
            main_text = main_path.read_text(encoding="ascii")

            deleted = root / "deleted.csv"
            deleted.write_text(
                "\n".join(
                    line for line in main_text.splitlines()
                    if not line.startswith("L,CROSS_ACC_PROTOCOL_LEDGER,")
                ) + "\n",
                encoding="ascii",
            )
            with self.assertRaises(verifier.ContractError):
                verifier.parse_ordered_summary(deleted)

            tampered = root / "tampered.csv"
            tampered.write_text(
                main_text.replace("L,CROSS_ACC_PROTOCOL_LEDGER,2,1,1,", "L,CROSS_ACC_PROTOCOL_LEDGER,2,0,2,"),
                encoding="ascii",
            )
            tampered_main = verifier.parse_ordered_summary(tampered)
            lower = verifier.parse_ordered_summary(lower_path)
            with self.assertRaises(verifier.ContractError):
                verifier.verify_cross_summary_pair(
                    tampered_main, lower, heads=1, output_tiles=1, addresses_per_tile=1
                )

            main = verifier.parse_ordered_summary(main_path)
            ledger = main.cross_protocol_ledger
            self.assertIsNotNone(ledger)
            swapped = verifier.summarize_cross_protocol_order(target, [(0, 0), (1, 0)])
            exchanged = root / "exchanged.csv"
            exchanged.write_text(
                main_text.replace(
                    f"{ledger.digest0:016x},{ledger.digest1:016x}",
                    f"{swapped.digest0:016x},{swapped.digest1:016x}",
                ),
                encoding="ascii",
            )
            exchanged_main = verifier.parse_ordered_summary(exchanged)
            with self.assertRaises(verifier.ContractError):
                verifier.verify_cross_summary_pair(
                    exchanged_main, lower, heads=1, output_tiles=1, addresses_per_tile=1
                )

    def test_tcfm_ledger_and_main_low_projection_match(self) -> None:
        target = "tb.u_tcfm5"
        topology = verifier.expected_tcfm5_topology(0, 7, 7, 0b11111)
        event = make_event(
            0,
            9,
            7 * 15 + 7,
            2,
            topology.expected_mask,
            topology.expected_mask,
            *topology.bank_addresses,
        )
        resource = make_resource("TCFM5_TERM_COMMIT", target, [event])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main_path = root / "main.csv"
            low_path = root / "tcfm.csv"
            write_main_summary(main_path, {resource.name: resource})
            write_lower_summary(low_path, verifier.TCFM5_SUMMARY_SCHEMA, resource)
            main = verifier.parse_ordered_summary(main_path)
            low = verifier.parse_ordered_summary(low_path)
            verifier.validate_observer_summary_binding(
                low,
                expected_schema=verifier.TCFM5_SUMMARY_SCHEMA,
                expected_target_instance=target,
            )
            verifier.verify_tcfm5_summary_pair(main, low)
            text = low_path.read_text(encoding="ascii")
            updates = topology.expected_mask.bit_count()
            low_path.write_text(
                text.replace(
                    f"L,TCFM5_TERM_LEDGER,1,{updates},0",
                    f"L,TCFM5_TERM_LEDGER,1,{updates - 1},0",
                ),
                encoding="ascii",
            )
            with self.assertRaises(verifier.ContractError):
                verifier.verify_tcfm5_summary_pair(
                    main, verifier.parse_ordered_summary(low_path)
                )


class IdentityTraceStreamingTests(unittest.TestCase):
    def write_trace(self, path: Path, *, omit_final: bool = False) -> None:
        rows = [
            [0, "tile_start", 0, -1, -1, -1, -1, 0, 0, "rtl_handshake", "-"],
            [1, "weight_accept", 0, 0, -1, 2, 3, 0, 0, "rtl_handshake", "-"],
            [2, "weight_response_accept", 0, 0, -1, 2, 3, 0, 0, "rtl_handshake", "0"],
            [3, "relation_accept", 0, 0, 5, -1, -1, 0, 0, "rtl_handshake", "-"],
            [4, "relation_response_accept", -1, 0, 5, -1, -1, 0, 0, "rtl_handshake", "0"],
        ]
        if not omit_final:
            rows.append([5, "final_accept", 0, -1, 6, -1, 7, 0, 0, "rtl_handshake", "0"])
        rows.append([6, "tile_done", 0, -1, -1, -1, -1, 0, 0, "rtl_handshake", "-"])
        with path.open("w", newline="", encoding="ascii") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(verifier.IDENTITY_TRACE_COLUMNS)
            writer.writerows(rows)

    def test_small_trace_streams_to_five_summary_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity_trace.csv"
            self.write_trace(path)
            instances = {name: "tb.u_main" for name in verifier.ALIGNED_RESOURCES}
            audit = verifier.stream_identity_trace_aligned_resources(path, instances)
            self.assertEqual(audit.rows_read, 7)
            self.assertEqual([audit.resources[name].count for name in verifier.ALIGNED_RESOURCES], [1] * 5)
            rsp = audit.resources["RELATION_RSP_ACCEPT"]
            self.assertEqual(rsp.first_anchor, (0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0))
            main_path = Path(temporary) / "main.csv"
            write_main_summary(main_path, dict(audit.resources))
            main = verifier.parse_ordered_summary(main_path)
            for name in verifier.ALIGNED_RESOURCES:
                self.assertEqual(main.resources[name], audit.resources[name])

            deleted = Path(temporary) / "identity_deleted.csv"
            self.write_trace(deleted, omit_final=True)
            deleted_audit = verifier.stream_identity_trace_aligned_resources(deleted, instances)
            self.assertEqual(deleted_audit.resources["FINAL_ACCEPT"].count, 0)
            self.assertNotEqual(
                deleted_audit.resources["FINAL_ACCEPT"], audit.resources["FINAL_ACCEPT"]
            )


class PhaseAndEnumTests(unittest.TestCase):
    def test_phase_ledger_order_duration_identity_and_cardinality(self) -> None:
        contract, _parsed, _report = verifier.load_and_verify_state_role_contract(ROOT)
        roles = verifier.head_phase_roles_from_state_contract(contract)
        with tempfile.TemporaryDirectory() as temporary:
            good = Path(temporary) / "good.csv"
            write_phase_ledger(good)
            ledger = verifier.parse_phase_interval_ledger(good)
            verifier.validate_phase_interval_ledger(
                ledger, verifier.PhaseIdentity(2, 3, 4), roles
            )
            bad = Path(temporary) / "bad.csv"
            write_phase_ledger(bad, disorder=True)
            with self.assertRaises(verifier.ContractError):
                verifier.validate_phase_interval_ledger(
                    verifier.parse_phase_interval_ledger(bad),
                    verifier.PhaseIdentity(2, 3, 4),
                    roles,
                )
            text = good.read_text(encoding="ascii").replace(",10,10,1,RTL_DIRECT", ",10,10,2,RTL_DIRECT", 1)
            bad.write_text(text, encoding="ascii")
            with self.assertRaises(verifier.ContractError):
                verifier.parse_phase_interval_ledger(bad)

    def test_phase_ledger_streams_against_identity_trace_and_tamper(self) -> None:
        contract, _parsed, _report = verifier.load_and_verify_state_role_contract(ROOT)
        identity = verifier.PhaseIdentity(2, 3, 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger_path = root / "phase.csv"
            trace_path = root / "identity.csv"
            write_phase_ledger(ledger_path)
            write_phase_identity_trace(trace_path)
            ledger = verifier.parse_phase_interval_ledger(ledger_path)
            audit = verifier.stream_compare_phase_ledger_to_identity_trace(
                ledger, trace_path, contract, identity
            )
            self.assertEqual(audit.intervals_compared, 8)

            real_order = root / "identity_real_same_cycle_order.csv"
            write_phase_identity_trace(real_order, tile_done_before_exit_state=True)
            real_order_audit = verifier.stream_compare_phase_ledger_to_identity_trace(
                ledger, real_order, contract, identity
            )
            self.assertEqual(real_order_audit.intervals_compared, 8)

            original = ledger_path.read_text(encoding="ascii")
            mutations = {
                "cycle": original.replace(",HEAD_WEIGHT,10,10,1,", ",HEAD_WEIGHT,9,10,2,", 1),
                "role": original.replace("HEAD_WEIGHT", "HEAD_FRONTEND", 1),
                "identity": original.replace("P,0,2,3,4,", "P,0,9,3,4,", 1),
            }
            for label, text in mutations.items():
                with self.subTest(label=label):
                    bad = root / f"bad_{label}.csv"
                    bad.write_text(text, encoding="ascii")
                    with self.assertRaises(verifier.ContractError):
                        verifier.stream_compare_phase_ledger_to_identity_trace(
                            verifier.parse_phase_interval_ledger(bad),
                            trace_path,
                            contract,
                            identity,
                        )

            bad_trace = root / "bad_identity.csv"
            write_phase_identity_trace(bad_trace, head_done_head=1)
            with self.assertRaises(verifier.ContractError):
                verifier.stream_compare_phase_ledger_to_identity_trace(
                    ledger, bad_trace, contract, identity
                )

    def make_contract_tree(self, temporary: str) -> tuple[Path, Path, Path]:
        root = Path(temporary)
        (root / "contracts").mkdir()
        (root / "rtl_qfit").mkdir()
        rtl = root / verifier.EXPECTED_RTL_SOURCE
        contract_path = root / "contracts/local5_phase_state_roles_v2.json"
        shutil.copyfile(ROOT / verifier.EXPECTED_RTL_SOURCE, rtl)
        shutil.copyfile(ROOT / "contracts/local5_phase_state_roles_v2.json", contract_path)
        return root, rtl, contract_path

    def test_enum_width_symbol_value_and_sha_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, rtl, contract_path = self.make_contract_tree(temporary)
            verifier.load_and_verify_state_role_contract(root)
            source = rtl.read_text(encoding="ascii").replace("ST_ERROR\n", "ST_TAMPERED\n", 1)
            rtl.write_text(source, encoding="ascii")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["rtl_source_sha256"] = hashlib.sha256(rtl.read_bytes()).hexdigest()
            contract_path.write_text(json.dumps(contract, ensure_ascii=True), encoding="ascii")
            with self.assertRaisesRegex(verifier.ContractError, "ordered RTL enum"):
                verifier.load_and_verify_state_role_contract(root)

        with tempfile.TemporaryDirectory() as temporary:
            root, rtl, contract_path = self.make_contract_tree(temporary)
            source = rtl.read_text(encoding="ascii").replace(
                "typedef enum logic [4:0]", "typedef enum logic [5:0]", 1
            )
            rtl.write_text(source, encoding="ascii")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["rtl_source_sha256"] = hashlib.sha256(rtl.read_bytes()).hexdigest()
            contract_path.write_text(json.dumps(contract, ensure_ascii=True), encoding="ascii")
            with self.assertRaisesRegex(verifier.ContractError, "enum width"):
                verifier.load_and_verify_state_role_contract(root)


class OracleTests(unittest.TestCase):
    def test_cross_acc_protocol_count_and_per_address_phase(self) -> None:
        commands = expected_cross_commands(2, 2, 3)
        audit = verifier.verify_cross_acc_protocol(
            commands, heads=2, output_tiles=2, addresses_per_tile=3
        )
        self.assertEqual((audit.total, audit.reads, audit.writes), (24, 12, 12))
        self.assertEqual(
            verifier.cross_acc_scalar_address(1, 14, 14, 31), 14_399
        )
        for label, mutation in (
            ("swap", [commands[1], commands[0], *commands[2:]]),
            ("duplicate", [commands[0], *commands]),
            ("delete", commands[:-1]),
        ):
            with self.subTest(label=label):
                with self.assertRaises(verifier.ContractError):
                    verifier.verify_cross_acc_protocol(
                        mutation, heads=2, output_tiles=2, addresses_per_tile=3
                    )

    def test_cross_acc_runtime_address_order_is_explicit(self) -> None:
        order = (2, 0, 1)
        commands: list[verifier.CrossAccCommand] = []
        for rw, address in [*( (1, value) for value in order), *( (0, value) for value in order)]:
            commands.append(verifier.CrossAccCommand(len(commands), len(commands), rw, address, 0))
        verifier.verify_cross_acc_protocol(
            commands,
            heads=1,
            output_tiles=1,
            addresses_per_tile=3,
            address_order_for_tile=lambda _tile: order,
        )

    def test_tcfm5_mask_and_bank_address_topology(self) -> None:
        topology = verifier.expected_tcfm5_topology(1, 7, 8, 0b11111)
        event = verifier.Tcfm5Projection(
            0,
            10,
            1 * 225 + 7 * 15 + 8,
            4,
            topology.expected_mask,
            topology.expected_mask,
            topology.bank_addresses,
        )
        verifier.verify_tcfm5_projection(
            event, plane=1, y=7, x=8, destination_mask=0b11111
        )
        bad = verifier.Tcfm5Projection(
            event.sequence,
            event.cycle,
            event.source,
            event.lane,
            event.expected_mask,
            event.actual_mask ^ 1,
            event.bank_addresses,
        )
        with self.assertRaises(verifier.ContractError):
            verifier.verify_tcfm5_projection(
                bad, plane=1, y=7, x=8, destination_mask=0b11111
            )
        boundary = verifier.expected_tcfm5_topology(0, 0, 0, 0b00001)
        self.assertEqual(boundary.bank_addresses, (0, 0, 3, 0, 0))
        with self.assertRaises(verifier.ContractError):
            verifier.expected_tcfm5_topology(0, 0, 0, 0b00100)


class PayloadAndAdmissionTests(unittest.TestCase):
    def test_payload_st_size_exclusions_hardlinks_and_compression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "payload.bin"
            data.write_bytes(b"0123456789")
            os.link(data, root / "payload.hardlink")
            (root / "build").mkdir()
            (root / "source").mkdir()
            (root / "build/ignored.bin").write_bytes(b"x" * 100)
            (root / "source/ignored.bin").write_bytes(b"y" * 100)
            compressed = root / "payload.gz"
            with gzip.open(compressed, "wb") as handle:
                handle.write(b"z" * 10_000)
            audit = verifier.audit_evidence_payload(root)
            self.assertEqual(audit.bytes, 10 + compressed.stat().st_size)
            self.assertEqual(audit.regular_files, 3)
            self.assertEqual(audit.unique_inodes, 2)

    def test_excluded_hardlink_cannot_hide_payload_and_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "build").mkdir()
            source = root / "build/excluded.bin"
            source.write_bytes(b"1234567")
            os.link(source, root / "visible.bin")
            self.assertEqual(verifier.audit_evidence_payload(root).bytes, 7)
            (root / "link.bin").symlink_to(root / "visible.bin")
            with self.assertRaises(verifier.ContractError):
                verifier.audit_evidence_payload(root)

    def admitted_receipt(self) -> dict[str, object]:
        identity = {key: value for key, value in zip(verifier.IDENTITY_KEYS, (2, 0, 0, 249, 24))}
        return {
            "schema": "run_v2",
            "status": "PASS_NOT_G0",
            "requested_identity": dict(identity),
            "actual_identity": dict(identity),
            "identity_status": "MATCH",
            "receipts": {"summary": {"status": "PASS"}, "acc32": {"status": "PASS"}},
        }

    def test_positive_admission_precedes_denylist(self) -> None:
        digest = "a" * 64
        required = {"summary": ("PASS",), "acc32": ("PASS",)}
        receipt = self.admitted_receipt()
        result = verifier.verify_receipt_admission(
            receipt,
            allowed_schemas=("run_v2",),
            allowed_statuses=("PASS_NOT_G0",),
            required_receipts=required,
            package_digest=digest,
        )
        self.assertEqual(result["denylist_status"], "CLEAR")
        denylist = {"entries": [{"package_digest": digest}]}
        with self.assertRaisesRegex(verifier.ContractError, "denylisted"):
            verifier.verify_receipt_admission(
                receipt,
                allowed_schemas=("run_v2",),
                allowed_statuses=("PASS_NOT_G0",),
                required_receipts=required,
                package_digest=digest,
                denylist=denylist,
            )
        receipt["schema"] = "bad"
        with self.assertRaisesRegex(verifier.ContractError, "schema"):
            verifier.verify_receipt_admission(
                receipt,
                allowed_schemas=("run_v2",),
                allowed_statuses=("PASS_NOT_G0",),
                required_receipts=required,
                package_digest=digest,
                denylist=denylist,
            )


class StaticPreflightTests(unittest.TestCase):
    def test_repository_static_preflight_passes(self) -> None:
        report = verifier.run_static_preflight(ROOT)
        self.assertEqual(report["status"], "STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION")
        self.assertEqual(report["blockers"], [])


if __name__ == "__main__":
    unittest.main()
