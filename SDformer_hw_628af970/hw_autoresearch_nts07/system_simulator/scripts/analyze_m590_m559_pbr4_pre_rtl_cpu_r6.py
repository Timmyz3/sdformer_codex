#!/usr/bin/env python3
"""M590 r6 repair of the M559 PBR4 pre-RTL CPU analyzer.

This immutable source is source-review-only until a later, independent launch
chain is complete.  Production mode is deliberately fail-closed on canonical
sealed inputs.  ``--self-test-static`` uses only synthetic scalar goldens.
"""

import argparse
from collections import Counter, deque
from decimal import Decimal, getcontext
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


getcontext().prec = 40

ARCHITECTURES = ("A1-SC8", "A1-ISO8", "A1-OSG", "PBR4")
A1_POINTS = ("A1-SC8", "A1-ISO8", "A1-OSG")
A1_TIE_ORDER = ("A1-OSG", "A1-SC8", "A1-ISO8")
PRIMARY_CLASSES = (
    "productive_source_or_group_issue", "source_scan_frontier_stall",
    "atomic_ingress_backpressure", "join_context_full",
    "phase_bank_conflict", "weight_refill_first_latency",
    "weight_refill_accepted_beat", "weight_refill_link_stall",
    "weight_L4_wait", "O8_full", "psum_1RW_conflict", "psum_L4_wait",
    "pending_write_RAW", "directory_1RW_conflict",
    "restore_first_latency", "restore_accepted_beat",
    "restore_link_stall", "writeback_first_latency",
    "writeback_accepted_beat", "writeback_link_stall", "final_zero_build",
    "final_output_first_latency", "final_output_accepted_beat",
    "final_output_sink_stall", "directory_set_or_clear_RMW",
    "block_transition_drain", "time_epoch_directory_clear",
)
LAYERS = (
    (0, 1536, 384, 15, 20, 30, 40, 4),
    (1, 770, 192, 30, 40, 60, 80, 2),
    (2, 386, 96, 60, 80, 120, 160, 1),
    (3, 194, 96, 120, 160, 240, 320, 1),
)
EXPECTED_RAW_BITS = 696_240_000
EXPECTED_REPLAY_BITS = 926_880_000
EXPECTED_DENSE_DESTINATIONS = 11_040_000
EXPECTED_ROWS = 1_600
MODELED_LOGICAL_BYTES = 239_636
LOGICAL_BUDGET_BYTES = 245_760
MANDATORY_EVENT_KEYS = (
    "source_scan_bits", "source_scan_active_reads", "source_scan_logical_bytes",
    "source_scan_padded_transport_bytes", "source_scan_base_issue_cycles",
    "source_scan_frontier_stall_cycles", "frontier_tokens_accepted",
    "numeric_source_events", "legal_descriptors", "descriptors_accepted",
    "descriptors_retired", "bundles_accepted", "groups", "weight_active_reads",
    "weight_refill_requests", "weight_refill_beats", "weight_refill_bytes",
    "psum_reads", "psum_writes", "psum_read_bytes", "psum_write_bytes",
    "persistent_read_commands", "persistent_read_beats", "persistent_read_bytes",
    "persistent_write_commands", "persistent_write_beats", "persistent_write_bytes",
    "dense_destinations", "zero_vectors", "nonzero_vectors", "commit_count",
    "output_logical_bytes", "output_padded_bytes", "output_accepted_beats",
    "output_sink_stalls", "directory_queries", "directory_set_rmw",
    "directory_clear_rmw", "directory_rmw", "epoch_directory_clear_words",
    "resident_hits", "resident_misses", "resident_evictions", "dirty_evictions",
    "restore_bytes", "writeback_bytes", "max_occupancy_ingress",
    "max_occupancy_contexts", "max_occupancy_context_slots", "max_occupancy_O8",
    "max_occupancy_FIFO4", "max_occupancy_pending_write",
    "max_occupancy_resident_destinations",
)
EXECUTION_CONTRACT_REL = "contracts/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r4_20260828.json"
EXECUTION_CONTRACT_SHA256 = "6a8a76f8d71188a115a44e9f0a6f0af2be897973d5c8eaa16d62b4e1fffbd56c"
SOURCE_CONTRACT_REL = "contracts/m590_m559_pbr4_pre_rtl_cpu_runner_source_contract_r3_20260828.json"
FUTURE_SCHEMA_REL = "reviews/m591_m590_m559_pbr4_pre_rtl_cpu_runner_source_author_handoff_r3_20260828/future_runner_schema_r6.json"
CONTRACT_STATIC_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_contract_static_hammer_r4_20260828"
SOURCE_STATIC_REL = "reviews/m590_m559_pbr4_pre_rtl_cpu_runner_static_hammer_r1_20260828"
LAUNCH_CANDIDATE_REL = "reviews/m590_m559_pbr4_pre_rtl_cpu_launch_candidate_hammer_r1_20260828"
FINAL_RELEASE_REL = "reviews/m590_m559_pbr4_pre_rtl_cpu_final_release_hammer_r1_20260828"
AUTH_REL = "contracts/m590_m559_pbr4_pre_rtl_cpu_final_launch_authorization_r1_20260828.json"
WRAPPER_REL = "system_simulator/scripts/launch_m590_m559_pbr4_pre_rtl_cpu_r6_authorized_r1.sh"
WRAPPER_REVIEW_REL = "reviews/m590_m559_pbr4_pre_rtl_cpu_post_auth_launcher_static_release_hammer_r1_20260828"
RUNNER_REL = "system_simulator/scripts/run_m590_m559_pbr4_pre_rtl_cpu_r6_exact_sha.sh"
ANALYZER_REL = "system_simulator/scripts/analyze_m590_m559_pbr4_pre_rtl_cpu_r6.py"
M511_REL = "system_handoff/outgoing/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827"
M511_VERIFY_REL = "results/m511_h67_ep35_convtranspose_payload_verify_r1_20260827"
WEIGHT_REL = "system_handoff/outgoing/m578_h67_ep35_decoder_signed_int8_weights_r2_20260828"
RESULT_REL = "results/m590_m559_pbr4_pre_rtl_cpu_r6_20260828"
ATTEMPT_REL = "results/.m590_m559_pbr4_pre_rtl_cpu_r6_attempt_consumed"
TERMINAL_GOLDEN_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_contract_author_handoff_r4_20260828/terminal_goldens.json"


class ContractFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def reject(token):
        raise ContractFailure("non-finite JSON token: " + token)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject)


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    require(member.parts and not member.is_absolute() and ".." not in member.parts and
            member.parts[0] not in ("", ".") and member.as_posix() == name,
            "unsafe or noncanonical sealed member: " + name)
    return member


def verify_directory(directory: Path) -> Mapping[str, object]:
    require(directory.is_dir() and not directory.is_symlink(),
            "missing/symlinked sealed directory: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(),
            "missing sealed-directory identity")
    members = {}
    resolved_members = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "malformed SHA256SUMS line")
        expected, name = fields
        require(name not in members and name not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "duplicate/recursive seal member")
        member = safe_member(name)
        path = directory.joinpath(*member.parts)
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected, "sealed member mismatch: " + name)
        resolved = path.resolve()
        require(resolved not in resolved_members and directory.resolve() in resolved.parents,
                "aliased or escaped sealed member")
        resolved_members.add(resolved)
        members[name] = expected
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(fields) == 2 and fields[1] == "SHA256SUMS" and
            fields[0] == sha256(manifest), "outer seal mismatch")
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "unsealed nested symlink")
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            actual.add(path.relative_to(directory).as_posix())
    require(actual == set(members), "sealed member-set mismatch")
    return {"members": members, "manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def verify_single_double_seal(member: Path) -> Tuple[str, str, str]:
    sidecar = Path(str(member) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(member.is_file() and sidecar.is_file() and outer.is_file() and
            not any(path.is_symlink() for path in (member, sidecar, outer)),
            "missing/symlinked member double seal")
    row = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(row) == 2 and row[1] == member.name and row[0] == sha256(member),
            "member sidecar mismatch")
    seal = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(seal) == 2 and seal[1] == sidecar.name and
            seal[0] == sha256(sidecar), "member outer seal mismatch")
    return sha256(member), sha256(sidecar), sha256(outer)


def write_directory_seal(directory: Path) -> None:
    members = sorted(path.relative_to(directory) for path in directory.rglob("*")
                     if path.is_file() and not path.is_symlink() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(directory / member), member.as_posix()) for member in members),
        encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def ratio(numerator: int, denominator: int) -> str:
    require(denominator > 0, "zero ratio denominator")
    return format(Decimal(numerator) / Decimal(denominator), ".12f")


def xorshift32(value: int) -> int:
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= value >> 17
    value ^= (value << 5) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def _apply_linear(columns: Sequence[int], value: int) -> int:
    result = 0
    index = 0
    while value:
        if value & 1:
            result ^= columns[index]
        value >>= 1
        index += 1
    return result & 0xFFFFFFFF


def advance_xorshift32(value: int, count: int) -> int:
    require(count >= 0, "negative xorshift advance")
    power = [xorshift32(1 << bit) for bit in range(32)]
    while count:
        if count & 1:
            value = _apply_linear(power, value)
        power = [_apply_linear(power, column) for column in power]
        count >>= 1
    return value & 0xFFFFFFFF


def wrap24(value: int) -> int:
    value &= (1 << 24) - 1
    return value - (1 << 24) if value & (1 << 23) else value


def int24_bytes(value: int) -> bytes:
    return int(value & 0xFFFFFF).to_bytes(3, byteorder="little", signed=False)


class CycleLedger:
    def __init__(self, architecture: str, capture_events: bool = False):
        self.architecture = architecture
        self.classes = Counter()
        self.events = Counter()
        self.cycle = 0
        self.ready_state = 0x53454217
        self.cycle_hash = hashlib.sha256()
        self.source_hash = hashlib.sha256()
        self.descriptor_accept_hash = hashlib.sha256()
        self.descriptor_retire_hash = hashlib.sha256()
        self.frontier_hash = hashlib.sha256()
        self.weight_hash = hashlib.sha256()
        self.group_hash = hashlib.sha256()
        self.rmw_hash = hashlib.sha256()
        self.commit_hash = hashlib.sha256()
        self.data_hash = hashlib.sha256()
        self.functional_mismatches = 0
        self.protocol_mismatches = 0
        self.transition_mismatches = 0
        self.conservation_mismatches = 0
        self.capture_events = capture_events
        self.event_trace = []
        self.terminal_hash = hashlib.sha256()
        self.directory_clear_hash = hashlib.sha256()
        # These latches are part of the executable resource model.  A caller
        # must never manufacture a zero stall count by passing a constant
        # predicate: contention is derived from the current owner set.
        self.resource_owners = set()

    @property
    def total_cycles(self):
        return sum(self.classes.values())

    @property
    def sink_ready(self):
        # Frozen M542 transcript: not-ready only when bits[2:0] are 000.
        return bool(self.ready_state & 7)

    def step(self, primary: str, event: str, count: int = 1) -> None:
        require(primary in PRIMARY_CLASSES and count > 0,
                "illegal primary class/count")
        self.cycle_hash.update(canonical_json_bytes(
            [self.cycle, primary, event, count]))
        if self.capture_events:
            self.event_trace.extend([event] * count)
        self.classes[primary] += count
        self.cycle += count
        self.ready_state = advance_xorshift32(self.ready_state, count)
        require(self.total_cycles == self.cycle, "exclusive cycle conservation")

    def optional_stall(self, primary: str, event: str, predicate: bool) -> None:
        if predicate:
            self.step(primary, event)

    def acquire(self, resource: str, stall_class: str, event: str) -> None:
        busy = resource in self.resource_owners
        self.optional_stall(stall_class, event, busy)
        require(not busy, "resource re-entry after charged stall: " + resource)
        self.resource_owners.add(resource)

    def release(self, resource: str) -> None:
        require(resource in self.resource_owners,
                "release of unowned resource: " + resource)
        self.resource_owners.remove(resource)

    def observe_occupancy(self, name: str, value: int) -> None:
        require(value >= 0, "negative occupancy")
        key = "max_occupancy_" + name
        self.events[key] = max(int(self.events.get(key, 0)), int(value))


class Descriptor:
    __slots__ = ("source_channel", "source_y", "source_x", "kernel_y",
                 "kernel_x", "destination_y", "destination_x", "ordinal",
                 "event_last", "output_block", "last_possible_ordinal",
                 "numeric_value", "source_sign_bit")

    def __init__(self, source_channel: int, source_y: int, source_x: int,
                 kernel_y: int, kernel_x: int, destination_y: int,
                 destination_x: int, ordinal: int, event_last: bool,
                 output_block: int, last_possible_ordinal: int,
                 numeric_value: int = 1, source_sign_bit: int = 0):
        self.source_channel = source_channel
        self.source_y = source_y
        self.source_x = source_x
        self.kernel_y = kernel_y
        self.kernel_x = kernel_x
        self.destination_y = destination_y
        self.destination_x = destination_x
        self.ordinal = ordinal
        self.event_last = event_last
        self.output_block = output_block
        self.last_possible_ordinal = last_possible_ordinal
        require(numeric_value == 1 and source_sign_bit == 0,
                "malformed typed binary source")
        self.numeric_value = numeric_value
        self.source_sign_bit = source_sign_bit

    @property
    def kernel_index(self):
        return self.kernel_y * 3 + self.kernel_x

    @property
    def phase(self):
        return ((self.destination_y & 1) << 1) | (self.destination_x & 1)

    @property
    def bank(self):
        return (self.source_channel * 9 + self.kernel_index) & 7

    @property
    def cin_tile(self):
        return self.source_channel >> 4

    @property
    def destination(self):
        return (self.destination_y, self.destination_x)

    @property
    def identity(self):
        return [self.output_block, self.ordinal, self.source_channel,
                self.kernel_index, self.destination_y, self.destination_x,
                {"numeric_activity": self.numeric_value,
                 "source_sign_bit": self.source_sign_bit}]


def last_source_ordinal_for_destination(dy: int, dx: int, cin: int,
                                        hin: int, win: int) -> int:
    candidates = []
    for ky in range(3):
        numerator_y = dy - ky + 1
        if numerator_y % 2:
            continue
        y = numerator_y // 2
        if not 0 <= y < hin:
            continue
        for kx in range(3):
            numerator_x = dx - kx + 1
            if numerator_x % 2:
                continue
            x = numerator_x // 2
            if 0 <= x < win:
                candidates.append(((cin - 1) * hin + y) * win + x)
    require(candidates, "destination has no legal source")
    return max(candidates)


def event_taps(channel: int, y: int, x: int, cin: int, height: int,
               width: int, ordinal: int, output_block: int) -> List[Descriptor]:
    slots = ((0, 0), (0, 2), (2, 0), (2, 2),
             (0, 1), (2, 1), (1, 0), (1, 2), (1, 1))
    taps = []
    for ky, kx in slots:
        dy, dx = 2 * y + ky - 1, 2 * x + kx - 1
        if 0 <= dy < 2 * height and 0 <= dx < 2 * width:
            taps.append((ky, kx, dy, dx))
    require(len(taps) in (4, 6, 9), "illegal K3/S2 tap count")
    result = []
    for index, (ky, kx, dy, dx) in enumerate(taps):
        result.append(Descriptor(
            channel, y, x, ky, kx, dy, dx, ordinal,
            index == len(taps) - 1, output_block,
            last_source_ordinal_for_destination(dy, dx, cin, height, width)))
    return result


def scan_set_ordinals(path: Path, time: int, plane_bits: int) -> Iterator[int]:
    require(plane_bits % 8 == 0, "unaligned frozen bitplane")
    plane_bytes = plane_bits // 8
    require(path.stat().st_size == plane_bytes * 10,
            "bitpack byte length is not exact literal T10")
    with path.open("rb", buffering=1 << 20) as handle:
        handle.seek(time * plane_bytes)
        base = 0
        remaining = plane_bytes
        while remaining:
            chunk = handle.read(min(1 << 20, remaining))
            require(chunk, "short bitpack read")
            for byte_index, packed_byte in enumerate(chunk):
                value = packed_byte
                while value:
                    lsb = value & -value
                    bit = lsb.bit_length() - 1
                    yield base + byte_index * 8 + bit
                    value ^= lsb
            base += len(chunk) * 8
            remaining -= len(chunk)


class WeightTensor:
    def __init__(self, package: Path, row: Mapping[str, object]):
        self.layer = int(row["layer"])
        self.shape = tuple(int(value) for value in row["shape"])
        self.path = package.joinpath(*safe_member(str(row["relative_path"])).parts)
        require(row["dtype"] == "int8" and row["layout"] == "COUT_CIN_KY_KX",
                "weight dtype/layout drift")
        require(self.path.stat().st_size == self.shape[0] * self.shape[1] * 9,
                "weight byte length mismatch")
        require(str(row["sha256"]) == sha256(self.path), "weight record hash mismatch")
        self.handle = self.path.open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, cout: int, cin: int, ky: int, kx: int) -> int:
        require(0 <= cout < self.shape[0] and 0 <= cin < self.shape[1] and
                0 <= ky < 3 and 0 <= kx < 3, "weight index out of range")
        offset = (((cout * self.shape[1] + cin) * 3 + ky) * 3 + kx)
        value = self.mapping[offset]
        return value - 256 if value >= 128 else value

    def close(self) -> None:
        self.mapping.close()
        self.handle.close()


class WeightSet:
    def __init__(self, package: Path, manifest: Mapping[str, object]):
        require(manifest.get("schema") == "m578_h67_decoder_signed_int8_weights_v2" and
                manifest.get("checkpoint_sha256") ==
                "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
                "decoder weight manifest schema/checkpoint mismatch")
        rows = {int(row["layer"]): row for row in manifest["records"]}
        require(set(rows) == {0, 1, 2, 3} and len(manifest["records"]) == 4,
                "decoder weight record population mismatch")
        self.tensors = {}
        for layer, cin, cout, hin, win, hout, wout, blocks in LAYERS:
            row = rows[layer]
            require(tuple(int(value) for value in row["shape"]) == (cout, cin, 3, 3),
                    "decoder weight shape mismatch")
            self.tensors[layer] = WeightTensor(package, row)

    def get(self, layer: int, cout: int, descriptor: Descriptor) -> int:
        require(descriptor.numeric_value == 1 and descriptor.source_sign_bit == 0,
                "candidate typed-source drift")
        return self.tensors[layer].get(
            cout, descriptor.source_channel, descriptor.kernel_y,
            descriptor.kernel_x) * descriptor.numeric_value

    def close(self) -> None:
        for tensor in self.tensors.values():
            tensor.close()


class IndependentWeightTensor:
    """Independent raw-byte weight view for the direct reference domain.

    This deliberately does not call WeightTensor.get or WeightSet.get.  It has
    its own bounds, byte offset and signed decoding path so candidate address
    bugs cannot self-miter.
    """
    def __init__(self, package: Path, row: Mapping[str, object]):
        self.shape = tuple(int(value) for value in row["shape"])
        self.path = package.joinpath(*safe_member(str(row["relative_path"])).parts)
        require(row["dtype"] == "int8" and row["layout"] == "COUT_CIN_KY_KX",
                "reference weight dtype/layout drift")
        require(self.path.stat().st_size == self.shape[0] * self.shape[1] * 9 and
                str(row["sha256"]) == sha256(self.path),
                "reference weight byte/hash drift")
        self.handle = self.path.open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def direct_get(self, output_channel: int, source_channel: int,
                   kernel_y: int, kernel_x: int) -> int:
        require(0 <= output_channel < self.shape[0] and
                0 <= source_channel < self.shape[1] and
                0 <= kernel_y < 3 and 0 <= kernel_x < 3,
                "reference weight coordinate out of range")
        flat = output_channel
        flat = flat * self.shape[1] + source_channel
        flat = flat * 3 + kernel_y
        flat = flat * 3 + kernel_x
        raw = self.mapping[flat]
        return raw - 256 if raw >= 128 else raw

    def close(self) -> None:
        self.mapping.close()
        self.handle.close()


class IndependentWeightSet:
    def __init__(self, package: Path, manifest: Mapping[str, object]):
        rows = {int(row["layer"]): row for row in manifest["records"]}
        require(set(rows) == {0, 1, 2, 3} and len(manifest["records"]) == 4,
                "reference weight population drift")
        self.tensors = {}
        for layer, cin, cout, hin, win, hout, wout, blocks in LAYERS:
            row = rows[layer]
            require(tuple(int(value) for value in row["shape"]) ==
                    (cout, cin, 3, 3), "reference weight shape drift")
            self.tensors[layer] = IndependentWeightTensor(package, row)

    def direct_get(self, layer: int, output_channel: int, source_channel: int,
                   kernel_y: int, kernel_x: int) -> int:
        return self.tensors[layer].direct_get(
            output_channel, source_channel, kernel_y, kernel_x)

    def close(self) -> None:
        for tensor in self.tensors.values():
            tensor.close()


class DirectReferenceAccumulator:
    def __init__(self, layer: int, output_block: int, cout: int,
                 weights: IndependentWeightSet):
        self.layer = layer
        self.output_block = output_block
        self.cout = cout
        self.weights = weights
        self.vectors = {}

    def add_source(self, source_channel: int, source_y: int, source_x: int,
                   input_height: int, input_width: int) -> None:
        # Direct kernel-index order; does not reuse candidate event_taps().
        for kernel_index in range(9):
            kernel_y, kernel_x = divmod(kernel_index, 3)
            destination_y = 2 * source_y + kernel_y - 1
            destination_x = 2 * source_x + kernel_x - 1
            if not (0 <= destination_y < 2 * input_height and
                    0 <= destination_x < 2 * input_width):
                continue
            destination = (destination_y, destination_x)
            vector = self.vectors.setdefault(destination, [0] * 96)
            base = self.output_block * 96
            for lane in range(96):
                output_channel = base + lane
                if output_channel < self.cout:
                    weight = self.weights.direct_get(
                        self.layer, output_channel, source_channel,
                        kernel_y, kernel_x)
                    vector[lane] = wrap24(vector[lane] + weight)


def directory_index(layer: int, output_block: int, y: int, x: int) -> int:
    spec = LAYERS[layer]
    hout, wout = spec[5], spec[6]
    base = (0, 4800, 14400, 33600)[layer]
    index = base + ((output_block * hout + y) * wout + x)
    require(0 <= index < 110400, "directory index out of aperture")
    return index


class PsumStore:
    def __init__(self, ledger: CycleLedger, layer: int, output_block: int,
                 generation: int):
        self.ledger = ledger
        self.layer = layer
        self.output_block = output_block
        self.generation = generation
        self.slots = [[None for _ in range(128)] for _ in range(4)]
        self.victims = [0, 0, 0, 0]
        self.directory = set()
        self.backing = {}
        self.committed_destinations = set()

    def _find(self, phase: int, destination: Tuple[int, int]):
        for index, entry in enumerate(self.slots[phase]):
            if entry is not None and entry["destination"] == destination:
                return index, entry
        return None, None

    def _six_read_drain(self, prefix: str) -> None:
        self.ledger.step("productive_source_or_group_issue", prefix + "_READ_ISSUE", 6)
        self.ledger.step("psum_L4_wait", prefix + "_L4_DRAIN", 4)

    def _directory_rmw(self, event: str) -> None:
        self.ledger.acquire("directory_port", "directory_1RW_conflict",
                            event + "_DIRECTORY_1RW_CONFLICT")
        try:
            self.ledger.step("directory_set_or_clear_RMW", event + "_READ")
            self.ledger.step("directory_set_or_clear_RMW", event + "_WRITE")
            self.ledger.events["directory_rmw"] += 2
            if event == "DIR_SET":
                self.ledger.events["directory_set_rmw"] += 1
            elif event == "DIR_CLEAR":
                self.ledger.events["directory_clear_rmw"] += 1
        finally:
            self.ledger.release("directory_port")

    def _directory_query(self, event: str) -> None:
        self.ledger.acquire("directory_port", "directory_1RW_conflict",
                            event + "_DIRECTORY_1RW_CONFLICT")
        try:
            self.ledger.step("productive_source_or_group_issue", event)
            self.ledger.events["directory_queries"] += 1
        finally:
            self.ledger.release("directory_port")

    def _external_write(self, vector: Sequence[int], index: int) -> None:
        self.ledger.acquire("writeback_link", "writeback_link_stall",
                            "WB_LINK_STALL")
        try:
            self.ledger.step("writeback_first_latency", "WB_FIRST_LAT", 32)
            self.ledger.step("writeback_accepted_beat", "WB_ACCEPTED_BEAT", 3)
            self.ledger.events["writeback_bytes"] += 384
            self.ledger.events["persistent_write_commands"] += 1
            self.ledger.events["persistent_write_beats"] += 3
            self.ledger.events["persistent_write_bytes"] += 384
            self.backing[index] = list(vector)
            self.directory.add(index)
            self._directory_rmw("DIR_SET")
        finally:
            self.ledger.release("writeback_link")

    def _restore(self, index: int) -> List[int]:
        require(index in self.directory and index in self.backing,
                "directory/backing restore mismatch")
        self.ledger.acquire("restore_link", "restore_link_stall",
                            "RESTORE_LINK_STALL")
        try:
            self.ledger.step("restore_first_latency", "RESTORE_FIRST_LAT", 32)
            self.ledger.step("restore_accepted_beat", "RESTORE_ACCEPTED_BEAT", 3)
            self.ledger.step("productive_source_or_group_issue", "RESTORE_PSUM_WRITE", 6)
            self.ledger.events["restore_bytes"] += 384
            self.ledger.events["persistent_read_commands"] += 1
            self.ledger.events["persistent_read_beats"] += 3
            self.ledger.events["persistent_read_bytes"] += 384
            return list(self.backing[index])
        finally:
            self.ledger.release("restore_link")

    def _evict(self, phase: int, slot: int) -> None:
        entry = self.slots[phase][slot]
        if entry is None:
            return
        if entry["dirty"]:
            self._six_read_drain("EVICT")
            index = directory_index(self.layer, self.output_block,
                                    entry["destination"][0], entry["destination"][1])
            self._external_write(entry["vector"], index)
            self.ledger.events["dirty_evictions"] += 1
        self.ledger.events["resident_evictions"] += 1
        self.slots[phase][slot] = None

    def ensure(self, destination: Tuple[int, int], directory_already_queried: bool = False) -> Mapping[str, object]:
        phase = ((destination[0] & 1) << 1) | (destination[1] & 1)
        slot, entry = self._find(phase, destination)
        if entry is not None:
            self.ledger.events["resident_hits"] += 1
            return entry
        self.ledger.events["resident_misses"] += 1
        if not directory_already_queried:
            self._directory_query("DIRECTORY_QUERY")
        free = next((index for index, value in enumerate(self.slots[phase])
                     if value is None), None)
        if free is None:
            free = self.victims[phase]
            self.victims[phase] = (free + 1) % 128
            self._evict(phase, free)
        index = directory_index(self.layer, self.output_block,
                                destination[0], destination[1])
        if index in self.directory:
            vector = self._restore(index)
        else:
            self.ledger.step("productive_source_or_group_issue", "PSUM_ZERO_FILL", 6)
            vector = [0] * 96
        entry = {"destination": destination, "vector": vector, "dirty": False,
                 "generation": self.generation}
        self.slots[phase][free] = entry
        self.ledger.observe_occupancy(
            "resident_destinations",
            sum(value is not None for bank in self.slots for value in bank))
        return entry

    def update_slice(self, destination: Tuple[int, int], output_slice: int,
                     contributors: Sequence[Descriptor], weights: WeightSet,
                     cout: int) -> None:
        entry = self.ensure(destination)
        base = self.output_block * 96 + output_slice * 16
        for lane in range(16):
            output_channel = base + lane
            if output_channel >= cout:
                continue
            value = entry["vector"][output_slice * 16 + lane]
            for descriptor in contributors:
                value = wrap24(value + weights.get(self.layer, output_channel, descriptor))
            entry["vector"][output_slice * 16 + lane] = value
        entry["dirty"] = True

    def _remove_resident(self, destination: Tuple[int, int]) -> None:
        phase = ((destination[0] & 1) << 1) | (destination[1] & 1)
        slot, entry = self._find(phase, destination)
        if entry is not None:
            self.slots[phase][slot] = None

    def final_commit(self, destination: Tuple[int, int], reference: Sequence[int]) -> None:
        require(destination not in self.committed_destinations,
                "duplicate dense destination commit")
        phase = ((destination[0] & 1) << 1) | (destination[1] & 1)
        slot, entry = self._find(phase, destination)
        index = directory_index(self.layer, self.output_block,
                                destination[0], destination[1])
        ever_existed = entry is not None or index in self.directory
        if entry is None:
            self._directory_query("FINAL_DIRECTORY_QUERY")
            if index in self.directory:
                entry = self.ensure(destination, directory_already_queried=True)
        if entry is None:
            vector = [0] * 96
            self.ledger.step("final_zero_build", "FINAL_ZERO_BUILD", 6)
            self.ledger.events["zero_vectors"] += 1
        else:
            self._six_read_drain("FINAL")
            vector = list(entry["vector"])
            self.ledger.events["nonzero_vectors"] += 1
        if list(reference) != vector:
            self.ledger.functional_mismatches += 1
        payload = b"".join(int24_bytes(value) for value in vector) + bytes(96)
        require(len(payload) == 384, "padded output vector size")
        identity = [self.layer, self.output_block, destination[0],
                    destination[1], index, self.generation]
        self.ledger.commit_hash.update(canonical_json_bytes(identity))
        self.ledger.data_hash.update(payload)
        self.ledger.step("final_output_first_latency", "COMMIT_FIRST_LAT", 32)
        command_owner = tuple(identity)
        retired_beats = set()
        for beat in range(3):
            address = 0x20000000 + index * 384 + beat * 128
            command = (command_owner, address, beat, 2 - beat,
                       hashlib.sha256(payload[beat * 128:(beat + 1) * 128]).hexdigest())
            while not self.ledger.sink_ready:
                held = command
                self.ledger.step("final_output_sink_stall", "HOLD_FINAL_OUTPUT")
                self.ledger.events["output_sink_stalls"] += 1
                if command != held:
                    self.ledger.protocol_mismatches += 1
            if beat in retired_beats or address != 0x20000000 + index * 384 + beat * 128:
                self.ledger.protocol_mismatches += 1
            self.ledger.step("final_output_accepted_beat", "COMMIT_BEAT{}".format(beat))
            retired_beats.add(beat)
            self.ledger.events["output_accepted_beats"] += 1
        require(retired_beats == {0, 1, 2}, "dense output beat retirement mismatch")
        self.ledger.events["output_bytes"] += 384
        self.ledger.events["output_logical_bytes"] += 288
        self.ledger.events["output_padded_bytes"] += 384
        self.ledger.events["dense_destinations"] += 1
        self.ledger.events["commit_count"] += 1
        if index in self.directory:
            self._directory_rmw("DIR_CLEAR")
            self.directory.remove(index)
            self.backing.pop(index, None)
        self._remove_resident(destination)
        self.committed_destinations.add(destination)
        if not ever_existed:
            require(all(value == 0 for value in reference),
                    "never-existed destination has nonzero reference")

    def close_epoch(self, expected_destinations: int) -> None:
        require(all(entry is None for bank in self.slots for entry in bank),
                "resident state not empty at epoch close")
        require(not self.directory and not self.backing,
                "backing state not empty at epoch close")
        require(len(self.committed_destinations) == expected_destinations,
                "dense commit population mismatch at block close")


class WeightResident:
    def __init__(self, ledger: CycleLedger):
        self.ledger = ledger
        self.identity = None

    def ensure(self, layer: int, output_block: int, cin_tile: int) -> None:
        identity = (layer, output_block, cin_tile)
        if identity == self.identity:
            return
        self.ledger.acquire("weight_refill_link", "weight_refill_link_stall",
                            "WEIGHT_LINK_STALL")
        try:
            self.ledger.events["weight_refill_requests"] += 1
            self.ledger.step("weight_refill_first_latency", "WEIGHT_REFILL_FIRST_LAT", 32)
            self.ledger.step("weight_refill_accepted_beat", "WEIGHT_REFILL_BEAT", 108)
            self.ledger.events["weight_refill_beats"] += 108
            self.ledger.events["weight_refill_bytes"] += 108 * 128
            self.ledger.weight_hash.update(canonical_json_bytes(identity))
            self.identity = identity
        finally:
            self.ledger.release("weight_refill_link")


class FrontierTracker:
    def __init__(self, ledger: CycleLedger):
        self.ledger = ledger
        self.next_frontier = 0
        self.known_upto = -1
        self.active_order = deque()
        self.remaining = {}
        self.last_accepted = -1

    def offer_event(self, ordinal: int, count: int) -> None:
        require(ordinal == self.known_upto + 1 or ordinal > self.known_upto,
                "nonmonotonic source ordinal")
        self.known_upto = ordinal
        if count:
            self.active_order.append(ordinal)
            self.remaining[ordinal] = count
            self.ledger.events["numeric_source_events"] += 1
            self.ledger.events["legal_descriptors"] += count

    def offer_zero_gap_through(self, ordinal: int) -> None:
        require(ordinal >= self.known_upto, "nonmonotonic known frontier")
        self.known_upto = ordinal

    def retire_descriptor(self, descriptor: Descriptor) -> None:
        if self.remaining.get(descriptor.ordinal, 0) <= 0:
            self.ledger.step("source_scan_frontier_stall",
                             "UNDEFINED_DESCRIPTOR_RETIRE")
            self.ledger.protocol_mismatches += 1
            raise ContractFailure("descriptor retire without pending identity")
        self.remaining[descriptor.ordinal] -= 1
        self.ledger.descriptor_retire_hash.update(
            canonical_json_bytes(descriptor.identity))
        self.ledger.events["descriptors_retired"] += 1

    def accept_ready(self, active_event: str = "FRONTIER_ACCEPT") -> None:
        while self.next_frontier <= self.known_upto:
            while self.active_order and self.active_order[0] < self.next_frontier:
                self.active_order.popleft()
            pending = self.active_order[0] if self.active_order else None
            if pending is not None and pending == self.next_frontier:
                if self.remaining[pending] != 0:
                    break
                self.ledger.step("productive_source_or_group_issue", active_event)
                self.ledger.frontier_hash.update(canonical_json_bytes(pending))
                self.ledger.events["frontier_tokens_accepted"] += 1
                self.remaining.pop(pending)
                self.active_order.popleft()
                self.last_accepted = pending
                self.next_frontier += 1
                continue
            end = self.known_upto + 1 if pending is None else min(pending, self.known_upto + 1)
            count = end - self.next_frontier
            if count <= 0:
                break
            self.ledger.step("productive_source_or_group_issue",
                             "ZERO_FRONTIER_ACCEPT_RUN", count)
            self.ledger.frontier_hash.update(canonical_json_bytes(
                [self.next_frontier, end - 1, count]))
            self.ledger.events["frontier_tokens_accepted"] += count
            self.last_accepted = end - 1
            self.next_frontier = end

    def finish(self, plane_bits: int) -> None:
        self.offer_zero_gap_through(plane_bits - 1)
        self.accept_ready()
        require(self.next_frontier == plane_bits and not self.remaining and
                not self.active_order, "source frontier did not close")


class ExplicitContext:
    def __init__(self, phase: int, index: int, descriptor: Descriptor):
        self.phase = phase
        self.index = index
        self.destination = descriptor.destination
        self.last_possible = descriptor.last_possible_ordinal
        self.contributors = []


class ArchitectureMachine:
    def __init__(self, architecture: str, ledger: CycleLedger, store: PsumStore,
                 weights: WeightSet, weight_resident: WeightResident, layer: int,
                 output_block: int, cout: int, tracker: FrontierTracker):
        self.architecture = architecture
        self.ledger = ledger
        self.store = store
        self.weights = weights
        self.weight_resident = weight_resident
        self.layer = layer
        self.output_block = output_block
        self.cout = cout
        self.tracker = tracker
        self.contexts = [[None for _ in range(4)] for _ in range(4)]
        self.ingress_active = False

    def _service_group(self, group: Sequence[Descriptor], lock_class: str,
                       lock_event: str) -> None:
        require(group and len(group) <= 8 and
                len({descriptor.bank for descriptor in group}) == len(group) and
                len({descriptor.cin_tile for descriptor in group}) == 1,
                "illegal bank/tile round")
        destinations = sorted({descriptor.destination for descriptor in group})
        destination_phases = [((y & 1) << 1) | (x & 1)
                              for y, x in destinations]
        phase_conflict = len(set(destination_phases)) != len(destination_phases)
        self.ledger.optional_stall("phase_bank_conflict", "PHASE_BANK_CONFLICT",
                                   phase_conflict)
        require(not phase_conflict, "same-phase group conflict")
        self.weight_resident.ensure(
            self.layer, self.output_block, group[0].cin_tile)
        for destination in destinations:
            self.store.ensure(destination)
        if lock_class == "join_context_full":
            self.ledger.step("join_context_full", lock_event)
        else:
            self.ledger.step(lock_class, lock_event)
        outstanding_slices = set()
        outstanding_issues = set()
        for output_slice in range(4):
            self.ledger.optional_stall("O8_full", "O8_FULL",
                                       len(outstanding_issues) >= 8)
            psum_conflict = len(set(destination_phases)) != len(destination_phases)
            self.ledger.optional_stall("psum_1RW_conflict", "PSUM_1RW_CONFLICT",
                                       psum_conflict)
            raw_identity = tuple((destination, output_slice)
                                 for destination in destinations)
            raw_conflict = any(identity in outstanding_slices
                               for identity in raw_identity)
            self.ledger.optional_stall("pending_write_RAW", "PENDING_WRITE_RAW",
                                       raw_conflict)
            require(not psum_conflict and not raw_conflict,
                    "illegal psum issue conflict")
            self.ledger.step("productive_source_or_group_issue",
                             "ISSUE_SLICE{}".format(output_slice))
            outstanding_slices.update(raw_identity)
            outstanding_issues.add(output_slice)
            self.ledger.observe_occupancy("O8", len(outstanding_issues))
            self.ledger.observe_occupancy("FIFO4", len(outstanding_issues))
        for output_slice in range(4):
            self._retire_slice(group, destinations, output_slice)
            for destination in destinations:
                outstanding_slices.remove((destination, output_slice))
            outstanding_issues.remove(output_slice)
        for output_slice in (4, 5):
            self.ledger.optional_stall("O8_full", "O8_FULL",
                                       len(outstanding_issues) >= 8)
            psum_conflict = len(set(destination_phases)) != len(destination_phases)
            self.ledger.optional_stall("psum_1RW_conflict", "PSUM_1RW_CONFLICT",
                                       psum_conflict)
            raw_identity = tuple((destination, output_slice)
                                 for destination in destinations)
            raw_conflict = any(identity in outstanding_slices
                               for identity in raw_identity)
            self.ledger.optional_stall("pending_write_RAW", "PENDING_WRITE_RAW",
                                       raw_conflict)
            require(not psum_conflict and not raw_conflict,
                    "illegal psum or pending-write conflict")
            self.ledger.step("productive_source_or_group_issue",
                             "ISSUE_SLICE{}".format(output_slice))
            outstanding_slices.update(raw_identity)
            outstanding_issues.add(output_slice)
            self.ledger.observe_occupancy("O8", len(outstanding_issues))
            self.ledger.observe_occupancy("FIFO4", len(outstanding_issues))
        self.ledger.step("weight_L4_wait", "WAIT_L4_SLICE4")
        self.ledger.step("weight_L4_wait", "WAIT_L4_SLICE5")
        self._retire_slice(group, destinations, 4)
        for destination in destinations:
            outstanding_slices.remove((destination, 4))
        outstanding_issues.remove(4)
        self._retire_slice(group, destinations, 5)
        for destination in destinations:
            outstanding_slices.remove((destination, 5))
        outstanding_issues.remove(5)
        require(not outstanding_slices and not outstanding_issues,
                "O8 state did not drain")
        identity = [descriptor.identity for descriptor in group]
        self.ledger.group_hash.update(canonical_json_bytes(identity))
        self.ledger.events["groups"] += 1
        self.ledger.events["group_size_{}".format(len(group))] += 1
        self.ledger.events["weight_active_reads"] += 6 * len(group)

    def _retire_slice(self, group: Sequence[Descriptor],
                      destinations: Sequence[Tuple[int, int]], output_slice: int) -> None:
        if output_slice == 5:
            event = ("RETIRE_SLICE5_GROUP_AND_SERVICE_DONE"
                     if self.architecture in ("A1-SC8", "A1-ISO8")
                     else "RETIRE_SLICE5_GROUP_DONE")
        else:
            event = "RETIRE_SLICE{}".format(output_slice)
        self.ledger.step("productive_source_or_group_issue", event)
        self.ledger.observe_occupancy("pending_write", 1)
        for destination in destinations:
            contributors = [descriptor for descriptor in group
                            if descriptor.destination == destination]
            self.store.update_slice(destination, output_slice, contributors,
                                    self.weights, self.cout)
            self.ledger.rmw_hash.update(canonical_json_bytes(
                [self.output_block, destination, output_slice,
                 [descriptor.identity for descriptor in contributors]]))
            self.ledger.events["psum_reads"] += 1
            self.ledger.events["psum_writes"] += 1

    def _sc8_groups(self, bundle: Sequence[Descriptor]):
        remaining = list(bundle)
        groups = []
        while remaining:
            tile = remaining[0].cin_tile
            banks, phases, selected = set(), set(), []
            for descriptor in remaining:
                if descriptor.cin_tile == tile and descriptor.bank not in banks and \
                        descriptor.phase not in phases:
                    selected.append(descriptor)
                    banks.add(descriptor.bank)
                    phases.add(descriptor.phase)
            require(selected, "SC8 no progress")
            groups.append(tuple(selected))
            identities = {id(descriptor) for descriptor in selected}
            remaining = [descriptor for descriptor in remaining
                         if id(descriptor) not in identities]
        return groups

    def _iso8_groups(self, bundle: Sequence[Descriptor]):
        groups, index = [], 0
        while index < len(bundle):
            head = bundle[index]
            group = [head]
            if index + 1 < len(bundle):
                nxt = bundle[index + 1]
                if nxt.destination == head.destination and \
                        nxt.cin_tile == head.cin_tile and nxt.bank != head.bank:
                    group.append(nxt)
            groups.append(tuple(group))
            index += len(group)
        return groups

    def _matching_context(self, descriptor: Descriptor):
        for context in self.contexts[descriptor.phase]:
            if context is not None and context.destination == descriptor.destination:
                return context
        return None

    def _lowest_free(self, phase: int):
        return next((index for index, context in enumerate(self.contexts[phase])
                     if context is None), None)

    def _context_round(self, context: ExplicitContext, lock_class: str,
                       lock_event: str) -> None:
        remaining = context.contributors
        require(remaining, "empty context service")
        tile = remaining[0].cin_tile
        banks, selected = set(), []
        for descriptor in remaining:
            if descriptor.cin_tile == tile and descriptor.bank not in banks:
                selected.append(descriptor)
                banks.add(descriptor.bank)
        self._service_group(selected, lock_class, lock_event)
        identities = {id(descriptor) for descriptor in selected}
        context.contributors = [descriptor for descriptor in remaining
                                if id(descriptor) not in identities]
        if not context.contributors:
            self.ledger.step("block_transition_drain", "CONTEXT_RELEASE")
            self.contexts[context.phase][context.index] = None

    def _drain_context(self, context: ExplicitContext, lock_class: str,
                       lock_event: str) -> None:
        first = True
        while self.contexts[context.phase][context.index] is not None:
            self._context_round(context, lock_class,
                                lock_event if first else lock_event + "_NEXT_ROUND")
            first = False

    def _move_lane(self, descriptor: Descriptor, lane_index: int) -> None:
        context = self._matching_context(descriptor)
        while context is not None and len(context.contributors) >= 8:
            self._context_round(context, "join_context_full",
                                "PRESSURE_GROUP_LOCK_C{}".format(context.index))
            context = self._matching_context(descriptor)
        allocated = False
        if context is None:
            free = self._lowest_free(descriptor.phase)
            while free is None:
                victim = next(context for context in self.contexts[descriptor.phase]
                              if context is not None)
                self._context_round(victim, "join_context_full",
                                    "PRESSURE_GROUP_LOCK_C{}".format(victim.index))
                free = self._lowest_free(descriptor.phase)
            context = ExplicitContext(descriptor.phase, free, descriptor)
            self.contexts[descriptor.phase][free] = context
            allocated = True
        move_event = "INGRESS_MOVE_LANE{}_{}_C{}".format(
            lane_index, "ALLOC" if allocated else "JOIN", context.index)
        self.ledger.step("productive_source_or_group_issue", move_event)
        context.contributors.append(descriptor)
        self.ledger.observe_occupancy(
            "contexts", sum(context is not None for bank in self.contexts
                            for context in bank))
        self.ledger.observe_occupancy(
            "context_slots", sum(len(context.contributors) if context is not None else 0
                                 for bank in self.contexts for context in bank))
        context.last_possible = max(context.last_possible,
                                    descriptor.last_possible_ordinal)
        self.tracker.retire_descriptor(descriptor)
        if len(context.contributors) == 8:
            self._context_round(context, "productive_source_or_group_issue",
                                "FULL_GROUP_LOCK_C{}".format(context.index))

    def _drain_closed_osg(self) -> None:
        while True:
            choices = [context for bank in self.contexts for context in bank
                       if context is not None and
                       context.last_possible <= self.tracker.last_accepted]
            if not choices:
                return
            context = min(choices, key=lambda item: (item.phase, item.index))
            self._drain_context(context, "block_transition_drain",
                                "CLOSE_GROUP_LOCK_C{}".format(context.index))

    def accept_bundle(self, bundle: Sequence[Descriptor]) -> None:
        require(bundle and len(bundle) <= 8, "illegal atomic bundle")
        self.ledger.optional_stall("atomic_ingress_backpressure",
                                   "ATOMIC_INGRESS_BACKPRESSURE",
                                   self.ingress_active)
        require(not self.ingress_active, "atomic ingress re-entry")
        self.ingress_active = True
        self.ledger.observe_occupancy("ingress", len(bundle))
        try:
            self.ledger.step("productive_source_or_group_issue", "BUNDLE_ACCEPT")
            self.ledger.events["bundles_accepted"] += 1
            self.ledger.events["descriptors_accepted"] += len(bundle)
            for descriptor in bundle:
                self.ledger.descriptor_accept_hash.update(
                    canonical_json_bytes(descriptor.identity))
            if self.architecture in ("A1-SC8", "A1-ISO8"):
                groups = (self._sc8_groups(bundle) if self.architecture == "A1-SC8"
                          else self._iso8_groups(bundle))
                for group in groups:
                    if self.architecture == "A1-ISO8":
                        lock_event = ("ADJACENT_PAIR_GROUP_LOCK" if len(group) == 2
                                      else "SINGLETON_GROUP_LOCK")
                    else:
                        lock_event = "GROUP_LOCK"
                    self._service_group(group, "productive_source_or_group_issue",
                                        lock_event)
                    for descriptor in group:
                        self.tracker.retire_descriptor(descriptor)
                self.tracker.accept_ready()
                self.ledger.step("block_transition_drain", "BUNDLE_RETIRE")
                return
            require(self.architecture in ("A1-OSG", "PBR4"), "unknown architecture")
            if self.architecture == "PBR4":
                require(all(context is None for bank in self.contexts for context in bank),
                        "PBR4 context crossed bundle epoch")
            for lane_index, descriptor in enumerate(bundle):
                self._move_lane(descriptor, lane_index)
            if self.architecture == "PBR4":
                for phase in range(4):
                    for index in range(4):
                        context = self.contexts[phase][index]
                        if context is not None:
                            self._drain_context(context, "block_transition_drain",
                                                "TAIL_GROUP_LOCK_C{}".format(context.index))
                self.tracker.accept_ready()
                self.ledger.step("block_transition_drain", "BUNDLE_EPOCH_RETIRE")
            else:
                close_candidates = [context for bank in self.contexts for context in bank
                                    if context is not None and
                                    context.last_possible <= self.tracker.next_frontier]
                frontier_event = ("FRONTIER_ACCEPT_MARK_CLOSE_C{}".format(
                    min(close_candidates, key=lambda item: (item.phase, item.index)).index)
                    if close_candidates else "FRONTIER_ACCEPT")
                self.tracker.accept_ready(frontier_event)
                self._drain_closed_osg()
                self.ledger.step("block_transition_drain", "BUNDLE_RETIRE")
        finally:
            self.ingress_active = False

    def block_drain(self) -> None:
        if self.architecture == "A1-OSG":
            for phase in range(4):
                for index in range(4):
                    context = self.contexts[phase][index]
                    if context is not None:
                        self._drain_context(context, "block_transition_drain",
                                            "BLOCK_DRAIN_GROUP_LOCK_C{}".format(context.index))
            self.ledger.step("block_transition_drain", "BLOCK_DRAIN_RETIRE")
        require(all(context is None for bank in self.contexts for context in bank),
                "context state crossed output block")


def terminal_tail(ledger: CycleLedger, output_block: int, last_block: int,
                  time: int, layer: int, sample: int,
                  committed_blocks: set, block_commit_count: int,
                  expected_block_commits: int) -> Sequence[Sequence[object]]:
    """Execute the common M559 T00--T14 terminal FSM.

    The function retains explicit state/counters even though a mandatory row is
    deterministic.  That makes skipped/fused/duplicated clear and owner edges
    impossible to hide behind a bulk cycle increment.
    """
    require(0 <= output_block <= last_block and 0 <= time < 10 and
            0 <= layer < 4 and 0 <= sample < 10,
            "terminal owner identity out of range")
    require(block_commit_count == expected_block_commits,
            "terminal block commit count mismatch")
    require(output_block not in committed_blocks,
            "terminal duplicate block receipt")
    committed_blocks.add(output_block)
    require(committed_blocks == set(range(output_block + 1)),
            "terminal committed-block bitmap mismatch")
    state = "ACTIVE_BLOCK"
    compact = []

    def edge(expected_state: str, next_state: str, primary: str,
             event: str, count: int = 1, compact_event: Optional[str] = None) -> None:
        nonlocal state
        require(state == expected_state, "terminal state mismatch before " + event)
        ledger.step(primary, event, count)
        ledger.terminal_hash.update(canonical_json_bytes(
            [sample, layer, time, output_block, state, event, count, next_state]))
        state = next_state
        compact.append([compact_event if compact_event is not None else event, count])

    if output_block < last_block:
        edge("ACTIVE_BLOCK", "NEXT_BLOCK_OWNER_LOAD", "block_transition_drain",
             "NONLAST_BLOCK_RETIRE")
        edge("NEXT_BLOCK_OWNER_LOAD", "ACTIVE_BLOCK", "block_transition_drain",
             "NEXT_BLOCK_OWNER_LOAD")
        require(state == "ACTIVE_BLOCK", "nonlast terminal did not reload owner")
        return compact

    require(committed_blocks == set(range(last_block + 1)),
            "last-block committed bitmap incomplete")
    edge("ACTIVE_BLOCK", "DIRECTORY_CLEAR_START", "block_transition_drain",
         "LAST_BLOCK_RETIRE")
    require("directory_port" not in ledger.resource_owners,
            "terminal directory port busy")
    ledger.resource_owners.add("directory_port")
    edge("DIRECTORY_CLEAR_START", "DIRECTORY_CLEAR_WORD",
         "time_epoch_directory_clear", "DIRECTORY_CLEAR_START")
    clear_index = 0
    clear_write_count = 0
    expected_clear_hash = hashlib.sha256()
    actual_clear_hash = hashlib.sha256()
    for index in range(1024):
        expected_clear_hash.update(canonical_json_bytes(index))
        require(state == "DIRECTORY_CLEAR_WORD" and index == clear_index and
                clear_write_count == index, "directory clear counter mismatch")
        ledger.step("time_epoch_directory_clear", "DIRECTORY_CLEAR_WORD")
        actual_clear_hash.update(canonical_json_bytes(clear_index))
        ledger.directory_clear_hash.update(canonical_json_bytes(
            [sample, layer, time, clear_index]))
        clear_write_count += 1
        if index < 1023:
            clear_index += 1
    compact.append(["DIRECTORY_CLEAR_WORD_INDEX_0_THROUGH_1023", 1024])
    require(clear_index == 1023 and clear_write_count == 1024 and
            actual_clear_hash.hexdigest() == expected_clear_hash.hexdigest(),
            "directory clear identity/count/hash mismatch")
    ledger.events["epoch_directory_clear_words"] += clear_write_count
    edge("DIRECTORY_CLEAR_WORD", "TIME_RETIRE", "time_epoch_directory_clear",
         "DIRECTORY_CLEAR_END")
    require("directory_port" in ledger.resource_owners,
            "terminal directory ownership lost")
    ledger.resource_owners.remove("directory_port")
    if time < 9:
        edge("TIME_RETIRE", "NEXT_TIME_OWNER_LOAD", "time_epoch_directory_clear",
             "TIME_RETIRE_NONFINAL")
        edge("NEXT_TIME_OWNER_LOAD", "ACTIVE_BLOCK", "time_epoch_directory_clear",
             "NEXT_TIME_OWNER_LOAD")
        return compact
    edge("TIME_RETIRE", "LAYER_RETIRE", "time_epoch_directory_clear",
         "TIME_RETIRE_FINAL")
    next_state = "NEXT_LAYER_OWNER_LOAD" if layer < 3 else "SAMPLE_RETIRE"
    edge("LAYER_RETIRE", next_state, "block_transition_drain", "LAYER_RETIRE")
    if layer < 3:
        edge("NEXT_LAYER_OWNER_LOAD", "ACTIVE_BLOCK", "block_transition_drain",
             "NEXT_LAYER_OWNER_LOAD")
        return compact
    next_state = "NEXT_SAMPLE_OWNER_LOAD" if sample < 9 else "COHORT_RETIRE"
    edge("SAMPLE_RETIRE", next_state, "block_transition_drain", "SAMPLE_RETIRE")
    if sample < 9:
        edge("NEXT_SAMPLE_OWNER_LOAD", "ACTIVE_BLOCK", "block_transition_drain",
             "NEXT_SAMPLE_OWNER_LOAD")
    else:
        edge("COHORT_RETIRE", "POINT_COMPLETE", "block_transition_drain",
             "COHORT_RETIRE")
    return compact


def simulate_row(architecture: str, sample: int, layer_spec,
                 time: int, bitpack: Path, weights: WeightSet,
                 reference_weights: IndependentWeightSet) -> Mapping[str, object]:
    layer, cin, cout, hin, win, hout, wout, blocks = layer_spec
    ledger = CycleLedger(architecture)
    generation = ((sample * 4 + layer) * 10 + time) & 0xFF
    plane_bits = cin * hin * win
    weight_resident = WeightResident(ledger)
    committed_blocks = set()
    terminal_compact = []
    for output_block in range(blocks):
        ledger.events["source_scan_bits"] += plane_bits
        ledger.events["source_scan_active_reads"] += plane_bits
        ledger.events["source_scan_logical_bytes"] += (plane_bits + 7) // 8
        ledger.events["source_scan_padded_transport_bytes"] += (
            (plane_bits + 127) // 128) * 16
        ledger.events["source_scan_base_issue_cycles"] += plane_bits
        store = PsumStore(ledger, layer, output_block, generation)
        tracker = FrontierTracker(ledger)
        machine = ArchitectureMachine(architecture, ledger, store, weights,
                                      weight_resident, layer, output_block,
                                      cout, tracker)
        reference = DirectReferenceAccumulator(
            layer, output_block, cout, reference_weights)
        fifo = deque()
        next_ordinal = 0
        for ordinal in scan_set_ordinals(bitpack, time, plane_bits):
            require(ordinal >= next_ordinal, "nonmonotonic active ordinal")
            if ordinal > next_ordinal:
                tracker.offer_zero_gap_through(ordinal - 1)
                tracker.accept_ready()
            channel, remainder = divmod(ordinal, hin * win)
            y, x = divmod(remainder, win)
            event = event_taps(channel, y, x, cin, hin, win,
                               ordinal, output_block)
            tracker.offer_event(ordinal, len(event))
            ledger.source_hash.update(canonical_json_bytes(
                [output_block, ordinal,
                 {"numeric_activity": 1, "source_sign_bit": 0}]))
            reference.add_source(channel, y, x, hin, win)
            for descriptor in event:
                fifo.append(descriptor)
            while len(fifo) >= 8:
                bundle = tuple(fifo.popleft() for _ in range(8))
                machine.accept_bundle(bundle)
            next_ordinal = ordinal + 1
        if next_ordinal < plane_bits:
            tracker.offer_zero_gap_through(plane_bits - 1)
        if fifo:
            machine.accept_bundle(tuple(fifo.popleft() for _ in range(len(fifo))))
        tracker.finish(plane_bits)
        machine.block_drain()
        for destination_y in range(hout):
            for destination_x in range(wout):
                destination = (destination_y, destination_x)
                store.final_commit(destination,
                                   reference.vectors.get(destination, [0] * 96))
        store.close_epoch(hout * wout)
        terminal_compact.append(list(terminal_tail(
            ledger, output_block, blocks - 1, time, layer, sample,
            committed_blocks, len(store.committed_destinations), hout * wout)))
    ledger.events["psum_read_bytes"] = ledger.events["psum_reads"] * 48
    ledger.events["psum_write_bytes"] = ledger.events["psum_writes"] * 48
    ledger.events["source_scan_frontier_stall_cycles"] = int(
        ledger.classes["source_scan_frontier_stall"])
    for key in MANDATORY_EVENT_KEYS:
        ledger.events[key] += 0
    for size in range(1, 9):
        ledger.events["group_size_{}".format(size)] += 0
    expected_replay = plane_bits * blocks
    expected_dense = hout * wout * blocks
    expected_logical = ((plane_bits + 7) // 8) * blocks
    expected_padded = ((plane_bits + 127) // 128) * 16 * blocks
    conservation = {
        "exclusive_cycles": ledger.total_cycles == ledger.cycle ==
                            sum(ledger.classes.values()),
        "source_scan_bits": ledger.events["source_scan_bits"] == expected_replay,
        "source_active_reads": ledger.events["source_scan_active_reads"] == expected_replay,
        "source_logical_bytes": ledger.events["source_scan_logical_bytes"] == expected_logical,
        "source_padded_bytes": ledger.events["source_scan_padded_transport_bytes"] == expected_padded,
        "source_base_cycles": ledger.events["source_scan_base_issue_cycles"] == expected_replay,
        "frontier_tokens": ledger.events["frontier_tokens_accepted"] == expected_replay,
        "descriptor_accept_retire": ledger.events["legal_descriptors"] ==
                                    ledger.events["descriptors_accepted"] ==
                                    ledger.events["descriptors_retired"],
        "dense_commits": ledger.events["dense_destinations"] ==
                         ledger.events["commit_count"] == expected_dense,
        "output_beats": ledger.events["output_accepted_beats"] == expected_dense * 3,
        "output_padded_bytes": ledger.events["output_padded_bytes"] == expected_dense * 384,
        "weight_refill_bytes": ledger.events["weight_refill_beats"] * 128 ==
                               ledger.events["weight_refill_bytes"],
        "persistent_read_bytes": ledger.events["persistent_read_beats"] * 128 ==
                                 ledger.events["persistent_read_bytes"],
        "persistent_write_bytes": ledger.events["persistent_write_beats"] * 128 ==
                                  ledger.events["persistent_write_bytes"],
        "psum_bytes": ledger.events["psum_read_bytes"] == ledger.events["psum_reads"] * 48 and
                      ledger.events["psum_write_bytes"] == ledger.events["psum_writes"] * 48,
        "terminal_clear_words": ledger.events["epoch_directory_clear_words"] == 1024,
        "terminal_blocks": committed_blocks == set(range(blocks)),
        "resources_empty": not ledger.resource_owners,
        "capacity": MODELED_LOGICAL_BYTES <= LOGICAL_BUDGET_BYTES,
    }
    ledger.conservation_mismatches = sum(not value for value in conservation.values())
    source_time_output_cycle_mismatches = (
        ledger.conservation_mismatches + ledger.transition_mismatches)
    require(source_time_output_cycle_mismatches == 0,
            "row source/time/output/cycle conservation failure")
    require(ledger.total_cycles == ledger.cycle and
            ledger.total_cycles == sum(ledger.classes.values()),
            "row cycle conservation failure")
    return {
        "sample_id": sample, "layer": layer, "time": time,
        "architecture": architecture, "total_cycles": ledger.total_cycles,
        "primary_cycles": {name: int(ledger.classes[name]) for name in PRIMARY_CLASSES},
        "events": dict(sorted((key, int(value)) for key, value in ledger.events.items())),
        "source_sha256": ledger.source_hash.hexdigest(),
        "descriptor_accept_sha256": ledger.descriptor_accept_hash.hexdigest(),
        "descriptor_retire_sha256": ledger.descriptor_retire_hash.hexdigest(),
        "frontier_sha256": ledger.frontier_hash.hexdigest(),
        "weight_sequence_sha256": ledger.weight_hash.hexdigest(),
        "group_sha256": ledger.group_hash.hexdigest(),
        "rmw_sha256": ledger.rmw_hash.hexdigest(),
        "commit_sha256": ledger.commit_hash.hexdigest(),
        "output_data_sha256": ledger.data_hash.hexdigest(),
        "cycle_sequence_sha256": ledger.cycle_hash.hexdigest(),
        "terminal_sequence_sha256": ledger.terminal_hash.hexdigest(),
        "directory_clear_sha256": ledger.directory_clear_hash.hexdigest(),
        "terminal_compact": terminal_compact,
        "conservation": conservation,
        "capacity": {"modeled_logical_bytes": MODELED_LOGICAL_BYTES,
                     "logical_budget_bytes": LOGICAL_BUDGET_BYTES,
                     "within_budget": MODELED_LOGICAL_BYTES <= LOGICAL_BUDGET_BYTES,
                     "foundry_cacti_mapped_ppa_ready": False},
        "final_state_empty": bool(not ledger.resource_owners and
                                  committed_blocks == set(range(blocks))),
        "functional_mismatches": ledger.functional_mismatches,
        "protocol_mismatches": ledger.protocol_mismatches,
        "transition_mismatches": ledger.transition_mismatches,
        "conservation_mismatches": ledger.conservation_mismatches,
        "source_time_output_cycle_mismatches": source_time_output_cycle_mismatches,
    }


class _SyntheticZeroWeights:
    def get(self, layer: int, output_channel: int, descriptor: Descriptor) -> int:
        require(descriptor.numeric_value == 1 and descriptor.source_sign_bit == 0,
                "synthetic typed-source drift")
        return 0


def production_resident_hit_trace(architecture: str) -> Sequence[str]:
    """Drive the actual production classes under the frozen minimal golden."""
    ledger = CycleLedger(architecture, capture_events=True)
    store = PsumStore(ledger, 0, 0, 0)
    tracker = FrontierTracker(ledger)
    weight_resident = WeightResident(ledger)
    weight_resident.identity = (0, 0, 0)
    if architecture == "A1-SC8":
        descriptors = [
            Descriptor(0, 0, 0, 0, 0, 0, 0, 0, False, 0, 0),
            Descriptor(0, 0, 0, 0, 1, 0, 1, 0, True, 0, 0),
        ]
    else:
        descriptors = [
            Descriptor(0, 0, 0, 0, 0, 0, 0, 0, False, 0, 0),
            Descriptor(0, 0, 0, 0, 1, 0, 0, 0, True, 0, 0),
        ]
    for descriptor in descriptors:
        phase = descriptor.phase
        if store._find(phase, descriptor.destination)[1] is None:
            free = next(index for index, entry in enumerate(store.slots[phase])
                        if entry is None)
            store.slots[phase][free] = {
                "destination": descriptor.destination,
                "vector": [0] * 96,
                "dirty": False,
                "generation": 0,
            }
    tracker.offer_event(0, len(descriptors))
    machine = ArchitectureMachine(
        architecture, ledger, store, _SyntheticZeroWeights(), weight_resident,
        0, 0, 96, tracker)
    machine.accept_bundle(tuple(descriptors))
    machine.block_drain()
    return tuple(ledger.event_trace)


def production_golden_self_test(execution_contract: Mapping[str, object],
                                terminal_goldens: Mapping[str, object],
                                r3: Mapping[str, object]) -> None:
    imported = execution_contract["golden_cycle_schedules_r4"][
        "imported_four_resident_hit_goldens_unchanged"]
    for architecture in ARCHITECTURES:
        row = r3["golden_cycle_schedules"][architecture]
        expected = strict_json_text(row["canonical_json_utf8"])
        actual = list(production_resident_hit_trace(architecture))
        require(actual == expected and
                hashlib.sha256(canonical_json_bytes(actual)).hexdigest() ==
                row["sha256"] == imported[architecture],
                "production resident-hit golden mismatch: " + architecture)
    nonlast_ledger = CycleLedger("PBR4")
    nonlast = terminal_tail(
        nonlast_ledger, 0, 3, 0, 0, 0, set(), 1, 1)
    last_ledger = CycleLedger("PBR4")
    last = terminal_tail(
        last_ledger, 3, 3, 0, 0, 0, {0, 1, 2}, 1, 1)
    expected_nonlast = strict_json_text(
        terminal_goldens["goldens"]["COMMON_NONLAST_BLOCK"]["canonical_json_utf8"])
    expected_last = strict_json_text(
        terminal_goldens["goldens"]["COMMON_LAST_BLOCK_TIME"]["canonical_json_utf8"])
    require(list(nonlast) == expected_nonlast["cycles"] and
            list(last) == expected_last["cycles"] and
            nonlast_ledger.total_cycles == expected_nonlast["total_cycles"] and
            last_ledger.total_cycles == expected_last["total_cycles"],
            "production terminal golden mismatch")


def strict_json_text(value: str):
    return json.loads(value, object_pairs_hook=lambda pairs: _unique_pairs(pairs),
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ContractFailure("non-finite embedded golden")))


AUTH_KEYS = {
    "schema", "status", "launch_now", "score_0_to_100", "p0_count", "p1_count",
    "execution_contract_path", "execution_contract_sha256",
    "execution_contract_member_sidecar_file_sha256",
    "execution_contract_outer_sidecar_file_sha256", "source_contract_path",
    "source_contract_sha256", "source_contract_member_sidecar_file_sha256",
    "source_contract_outer_sidecar_file_sha256", "future_runner_schema_path",
    "future_runner_schema_sha256", "runner_python_path", "runner_python_sha256",
    "runner_shell_path", "runner_shell_sha256", "contract_static_review_md_sha256",
    "contract_static_review_json_sha256", "contract_static_manifest_sha256",
    "contract_static_outer_seal_file_sha256", "source_static_review_md_sha256",
    "source_static_review_json_sha256", "source_static_manifest_sha256",
    "source_static_outer_seal_file_sha256", "launch_candidate_review_md_sha256",
    "launch_candidate_review_json_sha256", "launch_candidate_manifest_sha256",
    "launch_candidate_outer_seal_file_sha256", "final_release_review_md_sha256",
    "final_release_review_json_sha256", "final_release_manifest_sha256",
    "final_release_outer_seal_file_sha256", "m511_manifest_sha256",
    "m511_outer_seal_file_sha256", "payload_verifier_review_manifest_sha256",
    "payload_verifier_review_outer_seal_file_sha256",
    "decoder_int8_weight_manifest_sha256", "decoder_int8_weight_outer_seal_file_sha256",
    "result_path_absent", "attempt_marker_absent",
}


REVIEW_EXPECTATIONS = {
    "contract_static": (
        "m562_m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_contract_static_hammer_v4",
        "PASS_SOURCE_ONLY_R4_CONTRACT__RUNNER_SOURCE_ADMISSION_ONLY__NO_EXECUTION"),
    "source_static": (
        "m590_m559_pbr4_immutable_cpu_source_static_hammer_v1",
        "PASS_SOURCE_STATIC__NO_EXECUTION__LAUNCH_CANDIDATE_AUTHORING_ONLY"),
    "launch_candidate": (
        "m590_m559_pbr4_pre_rtl_cpu_launch_candidate_hammer_v1",
        "PASS_LAUNCH_CANDIDATE__NO_EXECUTION__FINAL_REVIEW_AUTHORING_ONLY"),
    "final_release": (
        "m590_m559_pbr4_pre_rtl_cpu_final_release_hammer_v1",
        "PASS_FINAL_RELEASE__AUTHORIZATION_AUTHORING_ONLY"),
}


def verify_review(directory: Path, auth: Mapping[str, object], prefix: str) -> Mapping[str, object]:
    identity = verify_directory(directory)
    require(set(identity["members"]) >= {"review.md", "review.json"},
            "review member population")
    review = strict_json(directory / "review.json")
    p0 = int(review.get("p0_count", review.get("findings", {}).get("p0", -1)))
    p1 = int(review.get("p1_count", review.get("findings", {}).get("p1", -1)))
    expected_schema, expected_status = REVIEW_EXPECTATIONS[prefix]
    require(review.get("schema") == expected_schema and
            review.get("status") == expected_status and
            int(review["score_0_to_100"]) == 100 and p0 == 0 and p1 == 0,
            "review schema/status/score predicate mismatch: " + prefix)
    require(auth[prefix + "_review_md_sha256"] == sha256(directory / "review.md") and
            auth[prefix + "_review_json_sha256"] == sha256(directory / "review.json") and
            auth[prefix + "_manifest_sha256"] == identity["manifest_sha256"] and
            auth[prefix + "_outer_seal_file_sha256"] == identity["outer_file_sha256"],
            "authorization/review binding mismatch: " + prefix)
    return review


def verify_wrapper_descriptor(argument: str, hw_root: Path) -> Mapping[str, object]:
    match = re.fullmatch(r"/proc/self/fd/([0-9]+)", argument)
    require(match is not None, "authorization descriptor must be inherited read-only fd")
    descriptor_fd = int(match.group(1))
    require((fcntl.fcntl(descriptor_fd, fcntl.F_GETFL) & os.O_ACCMODE) == os.O_RDONLY,
            "authorization descriptor fd is writable")
    with os.fdopen(os.dup(descriptor_fd), "r", encoding="utf-8") as handle:
        descriptor = json.loads(handle.read(), object_pairs_hook=lambda pairs: _unique_pairs(pairs),
                                parse_constant=lambda token: (_ for _ in ()).throw(
                                    ContractFailure("non-finite descriptor")))
    expected = {"schema", "authorization_path", "wrapper_review_path",
                "wrapper_path", "wrapper_sha256", "wrapper_pid",
                "wrapper_starttime_ticks"}
    require(set(descriptor) == expected and descriptor["schema"] ==
            "m590_pbr4_read_only_wrapper_attestation_v1", "descriptor schema/key drift")
    canonical_wrapper = (hw_root / WRAPPER_REL).resolve()
    require(Path(descriptor["authorization_path"]).resolve() == (hw_root / AUTH_REL).resolve() and
            Path(descriptor["wrapper_review_path"]).resolve() ==
            (hw_root / WRAPPER_REVIEW_REL).resolve() and
            Path(descriptor["wrapper_path"]).resolve() == canonical_wrapper,
            "descriptor canonical-path drift")
    parent = os.getppid()
    require(int(descriptor["wrapper_pid"]) == parent, "wrapper parent PID mismatch")
    stat = (Path("/proc") / str(parent) / "stat").read_text(encoding="utf-8").split()
    require(int(descriptor["wrapper_starttime_ticks"]) == int(stat[21]),
            "wrapper PID starttime mismatch")
    require(canonical_wrapper.is_file() and not canonical_wrapper.is_symlink() and
            sha256(canonical_wrapper) == descriptor["wrapper_sha256"],
            "canonical wrapper source mismatch")
    cmdline = (Path("/proc") / str(parent) / "cmdline").read_bytes().split(b"\0")
    require(str(canonical_wrapper).encode() in cmdline,
            "runner not invoked by canonical reviewed wrapper")
    return descriptor


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def verify_goldens(hw_root: Path, execution_contract: Mapping[str, object]) -> None:
    terminal = strict_json(hw_root / TERMINAL_GOLDEN_REL)
    for name in ("COMMON_NONLAST_BLOCK", "COMMON_LAST_BLOCK_TIME"):
        row = terminal["goldens"][name]
        require(hashlib.sha256(row["canonical_json_utf8"].encode("utf-8")).hexdigest() ==
                row["sha256"] == execution_contract["golden_cycle_schedules_r4"][name]["sha256"],
                "terminal golden mismatch: " + name)
    imported = execution_contract["golden_cycle_schedules_r4"][
        "imported_four_resident_hit_goldens_unchanged"]
    r3 = strict_json(hw_root /
        "contracts/m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r3_20260827.json")
    for architecture in ARCHITECTURES:
        row = r3["golden_cycle_schedules"][architecture]
        require(hashlib.sha256(row["canonical_json_utf8"].encode("utf-8")).hexdigest() ==
                row["sha256"] == imported[architecture],
                "resident-hit golden mismatch: " + architecture)
    production_golden_self_test(execution_contract, terminal, r3)


def validate_capture_manifest(capture: Mapping[str, object], package: Path) -> Mapping[Tuple[int, int], object]:
    require(capture.get("schema") == "m511_h67_ep35_convtranspose_binary_inputs_s10_v1" and
            capture.get("checkpoint_sha256") ==
            "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158" and
            capture.get("sequence") == "zurich_city_09_a" and
            capture.get("layout") == "T_B_C_H_W" and
            capture.get("bit_order") == "little",
            "M511 manifest schema/identity drift")
    records = {}
    for row in capture["records"]:
        sample, layer = int(row["sample_id"]), int(row["module_index"])
        require((sample, layer) not in records and 0 <= sample < 10 and 0 <= layer < 4,
                "M511 record key drift")
        spec = LAYERS[layer]
        require(tuple(int(value) for value in row["shape"]) ==
                (10, 1, spec[1], spec[3], spec[4]) and row["dtype"] == "bitpack-u1",
                "M511 record shape/dtype drift")
        path = package.joinpath(*safe_member(str(row["relative_path"])).parts)
        expected_bytes = 10 * spec[1] * spec[3] * spec[4] // 8
        require(path.stat().st_size == expected_bytes and
                sha256(path) == row["sha256"], "M511 payload byte/hash drift")
        records[(sample, layer)] = row
    require(len(records) == 40, "M511 record population mismatch")
    return records


def validate_payload_verifier(directory: Path, capture_identity: Mapping[str, object]) -> None:
    receipt = strict_json(directory / "verification.json")
    require(receipt.get("schema") == "m578_m511_payload_verification_v1" and
            receipt.get("status") == "PASS" and receipt.get("p0_count") == 0 and
            receipt.get("m511_manifest_sha256") == capture_identity["manifest_sha256"] and
            receipt.get("raw_bits_all_s10_t10") == EXPECTED_RAW_BITS and
            receipt.get("literal_timesteps") == 10 and
            receipt.get("numeric_values") == [0, 1],
            "M511 semantic verifier receipt drift")


def validate_weight_manifest(manifest: Mapping[str, object], package: Path) -> None:
    require(manifest.get("schema") == "m578_h67_decoder_signed_int8_weights_v2" and
            manifest.get("checkpoint_sha256") ==
            "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "decoder weight manifest schema/checkpoint mismatch")
    rows = {int(row["layer"]): row for row in manifest["records"]}
    require(set(rows) == {0, 1, 2, 3} and len(manifest["records"]) == 4,
            "decoder weight record population mismatch")
    for layer, cin, cout, hin, win, hout, wout, blocks in LAYERS:
        row = rows[layer]
        require(tuple(int(value) for value in row["shape"]) == (cout, cin, 3, 3) and
                row["dtype"] == "int8" and row["layout"] == "COUT_CIN_KY_KX",
                "decoder weight record shape/dtype/layout mismatch")
        path = package.joinpath(*safe_member(str(row["relative_path"])).parts)
        require(path.stat().st_size == cout * cin * 9 and
                sha256(path) == row["sha256"],
                "decoder weight record byte/hash mismatch")


def preflight(args, hw_root: Path) -> Mapping[str, object]:
    descriptor = verify_wrapper_descriptor(args.authorization_descriptor, hw_root)
    auth_path = hw_root / AUTH_REL
    auth_hashes = verify_single_double_seal(auth_path)
    auth = strict_json(auth_path)
    require(set(auth) == AUTH_KEYS and
            auth["schema"] == "m590_m559_pbr4_final_launch_authorization_v1" and
            auth["status"] == "PASS_FINAL_RELEASE" and auth["launch_now"] is True and
            auth["score_0_to_100"] == 100 and auth["p0_count"] == 0 and
            auth["p1_count"] == 0 and auth["result_path_absent"] is True and
            auth["attempt_marker_absent"] is True,
            "authorization closed predicate failed")
    execution_contract = hw_root / EXECUTION_CONTRACT_REL
    source_contract = hw_root / SOURCE_CONTRACT_REL
    require(auth["execution_contract_path"] == EXECUTION_CONTRACT_REL and
            auth["source_contract_path"] == SOURCE_CONTRACT_REL and
            args.contract.resolve() == execution_contract.resolve(),
            "contract canonical path drift")
    execution_hashes = verify_single_double_seal(execution_contract)
    source_hashes = verify_single_double_seal(source_contract)
    require(execution_hashes[0] == EXECUTION_CONTRACT_SHA256 ==
            auth["execution_contract_sha256"] and
            execution_hashes[1] == auth["execution_contract_member_sidecar_file_sha256"] and
            execution_hashes[2] == auth["execution_contract_outer_sidecar_file_sha256"] and
            source_hashes[0] == auth["source_contract_sha256"] and
            source_hashes[1] == auth["source_contract_member_sidecar_file_sha256"] and
            source_hashes[2] == auth["source_contract_outer_sidecar_file_sha256"],
            "contract identity binding drift")
    source_contract_json = strict_json(source_contract)
    future_schema = hw_root / FUTURE_SCHEMA_REL
    require(auth["future_runner_schema_path"] == FUTURE_SCHEMA_REL and
            auth["future_runner_schema_sha256"] == sha256(future_schema) ==
            source_contract_json["future_identity"]["future_runner_schema_sha256"],
            "future runner schema binding drift")
    analyzer = (hw_root / ANALYZER_REL).resolve()
    runner = (hw_root / RUNNER_REL).resolve()
    require(auth["runner_python_path"] == ANALYZER_REL and
            auth["runner_python_sha256"] == sha256(analyzer) and
            auth["runner_shell_path"] == RUNNER_REL and
            auth["runner_shell_sha256"] == sha256(runner),
            "runner self identity drift")
    verify_review(hw_root / CONTRACT_STATIC_REL, auth, "contract_static")
    verify_review(hw_root / SOURCE_STATIC_REL, auth, "source_static")
    verify_review(hw_root / LAUNCH_CANDIDATE_REL, auth, "launch_candidate")
    verify_review(hw_root / FINAL_RELEASE_REL, auth, "final_release")
    wrapper_identity = verify_directory(hw_root / WRAPPER_REVIEW_REL)
    wrapper_review = strict_json(hw_root / WRAPPER_REVIEW_REL / "review.json")
    require(wrapper_review.get("schema") ==
            "m590_m559_pbr4_post_auth_wrapper_static_release_hammer_v1" and
            wrapper_review.get("status") == "PASS_TERMINAL_WRAPPER_RELEASE" and
            wrapper_review["score_0_to_100"] == 100 and
            wrapper_review["p0_count"] == 0 and wrapper_review["p1_count"] == 0 and
            wrapper_review["launch_now"] is True and
            wrapper_review["wrapper_path"] == WRAPPER_REL and
            wrapper_review["wrapper_sha256"] == descriptor["wrapper_sha256"] and
            wrapper_review["authorization_json_sha256"] == auth_hashes[0] and
            wrapper_review["authorization_member_sidecar_file_sha256"] == auth_hashes[1] and
            wrapper_review["authorization_outer_seal_file_sha256"] == auth_hashes[2],
            "wrapper terminal review mismatch")
    execution_json = strict_json(execution_contract)
    verify_goldens(hw_root, execution_json)
    capture_id = verify_directory(args.m511_directory)
    verifier_id = verify_directory(args.m511_payload_verifier_directory)
    weight_id = verify_directory(args.decoder_int8_weight_package)
    require(auth["m511_manifest_sha256"] == capture_id["manifest_sha256"] and
            auth["m511_outer_seal_file_sha256"] == capture_id["outer_file_sha256"] and
            auth["payload_verifier_review_manifest_sha256"] == verifier_id["manifest_sha256"] and
            auth["payload_verifier_review_outer_seal_file_sha256"] == verifier_id["outer_file_sha256"] and
            auth["decoder_int8_weight_manifest_sha256"] == weight_id["manifest_sha256"] and
            auth["decoder_int8_weight_outer_seal_file_sha256"] == weight_id["outer_file_sha256"],
            "input authorization binding mismatch")
    capture = strict_json(args.m511_directory / "manifest.json")
    records = validate_capture_manifest(capture, args.m511_directory)
    validate_payload_verifier(args.m511_payload_verifier_directory, capture_id)
    weight_manifest = strict_json(args.decoder_int8_weight_package / "manifest.json")
    validate_weight_manifest(weight_manifest, args.decoder_int8_weight_package)
    require(not args.output_directory.exists() and not (hw_root / ATTEMPT_REL).exists(),
            "canonical result or attempt already exists")
    return {"authorization_sha256": auth_hashes[0], "capture": capture_id,
            "verifier": verifier_id, "weights": weight_id,
            "wrapper_review_manifest_sha256": wrapper_identity["manifest_sha256"],
            "records": records, "weight_manifest": weight_manifest}


def aggregate_events(rows: Sequence[Mapping[str, object]], architecture: str) -> Mapping[str, int]:
    result = Counter()
    for row in rows:
        if row["architecture"] == architecture:
            result.update(row["events"])
    return dict(sorted((key, int(value)) for key, value in result.items()))


def aggregate_hash(rows: Sequence[Mapping[str, object]], architecture: str,
                   field: str) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if row["architecture"] == architecture:
            digest.update(bytes.fromhex(row[field]))
    return digest.hexdigest()


def failure_close(attempt: Path, staging: Path, output: Path,
                  error: BaseException) -> None:
    if attempt.exists() and attempt.is_dir() and not attempt.is_symlink():
        for seal in (attempt / "SHA256SUMS", attempt / "SHA256SUMS.seal.sha256"):
            if seal.exists() and seal.is_file() and not seal.is_symlink():
                seal.unlink()
        (attempt / "ATTEMPT_FAILED_OR_INCOMPLETE.json").write_text(json.dumps({
            "schema": "m590_pbr4_attempt_failed_or_incomplete_v1",
            "status": "CONSUMED_FAILED_OR_INCOMPLETE_DO_NOT_CITE",
            "exception_type": type(error).__name__, "message": str(error),
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        write_directory_seal(attempt)
        verify_directory(attempt)
    failed_directories = []
    for failed in (staging, output):
        if failed.exists() and failed.is_dir() and not failed.is_symlink():
            failed_directories.append(failed)
    for failed in failed_directories:
        for seal in (failed / "SHA256SUMS", failed / "SHA256SUMS.seal.sha256"):
            if seal.exists() and seal.is_file() and not seal.is_symlink():
                seal.unlink()
        (failed / "RUN_FAILED_OR_INCOMPLETE.json").write_text(json.dumps({
            "schema": "m590_pbr4_failed_or_incomplete_v1",
            "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
            "exception_type": type(error).__name__, "message": str(error),
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        write_directory_seal(failed)
        suffix = 0
        quarantine = output.with_name(output.name +
            ".failed_or_incomplete.{}.quarantine".format(os.getpid()))
        while quarantine.exists():
            suffix += 1
            quarantine = output.with_name(output.name +
                ".failed_or_incomplete.{}.{}.quarantine".format(os.getpid(), suffix))
        os.replace(failed, quarantine)
        verify_directory(quarantine)
    require(not output.exists(), "canonical output survived failure closure")


def run_production(args, hw_root: Path) -> None:
    attempt = hw_root / ATTEMPT_REL
    staging = args.output_directory.parent / ("." + args.output_directory.name +
                                               ".staging.incomplete")
    weights = None
    reference_weights = None
    post_attempt = False
    try:
        identity = preflight(args, hw_root)
        require(not attempt.exists() and not staging.exists(), "attempt/staging collision")
        attempt.mkdir()
        post_attempt = True
        (attempt / "ATTEMPT_CONSUMED.json").write_text(json.dumps({
            "schema": "m590_pbr4_attempt_consumed_v1", "status": "CONSUMED",
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "runner_sha256": sha256(hw_root / RUNNER_REL),
            "authorization_sha256": identity["authorization_sha256"],
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        write_directory_seal(attempt)
        staging.mkdir()
        weights = WeightSet(args.decoder_int8_weight_package,
                            identity["weight_manifest"])
        reference_weights = IndependentWeightSet(
            args.decoder_int8_weight_package, identity["weight_manifest"])
        independent_raw_bits = sum(
            spec[1] * spec[3] * spec[4] * 10 * 10 for spec in LAYERS)
        require(independent_raw_bits == EXPECTED_RAW_BITS,
                "independent raw cohort count mismatch")
        rows = []
        totals = {}
        a1_receipt_hash = None
        for architecture in ARCHITECTURES:
            architecture_rows = []
            for sample in range(10):
                for layer_spec in LAYERS:
                    layer = layer_spec[0]
                    record = identity["records"][(sample, layer)]
                    bitpack = args.m511_directory.joinpath(
                        *safe_member(str(record["relative_path"])).parts)
                    for time in range(10):
                        architecture_rows.append(simulate_row(
                            architecture, sample, layer_spec, time, bitpack,
                            weights, reference_weights))
            rows.extend(architecture_rows)
            require(len(architecture_rows) == 400,
                    "architecture row population mismatch")
            totals[architecture] = sum(row["total_cycles"] for row in architecture_rows)
            require(sum(row["functional_mismatches"] + row["protocol_mismatches"] +
                        row["source_time_output_cycle_mismatches"]
                        for row in architecture_rows) == 0,
                    "A1/candidate mismatch before visibility")
            if architecture == "A1-OSG":
                a1_strong = min(A1_POINTS, key=lambda name:
                                (totals[name], A1_TIE_ORDER.index(name)))
                selection_dir = staging / "A1_ONLY_RECEIPT"
                selection_dir.mkdir()
                (selection_dir / "selection.json").write_text(json.dumps({
                    "schema": "m590_pbr4_a1_selection_v1", "status": "PASS_A1_ONLY",
                    "totals": {name: totals[name] for name in A1_POINTS},
                    "selected": a1_strong, "tie_order": list(A1_TIE_ORDER),
                    "candidate_visible": False,
                    "complete_rows": 1200,
                }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                write_directory_seal(selection_dir)
                a1_receipt_hash = verify_directory(selection_dir)["manifest_sha256"]
            if architecture == "PBR4":
                require(a1_receipt_hash is not None and
                        verify_directory(staging / "A1_ONLY_RECEIPT")["manifest_sha256"] ==
                        a1_receipt_hash, "A1 receipt mutated before/after PBR4")
        require(len(rows) == EXPECTED_ROWS, "mandatory row population mismatch")
        a1_receipt = strict_json(staging / "A1_ONLY_RECEIPT/selection.json")
        a1_strong = a1_receipt["selected"]
        sample_ratios = []
        for sample in range(10):
            baseline = sum(row["total_cycles"] for row in rows if
                           row["sample_id"] == sample and row["architecture"] == a1_strong)
            candidate = sum(row["total_cycles"] for row in rows if
                            row["sample_id"] == sample and row["architecture"] == "PBR4")
            sample_ratios.append({"sample_id": sample, "numerator": baseline,
                                  "denominator": candidate,
                                  "ratio_decimal_12": ratio(baseline, candidate)})
        traffic = {architecture: aggregate_events(rows, architecture)
                   for architecture in ARCHITECTURES}
        aggregate_conservation = {}
        for architecture in ARCHITECTURES:
            ledger = traffic[architecture]
            checks = {
                "rows": sum(row["architecture"] == architecture for row in rows) == 400,
                "replay_bits": ledger.get("source_scan_bits", 0) == EXPECTED_REPLAY_BITS,
                "active_reads": ledger.get("source_scan_active_reads", 0) == EXPECTED_REPLAY_BITS,
                "frontier_tokens": ledger.get("frontier_tokens_accepted", 0) == EXPECTED_REPLAY_BITS,
                "dense_destinations": ledger.get("dense_destinations", 0) ==
                                      EXPECTED_DENSE_DESTINATIONS,
                "commit_count": ledger.get("commit_count", 0) == EXPECTED_DENSE_DESTINATIONS,
                "output_beats": ledger.get("output_accepted_beats", 0) ==
                                EXPECTED_DENSE_DESTINATIONS * 3,
                "output_padded_bytes": ledger.get("output_padded_bytes", 0) ==
                                       EXPECTED_DENSE_DESTINATIONS * 384,
                "row_conservation": all(row["conservation_mismatches"] == 0 and
                                        row["final_state_empty"] for row in rows
                                        if row["architecture"] == architecture),
            }
            aggregate_conservation[architecture] = checks
            require(all(checks.values()),
                    "aggregate cohort conservation mismatch: " + architecture)
        common_traffic_keys = (
            "source_scan_bits", "source_scan_active_reads",
            "source_scan_logical_bytes", "source_scan_padded_transport_bytes",
            "source_scan_base_issue_cycles", "frontier_tokens_accepted",
            "numeric_source_events", "legal_descriptors",
            "descriptors_accepted", "descriptors_retired",
            "dense_destinations", "commit_count", "output_accepted_beats",
            "output_padded_bytes")
        source_common_gate = all(
            len({traffic[architecture].get(key, 0)
                 for architecture in ARCHITECTURES}) == 1
            for key in common_traffic_keys)
        common_hash_fields = ("source_sha256", "descriptor_accept_sha256",
                              "frontier_sha256", "commit_sha256",
                              "output_data_sha256")
        common_hash_gate = all(
            len({aggregate_hash(rows, architecture, field)
                 for architecture in ARCHITECTURES}) == 1
            for field in common_hash_fields)
        traffic_conservation_gate = (
            source_common_gate and common_hash_gate and
            all(all(checks.values()) for checks in aggregate_conservation.values()))
        hidden_resource_gate = (
            MODELED_LOGICAL_BYTES <= LOGICAL_BUDGET_BYTES and
            all(row["final_state_empty"] for row in rows))
        mismatch_total = sum(row["functional_mismatches"] + row["protocol_mismatches"] +
                             row["source_time_output_cycle_mismatches"] for row in rows)
        equivalence_fields = ("group_sha256", "rmw_sha256", "commit_sha256")
        osg_equivalent = all(aggregate_hash(rows, "A1-OSG", field) ==
                             aggregate_hash(rows, "PBR4", field)
                             for field in equivalence_fields)
        base_traffic = traffic[a1_strong]
        candidate_traffic = traffic["PBR4"]
        weight_gate = (candidate_traffic.get("weight_active_reads", 0) <=
                       base_traffic.get("weight_active_reads", 0) and
                       candidate_traffic.get("weight_refill_bytes", 0) <=
                       base_traffic.get("weight_refill_bytes", 0))
        aggregate_ratio = Decimal(ratio(totals[a1_strong], totals["PBR4"]))
        sample_speed_gate = all(Decimal(row["ratio_decimal_12"]) >= Decimal("1.10")
                                for row in sample_ratios)
        exact_gate = mismatch_total == 0
        go = (exact_gate and traffic_conservation_gate and hidden_resource_gate and
              aggregate_ratio >= Decimal("1.30") and sample_speed_gate and
              weight_gate and not osg_equivalent)
        baseline_psum = (base_traffic.get("psum_reads", 0) +
                         base_traffic.get("psum_writes", 0)) * 48
        candidate_psum = (candidate_traffic.get("psum_reads", 0) +
                          candidate_traffic.get("psum_writes", 0)) * 48
        reduction = (Decimal(0) if baseline_psum == 0 else
                     Decimal(baseline_psum - candidate_psum) / Decimal(baseline_psum))
        support_only = (not go and exact_gate and traffic_conservation_gate and
                        hidden_resource_gate and
                        all(Decimal(row["ratio_decimal_12"]) >= Decimal("1.0")
                            for row in sample_ratios) and reduction >= Decimal("0.30") and
                        weight_gate)
        result = {
            "schema": "m590_m559_pbr4_pre_rtl_cpu_result_v6",
            "status": ("PASS_CPU_GO" if go else
                       "PASS_CPU_SUPPORT_ONLY" if support_only else "PASS_CPU_NO_GO"),
            "identity": {key: value for key, value in identity.items()
                         if key not in ("records", "weight_manifest")},
            "model": {"architectures": list(ARCHITECTURES), "literal_timesteps": 10,
                      "raw_bits_all_s10_t10": EXPECTED_RAW_BITS,
                      "replay_bits_all_s10_t10": EXPECTED_REPLAY_BITS,
                      "dense_destinations_all_s10_t10": EXPECTED_DENSE_DESTINATIONS,
                      "modeled_logical_bytes": MODELED_LOGICAL_BYTES,
                      "logical_budget_bytes": LOGICAL_BUDGET_BYTES,
                      "foundry_cacti_mapped_ppa_ready": False},
            "totals": totals, "traffic": traffic, "a1_strong": a1_strong,
            "aggregate_conservation": aggregate_conservation,
            "a1_receipt_manifest_sha256": a1_receipt_hash,
            "ratio_of_sums": {"numerator": totals[a1_strong],
                              "denominator": totals["PBR4"],
                              "decimal_12": ratio(totals[a1_strong], totals["PBR4"])},
            "sample_ratios": sample_ratios,
            "aggregate_hashes": {architecture: {field: aggregate_hash(rows, architecture, field)
                                                  for field in equivalence_fields}
                                 for architecture in ARCHITECTURES},
            "decision": {"cpu_go": go, "support_only": support_only,
                         "mismatch_total": mismatch_total,
                         "aggregate_speed_gate": aggregate_ratio >= Decimal("1.30"),
                         "every_sample_speed_gate": sample_speed_gate,
                         "weight_and_refill_gate": weight_gate,
                         "source_common_ledger_gate": source_common_gate,
                         "common_source_commit_output_hash_gate": common_hash_gate,
                         "traffic_conservation_gate": traffic_conservation_gate,
                         "hidden_resource_capacity_gate": hidden_resource_gate,
                         "pbr4_not_osg_equivalent": not osg_equivalent,
                         "psum_traffic_reduction_decimal_12": format(reduction, ".12f"),
                         "cpu_go_authorizes_rtl": False},
            "claim_boundary": {"single_sequence_fast_kill": True,
                               "multi_sequence": False, "rtl": False,
                               "energy": False, "ppa": False,
                               "system_speedup": False, "paper_headline": False},
        }
        # Both mmap domains close before any result byte is published.  A close
        # exception is therefore a post-attempt/pre-publication failure and is
        # handled by the same failure FSM.
        weights.close()
        weights = None
        reference_weights.close()
        reference_weights = None
        with (staging / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (staging / "result.json").write_text(json.dumps(
            result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M590_M559_PBR4_PRE_RTL_CPU_R6\n", encoding="utf-8")
        write_directory_seal(staging)
        verify_directory(staging)
        os.replace(staging, args.output_directory)
        verify_directory(args.output_directory)
    except BaseException as error:
        for resource in (weights, reference_weights):
            if resource is not None:
                try:
                    resource.close()
                except BaseException:
                    pass
        if post_attempt:
            failure_close(attempt, staging, args.output_directory, error)
        raise


def static_self_test() -> None:
    require(advance_xorshift32(0x53454217, 0) == 0x53454217,
            "xorshift zero advance")
    state = 0x53454217
    for _ in range(137):
        state = xorshift32(state)
    require(advance_xorshift32(0x53454217, 137) == state,
            "xorshift jump mismatch")
    ready_probe = CycleLedger("PBR4")
    ready_probe.ready_state = 0x9BFC2D14
    require(ready_probe.sink_ready and not bool(ready_probe.ready_state & 1),
            "frozen low-three-bit ready predicate mismatch")
    require(wrap24((1 << 23) - 1 + 1) == -(1 << 23) and
            wrap24(-(1 << 23) - 1) == (1 << 23) - 1,
            "Acc24 modulo mismatch")
    taps = event_taps(0, 1, 1, 4, 3, 3, 4, 0)
    require(len(taps) == 9 and taps[-1].event_last and
            [descriptor.kernel_index for descriptor in taps] ==
            [0, 2, 6, 8, 1, 7, 3, 5, 4], "M523 phase-major golden")
    expected_cycles = {
        "A1-SC8": 18, "A1-ISO8": 18, "A1-OSG": 22, "PBR4": 21,
    }
    require(all(len(production_resident_hit_trace(architecture)) == cycles
                for architecture, cycles in expected_cycles.items()),
            "production architecture resident-hit cycle golden")
    terminal = CycleLedger("PBR4")
    compact = terminal_tail(terminal, 0, 0, 0, 0, 0, set(), 1, 1)
    require(terminal.total_cycles == 1029, "terminal nonfinal-time golden")
    require(compact == [
        ["LAST_BLOCK_RETIRE", 1], ["DIRECTORY_CLEAR_START", 1],
        ["DIRECTORY_CLEAR_WORD_INDEX_0_THROUGH_1023", 1024],
        ["DIRECTORY_CLEAR_END", 1], ["TIME_RETIRE_NONFINAL", 1],
        ["NEXT_TIME_OWNER_LOAD", 1]], "terminal event golden")
    require(len(PRIMARY_CLASSES) == 27 and len(set(PRIMARY_CLASSES)) == 27,
            "primary-class declaration closure")


def parse_args(argv: Sequence[str]):
    if argv == ["--self-test-static"]:
        return None
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--m511-directory", required=True, type=Path)
    parser.add_argument("--m511-payload-verifier-directory", required=True, type=Path)
    parser.add_argument("--decoder-int8-weight-package", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--authorization-descriptor", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args is None:
        static_self_test()
        print("PASS M590 M559 r6 repaired immutable analyzer static self-test")
        return 0
    hw_root = Path(__file__).resolve().parents[2]
    canonical = ((hw_root / EXECUTION_CONTRACT_REL), (hw_root / M511_REL),
                 (hw_root / M511_VERIFY_REL), (hw_root / WEIGHT_REL),
                 (hw_root / RESULT_REL))
    supplied = (args.contract, args.m511_directory,
                args.m511_payload_verifier_directory,
                args.decoder_int8_weight_package, args.output_directory)
    require(all(Path(os.path.abspath(value)) == expected.resolve()
                for value, expected in zip(supplied, canonical)),
            "noncanonical production CLI path")
    run_production(args, hw_root)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ContractFailure as error:
        print("M590_FAIL_CLOSED: " + str(error), file=sys.stderr)
        sys.exit(70)
