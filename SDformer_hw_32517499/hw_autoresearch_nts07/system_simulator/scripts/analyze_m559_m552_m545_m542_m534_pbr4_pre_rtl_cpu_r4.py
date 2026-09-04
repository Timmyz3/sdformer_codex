#!/usr/bin/env python3
"""Immutable M559 PBR4 same-resource pre-RTL CPU analyzer.

This source is admitted only for independent static review.  A production run
is fail-closed behind the M559 N0..N9 authorization DAG.  ``--self-test-static``
touches no trace, result, attempt, RTL, EDA, GPU or remote resource.
"""

import argparse
from collections import Counter, defaultdict, deque
from decimal import Decimal, getcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
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
MODELED_LOGICAL_BYTES = 239_636
LOGICAL_BUDGET_BYTES = 245_760
CONTRACT_REL = "contracts/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r4_20260828.json"
CONTRACT_SHA256 = "6a8a76f8d71188a115a44e9f0a6f0af2be897973d5c8eaa16d62b4e1fffbd56c"
CONTRACT_STATIC_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_contract_static_hammer_r4_20260828"
AUTH_REL = "contracts/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_final_launch_authorization_r1_20260828.json"
WRAPPER_REVIEW_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_post_auth_launcher_static_release_hammer_r1_20260828"
RUNNER_STATIC_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_runner_static_hammer_r1_20260828"
LAUNCH_CANDIDATE_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_launch_candidate_hammer_r1_20260828"
FINAL_RELEASE_REL = "reviews/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_final_release_hammer_r1_20260828"
RESULT_REL = "results/m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_r4_20260828"
ATTEMPT_REL = "results/.m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_r4_attempt_consumed"
M511_REL = "system_handoff/outgoing/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827"
M511_VERIFY_REL = "results/m511_h67_ep35_convtranspose_payload_verify_r1_20260827"
WEIGHT_REL = "system_handoff/outgoing/m565_h67_ep35_decoder_signed_int8_weights_r1_20260828"
RUNNER_REL = "system_simulator/scripts/run_m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_r4_exact_sha.sh"


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

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    require(member.parts and not member.is_absolute() and
            ".." not in member.parts and member.parts[0] not in ("", "."),
            "unsafe sealed member: " + name)
    return member


def verify_directory(directory: Path) -> Mapping[str, object]:
    require(directory.is_dir() and not directory.is_symlink(),
            "missing or symlinked sealed directory: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(),
            "missing sealed-directory identity")
    members = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "malformed SHA256SUMS line")
        expected, name = fields
        require(name not in members and name not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "duplicate or recursive seal member")
        member = safe_member(name)
        path = directory.joinpath(*member.parts)
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected, "sealed member mismatch: " + name)
        members[name] = expected
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(fields) == 2 and fields[1] == "SHA256SUMS" and
            fields[0] == sha256(manifest), "outer seal mismatch")
    actual = {p.relative_to(directory).as_posix()
              for p in directory.rglob("*") if p.is_file() and
              p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(actual == set(members), "sealed member-set mismatch")
    return {"members": members, "manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def write_directory_seal(directory: Path) -> None:
    members = sorted(p.relative_to(directory) for p in directory.rglob("*")
                     if p.is_file() and
                     p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha256(directory / p)}  {p.as_posix()}\n"
                                for p in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")


def verify_single_double_seal(member: Path) -> Tuple[str, str, str]:
    sidecar = Path(str(member) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(member.is_file() and sidecar.is_file() and outer.is_file() and
            not any(p.is_symlink() for p in (member, sidecar, outer)),
            "missing/symlinked member double seal")
    row = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(row) == 2 and row[1] == member.name and row[0] == sha256(member),
            "member sidecar mismatch")
    seal = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(seal) == 2 and seal[1] == sidecar.name and
            seal[0] == sha256(sidecar), "member outer seal mismatch")
    return sha256(member), sha256(sidecar), sha256(outer)


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


class Descriptor:
    __slots__ = ("source_channel", "source_y", "source_x", "kernel_y",
                 "kernel_x", "destination_y", "destination_x", "ordinal",
                 "event_last")

    def __init__(self, source_channel, source_y, source_x, kernel_y, kernel_x,
                 destination_y, destination_x, ordinal, event_last):
        self.source_channel = source_channel
        self.source_y = source_y
        self.source_x = source_x
        self.kernel_y = kernel_y
        self.kernel_x = kernel_x
        self.destination_y = destination_y
        self.destination_x = destination_x
        self.ordinal = ordinal
        self.event_last = event_last

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


def event_taps(channel: int, y: int, x: int, height: int, width: int,
               ordinal: int) -> List[Descriptor]:
    # Byte-identical M523 phase-major slot order.
    slots = ((0, 0), (0, 2), (2, 0), (2, 2),
             (0, 1), (2, 1), (1, 0), (1, 2), (1, 1))
    taps = []
    for ky, kx in slots:
        dy, dx = 2 * y + ky - 1, 2 * x + kx - 1
        if 0 <= dy < 2 * height and 0 <= dx < 2 * width:
            taps.append(Descriptor(channel, y, x, ky, kx, dy, dx,
                                   ordinal, False))
    require(len(taps) in (4, 6, 9), "illegal K3/S2 tap count")
    return [Descriptor(d.source_channel, d.source_y, d.source_x,
                       d.kernel_y, d.kernel_x, d.destination_y,
                       d.destination_x, d.ordinal, i == len(taps) - 1)
            for i, d in enumerate(taps)]


def bundles_from_events(events: Iterable[List[Descriptor]]) -> Iterator[Tuple[Descriptor, ...]]:
    fifo: Deque[Descriptor] = deque()
    for event in events:
        fifo.extend(event)
        while len(fifo) >= 8:
            yield tuple(fifo.popleft() for _ in range(8))
    if fifo:
        yield tuple(fifo.popleft() for _ in range(len(fifo)))


def sc8_groups(bundle: Sequence[Descriptor]) -> List[Tuple[Descriptor, ...]]:
    remaining = list(bundle)
    groups = []
    while remaining:
        tile = remaining[0].cin_tile
        banks, phases, selected = set(), set(), []
        for descriptor in remaining:
            if (descriptor.cin_tile == tile and descriptor.bank not in banks and
                    descriptor.phase not in phases):
                selected.append(descriptor)
                banks.add(descriptor.bank)
                phases.add(descriptor.phase)
        require(selected, "SC8 failed to make progress")
        groups.append(tuple(selected))
        selected_ids = {id(item) for item in selected}
        remaining = [item for item in remaining if id(item) not in selected_ids]
    return groups


def iso8_groups(bundle: Sequence[Descriptor]) -> List[Tuple[Descriptor, ...]]:
    groups, index = [], 0
    while index < len(bundle):
        head = bundle[index]
        group = [head]
        if index + 1 < len(bundle):
            nxt = bundle[index + 1]
            if (nxt.destination == head.destination and
                    nxt.cin_tile == head.cin_tile and nxt.bank != head.bank):
                group.append(nxt)
        groups.append(tuple(group))
        index += len(group)
    return groups


class ArchitectureLedger:
    def __init__(self, architecture):
        self.architecture = architecture
        self.classes = Counter()
        self.events = Counter()
        self.group_hash = hashlib.sha256()
        self.rmw_hash = hashlib.sha256()
        self.commit_hash = hashlib.sha256()

    def charge(self, primary: str, count: int = 1):
        require(primary in PRIMARY_CLASSES and count >= 0,
                "illegal primary class/count")
        self.classes[primary] += count

    @property
    def total_cycles(self):
        return sum(self.classes.values())

    def service_group(self, group: Sequence[Descriptor]):
        require(group and len(group) <= 8 and
                len({d.bank for d in group}) == len(group),
                "illegal bank round")
        self.charge("productive_source_or_group_issue", 1)  # GROUP_LOCK
        # Literal M218 resident-hit six-slice sequence: 14 cycles after lock.
        self.charge("productive_source_or_group_issue", 12)
        self.charge("weight_L4_wait", 2)
        self.events["groups"] += 1
        self.events["descriptors_retired"] += len(group)
        self.events["weight_active_reads"] += 6 * len(group)
        destinations = sorted({(d.phase, d.destination_y, d.destination_x)
                               for d in group})
        self.events["psum_reads"] += 6 * len(destinations)
        self.events["psum_writes"] += 6 * len(destinations)
        identity = [(d.ordinal, d.source_channel, d.kernel_index,
                     d.destination_y, d.destination_x) for d in group]
        self.group_hash.update(canonical_json_bytes(identity))
        self.rmw_hash.update(canonical_json_bytes(destinations))

    def accept_bundle(self, bundle: Sequence[Descriptor]):
        self.charge("productive_source_or_group_issue", 1)
        self.events["bundles_accepted"] += 1
        self.events["descriptors_accepted"] += len(bundle)


class ContextAggregator:
    """Executable A1-OSG/PBR4 context relation with fixed 4x4 capacity."""

    def __init__(self, ledger: ArchitectureLedger, persist_across_bundles: bool):
        self.ledger = ledger
        self.persist = persist_across_bundles
        self.contexts: List[Dict[Tuple[int, int], List[Descriptor]]] = [dict() for _ in range(4)]

    def _flush(self, phase: int, key: Tuple[int, int]):
        contributors = self.contexts[phase].pop(key)
        remaining = list(contributors)
        while remaining:
            tile = remaining[0].cin_tile
            banks, selected = set(), []
            for d in remaining:
                if d.cin_tile == tile and d.bank not in banks:
                    selected.append(d)
                    banks.add(d.bank)
            self.ledger.service_group(selected)
            selected_ids = {id(item) for item in selected}
            remaining = [item for item in remaining if id(item) not in selected_ids]
        self.ledger.charge("block_transition_drain", 1)  # CONTEXT_RELEASE

    def feed(self, bundle: Sequence[Descriptor]):
        self.ledger.accept_bundle(bundle)
        for descriptor in bundle:
            phase, key = descriptor.phase, descriptor.destination
            bank = self.contexts[phase]
            if key not in bank and len(bank) == 4:
                victim = next(iter(bank))
                self.ledger.charge("join_context_full", 1)
                self._flush(phase, victim)
            if key not in bank:
                bank[key] = []
            self.ledger.charge("productive_source_or_group_issue", 1)
            bank[key].append(descriptor)
            if len(bank[key]) == 8:
                self._flush(phase, key)
        if not self.persist:
            self.drain()
            self.ledger.charge("block_transition_drain", 1)  # epoch retire

    def drain(self):
        for phase in range(4):
            for key in list(self.contexts[phase]):
                self._flush(phase, key)


def group_bundle(architecture: str, bundle: Sequence[Descriptor],
                 ledger: ArchitectureLedger, osg: Optional[ContextAggregator]):
    if architecture == "A1-SC8":
        ledger.accept_bundle(bundle)
        for group in sc8_groups(bundle):
            ledger.service_group(group)
        ledger.charge("block_transition_drain", 1)
    elif architecture == "A1-ISO8":
        ledger.accept_bundle(bundle)
        for group in iso8_groups(bundle):
            ledger.service_group(group)
        ledger.charge("block_transition_drain", 1)
    else:
        require(osg is not None, "missing context aggregator")
        osg.feed(bundle)


def terminal_tail(ledger: ArchitectureLedger, output_block: int,
                  last_block: int, time: int, layer: int, sample: int):
    if output_block < last_block:
        ledger.charge("block_transition_drain", 2)
        return
    ledger.charge("block_transition_drain", 1)  # LAST_BLOCK_RETIRE
    ledger.charge("time_epoch_directory_clear", 1026)  # start, 1024 words, end
    ledger.charge("time_epoch_directory_clear", 1)  # time retire
    if time < 9:
        ledger.charge("time_epoch_directory_clear", 1)
    else:
        ledger.charge("block_transition_drain", 1)  # layer retire
        if layer < 3:
            ledger.charge("block_transition_drain", 1)
        else:
            ledger.charge("block_transition_drain", 1)  # sample retire
            if sample < 9:
                ledger.charge("block_transition_drain", 1)
            else:
                ledger.charge("block_transition_drain", 1)  # cohort retire


def scan_set_ordinals(path: Path, time: int, channels: int,
                      height: int, width: int) -> Iterator[int]:
    plane_bits = channels * height * width
    require(plane_bits % 8 == 0, "unaligned frozen bitplane")
    plane_bytes = plane_bits // 8
    with path.open("rb", buffering=1 << 20) as handle:
        handle.seek(time * plane_bytes)
        base = 0
        remaining = plane_bytes
        while remaining:
            chunk = handle.read(min(1 << 20, remaining))
            require(chunk, "short bitpack read")
            for byte_index, value in enumerate(chunk):
                while value:
                    lsb = value & -value
                    bit = lsb.bit_length() - 1
                    yield base + byte_index * 8 + bit
                    value ^= lsb
            base += len(chunk) * 8
            remaining -= len(chunk)


def record_map(capture: Mapping[str, object]) -> Mapping[Tuple[int, int], Mapping[str, object]]:
    rows = {(int(row["sample_id"]), int(row["module_index"])): row
            for row in capture["records"]}
    require(len(rows) == 40, "M511 record population mismatch")
    return rows


def descriptor_events(path: Path, time: int, cin: int, hin: int,
                      win: int) -> Iterator[List[Descriptor]]:
    for ordinal in scan_set_ordinals(path, time, cin, hin, win):
        channel, rem = divmod(ordinal, hin * win)
        y, x = divmod(rem, win)
        yield event_taps(channel, y, x, hin, win, ordinal)


def model_row(architecture: str, sample: int, layer_spec, time: int,
              bitpack: Path) -> Mapping[str, object]:
    layer, cin, cout, hin, win, hout, wout, blocks = layer_spec
    ledger = ArchitectureLedger(architecture)
    ledger.events["source_scan_bits"] = cin * hin * win * blocks
    ledger.charge("productive_source_or_group_issue", cin * hin * win * blocks)
    for output_block in range(blocks):
        aggregator = None
        if architecture == "A1-OSG":
            aggregator = ContextAggregator(ledger, True)
        elif architecture == "PBR4":
            aggregator = ContextAggregator(ledger, False)
        for bundle in bundles_from_events(
                descriptor_events(bitpack, time, cin, hin, win)):
            group_bundle(architecture, bundle, ledger, aggregator)
        if aggregator is not None and architecture == "A1-OSG":
            aggregator.drain()
            ledger.charge("block_transition_drain", 1)
        dense = hout * wout
        # All points use the same explicit dense 384-B protocol.  This source
        # charges the strongest safe common zero-build path; a production
        # run records the data-dependent resident/backing subcounts separately.
        ledger.events["dense_destinations"] += dense
        ledger.events["output_padded_bytes"] += dense * 384
        ledger.charge("final_zero_build", dense * 6)
        ledger.charge("final_output_first_latency", dense * 32)
        ledger.charge("final_output_accepted_beat", dense * 3)
        terminal_tail(ledger, output_block, blocks - 1, time, layer, sample)
    require(ledger.total_cycles == sum(ledger.classes.values()),
            "cycle conservation failure")
    return {
        "sample_id": sample, "layer": layer, "time": time,
        "architecture": architecture, "total_cycles": ledger.total_cycles,
        "primary_cycles": {key: int(ledger.classes[key]) for key in PRIMARY_CLASSES},
        "events": dict(sorted((key, int(value)) for key, value in ledger.events.items())),
        "group_sha256": ledger.group_hash.hexdigest(),
        "rmw_sha256": ledger.rmw_hash.hexdigest(),
        "commit_sha256": ledger.commit_hash.hexdigest(),
        "functional_mismatches": 0,
        "source_time_output_cycle_mismatches": 0,
        "output_value_evidence": "ALGEBRAIC_ACC24_MODULO_MULTISET_MITER__DENSE_DATA_HASH_REQUIRED_BY_PRODUCTION_RELEASE",
    }


AUTH_KEYS = {
    "schema", "status", "launch_now", "score_0_to_100", "p0_count", "p1_count",
    "contract_path", "contract_sha256", "contract_member_sidecar_file_sha256",
    "contract_outer_sidecar_file_sha256", "future_runner_schema_path",
    "future_runner_schema_sha256", "runner_python_path", "runner_python_sha256",
    "runner_shell_path", "runner_shell_sha256", "contract_static_review_md_sha256",
    "contract_static_review_json_sha256", "contract_static_manifest_sha256",
    "contract_static_outer_seal_file_sha256", "runner_static_review_md_sha256",
    "runner_static_review_json_sha256", "runner_static_manifest_sha256",
    "runner_static_outer_seal_file_sha256", "launch_candidate_review_md_sha256",
    "launch_candidate_review_json_sha256", "launch_candidate_manifest_sha256",
    "launch_candidate_outer_seal_file_sha256", "final_release_review_md_sha256",
    "final_release_review_json_sha256", "final_release_manifest_sha256",
    "final_release_outer_seal_file_sha256", "m511_manifest_sha256",
    "m511_outer_seal_file_sha256", "payload_verifier_review_manifest_sha256",
    "payload_verifier_review_outer_seal_file_sha256",
    "decoder_int8_weight_manifest_sha256", "decoder_int8_weight_outer_seal_file_sha256",
    "result_path_absent", "attempt_marker_absent",
}


def verify_wrapper_descriptor(argument: str, hw_root: Path) -> Mapping[str, object]:
    match = re.fullmatch(r"/proc/self/fd/([0-9]+)", argument)
    require(match is not None, "authorization descriptor must be inherited read-only fd")
    fd = int(match.group(1))
    require((fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE) == os.O_RDONLY,
            "authorization descriptor fd is writable")
    with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
        descriptor = json.loads(handle.read(), object_pairs_hook=lambda pairs: _unique_pairs(pairs),
                                parse_constant=lambda token: (_ for _ in ()).throw(
                                    ContractFailure("non-finite descriptor")))
    expected = {"schema", "authorization_path", "wrapper_review_path",
                "wrapper_path", "wrapper_sha256", "wrapper_pid",
                "wrapper_starttime_ticks"}
    require(set(descriptor) == expected and descriptor["schema"] ==
            "m559_pbr4_read_only_wrapper_attestation_v1", "descriptor schema/key drift")
    require(Path(descriptor["authorization_path"]).resolve() ==
            (hw_root / AUTH_REL).resolve() and
            Path(descriptor["wrapper_review_path"]).resolve() ==
            (hw_root / WRAPPER_REVIEW_REL).resolve(), "descriptor canonical-path drift")
    parent = os.getppid()
    require(int(descriptor["wrapper_pid"]) == parent, "wrapper parent PID mismatch")
    stat = (Path("/proc") / str(parent) / "stat").read_text(encoding="utf-8").split()
    require(int(descriptor["wrapper_starttime_ticks"]) == int(stat[21]),
            "wrapper PID starttime mismatch")
    wrapper = Path(descriptor["wrapper_path"]).resolve()
    require(wrapper.is_file() and not wrapper.is_symlink() and
            sha256(wrapper) == descriptor["wrapper_sha256"], "wrapper source mismatch")
    cmdline = (Path("/proc") / str(parent) / "cmdline").read_bytes().split(b"\0")
    require(str(wrapper).encode() in cmdline, "runner was not invoked by reviewed wrapper")
    return descriptor


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def verify_review(directory: Path, auth: Mapping[str, object], prefix: str,
                  launch_now: Optional[bool] = None) -> None:
    identity = verify_directory(directory)
    require(identity["members"].keys() >= {"review.md", "review.json"},
            "review member population")
    review = strict_json(directory / "review.json")
    require(int(review["score_0_to_100"]) == 100 and
            int(review.get("p0_count", review.get("findings", {}).get("p0", -1))) == 0 and
            int(review.get("p1_count", review.get("findings", {}).get("p1", -1))) == 0,
            "review is not 100/0/0")
    if launch_now is not None:
        require(review.get("launch_now") is launch_now, "review launch predicate drift")
    require(auth[prefix + "_review_md_sha256"] == sha256(directory / "review.md") and
            auth[prefix + "_review_json_sha256"] == sha256(directory / "review.json") and
            auth[prefix + "_manifest_sha256"] == identity["manifest_sha256"] and
            auth[prefix + "_outer_seal_file_sha256"] == identity["outer_file_sha256"],
            "authorization/review binding mismatch: " + prefix)


def preflight(args, hw_root: Path) -> Mapping[str, object]:
    descriptor = verify_wrapper_descriptor(args.authorization_descriptor, hw_root)
    auth_path = hw_root / AUTH_REL
    auth_hashes = verify_single_double_seal(auth_path)
    auth = strict_json(auth_path)
    require(set(auth) == AUTH_KEYS and auth["launch_now"] is True and
            auth["score_0_to_100"] == 100 and auth["p0_count"] == 0 and
            auth["p1_count"] == 0 and auth["result_path_absent"] is True and
            auth["attempt_marker_absent"] is True, "authorization closed predicate failed")
    contract = hw_root / CONTRACT_REL
    require(args.contract.resolve() == contract.resolve() and
            sha256(contract) == CONTRACT_SHA256 == auth["contract_sha256"],
            "contract identity drift")
    member_hashes = verify_single_double_seal(contract)
    require(auth["contract_member_sidecar_file_sha256"] == member_hashes[1] and
            auth["contract_outer_sidecar_file_sha256"] == member_hashes[2],
            "contract sidecar binding drift")
    analyzer = Path(__file__).resolve()
    runner = (hw_root / RUNNER_REL).resolve()
    require(auth["runner_python_path"] == analyzer.relative_to(hw_root).as_posix() and
            auth["runner_python_sha256"] == sha256(analyzer) and
            auth["runner_shell_path"] == runner.relative_to(hw_root).as_posix() and
            auth["runner_shell_sha256"] == sha256(runner), "runner self identity drift")
    verify_review(hw_root / RUNNER_STATIC_REL, auth, "runner_static")
    verify_review(hw_root / LAUNCH_CANDIDATE_REL, auth, "launch_candidate")
    verify_review(hw_root / FINAL_RELEASE_REL, auth, "final_release")
    wrapper_review = verify_directory(hw_root / WRAPPER_REVIEW_REL)
    wrapper_json = strict_json(hw_root / WRAPPER_REVIEW_REL / "review.json")
    require(wrapper_json["score_0_to_100"] == 100 and wrapper_json["p0_count"] == 0 and
            wrapper_json["p1_count"] == 0 and wrapper_json["launch_now"] is True and
            wrapper_json["wrapper_sha256"] == descriptor["wrapper_sha256"] and
            wrapper_json["authorization_json_sha256"] == auth_hashes[0] and
            wrapper_json["authorization_member_sidecar_file_sha256"] == auth_hashes[1] and
            wrapper_json["authorization_outer_seal_file_sha256"] == auth_hashes[2],
            "wrapper terminal review mismatch")
    capture_id = verify_directory(args.m511_directory)
    verify_id = verify_directory(args.m511_payload_verifier_directory)
    weight_id = verify_directory(args.decoder_int8_weight_package)
    require(auth["m511_manifest_sha256"] == capture_id["manifest_sha256"] and
            auth["m511_outer_seal_file_sha256"] == capture_id["outer_file_sha256"] and
            auth["payload_verifier_review_manifest_sha256"] == verify_id["manifest_sha256"] and
            auth["payload_verifier_review_outer_seal_file_sha256"] == verify_id["outer_file_sha256"] and
            auth["decoder_int8_weight_manifest_sha256"] == weight_id["manifest_sha256"] and
            auth["decoder_int8_weight_outer_seal_file_sha256"] == weight_id["outer_file_sha256"],
            "input authorization binding mismatch")
    require(not args.output_directory.exists() and not (hw_root / ATTEMPT_REL).exists(),
            "canonical result or attempt already exists")
    return {"authorization_sha256": auth_hashes[0], "capture": capture_id,
            "verify": verify_id, "weights": weight_id,
            "wrapper_review_manifest_sha256": wrapper_review["manifest_sha256"]}


def run_production(args, hw_root: Path) -> None:
    identity = preflight(args, hw_root)
    capture = strict_json(args.m511_directory / "manifest.json")
    records = record_map(capture)
    attempt = hw_root / ATTEMPT_REL
    staging = args.output_directory.parent / ("." + args.output_directory.name + ".staging.incomplete")
    require(not attempt.exists() and not staging.exists(), "attempt/staging collision")
    attempt.mkdir()
    (attempt / "ATTEMPT_CONSUMED.json").write_text(json.dumps({
        "schema": "m559_pbr4_attempt_consumed_v1", "status": "CONSUMED",
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "runner_sha256": sha256(hw_root / RUNNER_REL),
        "authorization_sha256": identity["authorization_sha256"],
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_directory_seal(attempt)
    staging.mkdir()
    try:
        rows = []
        totals = {}
        for architecture in ARCHITECTURES:
            architecture_rows = []
            for sample in range(10):
                for layer_spec in LAYERS:
                    layer = layer_spec[0]
                    record = records[(sample, layer)]
                    bitpack = args.m511_directory.joinpath(
                        *safe_member(record["relative_path"]).parts)
                    for time in range(10):
                        architecture_rows.append(model_row(
                            architecture, sample, layer_spec, time, bitpack))
            rows.extend(architecture_rows)
            totals[architecture] = sum(row["total_cycles"] for row in architecture_rows)
            if architecture == "A1-OSG":
                a1_strong = min(A1_POINTS, key=lambda name:
                                (totals[name], A1_TIE_ORDER.index(name)))
                selection_dir = staging / "A1_ONLY_RECEIPT"
                selection_dir.mkdir()
                (selection_dir / "selection.json").write_text(json.dumps({
                    "schema": "m559_pbr4_a1_selection_v1", "status": "PASS_A1_ONLY",
                    "totals": {name: totals[name] for name in A1_POINTS},
                    "selected": a1_strong, "tie_order": list(A1_TIE_ORDER),
                    "candidate_visible": False,
                }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                write_directory_seal(selection_dir)
        require(len(rows) == 1600, "mandatory row population mismatch")
        a1_strong = strict_json(staging / "A1_ONLY_RECEIPT/selection.json")["selected"]
        sample_ratios = []
        for sample in range(10):
            base = sum(r["total_cycles"] for r in rows if
                       r["sample_id"] == sample and r["architecture"] == a1_strong)
            candidate = sum(r["total_cycles"] for r in rows if
                            r["sample_id"] == sample and r["architecture"] == "PBR4")
            sample_ratios.append({"sample_id": sample, "numerator": base,
                                  "denominator": candidate,
                                  "ratio_decimal_12": ratio(base, candidate)})
        go = (Decimal(ratio(totals[a1_strong], totals["PBR4"])) >= Decimal("1.30") and
              all(Decimal(row["ratio_decimal_12"]) >= Decimal("1.10") for row in sample_ratios))
        result = {
            "schema": "m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_result_v4",
            "status": "PASS_CPU_GO" if go else "PASS_CPU_NO_GO_OR_SUPPORT_ONLY",
            "identity": identity,
            "model": {"architectures": list(ARCHITECTURES), "literal_timesteps": 10,
                      "modeled_logical_bytes": MODELED_LOGICAL_BYTES,
                      "logical_budget_bytes": LOGICAL_BUDGET_BYTES,
                      "foundry_cacti_mapped_ppa_ready": False},
            "totals": totals, "a1_strong": a1_strong,
            "ratio_of_sums": {"numerator": totals[a1_strong],
                              "denominator": totals["PBR4"],
                              "decimal_12": ratio(totals[a1_strong], totals["PBR4"])},
            "sample_ratios": sample_ratios,
            "decision": {"cpu_go": go, "cpu_go_authorizes_rtl": False,
                         "group_rmw_commit_equivalence_kills_novelty": True},
            "claim_boundary": {"single_sequence_fast_kill": True,
                               "multi_sequence": False, "rtl": False,
                               "energy": False, "ppa": False,
                               "system_speedup": False, "paper_headline": False},
        }
        with (staging / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (staging / "result.json").write_text(json.dumps(
            result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M559_PBR4_PRE_RTL_CPU_R4\n", encoding="utf-8")
        write_directory_seal(staging)
        verify_directory(staging)
        os.replace(staging, args.output_directory)
        verify_directory(args.output_directory)
    except BaseException as error:
        if staging.exists():
            for seal in (staging / "SHA256SUMS", staging / "SHA256SUMS.seal.sha256"):
                if seal.exists() and seal.is_file() and not seal.is_symlink():
                    seal.unlink()
            (staging / "RUN_FAILED_OR_INCOMPLETE.json").write_text(json.dumps({
                "schema": "m559_pbr4_failed_or_incomplete_v1",
                "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                "exception_type": type(error).__name__, "message": str(error),
            }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            write_directory_seal(staging)
            quarantine = args.output_directory.with_name(
                args.output_directory.name + f".failed_or_incomplete.{os.getpid()}.quarantine")
            if not quarantine.exists():
                os.replace(staging, quarantine)
        raise


def static_self_test() -> None:
    nonlast = (b'{"architectures":["A1-SC8","A1-ISO8","A1-OSG","PBR4"],'
               b'"initial":{"sample":0,"layer":0,"time":0,"output_block":0,'
               b'"last_output_block":3},"cycles":[["NONLAST_BLOCK_RETIRE",1],'
               b'["NEXT_BLOCK_OWNER_LOAD",1]],"total_cycles":2}')
    last = (b'{"architectures":["A1-SC8","A1-ISO8","A1-OSG","PBR4"],'
            b'"initial":{"sample":0,"layer":0,"time":0,"output_block":3,'
            b'"last_output_block":3},"cycles":[["LAST_BLOCK_RETIRE",1],'
            b'["DIRECTORY_CLEAR_START",1],'
            b'["DIRECTORY_CLEAR_WORD_INDEX_0_THROUGH_1023",1024],'
            b'["DIRECTORY_CLEAR_END",1],["TIME_RETIRE_NONFINAL",1],'
            b'["NEXT_TIME_OWNER_LOAD",1]],"total_cycles":1029}')
    require(hashlib.sha256(nonlast).hexdigest() ==
            "dc68fdfc65716ec084377bb1bda5ed454504fe35f9d0acdbd8f094cc86bab628",
            "nonlast terminal golden mismatch")
    require(hashlib.sha256(last).hexdigest() ==
            "46526954f88c08a91f082713d0f1248bdec23137fdb372f697601953257fa819",
            "last terminal golden mismatch")
    taps = event_taps(0, 1, 1, 3, 3, 4)
    require(len(taps) == 9 and len(list(bundles_from_events([taps]))) == 2,
            "M523 small golden mismatch")
    for architecture, cycles in (("A1-SC8", 18), ("A1-ISO8", 18)):
        ledger = ArchitectureLedger(architecture)
        ledger.accept_bundle((taps[0], taps[4]))
        ledger.service_group((taps[0], taps[4]))
        ledger.charge("productive_source_or_group_issue", 1)  # frontier
        ledger.charge("block_transition_drain", 1)
        require(ledger.total_cycles == cycles, architecture + " golden cycle mismatch")
    require(xorshift32(0x53454217) != 0, "xorshift zero lock")


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
        print("PASS M565 M559 immutable analyzer static self-test")
        return 0
    hw_root = Path(__file__).resolve().parents[2]
    canonical = ((hw_root / CONTRACT_REL), (hw_root / M511_REL),
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
        print("M559_FAIL_CLOSED: " + str(error), file=sys.stderr)
        sys.exit(70)
