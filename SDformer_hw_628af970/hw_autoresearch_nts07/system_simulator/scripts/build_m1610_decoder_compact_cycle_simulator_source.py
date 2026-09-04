#!/usr/bin/env python3
"""M1610 representation-only compact decoder cycle-simulator source.

This source implements only the L0/L1 gates authorized by M1572.  It replaces
the frozen M1539 scheduler's dynamically growing token/request representation
with fixed numeric state while preserving the M1539 resource and transition
rules exactly.  It cannot open the ep34 payload, run an actual prefix, launch a
pilot, or publish a paper result.

The compact hot path has 24 port-calendar entries, 129 outstanding-return
slots, eight address/bank scratch entries, eight counters per request kind and
a nine-entry numeric weight cache.  Dependency readiness is passed as an
integer by the future compact generator; the L1-only reference adapter may use
M1539 strings outside the compact hot path to prove request-by-request
equivalence.

Python syntax is deliberately compatible with CPython 3.6.
"""
from __future__ import print_function

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import stat
import struct
import textwrap


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1539_SOURCE = HERE / "build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
M1539_SOURCE_SHA256 = "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4"
M1572 = HW / "reviews/m1572_decoder_compact_cycle_simulator_design_review_r1_20260901"
M1572_REVIEW_SHA256 = "34e109794409ad0c1af56101862cd9ce2c21a3ae327a94e3044cf5cfc9b3f9d1"
M1572_OUTER_SHA256 = "a6f44cd77dbb278feee693e386f9a3587fb7f5906af82d7c47f80a33f89efdd6"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1610_decoder_compact_cycle_simulator_source_r1_v1"
STATUS = "SOURCE_ONLY__L0_L1_EXACT_MITER__NO_ACTUAL_PAYLOAD_NO_EXECUTION"
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
KIND_NAMES = ("external_read", "external_write", "weight_read", "weight_write",
              "psum_read", "psum_write", "compute", "commit")
PACKED_ADDRESS_SCHEMA = (
    "m1610_packed_address_v1:>BBBBBBIHIHQQBI:"
    "schema,config,kind,module,timestep,flags,destination,output_block,group,"
    "subordinal,request_ordinal,address,bank,width_bytes")
PACKED_ADDRESS_SCHEMA_SHA256 = hashlib.sha256(
    PACKED_ADDRESS_SCHEMA.encode("ascii")).hexdigest()
PACKED_ADDRESS = struct.Struct(">BBBBBBIHIHQQBI")
PACKED_COMMIT = struct.Struct(">QQI")
U8_SENTINEL = 0xff
U16_SENTINEL = 0xffff
U32_SENTINEL = 0xffffffff
MAX_ADDRESS_PAIRS = 8
NEXT_PORT_ENTRIES = 24
OUTSTANDING_SLOTS = 129
WEIGHT_CACHE_ENTRIES = 9


class M1610Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1610Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def verify_flat_seal(path, review_sha, outer_sha):
    path = Path(path)
    regular_exact(path / "review.json", review_sha, "M1572 review")
    regular_exact(path / "SHA256SUMS.seal.sha256", outer_sha,
                  "M1572 outer seal")
    require((path / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [sha256(path / "SHA256SUMS"), "SHA256SUMS"],
            "M1572 outer content drift")
    value = json.loads((path / "review.json").read_text(encoding="utf-8"))
    require(value.get("status") ==
            "GO_COMPACT_SOURCE_ONLY_AFTER_EXACT_M1539_MITER_CONTRACT__NO_EXECUTION_AUTHORIZED",
            "M1572 status drift")
    return {"review_sha256": review_sha, "outer_seal_file_sha256": outer_sha}


def load_m1539():
    regular_exact(M1539_SOURCE, M1539_SOURCE_SHA256, "M1539 source")
    spec = importlib.util.spec_from_file_location("m1610_bound_m1539",
                                                  str(M1539_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M1539")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG and
            module.validate_resource() ==
            "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10",
            "M1539 semantic boundary drift")
    return module


M = load_m1539()


def config_index(config):
    if config == CONFIGS[0]:
        return 0
    if config == CONFIGS[1]:
        return 1
    if config == CONFIGS[2]:
        return 2
    raise M1610Error("configuration is not admitted")


def kind_index(kind):
    for ordinal in range(8):
        if KIND_NAMES[ordinal] == kind:
            return ordinal
    raise M1610Error("unmapped request kind")


def validate_bank(kind_id, bank):
    bank = int(bank)
    if kind_id in (2, 3):
        require(0 <= bank < 8, "weight bank out of range")
    elif kind_id in (4, 5):
        require(0 <= bank < 6, "psum bank out of range")
    elif kind_id in (0, 1, 7):
        require(bank == 0, "external bank out of range")
    else:
        require(kind_id == 6 and bank == 0, "compute context out of range")


def port_index_for(kind_id, bank):
    validate_bank(kind_id, bank)
    if kind_id == 2:
        return int(bank)
    if kind_id == 3:
        return 8 + int(bank)
    if kind_id in (4, 5):
        return 16 + int(bank)
    if kind_id in (0, 1, 7):
        return 22
    return 23


def queue_base_for(kind_id, bank):
    validate_bank(kind_id, bank)
    if kind_id in (2, 3):
        return int(bank) * 8
    if kind_id in (4, 5):
        return 64 + int(bank) * 8
    if kind_id in (0, 1, 7):
        return 112
    return 128


def queue_capacity_for(kind_id):
    if kind_id in (2, 3, 4, 5):
        return 8
    if kind_id in (0, 1, 7):
        return 16
    require(kind_id == 6, "unmapped queue capacity")
    return 1


def service_bytes_for(kind_id):
    if kind_id in (2, 3):
        return 16
    if kind_id in (4, 5):
        return 48
    if kind_id in (0, 1, 7):
        return 192
    require(kind_id == 6, "unmapped service width")
    return 288


def latency_for(kind_id):
    if kind_id in (2, 3):
        return 4
    if kind_id == 4:
        return 2
    if kind_id in (5, 6):
        return 1
    if kind_id == 0:
        return 32
    require(kind_id in (1, 7), "unmapped request latency")
    return 3


class CompactScheduler(object):
    """Fixed-state numeric implementation of the frozen M1539 scheduler."""
    __slots__ = (
        "config_id", "next_port", "outstanding", "outstanding_count",
        "addresses", "banks", "address_count", "kind_counts", "byte_counts",
        "address_digest", "commit_digest", "last_cycle", "requests",
        "last_request_ordinal", "last_issue", "last_return", "last_dependency",
        "last_port_ready", "last_beats", "max_outstanding", "full_waits",
        "shared_1rw_serializations", "packed_scratch", "commit_scratch")

    def __init__(self, config):
        self.config_id = config_index(config)
        self.next_port = [0] * NEXT_PORT_ENTRIES
        self.outstanding = [0] * OUTSTANDING_SLOTS
        self.outstanding_count = [0] * 16
        self.addresses = [0] * MAX_ADDRESS_PAIRS
        self.banks = [0] * MAX_ADDRESS_PAIRS
        self.address_count = 0
        self.kind_counts = [0] * 8
        self.byte_counts = [0] * 8
        self.address_digest = hashlib.sha256()
        self.commit_digest = hashlib.sha256()
        self.last_cycle = -1
        self.requests = 0
        self.last_request_ordinal = -1
        self.last_issue = -1
        self.last_return = -1
        self.last_dependency = -1
        self.last_port_ready = -1
        self.last_beats = -1
        self.max_outstanding = 0
        self.full_waits = 0
        self.shared_1rw_serializations = 0
        self.packed_scratch = bytearray(PACKED_ADDRESS.size)
        self.commit_scratch = bytearray(PACKED_COMMIT.size)

    def begin_addresses(self):
        self.address_count = 0

    def push_address(self, address, bank):
        require(self.address_count < MAX_ADDRESS_PAIRS,
                "more than eight address-bank pairs")
        self.addresses[self.address_count] = int(address)
        self.banks[self.address_count] = int(bank)
        self.address_count += 1

    def _count_index(self, base):
        if base < 64:
            return base // 8
        if base < 112:
            return 8 + (base - 64) // 8
        if base == 112:
            return 14
        return 15

    def _active_compact(self, base, capacity, issue):
        count_index = self._count_index(base)
        count = self.outstanding_count[count_index]
        write = 0
        for read in range(count):
            value = self.outstanding[base + read]
            if value > issue:
                insert = write
                while insert > 0 and self.outstanding[base + insert - 1] > value:
                    self.outstanding[base + insert] = self.outstanding[base + insert - 1]
                    insert -= 1
                self.outstanding[base + insert] = value
                write += 1
        self.outstanding_count[count_index] = write
        require(write <= capacity, "outstanding fixed array overflow")
        return write

    def schedule_loaded(self, kind_id, width_bytes, earliest,
                        dependency_ready, schema_version, module, timestep,
                        flags, destination, output_block, group, subordinal,
                        request_ordinal):
        """Schedule one numeric request; returns no per-request object."""
        kind_id = int(kind_id)
        width_bytes = int(width_bytes)
        earliest = int(earliest)
        dependency_ready = int(dependency_ready)
        request_ordinal = int(request_ordinal)
        require(0 <= kind_id < 8 and width_bytes > 0 and earliest >= 0 and
                dependency_ready >= 0 and self.address_count > 0,
                "bad numeric request")
        require(request_ordinal == self.last_request_ordinal + 1,
                "request ordinal is not unique and monotonic")
        require(0 <= int(schema_version) <= U8_SENTINEL and
                0 <= int(module) <= U8_SENTINEL and
                0 <= int(timestep) <= U8_SENTINEL and
                0 <= int(flags) <= U8_SENTINEL and
                0 <= int(destination) <= U32_SENTINEL and
                0 <= int(output_block) <= U16_SENTINEL and
                0 <= int(group) <= U32_SENTINEL and
                0 <= int(subordinal) <= U16_SENTINEL and
                0 <= request_ordinal < (1 << 64) and
                width_bytes <= U32_SENTINEL,
                "packed coordinate exceeds fixed schema")
        port_ready = 0
        pair = 0
        while pair < self.address_count:
            require(0 <= self.addresses[pair] < (1 << 64) and
                    0 <= self.banks[pair] <= U8_SENTINEL,
                    "address-bank pair exceeds fixed schema")
            port_index = port_index_for(kind_id, self.banks[pair])
            if self.next_port[port_index] > port_ready:
                port_ready = self.next_port[port_index]
            other = 0
            while other < pair:
                require(self.banks[other] != self.banks[pair],
                        "duplicate bank in one request")
                other += 1
            pair += 1
        issue = max(earliest, dependency_ready, port_ready)
        if port_ready > max(earliest, dependency_ready) and kind_id in (4, 5, 0, 1, 7):
            self.shared_1rw_serializations += 1
        changed = True
        while changed:
            changed = False
            pair = 0
            while pair < self.address_count:
                base = queue_base_for(kind_id, self.banks[pair])
                capacity = queue_capacity_for(kind_id)
                count = self._active_compact(base, capacity, issue)
                if count >= capacity:
                    proposed = self.outstanding[base + count - capacity]
                    if proposed > issue:
                        issue = proposed
                        self.full_waits += 1
                        changed = True
                pair += 1
        service_bytes = service_bytes_for(kind_id)
        latency = latency_for(kind_id)
        initiation_interval = 1
        beats = max(1, (width_bytes + service_bytes - 1) // service_bytes)
        returned = issue + latency + beats - 1
        pair = 0
        while pair < self.address_count:
            port_index = port_index_for(kind_id, self.banks[pair])
            base = queue_base_for(kind_id, self.banks[pair])
            capacity = queue_capacity_for(kind_id)
            self.next_port[port_index] = issue + max(initiation_interval, beats)
            count_index = self._count_index(base)
            count = self._active_compact(base, capacity, issue)
            require(count < capacity, "outstanding append overflow")
            self.outstanding[base + count] = returned
            self.outstanding_count[count_index] = count + 1
            if count + 1 > self.max_outstanding:
                self.max_outstanding = count + 1
            pair += 1
        if kind_id in (4, 5):
            pair = 0
            while pair < self.address_count:
                require(0 <= self.addresses[pair] and
                        self.addresses[pair] + width_bytes <= 221184,
                        "psum address exceeds partition")
                pair += 1
        if kind_id in (2, 3):
            pair = 0
            while pair < self.address_count:
                require(0 <= self.addresses[pair] and
                        self.addresses[pair] + width_bytes <= 1728,
                        "weight address exceeds bank partition")
                pair += 1
        pair = 0
        while pair < self.address_count:
            PACKED_ADDRESS.pack_into(self.packed_scratch, 0,
                int(schema_version), self.config_id, kind_id, int(module),
                int(timestep), int(flags), int(destination), int(output_block),
                int(group), int(subordinal), request_ordinal,
                self.addresses[pair], self.banks[pair], width_bytes)
            self.address_digest.update(self.packed_scratch)
            pair += 1
        if kind_id == 7:
            pair = 0
            while pair < self.address_count:
                PACKED_COMMIT.pack_into(self.commit_scratch, 0,
                                        self.kind_counts[7],
                                        self.addresses[pair], width_bytes)
                self.commit_digest.update(self.commit_scratch)
                pair += 1
        self.last_cycle = max(self.last_cycle, returned)
        self.requests += 1
        self.kind_counts[kind_id] += 1
        self.byte_counts[kind_id] += width_bytes * self.address_count
        self.last_request_ordinal = request_ordinal
        self.last_issue = issue
        self.last_return = returned
        self.last_dependency = dependency_ready
        self.last_port_ready = port_ready
        self.last_beats = beats
        self.address_count = 0

    def summary(self):
        return {"configuration": CONFIGS[self.config_id],
                "resource_manifest_sha256": M.validate_resource(),
                "total_cycles": self.last_cycle + 1,
                "request_count": self.requests,
                "kind_counts": dict((KIND_NAMES[index], self.kind_counts[index])
                                    for index in range(8)
                                    if self.kind_counts[index]),
                "byte_counts": dict((KIND_NAMES[index], self.byte_counts[index])
                                    for index in range(8)
                                    if self.byte_counts[index]),
                "packed_transaction_address_sha256":
                    self.address_digest.hexdigest(),
                "packed_address_schema_sha256": PACKED_ADDRESS_SCHEMA_SHA256,
                "packed_commit_sequence_sha256": self.commit_digest.hexdigest(),
                "fixed_state": {"next_port_entries": len(self.next_port),
                    "outstanding_slots": len(self.outstanding),
                    "address_scratch_entries": len(self.addresses),
                    "max_active_outstanding_per_bank": self.max_outstanding}}


class NumericWeightTileCache(object):
    """Fixed nine-entry array cache with the exact M1539 replacement rule."""
    __slots__ = ("valid", "module", "output_block", "tap", "channel_tile",
                 "age", "tick", "load_module", "load_output_block",
                 "load_tap", "load_channel_tile", "load_count",
                 "unique_module", "unique_output_block", "unique_tap",
                 "unique_channel_tile", "unique_count", "miss_slot",
                 "miss_module", "miss_output_block", "miss_tap",
                 "miss_channel_tile", "miss_count")

    def __init__(self):
        self.valid = [0] * WEIGHT_CACHE_ENTRIES
        self.module = [0] * WEIGHT_CACHE_ENTRIES
        self.output_block = [0] * WEIGHT_CACHE_ENTRIES
        self.tap = [0] * WEIGHT_CACHE_ENTRIES
        self.channel_tile = [0] * WEIGHT_CACHE_ENTRIES
        self.age = [0] * WEIGHT_CACHE_ENTRIES
        self.tick = 0
        self.load_module = [0] * 8
        self.load_output_block = [0] * 8
        self.load_tap = [0] * 8
        self.load_channel_tile = [0] * 8
        self.load_count = 0
        self.unique_module = [0] * 8
        self.unique_output_block = [0] * 8
        self.unique_tap = [0] * 8
        self.unique_channel_tile = [0] * 8
        self.unique_count = 0
        self.miss_slot = [0] * 8
        self.miss_module = [0] * 8
        self.miss_output_block = [0] * 8
        self.miss_tap = [0] * 8
        self.miss_channel_tile = [0] * 8
        self.miss_count = 0

    def begin_group(self):
        self.load_count = 0

    def push_key(self, module, output_block, tap, channel_tile):
        require(self.load_count < 8, "one group exceeds eight weight keys")
        offset = self.load_count
        self.load_module[offset] = int(module)
        self.load_output_block[offset] = int(output_block)
        self.load_tap[offset] = int(tap)
        self.load_channel_tile[offset] = int(channel_tile)
        self.load_count += 1

    def _equal_loaded(self, left, right):
        return (self.unique_module[left] == self.unique_module[right] and
                self.unique_output_block[left] == self.unique_output_block[right] and
                self.unique_tap[left] == self.unique_tap[right] and
                self.unique_channel_tile[left] == self.unique_channel_tile[right])

    def _slot_matches_unique(self, slot, unique):
        return (self.valid[slot] and self.module[slot] == self.unique_module[unique] and
                self.output_block[slot] == self.unique_output_block[unique] and
                self.tap[slot] == self.unique_tap[unique] and
                self.channel_tile[slot] == self.unique_channel_tile[unique])

    def _slot_is_pinned(self, slot):
        unique = 0
        while unique < self.unique_count:
            if self._slot_matches_unique(slot, unique):
                return True
            unique += 1
        return False

    def _slot_key_less(self, left, right):
        if self.module[left] != self.module[right]:
            return self.module[left] < self.module[right]
        if self.output_block[left] != self.output_block[right]:
            return self.output_block[left] < self.output_block[right]
        if self.tap[left] != self.tap[right]:
            return self.tap[left] < self.tap[right]
        return self.channel_tile[left] < self.channel_tile[right]

    def prepare_loaded(self):
        self.unique_count = 0
        self.miss_count = 0
        loaded = 0
        while loaded < self.load_count:
            duplicate = False
            unique = 0
            while unique < self.unique_count:
                if (self.unique_module[unique] == self.load_module[loaded] and
                    self.unique_output_block[unique] == self.load_output_block[loaded] and
                    self.unique_tap[unique] == self.load_tap[loaded] and
                    self.unique_channel_tile[unique] == self.load_channel_tile[loaded]):
                    duplicate = True
                    break
                unique += 1
            if not duplicate:
                unique = self.unique_count
                self.unique_module[unique] = self.load_module[loaded]
                self.unique_output_block[unique] = self.load_output_block[loaded]
                self.unique_tap[unique] = self.load_tap[loaded]
                self.unique_channel_tile[unique] = self.load_channel_tile[loaded]
                self.unique_count += 1
            loaded += 1
        unique = 0
        while unique < self.unique_count:
            self.tick += 1
            hit = -1
            slot = 0
            while slot < WEIGHT_CACHE_ENTRIES:
                if self._slot_matches_unique(slot, unique):
                    hit = slot
                    break
                slot += 1
            if hit >= 0:
                self.age[hit] = self.tick
                unique += 1
                continue
            free = -1
            slot = 0
            while slot < WEIGHT_CACHE_ENTRIES:
                if not self.valid[slot]:
                    free = slot
                    break
                slot += 1
            if free < 0:
                victim = -1
                slot = 0
                while slot < WEIGHT_CACHE_ENTRIES:
                    if not self._slot_is_pinned(slot):
                        if (victim < 0 or self.age[slot] < self.age[victim] or
                            (self.age[slot] == self.age[victim] and
                             self._slot_key_less(slot, victim))):
                            victim = slot
                    slot += 1
                require(victim >= 0, "weight cache has no unpinned victim")
                free = victim
            self.valid[free] = 1
            self.module[free] = self.unique_module[unique]
            self.output_block[free] = self.unique_output_block[unique]
            self.tap[free] = self.unique_tap[unique]
            self.channel_tile[free] = self.unique_channel_tile[unique]
            self.age[free] = self.tick
            miss = self.miss_count
            self.miss_slot[miss] = free
            self.miss_module[miss] = self.unique_module[unique]
            self.miss_output_block[miss] = self.unique_output_block[unique]
            self.miss_tap[miss] = self.unique_tap[unique]
            self.miss_channel_tile[miss] = self.unique_channel_tile[unique]
            self.miss_count += 1
            unique += 1

    def slot_for(self, module, output_block, tap, channel_tile):
        slot = 0
        while slot < WEIGHT_CACHE_ENTRIES:
            if (self.valid[slot] and self.module[slot] == int(module) and
                self.output_block[slot] == int(output_block) and
                self.tap[slot] == int(tap) and
                self.channel_tile[slot] == int(channel_tile)):
                return slot
            slot += 1
        raise M1610Error("weight key is not resident")


FLAG_SOURCE = 1
FLAG_CONTROL_READ = 2
FLAG_CONTROL_WRITE = 3
FLAG_TYPED_DESC = 4
FLAG_K1_DESC = 5
FLAG_REFILL = 6
FLAG_REFILL_WRITE = 7
FLAG_TYPED_WEIGHT = 8
FLAG_K1_WEIGHT = 9
FLAG_PSUM_READ = 10
FLAG_COMPUTE = 11
FLAG_PSUM_WRITE = 12
FLAG_COMMIT = 13


def parse_synthetic_identifier(config, identifier, request_ordinal):
    """L1-only adapter; strings never enter CompactScheduler."""
    prefix = config + ":"
    require(identifier.startswith(prefix), "synthetic identifier config drift")
    tail = identifier[len(prefix):]
    if tail == "source":
        return (1, U8_SENTINEL, U8_SENTINEL, FLAG_SOURCE, U32_SENTINEL,
                U16_SENTINEL, U32_SENTINEL, 0, request_ordinal)
    if tail == "control_read":
        return (1, U8_SENTINEL, U8_SENTINEL, FLAG_CONTROL_READ, U32_SENTINEL,
                U16_SENTINEL, U32_SENTINEL, 0, request_ordinal)
    if tail == "control_write":
        return (1, U8_SENTINEL, U8_SENTINEL, FLAG_CONTROL_WRITE, U32_SENTINEL,
                U16_SENTINEL, U32_SENTINEL, 0, request_ordinal)
    parts = tail.split(":")
    if parts[0] == "commit":
        require(len(parts) == 3, "commit identifier grammar drift")
        return (1, 3, 0, FLAG_COMMIT, int(parts[1]), int(parts[2]),
                U32_SENTINEL, 0, request_ordinal)
    require(len(parts) == 6 and parts[0] == "m3" and parts[1] == "t0" and
            parts[2].startswith("d") and parts[3].startswith("ob") and
            parts[4].startswith("g"), "destination identifier grammar drift")
    destination = int(parts[2][1:])
    output_block = int(parts[3][2:])
    group = int(parts[4][1:])
    suffix = parts[5]
    subordinal = 0
    if suffix == "typed_desc":
        flag = FLAG_TYPED_DESC
    elif suffix.startswith("k1_desc"):
        flag = FLAG_K1_DESC; subordinal = int(suffix[len("k1_desc"):])
    elif suffix.startswith("refill") and suffix.endswith("_weight_write"):
        flag = FLAG_REFILL_WRITE
        subordinal = int(suffix[len("refill"):-len("_weight_write")])
    elif suffix.startswith("refill"):
        flag = FLAG_REFILL; subordinal = int(suffix[len("refill"):])
    elif suffix == "typed_weight":
        flag = FLAG_TYPED_WEIGHT
    elif suffix.startswith("k1_weight"):
        flag = FLAG_K1_WEIGHT; subordinal = int(suffix[len("k1_weight"):])
    elif suffix == "psum_read":
        flag = FLAG_PSUM_READ
    elif suffix == "compute":
        flag = FLAG_COMPUTE
    elif suffix == "psum_write":
        flag = FLAG_PSUM_WRITE
    else:
        raise M1610Error("unknown destination identifier suffix")
    return (1, 3, 0, flag, destination, output_block, group, subordinal,
            request_ordinal)


def reconstruct_synthetic_identifier(config, coordinate):
    (_version, module, timestep, flag, destination, output_block, group,
     subordinal, _request_ordinal) = coordinate
    if flag == FLAG_SOURCE:
        return config + ":source"
    if flag == FLAG_CONTROL_READ:
        return config + ":control_read"
    if flag == FLAG_CONTROL_WRITE:
        return config + ":control_write"
    if flag == FLAG_COMMIT:
        return "{}:commit:{}:{}".format(config, destination, output_block)
    base = "{}:m{}:t{}:d{}:ob{}:g{}:".format(
        config, module, timestep, destination, output_block, group)
    if flag == FLAG_TYPED_DESC:
        return base + "typed_desc"
    if flag == FLAG_K1_DESC:
        return base + "k1_desc{}".format(subordinal)
    if flag == FLAG_REFILL:
        return base + "refill{}".format(subordinal)
    if flag == FLAG_REFILL_WRITE:
        return base + "refill{}_weight_write".format(subordinal)
    if flag == FLAG_TYPED_WEIGHT:
        return base + "typed_weight"
    if flag == FLAG_K1_WEIGHT:
        return base + "k1_weight{}".format(subordinal)
    if flag == FLAG_PSUM_READ:
        return base + "psum_read"
    if flag == FLAG_COMPUTE:
        return base + "compute"
    if flag == FLAG_PSUM_WRITE:
        return base + "psum_write"
    raise M1610Error("unknown packed flag")


def reference_port_ready(row, scheduler):
    resource_name, operation = M.port_for(row["kind"])
    port = M.normalized_port(resource_name)
    key_operation = "rw" if port["mode"] == "1RW" else operation
    return max([scheduler.next_port.get((resource_name, bank, key_operation), 0)
                for bank in row["banks"]] or [0])


def compact_next_port_projection(scheduler):
    values = [0] * NEXT_PORT_ENTRIES
    for (resource_name, bank, operation), value in scheduler.next_port.items():
        if resource_name == "weight":
            index = int(bank) + (8 if operation == "write" else 0)
        elif resource_name == "psum":
            index = 16 + int(bank)
        elif resource_name == "external":
            index = 22
        else:
            require(resource_name == "compute", "reference port name drift")
            index = 23
        values[index] = int(value)
    return values


def compact_outstanding_projection(reference):
    values = [0] * OUTSTANDING_SLOTS
    counts = [0] * 16
    for (resource_name, bank), active in reference.outstanding.items():
        if resource_name == "weight":
            base = int(bank) * 8; count_index = int(bank); capacity = 8
        elif resource_name == "psum":
            base = 64 + int(bank) * 8; count_index = 8 + int(bank); capacity = 8
        elif resource_name == "external":
            base = 112; count_index = 14; capacity = 16
        else:
            require(resource_name == "compute", "reference outstanding name drift")
            base = 128; count_index = 15; capacity = 1
        ordered = sorted(int(item) for item in active)
        require(len(ordered) <= capacity, "reference outstanding overflow")
        counts[count_index] = len(ordered)
        for ordinal, item in enumerate(ordered):
            values[base + ordinal] = item
    return values, counts


def miter_rows(config, rows, label):
    reference = M.AddressTimedScheduler(config)
    compact = CompactScheduler(config)
    numeric_tokens = {}
    packed_reference = hashlib.sha256()
    packed_commit = hashlib.sha256()
    coordinates = set()
    row_count = 0
    for row in rows:
        ordinal = row_count
        coordinate = parse_synthetic_identifier(config, row["id"], ordinal)
        require(reconstruct_synthetic_identifier(config, coordinate) == row["id"],
                "packed coordinate is not reversible")
        require(coordinate not in coordinates, "packed coordinate collision")
        coordinates.add(coordinate)
        missing = [token for token in row["dependencies"]
                   if token not in numeric_tokens]
        require(not missing, "L1 adapter unresolved dependency")
        numeric_dependency = max(
            [numeric_tokens[token] for token in row["dependencies"]] or
            [row["earliest_issue_cycle"]])
        expected_port_ready = reference_port_ready(row, reference)
        receipt = reference.schedule_one(row)
        require(receipt["dependency_ready_cycle"] == numeric_dependency,
                "numeric dependency differs from M1539")
        compact.begin_addresses()
        for address, bank in zip(row["addresses"], row["banks"]):
            compact.push_address(address, bank)
        compact.schedule_loaded(
            kind_index(row["kind"]), row["width_bytes"],
            row["earliest_issue_cycle"], numeric_dependency, *coordinate)
        require((compact.last_issue, compact.last_return,
                 compact.last_dependency, compact.last_port_ready) ==
                (receipt["issue_cycle"], receipt["return_cycle"],
                 receipt["dependency_ready_cycle"], expected_port_ready),
                "request cycle miter failed at {} row {}".format(label, ordinal))
        if row["produces"]:
            require(row["produces"] not in numeric_tokens,
                    "L1 duplicate numeric producer")
            numeric_tokens[row["produces"]] = compact.last_return
        require(compact.next_port == compact_next_port_projection(reference),
                "port-calendar miter failed")
        expected_outstanding, expected_counts = compact_outstanding_projection(
            reference)
        # Compare only active values.  Stale fixed slots are intentionally
        # retained outside the active count and cannot affect the schedule.
        queue_layout = tuple((bank * 8, 8) for bank in range(8)) + \
            tuple((64 + bank * 8, 8) for bank in range(6)) + ((112, 16), (128, 1))
        for queue in range(16):
            require(compact.outstanding_count[queue] == expected_counts[queue],
                    "outstanding count miter failed")
            base, capacity = queue_layout[queue]
            count = compact.outstanding_count[queue]
            require(sorted(compact.outstanding[base:base + count]) ==
                    expected_outstanding[base:base + count] and count <= capacity,
                    "outstanding return multiset miter failed")
        require(compact.last_cycle == reference.last_cycle and
                compact.requests == reference.requests,
                "aggregate cycle/count prefix miter failed")
        version, module, timestep, flags, destination, output_block, group, subordinal, request_ordinal = coordinate
        for address, bank in zip(row["addresses"], row["banks"]):
            packed_reference.update(PACKED_ADDRESS.pack(
                version, config_index(config), kind_index(row["kind"]), module,
                timestep, flags, destination, output_block, group, subordinal,
                request_ordinal, int(address), int(bank), int(row["width_bytes"])))
        if row["kind"] == "commit":
            for address in row["addresses"]:
                packed_commit.update(PACKED_COMMIT.pack(
                    reference.kind_counts.get("commit", 0) - 1,
                    int(address), int(row["width_bytes"])))
        row_count += 1
    summary = compact.summary()
    require(summary["total_cycles"] == reference.last_cycle + 1 and
            summary["request_count"] == reference.requests and
            summary["kind_counts"] == reference.kind_counts and
            summary["byte_counts"] == reference.byte_counts and
            summary["packed_transaction_address_sha256"] ==
                packed_reference.hexdigest() and
            summary["packed_commit_sequence_sha256"] == packed_commit.hexdigest(),
            "final cycle/count/byte/address/commit miter failed")
    return {"label": label, "configuration": config,
            "requests": row_count, "total_cycles": summary["total_cycles"],
            "packed_transaction_address_sha256":
                summary["packed_transaction_address_sha256"],
            "packed_commit_sequence_sha256":
                summary["packed_commit_sequence_sha256"],
            "kind_counts": summary["kind_counts"],
            "byte_counts": summary["byte_counts"],
            "max_active_outstanding_per_bank": compact.max_outstanding,
            "outstanding_full_waits": compact.full_waits,
            "shared_1rw_serializations": compact.shared_1rw_serializations}


def cache_miter():
    reference = M.WeightTileCache()
    compact = NumericWeightTileCache()
    scenarios = (
        tuple((3, 0, 0, index) for index in range(8)),
        ((3, 0, 0, 0), (3, 0, 0, 0), (3, 0, 0, 7)),
        tuple((3, 0, 1, index) for index in range(8)),
        ((3, 0, 2, 0), (3, 0, 1, 7), (3, 0, 2, 0)),
    )
    evictions = 0
    for keys in scenarios:
        before = set(reference.key_to_slot)
        expected_misses = reference.prepare(keys)
        compact.begin_group()
        for key in keys:
            compact.push_key(*key)
        compact.prepare_loaded()
        actual_misses = []
        for ordinal in range(compact.miss_count):
            actual_misses.append(((compact.miss_module[ordinal],
                                   compact.miss_output_block[ordinal],
                                   compact.miss_tap[ordinal],
                                   compact.miss_channel_tile[ordinal]),
                                  compact.miss_slot[ordinal]))
        require(actual_misses == expected_misses, "cache miss/slot miter failed")
        actual = {}
        actual_age = {}
        for slot in range(WEIGHT_CACHE_ENTRIES):
            if compact.valid[slot]:
                key = (compact.module[slot], compact.output_block[slot],
                       compact.tap[slot], compact.channel_tile[slot])
                actual[key] = slot
                actual_age[key] = compact.age[slot]
        require(actual == reference.key_to_slot and actual_age == reference.age and
                compact.tick == reference.tick,
                "cache state/age/tick miter failed")
        evictions += len([key for key in before if key not in reference.key_to_slot])
    require(evictions > 0, "cache eviction was not covered")
    return {"scenarios": len(scenarios), "evictions": evictions,
            "final_entries": len(reference.key_to_slot),
            "tick": reference.tick}


def manual_port_pressure_rows(config):
    rows = []
    for ordinal in range(18):
        rows.append(M.request(config + ":source", config, "external_read",
                              [(6 << 60) | ordinal], [0], 192))
    rows.append(M.request(config + ":m3:t0:d0:ob0:g0:psum_read", config,
                          "psum_read", [bank * 48 for bank in range(6)],
                          range(6), 48))
    rows.append(M.request(config + ":m3:t0:d0:ob0:g0:psum_write", config,
                          "psum_write", [bank * 48 for bank in range(6)],
                          range(6), 48))
    return rows


def make_bits(active):
    bits = [[[0 for _x in range(2)] for _y in range(2)] for _c in range(8)]
    for channel, y, x in active:
        bits[channel][y][x] = 1
    return bits


def validate_l0():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    seal = verify_flat_seal(M1572, M1572_REVIEW_SHA256, M1572_OUTER_SHA256)
    require(M.validate_resource() ==
            "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10",
            "common resource digest drift")
    scheduler = CompactScheduler(CONFIGS[0])
    cache = NumericWeightTileCache()
    require(len(scheduler.next_port) == 24 and
            len(scheduler.outstanding) == 129 and
            len(scheduler.addresses) == 8 and len(cache.valid) == 9 and
            PACKED_ADDRESS.size == 39,
            "fixed state capacity/schema width drift")
    checked = []
    for function in (validate_bank, port_index_for, queue_base_for,
                     queue_capacity_for, service_bytes_for, latency_for,
                     CompactScheduler.begin_addresses,
                     CompactScheduler.push_address,
                     CompactScheduler._active_compact,
                     CompactScheduler.schedule_loaded,
                     NumericWeightTileCache.begin_group,
                     NumericWeightTileCache.push_key,
                     NumericWeightTileCache.prepare_loaded,
                     NumericWeightTileCache.slot_for):
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    require(node.func.id not in ("dict", "set"),
                            "hot path constructs dict/set")
                if isinstance(node.func, ast.Attribute):
                    require(node.func.attr not in ("dumps", "format"),
                            "hot path uses JSON/string formatting")
        require(not any(isinstance(node, (ast.Dict, ast.Set, ast.List,
                                          ast.ListComp,
                                          ast.SetComp, ast.DictComp,
                                          ast.GeneratorExp)) for node in ast.walk(tree)),
                "hot path contains dynamic container construction")
        checked.append(function.__name__)
    return {"status": "PASS_M1610_L0_STATIC_SCHEMA_AND_BOUNDS",
            "resource_manifest_sha256": M.validate_resource(),
            "packed_address_schema_sha256": PACKED_ADDRESS_SCHEMA_SHA256,
            "fixed_state": {"next_port_entries": 24,
                "outstanding_slots": 129, "weight_cache_entries": 9,
                "address_scratch_entries": 8},
            "hot_paths_checked": checked, "m1572": seal,
            "actual_payload": False, "execution": False}


def synthetic_self_test():
    l0 = validate_l0()
    cases = (
        ("all_zero", make_bits(())),
        ("single_boundary_source", make_bits(((0, 0, 0),))),
        ("full_eight_bank_group", make_bits(tuple((c, 0, 0)
                                                   for c in range(8)))),
        ("opposite_boundaries", make_bits(((0, 0, 0), (7, 1, 1)))),
    )
    rows = []
    for label, bits in cases:
        for config in CONFIGS:
            rows.append(miter_rows(
                config, M.synthetic_config_transactions(config, bits), label))
    pressure = miter_rows(CONFIGS[2],
                          manual_port_pressure_rows(CONFIGS[2]),
                          "outstanding_and_1rw_pressure")
    require(pressure["outstanding_full_waits"] > 0 and
            pressure["shared_1rw_serializations"] > 0,
            "outstanding-full or shared-1RW conflict was not covered")
    cache = cache_miter()
    return {"schema": SCHEMA,
            "status": "PASS_M1610_L0_L1_SYNTHETIC_REQUEST_EXACT_MITER__NO_L2_NO_L3",
            "l0": l0, "l1": {"rows": rows, "pressure": pressure,
                "cache": cache, "configurations": list(CONFIGS),
                "cycle_exact": True, "count_exact": True,
                "bytes_exact": True, "commit_exact": True,
                "address_exact": True},
            "actual_payload": False, "l2_actual_prefix": False,
            "l3_full_diagnostic": False, "pilot": False,
            "production": False, "paper_result": False}


def production_release(_token=None):
    raise M1610Error(
        "M1610 is L0/L1 source-only; actual payload, L2/L3, pilot and production are forbidden")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "configurations": list(CONFIGS),
            "forbidden_configuration": FORBIDDEN_CONFIG,
            "representation": {"numeric_dependency_ready": True,
                "next_port_entries": 24, "outstanding_slots": 129,
                "weight_cache_entries": 9, "address_scratch_entries": 8,
                "packed_address_schema_sha256": PACKED_ADDRESS_SCHEMA_SHA256,
                "hardware_resource_or_schedule_change": False},
            "implemented_miter_levels": ["L0", "L1"],
            "missing_miter_levels": ["L2", "L3"],
            "claim_boundary": {"source_only": True,
                "actual_payload": False, "execution": False,
                "cycles": False, "traffic": False, "speedup": False,
                "energy": False, "system_speedup": False,
                "rtl": False, "eda": False, "ppa": False,
                "table_a": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--l0", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        result = describe()
    elif args.l0:
        result = validate_l0()
    else:
        result = synthetic_self_test()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
