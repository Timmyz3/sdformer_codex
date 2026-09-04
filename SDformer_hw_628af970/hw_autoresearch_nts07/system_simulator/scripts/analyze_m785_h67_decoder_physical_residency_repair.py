#!/usr/bin/env python3
"""M785 source-only physical-residency repair of frozen M777.

M777 and M768 remain immutable.  This additive layer makes dirty psum slot
reuse depend on local-read plus external-evict completion, gives the nine
weight tiles real physical slots and refill/write/read dependencies, charges
all local backing/refill port actions, and separates the M722 line-buffer
contributor oracle from an independently pinned global-vector storage oracle.

Production replay, production cycles, speedup and result writing are refused.
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
M777_PATH = HERE / "analyze_m777_h67_decoder_a1_k8_address_timed_repair.py"
M777_SHA256 = "237b94683271feb96dddeb63f9742f163372b24c70009c266a9c51fd4872eb58"
STORAGE_ORACLE_PATH = HERE / "oracle_m785_decoder_global_vector_storage.py"
STORAGE_ORACLE_SHA256 = "422da36ad1414d2dfa70363607c27bb99dee2f2505d1ceee2142a6023c162db5"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen(path: Path, expected: str, name: str):
    if _file_sha256(path) != expected:
        raise RuntimeError(name + " identity drift")
    spec = importlib.util.spec_from_file_location("m785_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M777 = _load_frozen(M777_PATH, M777_SHA256, "frozen_m777")
STORAGE = _load_frozen(
    STORAGE_ORACLE_PATH, STORAGE_ORACLE_SHA256, "storage_oracle")

Failure = M777.Failure
require = M777.require
sha256 = M777.sha256
strict_json = M777.strict_json
verify_sealed_directory = M777.verify_sealed_directory
safe_member = M777.safe_member
PortSpec = M777.PortSpec
CommonResource = M777.CommonResource
Request = M777.Request
canonical_sha256 = M777.canonical_sha256
DOCS359_SHA256 = M777.DOCS359_SHA256
CONFIGS = M777.CONFIGS
HEADLINE_PAIR = M777.HEADLINE_PAIR
MODULE_GEOMETRY = M777.MODULE_GEOMETRY
PSUM_VECTOR_BYTES = M777.PSUM_VECTOR_BYTES
PSUM_BANKS = M777.PSUM_BANKS
PSUM_BANK_ROW_BYTES = M777.PSUM_BANK_ROW_BYTES
WEIGHT_SOURCE_TILE = M777.WEIGHT_SOURCE_TILE
WEIGHT_TILE_BYTES = M777.WEIGHT_TILE_BYTES
OUTPUT_COMMIT_BYTES = M777.OUTPUT_COMMIT_BYTES
WEIGHT_BANKS = 8
WEIGHT_SLOT_BYTES_PER_BANK = WEIGHT_TILE_BYTES // WEIGHT_BANKS
WEIGHT_ROWS_PER_SOURCE = 96 // 16
CONTRACT_SCHEMA = "m785_h67_decoder_physical_residency_repair_contract_v1"

route_for_record = M777.route_for_record
headline_ratio_allowed = M777.headline_ratio_allowed
dense_commit_addresses = M777.dense_commit_addresses
commit_address_hash = M777.commit_address_hash
bank_unique_groups = M777.bank_unique_groups
weight_bank_and_local_row = M777.weight_bank_and_local_row
weight_group_layout = M777.weight_group_layout
Stripe = M777.Stripe
psum_stripes = M777.psum_stripes
ResidencyEvent = M777.ResidencyEvent
PsumResidency = M777.PsumResidency
load_pinned_module = M777.load_pinned_module
unpack_timestep = M777.unpack_timestep
service_groups = M777.service_groups
_source_read = M777._source_read
_descriptor_transactions = M777._descriptor_transactions
_d1_transactions = M777._d1_transactions
collect_mapper_contributors = M777.collect_mapper_contributors
resource_from_contract = M777.resource_from_contract
normalized_population_records = M777.normalized_population_records
assert_population_isolation = M777.assert_population_isolation
assert_fair_configs = M777.assert_fair_configs


@dataclass(frozen=True)
class CompressedTransaction(M777.CompressedTransaction):
    """M777 transaction extended only with a physical weight-bank write."""

    def validate(self) -> None:
        require(self.transaction_id, "transaction_id must be nonempty")
        require(self.population_id, "population_id must be nonempty")
        require(self.config in CONFIGS, "unknown configuration")
        require(self.kind in (
            "weight_read", "weight_write", "psum_read", "psum_write",
            "external_read", "external_write", "compute", "commit"),
            "unknown transaction kind")
        require(self.base_address >= 0 and self.address_stride_bytes >= 0,
                "negative transaction address")
        require(self.count >= 1 and self.width_bytes >= 1,
                "transaction count/width must be positive")
        require(self.bank_pattern, "bank pattern must be explicit")
        require(len(self.bank_pattern) == len(set(self.bank_pattern)),
                "one request cannot use one bank twice")
        require(not self.address_offsets or
                len(self.address_offsets) == len(self.bank_pattern),
                "explicit address offsets must match bank pattern")
        require(all(value >= 0 for value in self.address_offsets),
                "negative explicit address offset")
        require(self.earliest_issue_cycle >= 0, "negative earliest issue")


expand_transactions = M777.expand_transactions


class AddressTimedScheduler(M777.AddressTimedScheduler):
    """M777 scheduler with physical local weight-write and range checks."""

    def _resource(self, kind: str):
        if kind == "weight_write":
            return "weight", self.resource.weight, "write"
        return super()._resource(kind)

    def schedule(self, requests: Iterable[Request]) -> Dict[str, object]:
        rows = list(requests)
        per_weight_bank = self.resource.weight_bytes_logical // WEIGHT_BANKS
        require(self.resource.weight_bytes_logical % WEIGHT_BANKS == 0,
                "weight partition is not evenly banked")
        for request in rows:
            if request.kind in ("psum_read", "psum_write"):
                for address in request.addresses:
                    require(0 <= address and
                            address + request.width_bytes <=
                            self.resource.psum_bytes_logical,
                            "psum address exceeds 221184-byte physical partition")
            if request.kind in ("weight_read", "weight_write"):
                for address in request.addresses:
                    require(0 <= address and
                            address + request.width_bytes <= per_weight_bank,
                            "weight address exceeds physical bank partition")
        # Call the M768 scheduler directly: M777.schedule would reject a local
        # weight write in its precheck before our extended _resource is used.
        return M777.M768.AddressTimedScheduler.schedule(self, rows)


@dataclass(frozen=True)
class WeightAccess:
    miss: bool
    slot: int
    evicted_key: Optional[Tuple[int, int, int, int]]


class WeightResidency:
    """Nine physical 1536-byte slots with deterministic key/slot/LRU state."""

    def __init__(self, weight_bytes: int = 13824,
                 tile_bytes: int = WEIGHT_TILE_BYTES):
        require(weight_bytes > 0 and tile_bytes > 0 and
                weight_bytes % tile_bytes == 0,
                "invalid weight tile partition")
        self.weight_bytes = int(weight_bytes)
        self.tile_bytes = int(tile_bytes)
        self.capacity = self.weight_bytes // self.tile_bytes
        require(self.capacity == 9, "M785 freezes nine physical weight slots")
        self.key_to_slot: Dict[Tuple[int, int, int, int], int] = {}
        self.slot_to_key: Dict[int, Tuple[int, int, int, int]] = {}
        self.age: Dict[Tuple[int, int, int, int], int] = {}
        self.tick = 0

    def access(self, key: Tuple[int, int, int, int]) -> WeightAccess:
        key = tuple(int(value) for value in key)
        require(len(key) == 4 and min(key) >= 0, "invalid weight key")
        self.tick += 1
        if key in self.key_to_slot:
            self.age[key] = self.tick
            return WeightAccess(False, self.key_to_slot[key], None)
        evicted = None
        if len(self.slot_to_key) < self.capacity:
            slot = min(set(range(self.capacity)) - set(self.slot_to_key))
        else:
            evicted = min(self.key_to_slot,
                          key=lambda value: (self.age[value], value))
            slot = self.key_to_slot.pop(evicted)
            self.slot_to_key.pop(slot)
            self.age.pop(evicted)
        self.key_to_slot[key] = slot
        self.slot_to_key[slot] = key
        self.age[key] = self.tick
        require(self.slot_to_key[slot] == key and
                self.key_to_slot[key] == slot,
                "weight physical slot bijection failure")
        return WeightAccess(True, slot, evicted)


def weight_slot_bank_and_local_row(
    slot: int, flat_k: int, channels: int,
) -> Tuple[int, int]:
    """Map a logical K source to its physical cached tile slot.

    Each slot has 192 bytes per bank: two source channels, each holding six
    16-byte rows for 96 output lanes.  Tap/source-tile select the cache key;
    therefore only channel-within-that-16-source-tile participates locally.
    """
    require(0 <= int(slot) < 9, "weight slot out of range")
    flat_k = int(flat_k)
    channels = int(channels)
    require(flat_k >= 0 and channels > 0, "invalid flattened K")
    _tap, channel = divmod(flat_k, channels)
    bank = channel % WEIGHT_BANKS
    channel_within_tile = channel % WEIGHT_SOURCE_TILE
    half = channel_within_tile // WEIGHT_BANKS
    local_row = (int(slot) * WEIGHT_SLOT_BYTES_PER_BANK +
                 half * WEIGHT_ROWS_PER_SOURCE * 16)
    require(local_row + WEIGHT_ROWS_PER_SOURCE * 16 <=
            9 * WEIGHT_SLOT_BYTES_PER_BANK,
            "weight slot local row exceeds physical bank")
    return bank, local_row


@dataclass(frozen=True)
class OracleBundle:
    m712: object
    m722r2: object
    storage: object


def load_pinned_oracles(m712_path: Path, m712_sha256: str,
                        m722r2_path: Path, m722r2_sha256: str,
                        storage_path: Path = STORAGE_ORACLE_PATH,
                        storage_sha256: str = STORAGE_ORACLE_SHA256) -> OracleBundle:
    return OracleBundle(
        load_pinned_module(m712_path, m712_sha256, "m712_oracle"),
        load_pinned_module(m722r2_path, m722r2_sha256, "m722r2_oracle"),
        load_pinned_module(storage_path, storage_sha256, "m785_storage_oracle"),
    )


def _expected_m722_line_buffer_plan(
    module_index: int,
    geometry: Tuple[int, int, int, int, int, int],
) -> Dict[str, object]:
    """Independent strict recompute of every M722 a1_storage_plan field."""
    cin, cout, hin, win, hout, wout = geometry
    blocks = math.ceil(cout / 96)
    name = "D{}".format(module_index)
    budget, control, weight = 240 * 1024, 8192, 16 * 96 * 3 * 3
    available = budget - control - weight
    full = 3 * (((wout * 96 * 3 + 127) // 128) * 128)
    if full <= available:
        stripe_width = wout
    else:
        raw_width = available // (3 * 96 * 3)
        stripe_width = (raw_width // 64) * 64
        require(stripe_width >= 64, "no legal M722 line-buffer stripe")
    stripes = [(lo, min(wout, lo + stripe_width))
               for lo in range(0, wout, stripe_width)]

    def columns(lo: int, hi: int) -> List[int]:
        return [sx for sx in range(win)
                if any(lo <= ox < hi
                       for ox in (2 * sx - 1, 2 * sx, 2 * sx + 1))]

    source_columns = [columns(lo, hi) for lo, hi in stripes]
    backing = 3 * (((min(stripe_width, wout) * 96 * 3 + 127) // 128) * 128)
    require(backing + control + weight <= budget,
            "independent M722 storage recompute exceeds budget")
    return {
        "module": name,
        "accumulator": "Acc24",
        "stripe_width": stripe_width,
        "stripe_count": len(stripes),
        "stripes": [list(pair) for pair in stripes],
        "summed_source_columns": sum(len(value) for value in source_columns),
        "unique_source_columns": len(set(
            column for value in source_columns for column in value)),
        "source_column_overlap": (
            sum(len(value) for value in source_columns) - win),
        "onchip_psum_backing_bytes": backing,
        "control_bytes": control,
        "weight_tile_bytes": weight,
        "total_bytes": backing + control + weight,
        "offchip_psum_spill_bytes": 0,
        "model": True,
    }


def verify_contributor_and_storage_oracles(
    bits: np.ndarray,
    module_index: int,
    geometry: Tuple[int, int, int, int, int, int],
    observed_contributors_per_block: int,
    observed_groups_per_block: int,
    oracles: OracleBundle,
    psum_bytes: int = 221184,
) -> Dict[str, object]:
    """Strictly validate contributor and both non-equivalent storage plans."""
    cin, cout, hin, win, hout, wout = geometry
    blocks = math.ceil(cout / 96)
    require(tuple(bits.shape) == (cin, hin, win), "oracle plane shape drift")
    _counts, _active, m712_contributors, _groups = (
        oracles.m712.descriptor_counts(bits, blocks))
    m722_counts = oracles.m722r2.R1.group_counts(bits, blocks)
    expected_contributors = int(observed_contributors_per_block) * blocks
    expected_groups = int(observed_groups_per_block) * blocks
    require(int(m712_contributors) == expected_contributors,
            "M712 contributor oracle mismatch")
    require(int(m722_counts["contributors"]) == expected_contributors,
            "M722 contributor oracle mismatch")
    require(int(m722_counts["osg_groups"]) == expected_groups,
            "M722 OSG group oracle mismatch")

    m722_spec = ("D{}".format(module_index), cin, cout, hin, win,
                 hout, wout, blocks)
    m722_plan = oracles.m722r2.R1.a1_storage_plan(m722_spec)
    require(dict(m722_plan) == _expected_m722_line_buffer_plan(
        module_index, geometry),
        "M722 line-buffer storage plan mismatch")

    storage_plan = oracles.storage.plan(geometry, psum_bytes)
    independently_recomputed = STORAGE.plan(geometry, psum_bytes)
    require(dict(storage_plan) == independently_recomputed,
            "M785 independent storage oracle mismatch")
    require(storage_plan["model"] == "GLOBAL_VECTOR_LRU_DIRTY_BACKING" and
            storage_plan["m722_line_buffer_storage_equivalent"] is False,
            "M785 storage model boundary drift")
    require(int(storage_plan["stripe_count"]) ==
            len(psum_stripes(hout * wout * blocks, psum_bytes)),
            "M785 global-vector stripe count mismatch")
    require(int(storage_plan["psum_partition_bytes"]) == psum_bytes and
            int(storage_plan["offchip_backing_address_span_bytes"]) ==
            ((hout * wout * blocks * PSUM_VECTOR_BYTES)
             if hout * wout * blocks > psum_bytes // PSUM_VECTOR_BYTES else 0),
            "M785 storage bytes/offchip mismatch")
    return {
        "m712_contributors": int(m712_contributors),
        "m722_contributors": int(m722_counts["contributors"]),
        "m722_osg_groups": int(m722_counts["osg_groups"]),
        "m722_role": "CONTRIBUTOR_GROUP_ORACLE_ONLY",
        "m722_line_buffer_plan_sha256": canonical_sha256(m722_plan),
        "m722_storage_equivalent_to_m785": False,
        "m785_storage_plan_sha256": storage_plan["plan_sha256"],
        "m785_stripe_count": int(storage_plan["stripe_count"]),
        "m785_offchip_backing_address_span_bytes": int(
            storage_plan["offchip_backing_address_span_bytes"]),
    }


def _terminal_token(transaction: M777.CompressedTransaction) -> str:
    require(bool(transaction.produces_token_prefix),
            "transaction lacks completion token")
    return "{}:{}".format(transaction.produces_token_prefix,
                          transaction.count - 1)


def residency_transactions(
    prefix: str, population_id: str, config: str,
    events: Sequence[ResidencyEvent],
    dependency_by_key: Optional[Mapping[int, str]] = None,
) -> List[M777.CompressedTransaction]:
    """Charge local+external psum moves and serialize shared-slot reuse."""
    output: List[M777.CompressedTransaction] = []
    dependency_by_key = dependency_by_key or {}
    slot_terminal: Dict[int, str] = {}
    for ordinal, event in enumerate(events):
        dependencies = []
        victim = dependency_by_key.get(event.key)
        if victim:
            dependencies.append(victim)
        if event.slot in slot_terminal:
            dependencies.append(slot_terminal[event.slot])
        local_base = event.slot * PSUM_VECTOR_BYTES
        root = "{}:res{}:{}".format(prefix, ordinal, event.kind)
        if event.kind == "evict":
            local_id = root + ":local_psum_read"
            local = CompressedTransaction(
                local_id, population_id, config, "psum_read", local_base, 0,
                1, tuple(range(PSUM_BANKS)), PSUM_BANK_ROW_BYTES,
                dependency_tokens=tuple(dependencies),
                produces_token_prefix=local_id + ":done")
            output.append(local)
            external = M777._external_transaction(
                root + ":external_write", population_id, config,
                "external_write", event.backing_address, PSUM_VECTOR_BYTES,
                (_terminal_token(local),), root + ":external_write:done")
            output.append(external)
            slot_terminal[event.slot] = _terminal_token(external)
        elif event.kind == "restore":
            external = M777._external_transaction(
                root + ":external_read", population_id, config,
                "external_read", event.backing_address, PSUM_VECTOR_BYTES,
                tuple(dependencies), root + ":external_read:done")
            output.append(external)
            local_id = root + ":local_psum_write"
            local = CompressedTransaction(
                local_id, population_id, config, "psum_write", local_base, 0,
                1, tuple(range(PSUM_BANKS)), PSUM_BANK_ROW_BYTES,
                dependency_tokens=(_terminal_token(external),),
                produces_token_prefix=local_id + ":done")
            output.append(local)
            slot_terminal[event.slot] = _terminal_token(local)
        else:
            raise Failure("unknown psum residency event")
    return output


def _weight_refill_transactions(
    prefix: str, population_id: str, config: str,
    output_block: int, tap: int, source_tile: int, slot: int,
    source_dependency: str, overwrite_dependencies: Sequence[str],
    refill_ordinal: int,
) -> List[M777.CompressedTransaction]:
    root = "{}:rf{}:ob{}:tap{}:st{}:slot{}".format(
        prefix, refill_ordinal, output_block, tap, source_tile, slot)
    external = M777._external_transaction(
        root + ":weight_refill_external", population_id, config,
        "external_read", ((4 << 60) | (output_block << 48) |
                          (tap << 40) | (source_tile << 16)),
        WEIGHT_TILE_BYTES, (source_dependency,),
        root + ":weight_refill_external:done")
    dependencies = [_terminal_token(external)]
    dependencies.extend(sorted(set(overwrite_dependencies)))
    local_id = root + ":weight_refill_local_write"
    local = CompressedTransaction(
        local_id, population_id, config, "weight_write",
        slot * WEIGHT_SLOT_BYTES_PER_BANK, 16, 12,
        tuple(range(WEIGHT_BANKS)), 16,
        address_offsets=(0,) * WEIGHT_BANKS,
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=local_id + ":done")
    return [external, local]


def iter_record_transactions(
    mapper, record: Mapping[str, object], payload_root: Path,
    population_id: str, config: str, timestep: int, oracles: OracleBundle,
    tile_m: int = 256,
    geometry: Mapping[int, Tuple[int, int, int, int, int, int]] = MODULE_GEOMETRY,
    psum_bytes: int = 221184,
) -> Iterator[M777.CompressedTransaction]:
    """Generate M785 physical, dependency-closed source-only transactions."""
    require(config in CONFIGS, "unknown configuration")
    module_index = int(record["module_index"])
    require(module_index in geometry, "unknown decoder module")
    spec = geometry[module_index]
    cin, cout, hin, win, hout, wout = spec
    shape = tuple(int(value) for value in record["input_shape"])
    require(shape == (10, 1, cin, hin, win), "decoder record shape drift")
    require(0 <= timestep < shape[0], "timestep out of range")
    payload = Path(payload_root).joinpath(
        *safe_member(str(record["relative_path"])).parts)
    require(payload.is_file() and not payload.is_symlink(),
            "decoder payload is not a regular file")
    require(sha256(payload) == str(record["packed_sha256"]),
            "decoder payload identity drift")
    if module_index == 1:
        yield from _d1_transactions(record, population_id, config,
                                    timestep, spec)
        return

    blocks = math.ceil(cout / 96)
    prefix = "{}:{}:m{}:t{}".format(
        population_id, config, module_index, timestep)
    source_tx = _source_read(
        prefix, population_id, config, module_index, timestep,
        math.ceil(cin * hin * win / 8))
    yield source_tx
    source_done = "{}:source_fetch_done:{}".format(
        prefix, source_tx.count - 1)
    contributors = collect_mapper_contributors(
        mapper, payload, shape, timestep, tile_m, Path(payload_root))
    destinations = sorted(contributors)
    observed = sum(len(contributors[key]) for key in destinations)
    bits = unpack_timestep(payload, shape, timestep)
    osg_groups = sum(len(service_groups(contributors[key], "A1_OSG", cin))
                     for key in destinations)
    verify_contributor_and_storage_oracles(
        bits, module_index, spec, observed, osg_groups, oracles, psum_bytes)

    total_vectors = hout * wout * blocks
    stripe_by_vector: Dict[int, int] = {}
    for stripe in psum_stripes(total_vectors, psum_bytes):
        for vector in range(stripe.vector_lo, stripe.vector_hi):
            stripe_by_vector[vector] = stripe.index
    residency = PsumResidency(psum_bytes)
    weight_residency = WeightResidency()
    weight_ready: Dict[Tuple[int, int, int, int], str] = {}
    # A slot refill writes all eight banks, so eviction must wait for the most
    # recent read token of every bank used by the victim key, not merely the
    # most recently generated read in Python order.
    weight_last_use: Dict[
        Tuple[int, int, int, int], Dict[int, str]
    ] = {}
    commit_dependency: Dict[int, str] = {}
    group_ordinal = refill_ordinal = descriptor_count = source_services = 0

    for destination in destinations:
        for output_block in range(blocks):
            vector_key = destination * blocks + output_block
            require(vector_key in stripe_by_vector, "missing legal psum stripe")
            psum_slot, events = residency.acquire(vector_key)
            moves = residency_transactions(
                prefix + ":v{}".format(vector_key), population_id, config,
                events, commit_dependency)
            for row in moves:
                yield row
            slot_ready = _terminal_token(moves[-1]) if moves else None
            previous = commit_dependency.get(vector_key)

            for group in service_groups(contributors[destination], config, cin):
                descriptors = _descriptor_transactions(
                    prefix, population_id, config, group, group_ordinal,
                    source_done)
                descriptor_count += len(descriptors)
                for row in descriptors:
                    yield row
                descriptor_tokens = tuple(_terminal_token(row)
                                          for row in descriptors)

                item_state = []
                for flat_k, _source in group:
                    tap, channel = divmod(flat_k, cin)
                    source_tile = channel // WEIGHT_SOURCE_TILE
                    key = (stripe_by_vector[vector_key], output_block,
                           tap, source_tile)
                    access = weight_residency.access(key)
                    if access.evicted_key is not None:
                        weight_ready.pop(access.evicted_key, None)
                    if access.miss:
                        refill = _weight_refill_transactions(
                            prefix, population_id, config, output_block, tap,
                            source_tile, access.slot, source_done,
                            tuple(weight_last_use.pop(
                                access.evicted_key, {}).values())
                            if access.evicted_key is not None else (),
                            refill_ordinal)
                        refill_ordinal += 1
                        for row in refill:
                            yield row
                        weight_ready[key] = _terminal_token(refill[-1])
                    require(key in weight_ready,
                            "resident weight tile lacks local refill completion")
                    bank, offset = weight_slot_bank_and_local_row(
                        access.slot, flat_k, cin)
                    item_state.append((key, bank, offset, weight_ready[key]))

                weight_tokens: List[str] = []
                if config == "TYPED_SIGNED_K8":
                    banks = tuple(value[1] for value in item_state)
                    require(len(banks) == len(set(banks)),
                            "typed K8 physical weight bank collision")
                    weight_id = prefix + ":g{}:typed_weight".format(group_ordinal)
                    row = CompressedTransaction(
                        weight_id, population_id, config, "weight_read", 0, 16,
                        WEIGHT_ROWS_PER_SOURCE, banks, 16,
                        address_offsets=tuple(value[2] for value in item_state),
                        dependency_tokens=descriptor_tokens + tuple(
                            value[3] for value in item_state),
                        produces_token_prefix=weight_id + ":done")
                    yield row
                    token = _terminal_token(row)
                    weight_tokens.append(token)
                    for key, bank, _offset, _ready in item_state:
                        weight_last_use.setdefault(key, {})[bank] = token
                else:
                    for lane, (key, bank, offset, ready) in enumerate(item_state):
                        weight_id = prefix + ":g{}:k1_weight{}".format(
                            group_ordinal, lane)
                        row = CompressedTransaction(
                            weight_id, population_id, config, "weight_read",
                            offset, 16, WEIGHT_ROWS_PER_SOURCE, (bank,), 16,
                            address_offsets=(0,),
                            dependency_tokens=descriptor_tokens + (ready,),
                            produces_token_prefix=weight_id + ":done")
                        yield row
                        token = _terminal_token(row)
                        weight_tokens.append(token)
                        weight_last_use.setdefault(key, {})[bank] = token

                local_base = residency.local_base(psum_slot)
                read_id = prefix + ":g{}:psum_read".format(group_ordinal)
                read_dependencies = ([previous] if previous else [])
                if slot_ready:
                    read_dependencies.append(slot_ready)
                read = CompressedTransaction(
                    read_id, population_id, config, "psum_read", local_base, 0,
                    1, tuple(range(PSUM_BANKS)), PSUM_BANK_ROW_BYTES,
                    dependency_tokens=tuple(read_dependencies),
                    produces_token_prefix=read_id + ":done")
                yield read
                compute_id = prefix + ":g{}:compute".format(group_ordinal)
                compute = CompressedTransaction(
                    compute_id, population_id, config, "compute", 0, 0, 1,
                    (0,), 288,
                    dependency_tokens=tuple(weight_tokens) +
                    (_terminal_token(read),),
                    produces_token_prefix=compute_id + ":done")
                yield compute
                write_id = prefix + ":g{}:psum_write".format(group_ordinal)
                write = CompressedTransaction(
                    write_id, population_id, config, "psum_write", local_base,
                    0, 1, tuple(range(PSUM_BANKS)), PSUM_BANK_ROW_BYTES,
                    dependency_tokens=(_terminal_token(compute),),
                    produces_token_prefix=write_id + ":done")
                yield write
                previous = _terminal_token(write)
                residency.mark_dirty(vector_key)
                source_services += len(group)
                group_ordinal += 1
            if previous:
                commit_dependency[vector_key] = previous

    require(source_services == observed * blocks,
            "configuration source-service conservation mismatch")
    require(descriptor_count > 0 or observed == 0,
            "descriptor construction vanished")
    for ordinal, address in enumerate(dense_commit_addresses(
            module_index, timestep, hout, wout, blocks)):
        vector_key = ordinal
        _slot, events = residency.acquire(vector_key)
        moves = residency_transactions(
            prefix + ":commit_v{}".format(vector_key), population_id, config,
            events, commit_dependency)
        for row in moves:
            yield row
        dependencies = []
        if commit_dependency.get(vector_key):
            dependencies.append(commit_dependency[vector_key])
        if moves:
            dependencies.append(_terminal_token(moves[-1]))
        commit_id = prefix + ":commit{}".format(ordinal)
        commit = CompressedTransaction(
            commit_id, population_id, config, "commit",
            (1 << 60) | address, 0, 1, (0,), OUTPUT_COMMIT_BYTES,
            dependency_tokens=tuple(dependencies),
            produces_token_prefix=commit_id + ":done")
        yield commit
        residency.mark_committed(vector_key)


def validate_source_contract(repo_root: Path,
                             contract_path: Path) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(isinstance(contract, dict), "contract must be an object")
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema drift")
    require(contract.get("launch_now") is False and
            contract.get("production_speedup_allowed") is False,
            "M785 source contract must remain fail-closed")
    hw = repo_root / "hw_autoresearch_nts07"
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    checked: Dict[str, object] = {}
    for name in ("m781_failure_review", "primary_m686", "primary_m692_review",
                 "secondary_m699", "secondary_m705_review"):
        row = contract["inputs"][name]
        directory = hw / row["directory"]
        identity = verify_sealed_directory(directory)
        if "review_json_sha256" in row:
            require(sha256(directory / "review.json") ==
                    row["review_json_sha256"], name + " review drift")
        if "manifest_sha256" in row:
            require(sha256(directory / "manifest.json") ==
                    row["manifest_sha256"], name + " manifest drift")
        if "outer_seal_file_sha256" in row:
            require(identity["outer_seal_file_sha256"] ==
                    row["outer_seal_file_sha256"], name + " outer seal drift")
        checked[name] = identity
    for name in ("m777_substrate", "m768_substrate", "m672_mapper",
                 "m712_oracle", "m722r2_oracle", "m785_storage_oracle"):
        row = contract["inputs"][name]
        require(sha256(hw / row["path"]) == row["sha256"],
                name + " identity drift")
        checked[name] = row["sha256"]
    for name, row in contract["source_files"].items():
        require(sha256(hw / row["path"]) == row["sha256"],
                "M785 source identity drift: " + name)
        checked["source_" + name] = row["sha256"]
    resource = resource_from_contract(contract)
    return {
        "status": "PASS_M785_SOURCE_IDENTITY_ONLY__NO_PRODUCTION_RUN",
        "contract_sha256": sha256(contract_path),
        "resource": resource.identity(),
        "checked_inputs": checked,
        "launch_now": False,
        "production_cycles": None,
        "production_speedup": None,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
    }


def synthetic_self_test() -> Dict[str, object]:
    residency = PsumResidency(PSUM_VECTOR_BYTES)
    residency.acquire(0)
    residency.mark_dirty(0)
    slot, events = residency.acquire(1)
    moves = residency_transactions(
        "m785", "SYNTHETIC", "TYPED_SIGNED_K8", events,
        {0: "victim:done:0"})
    require(slot == 0 and [row.kind for row in moves] ==
            ["psum_read", "external_write"],
            "psum movement self-test")
    cache = WeightResidency()
    access0 = cache.access((0, 0, 0, 0))
    access1 = cache.access((0, 1, 0, 0))
    require(access0.slot != access1.slot,
            "weight physical slot alias self-test")
    storage = STORAGE.plan((96, 96, 120, 160, 240, 320), 221184)
    require(storage["stripe_count"] == 100,
            "D3 storage oracle self-test")
    return {
        "status": "PASS_M785_SYNTHETIC_SOURCE_SELF_TEST",
        "production_cycles": None,
        "production_speedup": None,
        "launch_now": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source-contract", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-production", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(synthetic_self_test(), sort_keys=True))
        return 0
    require(args.validate_source_contract and args.contract,
            "only self-test/source validation is authorized")
    require(not args.run_production,
            "production replay is fail-closed in M785")
    require(args.output is None,
            "source-only validation refuses result output")
    print(json.dumps(validate_source_contract(args.repo_root, args.contract),
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
