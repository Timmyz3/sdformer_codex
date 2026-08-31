#!/usr/bin/env python3
"""M777 source-only repair of the M768 decoder address-timed model.

M768 is retained as an immutable identity/input/scheduler substrate.  This
additive module closes the three M773 P0 findings without authorising a
population replay:

* A1-OSG, equal-service K1x8 and typed K8 have different descriptor and
  transaction construction, while conserving the same contributors, 96
  product lanes, resource manifest and dense commit sequence;
* Acc24 partial sums use a bounded 221184-byte, six-bank residency and a
  deterministic stripe/LRU backing protocol rather than unbounded addresses;
* flattened K uses tap-major, source-channel-div-8 bank-local rows.

Binary source reads, descriptor reads and weight refills are charged on the
frozen 192-byte/cycle external port.  M712 contributor and M722-r2
contributor/storage routines are executable oracles, not SHA-only labels.
D1 uses one common full-shape/density fallback for all configurations and is
still diagnostic because folded-theta checkpoint numerics are not admitted.

This file deliberately has no production-result writer.  ``launch_now`` is
false and ``--run-production`` is rejected.
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
M768_PATH = HERE / "analyze_m768_h67_decoder_a1_k8_address_timed_cycles.py"
M768_SHA256 = "926069762c6274bae3aa7b88352e29fff8219cbbceba2f2be0ec46ee304a3f37"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if _file_sha256(M768_PATH) != M768_SHA256:
    raise RuntimeError("frozen M768 substrate identity drift")
_M768_SPEC = importlib.util.spec_from_file_location("m777_frozen_m768", M768_PATH)
if _M768_SPEC is None or _M768_SPEC.loader is None:
    raise RuntimeError("cannot import frozen M768 substrate")
M768 = importlib.util.module_from_spec(_M768_SPEC)
_M768_SPEC.loader.exec_module(M768)


Failure = M768.Failure
require = M768.require
sha256 = M768.sha256
strict_json = M768.strict_json
verify_sealed_directory = M768.verify_sealed_directory
safe_member = M768.safe_member
PortSpec = M768.PortSpec
CommonResource = M768.CommonResource
CompressedTransaction = M768.CompressedTransaction
Request = M768.Request
expand_transactions = M768.expand_transactions
canonical_sha256 = M768.canonical_sha256
DOCS359_SHA256 = M768.DOCS359_SHA256
CONFIGS = M768.CONFIGS
HEADLINE_PAIR = M768.HEADLINE_PAIR

CONTRACT_SCHEMA = "m777_h67_decoder_a1_k8_address_timed_repair_contract_v1"
MODULE_GEOMETRY = M768.MODULE_GEOMETRY
PSUM_VECTOR_BYTES = 96 * 3
PSUM_BANKS = 6
PSUM_BANK_ROW_BYTES = 48
DESCRIPTOR_BYTES = 16
K8_DESCRIPTOR_BASE_BYTES = 16
K8_DESCRIPTOR_PER_SOURCE_BYTES = 4
WEIGHT_SOURCE_TILE = 16
WEIGHT_TILE_BYTES = WEIGHT_SOURCE_TILE * 96
OUTPUT_COMMIT_BYTES = 384


class AddressTimedScheduler(M768.AddressTimedScheduler):
    """M768 scheduler plus physical on-chip partition range checks."""

    def schedule(self, requests: Iterable[Request]) -> Dict[str, object]:
        rows = list(requests)
        for request in rows:
            if request.kind in ("psum_read", "psum_write"):
                for address in request.addresses:
                    require(
                        0 <= address
                        and address + request.width_bytes
                        <= self.resource.psum_bytes_logical,
                        "psum address exceeds 221184-byte physical partition",
                    )
            if request.kind == "weight_read":
                for address in request.addresses:
                    require(
                        0 <= address
                        and address + request.width_bytes
                        <= self.resource.weight_bytes_logical,
                        "weight address exceeds physical partition",
                    )
        return super().schedule(rows)


def route_for_record(module_index: int, config: str) -> Dict[str, object]:
    return M768.route_for_record(module_index, config)


def headline_ratio_allowed(numerator: str, denominator: str) -> bool:
    return M768.headline_ratio_allowed(numerator, denominator)


def dense_commit_addresses(module_index: int, timestep: int,
                           output_height: int, output_width: int,
                           output_blocks: int) -> List[int]:
    return M768.dense_commit_addresses(
        module_index, timestep, output_height, output_width, output_blocks)


def commit_address_hash(addresses: Sequence[int], width_bytes: int = 384) -> str:
    return M768.commit_address_hash(addresses, width_bytes)


def bank_unique_groups(flat_k_indices: Sequence[int], channels: int,
                       bank_count: int = 8) -> List[Tuple[int, ...]]:
    return M768.bank_unique_groups(flat_k_indices, channels, bank_count)


def weight_bank_and_local_row(flat_k: int, channels: int,
                              bank_count: int = 8) -> Tuple[int, int]:
    """Return (bank, bank-local byte row) for tap-major flattened K.

    The row is ``tap -> source_channel_div_8``.  The same local row exists in
    all eight banks.  K=24 and K=25 with C=8 therefore both select byte row
    48, in banks 0 and 1 respectively; using ``flat_k*16`` would incorrectly
    produce 384 and 400.
    """
    require(channels > 0 and bank_count == 8, "invalid weight layout")
    flat_k = int(flat_k)
    require(flat_k >= 0, "negative flattened K")
    tap, channel = divmod(flat_k, channels)
    bank = channel % bank_count
    channel_div8 = channel // bank_count
    rows_per_tap = math.ceil(channels / bank_count)
    row = tap * rows_per_tap + channel_div8
    return bank, row * 16


def weight_group_layout(group: Sequence[int], channels: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    pairs = [weight_bank_and_local_row(value, channels) for value in group]
    banks = tuple(pair[0] for pair in pairs)
    require(len(banks) == len(set(banks)),
            "weight service group contains a bank collision")
    return banks, tuple(pair[1] for pair in pairs)


@dataclass(frozen=True)
class Stripe:
    index: int
    vector_lo: int
    vector_hi: int


def psum_stripes(total_vectors: int, psum_bytes: int = 221184) -> List[Stripe]:
    require(total_vectors >= 1, "empty psum population")
    require(psum_bytes % PSUM_VECTOR_BYTES == 0,
            "psum partition is not an integral Acc24 vector population")
    capacity = psum_bytes // PSUM_VECTOR_BYTES
    require(capacity >= 1, "no Acc24 vector fits")
    return [Stripe(index, lo, min(total_vectors, lo + capacity))
            for index, lo in enumerate(range(0, total_vectors, capacity))]


@dataclass(frozen=True)
class ResidencyEvent:
    kind: str
    key: int
    slot: int
    backing_address: int


class PsumResidency:
    """Deterministic bounded LRU with explicit dirty backing actions."""

    def __init__(self, psum_bytes: int = 221184,
                 backing_base: int = 1 << 61):
        require(psum_bytes > 0 and psum_bytes % PSUM_VECTOR_BYTES == 0,
                "invalid psum physical partition")
        self.psum_bytes = int(psum_bytes)
        self.capacity = self.psum_bytes // PSUM_VECTOR_BYTES
        self.backing_base = int(backing_base)
        self.key_to_slot: Dict[int, int] = {}
        self.slot_to_key: Dict[int, int] = {}
        self.dirty: Set[int] = set()
        self.backed: Set[int] = set()
        self.age: Dict[int, int] = {}
        self.tick = 0

    def local_base(self, slot: int) -> int:
        require(0 <= slot < self.capacity, "psum slot out of range")
        value = slot * PSUM_VECTOR_BYTES
        require(value + PSUM_VECTOR_BYTES <= self.psum_bytes,
                "psum local address exceeds partition")
        return value

    def _backing_address(self, key: int) -> int:
        return self.backing_base + int(key) * PSUM_VECTOR_BYTES

    def acquire(self, key: int) -> Tuple[int, List[ResidencyEvent]]:
        key = int(key)
        require(key >= 0, "negative psum key")
        self.tick += 1
        if key in self.key_to_slot:
            self.age[key] = self.tick
            return self.key_to_slot[key], []
        events: List[ResidencyEvent] = []
        if len(self.slot_to_key) < self.capacity:
            slot = min(set(range(self.capacity)) - set(self.slot_to_key))
        else:
            victim = min(self.key_to_slot, key=lambda value: (self.age[value], value))
            slot = self.key_to_slot.pop(victim)
            self.slot_to_key.pop(slot)
            if victim in self.dirty:
                events.append(ResidencyEvent(
                    "evict", victim, slot, self._backing_address(victim)))
                self.backed.add(victim)
                self.dirty.remove(victim)
        self.key_to_slot[key] = slot
        self.slot_to_key[slot] = key
        self.age[key] = self.tick
        if key in self.backed:
            events.append(ResidencyEvent(
                "restore", key, slot, self._backing_address(key)))
        return slot, events

    def mark_dirty(self, key: int) -> None:
        require(int(key) in self.key_to_slot, "dirty psum is not resident")
        self.dirty.add(int(key))

    def mark_committed(self, key: int) -> None:
        require(int(key) in self.key_to_slot, "committed psum is not resident")
        self.dirty.discard(int(key))
        self.backed.discard(int(key))


class WeightResidency:
    """Read-only nine-tile LRU for the frozen 13824-byte weight partition."""

    def __init__(self, weight_bytes: int = 13824,
                 tile_bytes: int = WEIGHT_TILE_BYTES):
        require(weight_bytes > 0 and tile_bytes > 0 and
                weight_bytes % tile_bytes == 0,
                "invalid weight tile partition")
        self.capacity = weight_bytes // tile_bytes
        require(self.capacity == 9, "M777 freezes nine resident weight tiles")
        self.age: Dict[Tuple[int, int, int, int], int] = {}
        self.tick = 0

    def access(self, key: Tuple[int, int, int, int]) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        self.tick += 1
        if key in self.age:
            self.age[key] = self.tick
            return False, None
        evicted = None
        if len(self.age) == self.capacity:
            evicted = min(self.age, key=lambda value: (self.age[value], value))
            del self.age[evicted]
        self.age[key] = self.tick
        return True, evicted


def load_pinned_module(path: Path, expected_sha256: str, name: str):
    require(sha256(path) == expected_sha256, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m777_" + name, path)
    require(spec is not None and spec.loader is not None,
            "cannot import pinned " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class OracleBundle:
    m712: object
    m722r2: object


def load_pinned_oracles(m712_path: Path, m712_sha256: str,
                        m722r2_path: Path, m722r2_sha256: str) -> OracleBundle:
    return OracleBundle(
        load_pinned_module(m712_path, m712_sha256, "m712_oracle"),
        load_pinned_module(m722r2_path, m722r2_sha256, "m722r2_oracle"),
    )


def unpack_timestep(payload: Path, shape: Sequence[int], timestep: int) -> np.ndarray:
    logical = math.prod(int(value) for value in shape)
    packed = np.fromfile(str(payload), dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")[:logical]
    require(bits.size == logical, "bitpack length mismatch")
    tensor = bits.reshape(tuple(int(value) for value in shape))
    return tensor[int(timestep), 0].astype(np.uint8, copy=False)


def verify_contributor_and_storage_oracles(
    bits: np.ndarray,
    module_index: int,
    geometry: Tuple[int, int, int, int, int, int],
    observed_contributors_per_block: int,
    observed_groups_per_block: int,
    oracles: OracleBundle,
) -> Dict[str, object]:
    """Execute both frozen oracles and fail on any conservation mismatch."""
    cin, cout, hin, win, hout, wout = geometry
    blocks = math.ceil(cout / 96)
    require(tuple(bits.shape) == (cin, hin, win), "oracle plane shape drift")
    _counts, _active, m712_contributors, _m712_groups = (
        oracles.m712.descriptor_counts(bits, blocks))
    m722 = oracles.m722r2.R1
    m722_counts = m722.group_counts(bits, blocks)
    spec = ("D{}".format(module_index), cin, cout, hin, win,
            hout, wout, blocks)
    storage = m722.a1_storage_plan(spec)
    expected = int(observed_contributors_per_block) * blocks
    expected_groups = int(observed_groups_per_block) * blocks
    require(int(m712_contributors) == expected,
            "M712 contributor oracle mismatch")
    require(int(m722_counts["contributors"]) == expected,
            "M722 contributor oracle mismatch")
    require(int(m722_counts["osg_groups"]) == expected_groups,
            "M722 OSG group oracle mismatch")
    require(int(storage["onchip_psum_backing_bytes"]) <= 221184,
            "M722 storage oracle exceeds psum partition")
    require(int(storage["offchip_psum_spill_bytes"]) == 0,
            "M722 storage oracle unexpectedly spills")
    return {
        "m712_contributors": int(m712_contributors),
        "m722_contributors": int(m722_counts["contributors"]),
        "m722_osg_groups": int(m722_counts["osg_groups"]),
        "m722_storage_plan_sha256": canonical_sha256(storage),
        "m722_stripe_count": int(storage["stripe_count"]),
    }


def _chunk(values: Sequence[Tuple[int, int]], count: int) -> Iterator[List[Tuple[int, int]]]:
    for start in range(0, len(values), count):
        yield list(values[start:start + count])


def service_groups(contributors: Sequence[Tuple[int, int]], config: str,
                   channels: int) -> List[Tuple[Tuple[int, int], ...]]:
    """Construct configuration-specific executable service groups.

    Contributors are ``(flat_k, source_flat_index)``.  A1 preserves local
    source order and splits every eight-source OSG packet at bank collisions.
    K1x8/K8 use the same bank-unique grouping (equal physical source service),
    but K1x8 fetches eight independent descriptors/weight requests while K8
    fetches and issues one typed bundle.
    """
    require(config in CONFIGS, "unknown configuration")
    values = [(int(k), int(source)) for k, source in contributors]
    require(all(k >= 0 and source >= 0 for k, source in values),
            "negative contributor identity")
    if config == "A1_OSG":
        output: List[Tuple[Tuple[int, int], ...]] = []
        # The frozen M712/M722 OSG oracle forms groups independently inside
        # each 16-channel source tile.  A logical group may contain duplicate
        # physical weight banks; those accesses are emitted as independent K1
        # requests below and therefore serialize honestly at the bank port.
        by_source_tile: Dict[int, List[Tuple[int, int]]] = {}
        for item in values:
            channel = item[0] % channels
            by_source_tile.setdefault(channel // 16, []).append(item)
        for source_tile in sorted(by_source_tile):
            output.extend(tuple(packet) for packet in
                          _chunk(by_source_tile[source_tile], 8))
        return output
    by_flat = {k: source for k, source in values}
    require(len(by_flat) == len(values), "duplicate flattened K contributor")
    return [tuple((k, by_flat[k]) for k in group)
            for group in bank_unique_groups(list(by_flat), channels, 8)]


def _external_transaction(identifier: str, population_id: str, config: str,
                          kind: str, address: int, byte_count: int,
                          dependencies: Sequence[str] = (),
                          produces: str = "") -> CompressedTransaction:
    require(kind in ("external_read", "external_write"),
            "bad external transaction kind")
    require(byte_count >= 1, "empty external transaction")
    return CompressedTransaction(
        transaction_id=identifier,
        population_id=population_id,
        config=config,
        kind=kind,
        base_address=int(address),
        address_stride_bytes=192,
        count=math.ceil(byte_count / 192),
        bank_pattern=(0,),
        width_bytes=min(192, int(byte_count)),
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=produces,
    )


def residency_transactions(
    prefix: str, population_id: str, config: str,
    events: Sequence[ResidencyEvent],
    dependency_by_key: Optional[Mapping[int, str]] = None,
) -> List[CompressedTransaction]:
    output = []
    dependency_by_key = dependency_by_key or {}
    for ordinal, event in enumerate(events):
        identifier = "{}:res{}:{}".format(prefix, ordinal, event.kind)
        dependency = dependency_by_key.get(event.key)
        output.append(_external_transaction(
            identifier, population_id, config,
            "external_write" if event.kind == "evict" else "external_read",
            event.backing_address, PSUM_VECTOR_BYTES,
            dependencies=((dependency,) if dependency else ()),
            produces=identifier + ":done",
        ))
    return output


def _source_read(prefix: str, population_id: str, config: str,
                 module_index: int, timestep: int, source_bytes: int) -> CompressedTransaction:
    return _external_transaction(
        prefix + ":source_fetch", population_id, config, "external_read",
        (2 << 60) | (module_index << 48) | (timestep << 40), source_bytes,
        produces=prefix + ":source_fetch_done")


def _descriptor_transactions(prefix: str, population_id: str, config: str,
                             group: Sequence[Tuple[int, int]], ordinal: int,
                             source_dependency: str) -> List[CompressedTransaction]:
    base = (3 << 60) | (ordinal << 12)
    if config == "TYPED_SIGNED_K8":
        size = K8_DESCRIPTOR_BASE_BYTES + K8_DESCRIPTOR_PER_SOURCE_BYTES * len(group)
        identifier = prefix + ":g{}:typed_k8_descriptor".format(ordinal)
        return [_external_transaction(
            identifier, population_id, config, "external_read", base, size,
            (source_dependency,), identifier + ":done")]
    output = []
    if config == "A1_OSG":
        identifier = prefix + ":g{}:osg_header".format(ordinal)
        output.append(_external_transaction(
            identifier, population_id, config, "external_read",
            base + 0x800, 2 * DESCRIPTOR_BYTES,
            (source_dependency,), identifier + ":done"))
    for lane, _item in enumerate(group):
        identifier = prefix + ":g{}:k1_descriptor{}".format(ordinal, lane)
        output.append(_external_transaction(
            identifier, population_id, config, "external_read",
            base + lane * DESCRIPTOR_BYTES, DESCRIPTOR_BYTES,
            (source_dependency,), identifier + ":done"))
    return output


def _weight_refill(prefix: str, population_id: str, config: str,
                   output_block: int, tap: int, source_tile: int,
                   dependency: str, refill_ordinal: int) -> CompressedTransaction:
    identifier = "{}:rf{}:ob{}:tap{}:st{}:weight_refill".format(
        prefix, refill_ordinal, output_block, tap, source_tile)
    address = ((4 << 60) | (output_block << 48) |
               (tap << 40) | (source_tile << 16))
    return _external_transaction(
        identifier, population_id, config, "external_read", address,
        WEIGHT_TILE_BYTES, (dependency,), identifier + ":done")


def _d1_transactions(record: Mapping[str, object], population_id: str,
                     config: str, timestep: int,
                     geometry: Tuple[int, int, int, int, int, int]) -> Iterator[CompressedTransaction]:
    cin, cout, hin, win, hout, wout = geometry
    blocks = math.ceil(cout / 96)
    prefix = "{}:{}:m1:t{}:d1_fullshape".format(population_id, config, timestep)
    input_bytes = cin * hin * win * 4
    weight_bytes = cin * cout * 9 * 4
    input_tx = _external_transaction(
        prefix + ":input", population_id, config, "external_read",
        (5 << 60) | (timestep << 44), input_bytes,
        produces=prefix + ":input_done")
    yield input_tx
    weight_tx = _external_transaction(
        prefix + ":weights", population_id, config, "external_read",
        (6 << 60) | (timestep << 44), weight_bytes,
        ("{}:{}".format(prefix + ":input_done", input_tx.count - 1),),
        prefix + ":weights_done")
    yield weight_tx
    dense_products = cin * hin * win * 9 * cout
    compute_count = math.ceil(dense_products / 96)
    compute_id = prefix + ":compute"
    yield CompressedTransaction(
        transaction_id=compute_id,
        population_id=population_id,
        config=config,
        kind="compute",
        base_address=0,
        address_stride_bytes=0,
        count=compute_count,
        bank_pattern=(0,),
        width_bytes=288,
        dependency_tokens=("{}:{}".format(
            prefix + ":weights_done", weight_tx.count - 1),),
        produces_token_prefix=compute_id + ":done",
    )
    final_compute = "{}:done:{}".format(compute_id, compute_count - 1)
    commit_id = prefix + ":commit"
    yield CompressedTransaction(
        transaction_id=commit_id,
        population_id=population_id,
        config=config,
        kind="commit",
        base_address=(1 << 60) | (1 << 52) | (timestep << 44),
        address_stride_bytes=OUTPUT_COMMIT_BYTES,
        count=hout * wout * blocks,
        bank_pattern=(0,),
        width_bytes=OUTPUT_COMMIT_BYTES,
        dependency_tokens=(final_compute,),
    )


def collect_mapper_contributors(mapper, payload: Path, shape: Sequence[int],
                                timestep: int, tile_m: int,
                                trusted_root: Path) -> Dict[int, List[Tuple[int, int]]]:
    _t, _batch, cin, _hin, win = tuple(int(value) for value in shape)
    output_width = 2 * win
    output: Dict[int, List[Tuple[int, int]]] = {}
    for tile in mapper.iter_polyphase_tiles(
            payload, shape, tile_m=tile_m, trusted_root=trusted_root.resolve()):
        values = tile["values"][timestep]
        sources = tile["source_flat_index"]
        for local_m, (dy, dx) in enumerate(zip(
                tile["destination_y"], tile["destination_x"])):
            destination = int(dy) * output_width + int(dx)
            active = np.flatnonzero(values[local_m])
            rows = output.setdefault(destination, [])
            for flat_k in active:
                source = int(sources[local_m, int(flat_k)])
                require(source >= 0, "active contributor has invalid source index")
                rows.append((int(flat_k), source))
    return output


def iter_record_transactions(
    mapper,
    record: Mapping[str, object],
    payload_root: Path,
    population_id: str,
    config: str,
    timestep: int,
    oracles: OracleBundle,
    tile_m: int = 256,
    geometry: Mapping[int, Tuple[int, int, int, int, int, int]] = MODULE_GEOMETRY,
    psum_bytes: int = 221184,
) -> Iterator[CompressedTransaction]:
    """Generate a source-only, bounded and externally charged transaction stream."""
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
        yield from _d1_transactions(record, population_id, config, timestep, spec)
        return

    blocks = math.ceil(cout / 96)
    prefix = "{}:{}:m{}:t{}".format(
        population_id, config, module_index, timestep)
    source_bytes = math.ceil(cin * hin * win / 8)
    source_tx = _source_read(
        prefix, population_id, config, module_index, timestep, source_bytes)
    yield source_tx
    source_done = "{}:source_fetch_done:{}".format(prefix, source_tx.count - 1)

    contributors = collect_mapper_contributors(
        mapper, payload, shape, timestep, tile_m, Path(payload_root))
    ordered_destinations = sorted(contributors)
    observed_contributors = sum(len(contributors[key])
                                for key in ordered_destinations)
    # M722 OSG packs within 16-channel source tiles.  Recompute its exact
    # group count from the executable oracle rather than assuming ceil(total/8).
    bits = unpack_timestep(payload, shape, timestep)
    observed_osg_groups_per_block = sum(
        len(service_groups(contributors[key], "A1_OSG", cin))
        for key in ordered_destinations)
    verify_contributor_and_storage_oracles(
        bits, module_index, spec, observed_contributors,
        observed_osg_groups_per_block, oracles)

    total_vectors = hout * wout * blocks
    stripes = psum_stripes(total_vectors, psum_bytes)
    stripe_by_vector = {}
    for stripe in stripes:
        for vector in range(stripe.vector_lo, stripe.vector_hi):
            stripe_by_vector[vector] = stripe.index
    residency = PsumResidency(psum_bytes)
    group_ordinal = 0
    refill_ordinal = 0
    weight_residency = WeightResidency()
    weight_ready_token: Dict[Tuple[int, int, int, int], str] = {}
    commit_dependency: Dict[int, str] = {}
    descriptor_count = 0
    source_services = 0

    for destination in ordered_destinations:
        for output_block in range(blocks):
            vector_key = destination * blocks + output_block
            require(vector_key in stripe_by_vector, "missing legal psum stripe")
            slot, residency_events = residency.acquire(vector_key)
            residency_txs = residency_transactions(
                prefix + ":v{}".format(vector_key), population_id, config,
                residency_events, commit_dependency)
            for row in residency_txs:
                yield row
            previous = commit_dependency.get(vector_key)
            restore_tokens = tuple(
                "{}:done:{}".format(row.transaction_id, row.count - 1)
                for row in residency_txs if row.kind == "external_read")
            groups = service_groups(contributors[destination], config, cin)
            for group in groups:
                descriptors = _descriptor_transactions(
                    prefix, population_id, config, group, group_ordinal,
                    source_done)
                descriptor_count += len(descriptors)
                for descriptor in descriptors:
                    yield descriptor
                descriptor_tokens = tuple(
                    "{}:done:{}".format(tx.transaction_id, tx.count - 1)
                    for tx in descriptors)
                refill_tokens: List[str] = []
                for flat_k, _source in group:
                    tap, channel = divmod(flat_k, cin)
                    source_tile = channel // WEIGHT_SOURCE_TILE
                    key = (stripe_by_vector[vector_key], output_block,
                           tap, source_tile)
                    miss, evicted = weight_residency.access(key)
                    if evicted is not None:
                        weight_ready_token.pop(evicted, None)
                    if miss:
                        refill = _weight_refill(
                            prefix, population_id, config, output_block,
                            tap, source_tile, source_done, refill_ordinal)
                        yield refill
                        refill_ordinal += 1
                        weight_ready_token[key] = "{}:done:{}".format(
                            refill.transaction_id, refill.count - 1)
                    require(key in weight_ready_token,
                            "resident weight tile lacks refill completion token")
                    refill_tokens.append(weight_ready_token[key])
                if config == "TYPED_SIGNED_K8":
                    banks, offsets = weight_group_layout(
                        [item[0] for item in group], cin)
                else:
                    pairs = [weight_bank_and_local_row(item[0], cin)
                             for item in group]
                    banks = tuple(pair[0] for pair in pairs)
                    offsets = tuple(pair[1] for pair in pairs)
                weight_tokens: List[str] = []
                if config == "TYPED_SIGNED_K8":
                    weight_id = prefix + ":g{}:typed_weight".format(group_ordinal)
                    yield CompressedTransaction(
                        transaction_id=weight_id,
                        population_id=population_id,
                        config=config,
                        kind="weight_read",
                        base_address=0,
                        address_stride_bytes=0,
                        count=1,
                        bank_pattern=banks,
                        width_bytes=16,
                        address_offsets=offsets,
                        dependency_tokens=descriptor_tokens + tuple(refill_tokens),
                        produces_token_prefix=weight_id + ":done",
                    )
                    weight_tokens.append(weight_id + ":done:0")
                else:
                    for lane, (bank, offset) in enumerate(zip(banks, offsets)):
                        weight_id = prefix + ":g{}:k1_weight{}".format(
                            group_ordinal, lane)
                        yield CompressedTransaction(
                            transaction_id=weight_id,
                            population_id=population_id,
                            config=config,
                            kind="weight_read",
                            base_address=offset,
                            address_stride_bytes=0,
                            count=1,
                            bank_pattern=(bank,),
                            width_bytes=16,
                            dependency_tokens=descriptor_tokens + tuple(refill_tokens),
                            produces_token_prefix=weight_id + ":done",
                        )
                        weight_tokens.append(weight_id + ":done:0")
                local_base = residency.local_base(slot)
                read_id = prefix + ":g{}:psum_read".format(group_ordinal)
                yield CompressedTransaction(
                    transaction_id=read_id,
                    population_id=population_id,
                    config=config,
                    kind="psum_read",
                    base_address=local_base,
                    address_stride_bytes=0,
                    count=1,
                    bank_pattern=tuple(range(PSUM_BANKS)),
                    width_bytes=PSUM_BANK_ROW_BYTES,
                    dependency_tokens=((previous,) if previous else ()) +
                    restore_tokens,
                    produces_token_prefix=read_id + ":done",
                )
                compute_id = prefix + ":g{}:compute".format(group_ordinal)
                yield CompressedTransaction(
                    transaction_id=compute_id,
                    population_id=population_id,
                    config=config,
                    kind="compute",
                    base_address=0,
                    address_stride_bytes=0,
                    count=1,
                    bank_pattern=(0,),
                    width_bytes=288,
                    dependency_tokens=tuple(weight_tokens) +
                    (read_id + ":done:0",),
                    produces_token_prefix=compute_id + ":done",
                )
                write_id = prefix + ":g{}:psum_write".format(group_ordinal)
                yield CompressedTransaction(
                    transaction_id=write_id,
                    population_id=population_id,
                    config=config,
                    kind="psum_write",
                    base_address=local_base,
                    address_stride_bytes=0,
                    count=1,
                    bank_pattern=tuple(range(PSUM_BANKS)),
                    width_bytes=PSUM_BANK_ROW_BYTES,
                    dependency_tokens=(compute_id + ":done:0",),
                    produces_token_prefix=write_id + ":done",
                )
                previous = write_id + ":done:0"
                residency.mark_dirty(vector_key)
                source_services += len(group)
                group_ordinal += 1
            if previous:
                commit_dependency[vector_key] = previous

    require(source_services == observed_contributors * blocks,
            "configuration source-service conservation mismatch")
    require(descriptor_count > 0 or observed_contributors == 0,
            "descriptor construction vanished")
    for ordinal, address in enumerate(dense_commit_addresses(
            module_index, timestep, hout, wout, blocks)):
        vector_key = ordinal
        slot, events = residency.acquire(vector_key)
        residency_txs = residency_transactions(
            prefix + ":commit_v{}".format(vector_key), population_id, config,
            events, commit_dependency)
        for row in residency_txs:
            yield row
        dependency = commit_dependency.get(vector_key)
        restore_tokens = tuple(
            "{}:done:{}".format(row.transaction_id, row.count - 1)
            for row in residency_txs if row.kind == "external_read")
        commit_id = prefix + ":commit{}".format(ordinal)
        yield CompressedTransaction(
            transaction_id=commit_id,
            population_id=population_id,
            config=config,
            kind="commit",
            base_address=(1 << 60) | address,
            address_stride_bytes=0,
            count=1,
            bank_pattern=(0,),
            width_bytes=OUTPUT_COMMIT_BYTES,
            dependency_tokens=((dependency,) if dependency else ()) +
            restore_tokens,
            produces_token_prefix=commit_id + ":done",
        )
        residency.mark_committed(vector_key)


def resource_from_contract(contract: Mapping[str, object]) -> CommonResource:
    return M768.resource_from_contract(contract)


def normalized_population_records(manifest: Mapping[str, object],
                                  population_id: str) -> List[Dict[str, object]]:
    return M768.normalized_population_records(manifest, population_id)


def assert_population_isolation(population_ids: Iterable[str]) -> str:
    return M768.assert_population_isolation(population_ids)


def assert_fair_configs(resource_hash_by_config: Mapping[str, str],
                        commit_hash_by_config: Mapping[str, str],
                        fallback_policy_by_config: Mapping[str, str]) -> None:
    M768.assert_fair_configs(
        resource_hash_by_config, commit_hash_by_config,
        fallback_policy_by_config)


def validate_source_contract(repo_root: Path,
                             contract_path: Path) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(isinstance(contract, dict), "contract must be an object")
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema drift")
    require(contract.get("launch_now") is False,
            "M777 source contract must not authorize production")
    require(contract.get("production_speedup_allowed") is False,
            "M777 source contract must forbid speedup")
    hw = repo_root / "hw_autoresearch_nts07"
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    resource = resource_from_contract(contract)
    inputs = contract["inputs"]
    checked: Dict[str, object] = {}
    m773 = inputs["m773_failure_review"]
    m773_directory = hw / m773["directory"]
    m773_identity = verify_sealed_directory(m773_directory)
    require(sha256(m773_directory / "review.json") ==
            m773["review_json_sha256"], "M773 review identity drift")
    checked["m773_failure_review"] = m773_identity
    substrate = inputs["m768_substrate"]
    require(sha256(hw / substrate["path"]) == substrate["sha256"],
            "M768 substrate identity drift")
    checked["m768_substrate"] = substrate["sha256"]
    for name in ("primary_m686", "secondary_m699"):
        row = inputs[name]
        directory = hw / row["directory"]
        identity = verify_sealed_directory(directory)
        require(sha256(directory / "manifest.json") == row["manifest_sha256"],
                name + " manifest drift")
        require(identity["outer_seal_file_sha256"] ==
                row["outer_seal_file_sha256"], name + " outer seal drift")
        checked[name] = identity
    for name in ("primary_m692_review", "secondary_m705_review"):
        row = inputs[name]
        directory = hw / row["directory"]
        identity = verify_sealed_directory(directory)
        require(sha256(directory / "review.json") == row["review_json_sha256"],
                name + " review drift")
        require(identity["outer_seal_file_sha256"] ==
                row["outer_seal_file_sha256"], name + " outer seal drift")
        checked[name] = identity
    for name in ("m672_mapper", "m712_oracle", "m722r2_oracle"):
        row = inputs[name]
        path = hw / row["path"]
        require(sha256(path) == row["sha256"], name + " identity drift")
        checked[name] = row["sha256"]
    for name, row in contract["source_files"].items():
        path = hw / row["path"]
        require(sha256(path) == row["sha256"],
                "M777 source identity drift: " + name)
        checked["source_" + name] = row["sha256"]
    return {
        "status": "PASS_M777_SOURCE_IDENTITY_ONLY__NO_PRODUCTION_RUN",
        "contract_sha256": sha256(contract_path),
        "resource": resource.identity(),
        "checked_inputs": checked,
        "launch_now": False,
        "production_cycles": None,
        "production_speedup": None,
        "decoder_complete": False,
        "table_a_insertion_allowed": False,
        "full_network_completion": False,
    }


def synthetic_self_test() -> Dict[str, object]:
    residency = PsumResidency(PSUM_VECTOR_BYTES * 2)
    slot0, events0 = residency.acquire(0)
    residency.mark_dirty(0)
    _slot1, _events1 = residency.acquire(1)
    residency.mark_dirty(1)
    _slot2, evict = residency.acquire(2)
    _slot0_again, restore = residency.acquire(0)
    require(slot0 == 0 and any(row.kind == "evict" for row in evict),
            "residency eviction self-test")
    require(any(row.kind == "restore" for row in restore),
            "residency restore self-test")
    require(weight_bank_and_local_row(24, 8) == (0, 48) and
            weight_bank_and_local_row(25, 8) == (1, 48),
            "weight bank-local golden self-test")
    return {
        "status": "PASS_M777_SYNTHETIC_SOURCE_SELF_TEST",
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
            "production replay is fail-closed in M777")
    require(args.output is None,
            "source-only validation refuses result output")
    print(json.dumps(validate_source_contract(args.repo_root, args.contract),
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
