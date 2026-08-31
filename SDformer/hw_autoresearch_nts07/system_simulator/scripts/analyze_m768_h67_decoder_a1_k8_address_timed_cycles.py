#!/usr/bin/env python3
"""M768 source-only decoder-component address-timed analyzer.

This module implements the minimum executable semantics authorized by M766.
It deliberately does *not* publish production cycles or speedups: the frozen
source contract has ``launch_now=false`` and a separately reviewed release is
required before the M686/M699 populations may be replayed.

The reusable part is intentionally small and attackable:

* strict sealed-input and population normalizers for M686 and M699;
* one macro-rounded common resource shared by A1-OSG, equal-service K1x8 and
  typed signed K8;
* compressed transactions with explicit address/bank/dependency semantics;
* a discrete-event bank/port scheduler with issue, return and commit times;
* exact commit/address hashes and fail-closed population/comparator guards.

M700 is neither imported nor accepted as an input.  D1 always takes the same
charged exact fallback in all three configurations until a separate numeric
admission closes the folded-theta gap.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
CONTRACT_SCHEMA = "m768_h67_decoder_a1_k8_address_timed_cycle_contract_v1"
CONFIGS = ("A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8")
HEADLINE_PAIR = ("TYPED_SIGNED_K8", "EQUAL_SERVICE_K1X8")
HEADLINE_MODULES = (0, 2, 3)
DIAGNOSTIC_MODULES = (1,)


class Failure(RuntimeError):
    """Fail-closed contract or scheduling violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_json(path: Path) -> object:
    def pairs(rows: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("non-finite JSON token: " + token)
        ),
    )


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    require(
        member.parts
        and not member.is_absolute()
        and ".." not in member.parts
        and member.as_posix() == name,
        "unsafe sealed member: " + name,
    )
    return member


def verify_sealed_directory(path: Path) -> Dict[str, str]:
    """Verify every member and both root seals, including nested seal files."""
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(
        manifest.is_file()
        and not manifest.is_symlink()
        and outer.is_file()
        and not outer.is_symlink(),
        "missing root seals",
    )
    expected_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(
            len(fields) == 2 and len(fields[0]) == 64,
            "malformed SHA256SUMS",
        )
        expected, name = fields
        require(name not in expected_names, "duplicate sealed member")
        expected_names.add(name)
        member = path.joinpath(*safe_member(name).parts)
        require(
            member.is_file()
            and not member.is_symlink()
            and sha256(member) == expected,
            "sealed member mismatch: " + name,
        )
    root_seals = {manifest.resolve(), outer.resolve()}
    actual_names = set()
    for member in path.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file() and member.resolve() not in root_seals:
            actual_names.add(member.relative_to(path).as_posix())
    require(actual_names == expected_names, "sealed population mismatch")
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(fields == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
    }


def round_up(value: int, quantum: int) -> int:
    require(value >= 0 and quantum > 0, "invalid round-up arguments")
    return ((value + quantum - 1) // quantum) * quantum


@dataclass(frozen=True)
class PortSpec:
    banks: int
    port_mode: str
    row_bytes: int
    read_latency: int
    write_latency: int
    initiation_interval: int
    outstanding_per_bank: int

    def validate(self) -> None:
        require(self.banks >= 1, "resource requires at least one bank")
        require(self.port_mode in ("1RW", "1R1W"), "unsupported port mode")
        require(self.row_bytes >= 1, "row width must be positive")
        require(self.read_latency >= 1 and self.write_latency >= 1,
                "latencies must be positive")
        require(self.initiation_interval >= 1, "II must be positive")
        require(self.outstanding_per_bank >= 1,
                "outstanding limit must be positive")


@dataclass(frozen=True)
class CommonResource:
    lanes: int
    accumulator_bits: int
    clock_ns: float
    external_bytes_per_cycle: int
    onchip_budget_bytes_macro_rounded: int
    macro_round_bytes: int
    weight_bytes_logical: int
    psum_bytes_logical: int
    descriptor_control_bytes_logical: int
    reserved_unallocated_bytes: int
    weight: PortSpec
    psum: PortSpec
    external: PortSpec
    compute: PortSpec

    @property
    def allocated_macro_rounded_bytes(self) -> int:
        return sum(
            round_up(value, self.macro_round_bytes)
            for value in (
                self.weight_bytes_logical,
                self.psum_bytes_logical,
                self.descriptor_control_bytes_logical,
            )
        )

    def validate(self) -> None:
        require(self.lanes == 96, "M768 freezes 96 product lanes")
        require(self.accumulator_bits == 24, "M768 freezes Acc24")
        require(self.clock_ns == 3.0, "M768 freezes 3.000 ns")
        require(self.external_bytes_per_cycle == 192,
                "M768 primary point freezes 192 B/cycle")
        require(self.onchip_budget_bytes_macro_rounded == 245760,
                "M768 freezes 245760 B total SRAM")
        for port in (self.weight, self.psum, self.external, self.compute):
            port.validate()
        allocated = self.allocated_macro_rounded_bytes
        require(
            allocated + self.reserved_unallocated_bytes
            == self.onchip_budget_bytes_macro_rounded,
            "macro-rounded partitions and reserved bytes must exactly conserve budget",
        )
        require(allocated <= self.onchip_budget_bytes_macro_rounded,
                "capacity cliff: macro-rounded storage exceeds 245760 B")

    def identity(self) -> Dict[str, object]:
        value = asdict(self)
        value["allocated_macro_rounded_bytes"] = (
            self.allocated_macro_rounded_bytes
        )
        value["resource_manifest_sha256"] = canonical_sha256(value)
        return value


def resource_from_contract(contract: Mapping[str, object]) -> CommonResource:
    row = contract["common_resource"]
    require(isinstance(row, dict), "common_resource must be an object")
    partitions = row["partitions"]
    ports = row["ports"]
    require(isinstance(partitions, dict) and isinstance(ports, dict),
            "resource partitions/ports must be objects")

    def port(name: str) -> PortSpec:
        value = ports[name]
        require(isinstance(value, dict), "port spec must be an object")
        return PortSpec(
            banks=int(value["banks"]),
            port_mode=str(value["port_mode"]),
            row_bytes=int(value["row_bytes"]),
            read_latency=int(value["read_latency"]),
            write_latency=int(value["write_latency"]),
            initiation_interval=int(value["initiation_interval"]),
            outstanding_per_bank=int(value["outstanding_per_bank"]),
        )

    resource = CommonResource(
        lanes=int(row["lanes"]),
        accumulator_bits=int(row["accumulator_bits"]),
        clock_ns=float(row["clock_ns"]),
        external_bytes_per_cycle=int(row["external_bytes_per_cycle"]),
        onchip_budget_bytes_macro_rounded=int(
            row["onchip_sram_bytes_macro_rounded"]
        ),
        macro_round_bytes=int(row["macro_round_bytes"]),
        weight_bytes_logical=int(partitions["weight_bytes"]),
        psum_bytes_logical=int(partitions["psum_bytes"]),
        descriptor_control_bytes_logical=int(
            partitions["descriptor_control_bytes"]
        ),
        reserved_unallocated_bytes=int(partitions["reserved_unallocated_bytes"]),
        weight=port("weight"),
        psum=port("psum"),
        external=port("external"),
        compute=port("compute"),
    )
    resource.validate()
    return resource


@dataclass(frozen=True)
class CompressedTransaction:
    transaction_id: str
    population_id: str
    config: str
    kind: str
    base_address: int
    address_stride_bytes: int
    count: int
    bank_pattern: Tuple[int, ...]
    width_bytes: int
    address_offsets: Tuple[int, ...] = ()
    dependency_tokens: Tuple[str, ...] = ()
    produces_token_prefix: str = ""
    earliest_issue_cycle: int = 0

    def validate(self) -> None:
        require(self.transaction_id, "transaction_id must be nonempty")
        require(self.population_id, "population_id must be nonempty")
        require(self.config in CONFIGS, "unknown configuration")
        require(
            self.kind
            in (
                "weight_read",
                "psum_read",
                "psum_write",
                "external_read",
                "external_write",
                "compute",
                "commit",
            ),
            "unknown transaction kind",
        )
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


@dataclass(frozen=True)
class Request:
    request_id: str
    transaction_id: str
    population_id: str
    config: str
    kind: str
    addresses: Tuple[int, ...]
    banks: Tuple[int, ...]
    width_bytes: int
    dependency_tokens: Tuple[str, ...]
    produces_token: str
    earliest_issue_cycle: int


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    transaction_id: str
    population_id: str
    config: str
    kind: str
    addresses: Tuple[int, ...]
    banks: Tuple[int, ...]
    width_bytes: int
    dependency_tokens: Tuple[str, ...]
    earliest_issue_cycle: int
    dependency_ready_cycle: int
    issue_cycle: int
    return_cycle: int
    commit_cycle: int
    wait_reason: str
    produces_token: str


def expand_transactions(
    transactions: Iterable[CompressedTransaction],
) -> Iterator[Request]:
    for transaction in transactions:
        transaction.validate()
        for index in range(transaction.count):
            base = transaction.base_address + index * transaction.address_stride_bytes
            offsets = (transaction.address_offsets
                       if transaction.address_offsets
                       else tuple(bank * transaction.width_bytes
                                  for bank in transaction.bank_pattern))
            addresses = tuple(base + offset for offset in offsets)
            token = (
                "{}:{}".format(transaction.produces_token_prefix, index)
                if transaction.produces_token_prefix
                else ""
            )
            yield Request(
                request_id="{}:{}".format(transaction.transaction_id, index),
                transaction_id=transaction.transaction_id,
                population_id=transaction.population_id,
                config=transaction.config,
                kind=transaction.kind,
                addresses=addresses,
                banks=transaction.bank_pattern,
                width_bytes=transaction.width_bytes,
                dependency_tokens=transaction.dependency_tokens,
                produces_token=token,
                earliest_issue_cycle=transaction.earliest_issue_cycle,
            )


class AddressTimedScheduler:
    """Deterministic fixed-latency scheduler with same-cycle slot release."""

    def __init__(self, resource: CommonResource):
        resource.validate()
        self.resource = resource
        self.token_ready: Dict[str, int] = {}
        self.next_port_cycle: Dict[Tuple[str, int, str], int] = {}
        self.outstanding_returns: Dict[Tuple[str, int], List[int]] = {}

    def _resource(self, kind: str) -> Tuple[str, PortSpec, str]:
        if kind == "weight_read":
            return "weight", self.resource.weight, "read"
        if kind in ("psum_read",):
            return "psum", self.resource.psum, "read"
        if kind in ("psum_write",):
            return "psum", self.resource.psum, "write"
        if kind in ("external_read",):
            return "external", self.resource.external, "read"
        if kind in ("external_write", "commit"):
            return "external", self.resource.external, "write"
        require(kind == "compute", "unmapped transaction kind")
        return "compute", self.resource.compute, "write"

    @staticmethod
    def _port_name(port: PortSpec, operation: str) -> str:
        return "rw" if port.port_mode == "1RW" else operation

    def _outstanding_bound(
        self, resource_name: str, banks: Tuple[int, ...], candidate: int,
        limit: int,
    ) -> int:
        bound = candidate
        changed = True
        while changed:
            changed = False
            for bank in banks:
                values = sorted(
                    value
                    for value in self.outstanding_returns.get(
                        (resource_name, bank), []
                    )
                    if value > bound
                )
                occupied = [
                    value
                    for value in self.outstanding_returns.get(
                        (resource_name, bank), []
                    )
                    if value > candidate
                ]
                if len(occupied) >= limit:
                    proposed = sorted(occupied)[len(occupied) - limit]
                    if proposed > bound:
                        bound = proposed
                        changed = True
        return bound

    def schedule(self, requests: Iterable[Request]) -> Dict[str, object]:
        scheduled: List[ScheduledRequest] = []
        address_digest = hashlib.sha256()
        commit_digest = hashlib.sha256()
        population_ids = set()
        configs = set()
        commit_ordinal = 0
        for request in requests:
            resource_name, port, operation = self._resource(request.kind)
            require(request.config in CONFIGS, "request configuration drift")
            require(request.banks and len(request.banks) == len(request.addresses),
                    "address/bank arity mismatch")
            require(all(0 <= bank < port.banks for bank in request.banks),
                    "bank index out of range")
            missing = [token for token in request.dependency_tokens
                       if token not in self.token_ready]
            require(not missing, "unresolved dependency token: " + repr(missing))
            dependency_ready = max(
                (self.token_ready[token] for token in request.dependency_tokens),
                default=request.earliest_issue_cycle,
            )
            port_name = self._port_name(port, operation)
            port_bound = max(
                (self.next_port_cycle.get((resource_name, bank, port_name), 0)
                 for bank in request.banks),
                default=0,
            )
            initial = max(request.earliest_issue_cycle,
                          dependency_ready, port_bound)
            outstanding_bound = self._outstanding_bound(
                resource_name, request.banks, initial,
                port.outstanding_per_bank,
            )
            issue = max(initial, outstanding_bound)
            if issue == request.earliest_issue_cycle:
                reason = "none"
            elif dependency_ready == issue:
                reason = "dependency_completion"
            elif outstanding_bound == issue and outstanding_bound > initial:
                reason = "memory"
            elif port_bound == issue:
                reason = (
                    "weight_bank" if resource_name == "weight"
                    else "psum_bank" if resource_name == "psum"
                    else "memory" if resource_name == "external"
                    else "compute"
                )
            else:
                reason = "compute"
            latency = port.read_latency if operation == "read" else port.write_latency
            beats = max(1, math.ceil(request.width_bytes /
                                     (self.resource.external_bytes_per_cycle
                                      if resource_name == "external"
                                      else port.row_bytes)))
            return_cycle = issue + latency + beats - 1
            commit_cycle = return_cycle
            for bank in request.banks:
                self.next_port_cycle[(resource_name, bank, port_name)] = (
                    issue + max(port.initiation_interval, beats)
                )
                key = (resource_name, bank)
                current = [value for value in self.outstanding_returns.get(key, [])
                           if value > issue]
                current.append(return_cycle)
                self.outstanding_returns[key] = current
            if request.produces_token:
                require(request.produces_token not in self.token_ready,
                        "duplicate produced token")
                self.token_ready[request.produces_token] = return_cycle
            for bank, address in zip(request.banks, request.addresses):
                address_digest.update(
                    json.dumps(
                        [request.request_id, request.kind, address, bank],
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            if request.kind == "commit":
                for address in request.addresses:
                    commit_digest.update(
                        json.dumps(
                            [commit_ordinal, address, request.width_bytes],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    commit_ordinal += 1
            population_ids.add(request.population_id)
            configs.add(request.config)
            scheduled.append(ScheduledRequest(
                request_id=request.request_id,
                transaction_id=request.transaction_id,
                population_id=request.population_id,
                config=request.config,
                kind=request.kind,
                addresses=request.addresses,
                banks=request.banks,
                width_bytes=request.width_bytes,
                dependency_tokens=request.dependency_tokens,
                earliest_issue_cycle=request.earliest_issue_cycle,
                dependency_ready_cycle=dependency_ready,
                issue_cycle=issue,
                return_cycle=return_cycle,
                commit_cycle=commit_cycle,
                wait_reason=reason,
                produces_token=request.produces_token,
            ))
        require(scheduled, "empty schedule")
        total_cycles = max(row.commit_cycle for row in scheduled) + 1
        issue_cycles = {row.issue_cycle for row in scheduled}
        stalls = {
            "compute": 0,
            "weight_bank": 0,
            "psum_bank": 0,
            "memory": 0,
            "dependency_completion": 0,
            "active_service": len(issue_cycles),
        }
        for cycle in range(total_cycles):
            if cycle in issue_cycles:
                continue
            waiting = [row for row in scheduled
                       if row.earliest_issue_cycle <= cycle < row.issue_cycle]
            inflight = [row for row in scheduled
                        if row.issue_cycle <= cycle < row.return_cycle]
            reasons = [row.wait_reason for row in waiting
                       if row.wait_reason != "none"]
            dependency_wait = any(row.dependency_ready_cycle > cycle
                                  for row in waiting)
            if dependency_wait or "dependency_completion" in reasons or inflight:
                stalls["dependency_completion"] += 1
            elif "weight_bank" in reasons:
                stalls["weight_bank"] += 1
            elif "psum_bank" in reasons:
                stalls["psum_bank"] += 1
            elif "memory" in reasons:
                stalls["memory"] += 1
            else:
                stalls["compute"] += 1
        require(sum(stalls.values()) == total_cycles,
                "mutually-exclusive timeline conservation failure")
        compressed_rows = compress_scheduled_rows(scheduled)
        return {
            "total_cycles": total_cycles,
            "expanded_request_count": len(scheduled),
            "compressed_transaction_count": len(compressed_rows),
            "scheduled_requests": [asdict(row) for row in scheduled],
            "compressed_schedule": compressed_rows,
            "transaction_address_sha256": address_digest.hexdigest(),
            "commit_sequence_sha256": commit_digest.hexdigest(),
            "population_ids": sorted(population_ids),
            "configs": sorted(configs),
            "cycle_classes": stalls,
            "same_cycle_response_slot_reuse": True,
        }


def compress_scheduled_rows(rows: Sequence[ScheduledRequest]) -> List[Dict[str, object]]:
    """Compress consecutive members without dropping explicit time endpoints."""
    output: List[Dict[str, object]] = []
    for row in rows:
        item = {
            "transaction_id": row.transaction_id,
            "population_id": row.population_id,
            "config": row.config,
            "kind": row.kind,
            "address_first": list(row.addresses),
            "address_last": list(row.addresses),
            "banks": list(row.banks),
            "count": 1,
            "issue_first": row.issue_cycle,
            "issue_last": row.issue_cycle,
            "return_first": row.return_cycle,
            "return_last": row.return_cycle,
            "dependency_ready_first": row.dependency_ready_cycle,
            "dependency_ready_last": row.dependency_ready_cycle,
            "earliest_issue_first": row.earliest_issue_cycle,
            "earliest_issue_last": row.earliest_issue_cycle,
            "commit_first": row.commit_cycle,
            "commit_last": row.commit_cycle,
            "dependency_tokens": list(row.dependency_tokens),
            "width_bytes": row.width_bytes,
        }
        if output and all(
            output[-1][key] == item[key]
            for key in (
                "transaction_id", "population_id", "config", "kind",
                "banks", "dependency_tokens", "width_bytes",
            )
        ):
            previous = output[-1]
            previous["count"] = int(previous["count"]) + 1
            previous["address_last"] = item["address_last"]
            previous["issue_last"] = item["issue_last"]
            previous["return_last"] = item["return_last"]
            previous["dependency_ready_last"] = item["dependency_ready_last"]
            previous["earliest_issue_last"] = item["earliest_issue_last"]
            previous["commit_last"] = item["commit_last"]
        else:
            output.append(item)
    return output


def assert_population_isolation(population_ids: Iterable[str]) -> str:
    observed = set(population_ids)
    require(len(observed) == 1,
            "primary and secondary populations must never be mixed")
    return next(iter(observed))


def assert_fair_configs(
    resource_hash_by_config: Mapping[str, str],
    commit_hash_by_config: Mapping[str, str],
    fallback_policy_by_config: Mapping[str, str],
) -> None:
    require(set(resource_hash_by_config) == set(CONFIGS),
            "resource tuple missing a configuration")
    require(set(commit_hash_by_config) == set(CONFIGS),
            "commit hash missing a configuration")
    require(set(fallback_policy_by_config) == set(CONFIGS),
            "fallback policy missing a configuration")
    require(len(set(resource_hash_by_config.values())) == 1,
            "resource tuple differs across configurations")
    require(len(set(commit_hash_by_config.values())) == 1,
            "commit sequence differs across configurations")
    require(len(set(fallback_policy_by_config.values())) == 1,
            "fallback policy differs across configurations")


def headline_ratio_allowed(numerator: str, denominator: str) -> bool:
    return (numerator, denominator) == HEADLINE_PAIR


def route_for_record(module_index: int, config: str) -> Dict[str, object]:
    require(config in CONFIGS, "unknown configuration")
    if module_index == 1:
        return {
            "effective_config": "D1_EXACT_DENSE_FP32_FALLBACK",
            "fallback": True,
            "headline_eligible": False,
            "fallback_policy": "COMMON_CHARGED_DENSE_FP32_FALLBACK_ALL_CONFIGS",
        }
    require(module_index in HEADLINE_MODULES, "unknown decoder module")
    return {
        "effective_config": config,
        "fallback": False,
        "headline_eligible": True,
        "fallback_policy": "COMMON_CHARGED_DENSE_FP32_FALLBACK_ALL_CONFIGS",
    }


def normalized_population_records(
    manifest: Mapping[str, object], population_id: str
) -> List[Dict[str, object]]:
    """Normalize M686 or M699 without allowing cross-population splicing."""
    schema = manifest.get("schema")
    result: List[Dict[str, object]] = []
    if schema == "m660_h67_ep35_layer_static_decoder_payload_v1":
        binary = manifest.get("d0_d2_d3_binary_records")
        d1 = manifest.get("d1_records")
        require(isinstance(binary, list) and len(binary) == 30,
                "M686 exact binary population drift")
        require(isinstance(d1, list) and len(d1) == 10,
                "M686 D1 diagnostic population drift")
        rows = list(binary) + list(d1)
        for row in rows:
            index = int(row["module_index"])
            packed = (row["theta_binary_candidate"] if index == 1
                      else row["input"])
            result.append({
                "population_id": population_id,
                "sequence": row["sequence_key"],
                "sample_id": int(row["sample_id"]),
                "module_index": index,
                "route": row["route"],
                "input_shape": list(row["input_shape"]),
                "relative_path": row["relative_path"],
                "packed_sha256": packed["packed_sha256"],
            })
        require(len(result) == 40, "M686 normalized population drift")
    elif schema == "m699_h67_ep35_multisequence_decoder_payload_v1":
        rows = manifest.get("records")
        require(isinstance(rows, list) and len(rows) == 120,
                "M699 record population drift")
        for row in rows:
            index = int(row["module_index"])
            packed = (row["statistics"]["scaled_binary_audit"]
                      if index == 1 else row["statistics"])
            result.append({
                "population_id": population_id,
                "sequence": row["sequence"],
                "sample_id": int(row["sequence_sample_id"]),
                "module_index": index,
                "route": row["route"],
                "input_shape": list(row["input_shape"]),
                "relative_path": row["relative_path"],
                "packed_sha256": packed["packed_sha256"],
            })
        require(len(result) == 120, "M699 normalized population drift")
    else:
        raise Failure("unsupported population manifest schema")
    require(all(row["population_id"] == population_id for row in result),
            "population identity injection failed")
    for row in result:
        index = int(row["module_index"])
        expected = ("EXACT_SCALED_BINARY_BITPACK" if index == 1
                    else "EXACT_BINARY_BITPACK")
        require(row["route"] == expected, "decoder route drift")
    return sorted(result, key=lambda row: (
        str(row["sequence"]), int(row["sample_id"]), int(row["module_index"])
    ))


def load_pinned_mapper(path: Path, expected_sha256: str):
    require(sha256(path) == expected_sha256, "M672 mapper identity drift")
    spec = importlib.util.spec_from_file_location("m768_pinned_m672", path)
    require(spec is not None and spec.loader is not None,
            "cannot import pinned M672 mapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank_unique_groups(flat_k_indices: Sequence[int], channels: int,
                       bank_count: int = 8) -> List[Tuple[int, ...]]:
    """Pack one flattened (tap,channel) source from each bank per group."""
    queues: List[List[int]] = [[] for _ in range(bank_count)]
    for value in flat_k_indices:
        value = int(value)
        require(value >= 0, "negative flattened K index")
        channel = value % channels
        queues[channel % bank_count].append(value)
    groups = []
    for ordinal in range(max((len(queue) for queue in queues), default=0)):
        groups.append(tuple(queue[ordinal] for queue in queues
                            if ordinal < len(queue)))
    require(sum(len(group) for group in groups) == len(flat_k_indices),
            "bank packing does not conserve sources")
    return groups


def dense_commit_addresses(module_index: int, timestep: int, output_height: int,
                           output_width: int, output_blocks: int) -> List[int]:
    base = (module_index << 48) | (timestep << 40)
    return [
        base | (((oy * output_width + ox) * output_blocks + block) * 384)
        for oy in range(output_height)
        for ox in range(output_width)
        for block in range(output_blocks)
    ]


def commit_address_hash(addresses: Sequence[int], width_bytes: int = 384) -> str:
    digest = hashlib.sha256()
    for ordinal, address in enumerate(addresses):
        digest.update(json.dumps(
            [ordinal, int(address), int(width_bytes)],
            separators=(",", ":"),
        ).encode("utf-8"))
    return digest.hexdigest()


MODULE_GEOMETRY = {
    0: (1536, 384, 15, 20, 30, 40),
    1: (770, 192, 30, 40, 60, 80),
    2: (386, 96, 60, 80, 120, 160),
    3: (194, 96, 120, 160, 240, 320),
}


def iter_record_transactions(
    mapper,
    record: Mapping[str, object],
    payload_root: Path,
    population_id: str,
    config: str,
    timestep: int,
    tile_m: int = 256,
    geometry: Mapping[int, Tuple[int, int, int, int, int, int]] = MODULE_GEOMETRY,
) -> Iterator[CompressedTransaction]:
    """Generate one record/timestep's exact-address transaction stream.

    The function is an iterator so a future reviewed production runner need
    not materialize the complete decoder request population.  It uses M672's
    destination/tap mapper directly.  The current source contract never calls
    this function on the full M686/M699 population.
    """
    require(config in CONFIGS, "unknown configuration")
    module_index = int(record["module_index"])
    require(module_index in geometry, "unknown decoder module")
    cin, cout, hin, win, hout, wout = geometry[module_index]
    shape = tuple(int(value) for value in record["input_shape"])
    require(shape == (10, 1, cin, hin, win), "decoder record shape drift")
    require(0 <= timestep < shape[0], "timestep out of range")
    payload = Path(payload_root).joinpath(
        *safe_member(str(record["relative_path"])).parts
    )
    require(payload.is_file() and not payload.is_symlink(),
            "decoder payload is not a regular file")
    require(sha256(payload) == str(record["packed_sha256"]),
            "decoder payload identity drift")
    route = route_for_record(module_index, config)
    output_blocks = math.ceil(cout / 96)
    prefix = "{}:{}:m{}:t{}".format(
        population_id, config, module_index, timestep
    )

    if route["fallback"]:
        input_bytes = cin * hin * win * 4
        input_beats = math.ceil(input_bytes / 192)
        input_prefix = prefix + ":fallback_input"
        yield CompressedTransaction(
            transaction_id=input_prefix,
            population_id=population_id,
            config=config,
            kind="external_read",
            base_address=(module_index << 52) | (timestep << 44),
            address_stride_bytes=192,
            count=input_beats,
            bank_pattern=(0,),
            width_bytes=192,
            produces_token_prefix=input_prefix + ":done",
        )
        compute_prefix = prefix + ":fallback_compute"
        yield CompressedTransaction(
            transaction_id=compute_prefix,
            population_id=population_id,
            config=config,
            kind="compute",
            base_address=(module_index << 52) | (timestep << 44),
            address_stride_bytes=0,
            count=1,
            bank_pattern=(0,),
            width_bytes=288,
            dependency_tokens=(
                "{}:done:{}".format(input_prefix, input_beats - 1),
            ),
            produces_token_prefix=compute_prefix + ":done",
        )
        commit_prefix = prefix + ":fallback_commit"
        yield CompressedTransaction(
            transaction_id=commit_prefix,
            population_id=population_id,
            config=config,
            kind="commit",
            base_address=(1 << 60) | (module_index << 52) | (timestep << 44),
            address_stride_bytes=384,
            count=hout * wout * output_blocks,
            bank_pattern=(0,),
            width_bytes=384,
            dependency_tokens=(compute_prefix + ":done:0",),
        )
        return

    group_ordinal = 0
    commit_tokens: Dict[int, str] = {}
    for tile in mapper.iter_polyphase_tiles(
        payload,
        shape,
        tile_m=tile_m,
        trusted_root=Path(payload_root).resolve(),
    ):
        values = tile["values"][timestep]
        source_flat = tile["source_flat_index"]
        for local_m, (dy, dx) in enumerate(zip(
                tile["destination_y"], tile["destination_x"])):
            destination = int(dy) * wout + int(dx)
            active_columns = np.flatnonzero(values[local_m])
            active_flat_k = [int(column) for column in active_columns]
            groups = bank_unique_groups(active_flat_k, cin, 8)
            for output_block in range(output_blocks):
                psum_base = ((module_index << 52) | (timestep << 44) |
                             ((destination * output_blocks + output_block) * 384))
                previous = commit_tokens.get(destination * output_blocks +
                                             output_block)
                for group in groups:
                    banks = tuple((flat_k % cin) % 8 for flat_k in group)
                    require(len(banks) == len(set(banks)),
                            "mapper group contains a weight-bank collision")
                    weight_id = prefix + ":g{}:w".format(group_ordinal)
                    weight_base = ((module_index << 52) |
                                   (output_block << 44))
                    yield CompressedTransaction(
                        transaction_id=weight_id,
                        population_id=population_id,
                        config=config,
                        kind="weight_read",
                        base_address=weight_base,
                        address_stride_bytes=0,
                        count=1,
                        bank_pattern=banks,
                        width_bytes=16,
                        address_offsets=tuple(flat_k * 16 for flat_k in group),
                        produces_token_prefix=weight_id + ":done",
                    )
                    read_id = prefix + ":g{}:pr".format(group_ordinal)
                    yield CompressedTransaction(
                        transaction_id=read_id,
                        population_id=population_id,
                        config=config,
                        kind="psum_read",
                        base_address=psum_base,
                        address_stride_bytes=0,
                        count=1,
                        bank_pattern=tuple(range(6)),
                        width_bytes=48,
                        dependency_tokens=((previous,) if previous else ()),
                        produces_token_prefix=read_id + ":done",
                    )
                    compute_id = prefix + ":g{}:c".format(group_ordinal)
                    yield CompressedTransaction(
                        transaction_id=compute_id,
                        population_id=population_id,
                        config=config,
                        kind="compute",
                        base_address=psum_base,
                        address_stride_bytes=0,
                        count=1,
                        bank_pattern=(0,),
                        width_bytes=288,
                        dependency_tokens=(weight_id + ":done:0",
                                           read_id + ":done:0"),
                        produces_token_prefix=compute_id + ":done",
                    )
                    write_id = prefix + ":g{}:pw".format(group_ordinal)
                    yield CompressedTransaction(
                        transaction_id=write_id,
                        population_id=population_id,
                        config=config,
                        kind="psum_write",
                        base_address=psum_base,
                        address_stride_bytes=0,
                        count=1,
                        bank_pattern=tuple(range(6)),
                        width_bytes=48,
                        dependency_tokens=(compute_id + ":done:0",),
                        produces_token_prefix=write_id + ":done",
                    )
                    previous = write_id + ":done:0"
                    group_ordinal += 1
                if previous:
                    commit_tokens[destination * output_blocks + output_block] = previous

    for ordinal, address in enumerate(dense_commit_addresses(
            module_index, timestep, hout, wout, output_blocks)):
        dependency = commit_tokens.get(ordinal)
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
            width_bytes=384,
            dependency_tokens=((dependency,) if dependency else ()),
        )


def validate_source_contract(repo_root: Path, contract_path: Path) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(isinstance(contract, dict), "contract must be an object")
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema drift")
    require(contract.get("launch_now") is False,
            "source-only M768 contract must not authorize production")
    require(contract.get("production_speedup_allowed") is False,
            "source package must forbid production speedup")
    require("m700" not in json.dumps(contract, sort_keys=True).lower(),
            "M700 must not enter candidate inputs")
    hw = repo_root / "hw_autoresearch_nts07"
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA256,
            "docs359 drift")
    resource = resource_from_contract(contract)
    inputs = contract["inputs"]
    checked = {}
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
                name + " review identity drift")
        require(identity["outer_seal_file_sha256"] ==
                row["outer_seal_file_sha256"], name + " outer seal drift")
        review = strict_json(directory / "review.json")
        require(isinstance(review, dict), name + " review must be an object")
        checked[name] = identity
    for name in ("m672_mapper", "m712_oracle", "m722r2_oracle", "m218_oracle"):
        row = inputs[name]
        path = hw / row["path"]
        require(sha256(path) == row["sha256"], name + " identity drift")
        checked[name] = row["sha256"]
    for name, row in contract["source_files"].items():
        path = hw / row["path"]
        require(sha256(path) == row["sha256"],
                "M768 source identity drift: " + name)
        checked["source_" + name] = row["sha256"]
    primary_manifest = strict_json(
        hw / inputs["primary_m686"]["directory"] / "manifest.json"
    )
    secondary_manifest = strict_json(
        hw / inputs["secondary_m699"]["directory"] / "manifest.json"
    )
    primary = normalized_population_records(
        primary_manifest, contract["populations"]["primary_id"]
    )
    secondary = normalized_population_records(
        secondary_manifest, contract["populations"]["secondary_id"]
    )
    require(len(primary) == 40 and len(secondary) == 120,
            "normalized population count drift")
    return {
        "status": "PASS_SOURCE_INPUT_IDENTITY_ONLY__NO_PRODUCTION_RUN",
        "contract_sha256": sha256(contract_path),
        "resource": resource.identity(),
        "primary_records": len(primary),
        "secondary_records": len(secondary),
        "checked_inputs": checked,
        "launch_now": False,
        "production_cycles": None,
        "production_speedup": None,
        "table_a_insertion_allowed": False,
        "full_network_completion": False,
    }


def synthetic_self_test() -> Dict[str, object]:
    resource = CommonResource(
        lanes=96,
        accumulator_bits=24,
        clock_ns=3.0,
        external_bytes_per_cycle=192,
        onchip_budget_bytes_macro_rounded=245760,
        macro_round_bytes=128,
        weight_bytes_logical=13824,
        psum_bytes_logical=221184,
        descriptor_control_bytes_logical=8192,
        reserved_unallocated_bytes=2560,
        weight=PortSpec(8, "1R1W", 16, 4, 1, 1, 8),
        psum=PortSpec(6, "1RW", 48, 2, 1, 1, 8),
        external=PortSpec(1, "1RW", 192, 32, 3, 1, 16),
        compute=PortSpec(1, "1RW", 288, 1, 1, 1, 1),
    )
    resource.validate()
    tx = [
        CompressedTransaction(
            "w", "SYNTHETIC_PRIMARY", "TYPED_SIGNED_K8",
            "weight_read", 0x1000, 128, 2, (0, 1), 16,
            produces_token_prefix="w_done",
        )
    ]
    result = AddressTimedScheduler(resource).schedule(expand_transactions(tx))
    require(result["expanded_request_count"] == 2, "self-test request count")
    require(sum(result["cycle_classes"].values()) == result["total_cycles"],
            "self-test conservation")
    return {
        "status": "PASS_M768_SYNTHETIC_SOURCE_SELF_TEST",
        "resource_manifest_sha256": resource.identity()[
            "resource_manifest_sha256"
        ],
        "expanded_requests": result["expanded_request_count"],
        "production_cycles": None,
        "production_speedup": None,
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
            "only self-test or source-contract validation is authorized")
    require(not args.run_production,
            "production replay is fail-closed in the source-only contract")
    require(args.output is None,
            "source-only validation refuses a result output")
    print(json.dumps(
        validate_source_contract(args.repo_root, args.contract),
        indent=2, sort_keys=True, allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
