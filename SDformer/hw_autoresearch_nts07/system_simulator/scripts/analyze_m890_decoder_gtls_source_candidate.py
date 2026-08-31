#!/usr/bin/env python3
"""M890 source-only group/transaction-level exact decoder scheduler.

This additive candidate keeps the frozen M785/M768 resource and scheduling
equations.  It changes only host-side representation:

* frozen ``CompressedTransaction`` objects form a lossless packed group IR;
* only produced tokens referenced by a future dependency are live, and each
  is retired after its statically verified last use;
* an exact closed form is used only for an isolated, homogeneous transaction;
  every other transaction falls back to the frozen per-request recurrence;
* cycle-class events are retained in packed int64 endpoint arrays and reduced
  with the exact frozen priority order.

The module is deliberately source-only.  It refuses the sealed full row,
production population, result publication, VCS/EDA, GPU and remote execution.
Only bounded synthetic and real prefixes may be exercised before a fresh
independent source hammer.
"""

import argparse
from array import array
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M861_PATH = HERE / "analyze_m861_decoder_streaming_event_sweep.py"
M861_SHA256 = "f72ed3b820051d624699152b784c05fa674106556ab73f452a2cf96a9f72d7a4"
M883_DIR = HW / "reviews/m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_r1_20260829"
M886_DIR = HW / "reviews/m886_m883_decoder_scalable_exact_successor_first_principles_review_r1_20260829"
M887_DIR = HW / "reviews/m887_m886_decoder_gtls_source_author_handoff_r1_20260829"
M888_DIR = HW / "reviews/m888_m887_m886_decoder_gtls_source_fresh_hammer_REQUEST_r1_20260829"
M883_IDENTITY = (
    "ae443b36084a3361548ec6a950dbc0a962cf60ec650000c9638db61854c02f88",
    "3cdd7be9cde8177e4cce6dfd16fc42dda5a84ba729757c92638eb242fe6fed0d",
    "4ddece71698ee0b83c18d039eb34205a0f2c93b4e5b95fd349f011686ab8d5a1",
)
M886_IDENTITY = (
    "009915ecc3524ba553edaef6c82cd615884db464440eef5a00e4df2531fc16b0",
    "9089dc440cf152fcc7df879f7b754d094e6745dcc7a24f7b576ad430587191ea",
    "98f0adb69f41f07e578e4ed0f66d2db99b981b868359ea5f1cfa37801f7b5ad4",
)
M887_IDENTITY = (
    "844ca9fe995f8a31242b17234a25373c10946a0d5597ce1875e534ebc3a6389b",
    "37efafd72181105a35f4281ce9714995f9e88c4ac7bcb9f9fa1ae76f070df1fa",
    "d00f00f4cb9bece1878e99abd1d1c3843804baeb252d5b588632281623684c46",
)
M888_IDENTITY = (
    "ea2815b894a50831b93471ce78cf9291c2c30571831737ba51169c7dccf3b8e9",
    "703f0945e2ae04c5860ca3e717a3df684d2ce96fc774af5825ff0bdba3b4ce17",
    "bbb0610f454eecb31a5315b7c0c02c259ea0e339657ce0295f89c8076963a137",
)
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT_SCHEMA = "m890_decoder_gtls_source_candidate_v1"
CYCLE_CLASS_ORDER = (
    "active_service", "dependency_completion", "weight_bank",
    "psum_bank", "memory", "compute",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen(path: Path, expected: str, name: str):
    if sha256(path) != expected:
        raise RuntimeError(name + " identity drift")
    spec = importlib.util.spec_from_file_location("m890_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M861 = _load_frozen(M861_PATH, M861_SHA256, "frozen_m861")
M785 = M861.M785
Failure = M785.Failure
require = M785.require
CompressedTransaction = M785.CompressedTransaction
ScheduledRequest = M861.ScheduledRequest


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _transaction_dict(tx: CompressedTransaction) -> Dict[str, object]:
    return {
        "transaction_id": tx.transaction_id,
        "population_id": tx.population_id,
        "config": tx.config,
        "kind": tx.kind,
        "base_address": int(tx.base_address),
        "address_stride_bytes": int(tx.address_stride_bytes),
        "count": int(tx.count),
        "bank_pattern": list(tx.bank_pattern),
        "width_bytes": int(tx.width_bytes),
        "address_offsets": list(tx.address_offsets),
        "dependency_tokens": list(tx.dependency_tokens),
        "produces_token_prefix": tx.produces_token_prefix,
        "earliest_issue_cycle": int(tx.earliest_issue_cycle),
    }


def token_for(tx: CompressedTransaction, index: int) -> str:
    if not tx.produces_token_prefix:
        return ""
    return "{}:{}".format(tx.produces_token_prefix, int(index))


def terminal_token(tx: CompressedTransaction) -> str:
    return token_for(tx, int(tx.count) - 1)


class PackedGroupIR:
    """Lossless bounded IR plus exact dependency-use/liveness metadata."""

    def __init__(self, transactions: Sequence[CompressedTransaction],
                 shard_key: Tuple[str, str, int, int, int]):
        require(transactions, "empty packed group IR")
        self.transactions = tuple(transactions)
        self.shard_key = tuple(shard_key)
        require(len(self.shard_key) == 5, "row shard key arity drift")
        self.expanded_count = 0
        self.dependency_uses: Counter = Counter()
        producer: Dict[str, Tuple[int, bool]] = {}
        digest = hashlib.sha256()
        terminal_only_dependencies = True
        for ordinal, tx in enumerate(self.transactions):
            tx.validate()
            row = _transaction_dict(tx)
            digest.update(_canonical_bytes(row))
            self.expanded_count += int(tx.count)
            for dependency in tx.dependency_tokens:
                require(dependency in producer,
                        "dependency precedes producer: " + dependency)
                self.dependency_uses[dependency] += int(tx.count)
                terminal_only_dependencies &= producer[dependency][1]
            if tx.produces_token_prefix:
                for index in range(int(tx.count)):
                    name = token_for(tx, index)
                    require(name not in producer, "duplicate produced token")
                    producer[name] = (ordinal, index == int(tx.count) - 1)
        self.producer = producer
        self.terminal_only_dependencies = bool(terminal_only_dependencies)
        self.compressed_group_ir_sha256 = digest.hexdigest()
        require(self.terminal_only_dependencies,
                "frozen M785 grammar unexpectedly depends on non-terminal token")

    def deterministic_shard(self, shard_count: int) -> int:
        shard_count = int(shard_count)
        require(shard_count >= 1, "invalid shard count")
        value = int(hashlib.sha256(_canonical_bytes(
            list(self.shard_key))).hexdigest()[:16], 16)
        return value % shard_count

    @property
    def packed_ir_bytes(self) -> int:
        return sum(len(_canonical_bytes(_transaction_dict(tx)))
                   for tx in self.transactions)


class TerminalLiveness:
    """Exact future-use ledger with explicit retirement attacks."""

    def __init__(self, uses: Mapping[str, int]):
        self.remaining = {str(key): int(value)
                          for key, value in uses.items()}
        require(all(value > 0 for value in self.remaining.values()),
                "nonpositive dependency-use count")
        self.ready: Dict[str, int] = {}
        self.retired = set()
        self.live_peak = 0

    def produce(self, token: str, ready: int) -> None:
        if not token or token not in self.remaining:
            return
        require(token not in self.ready and token not in self.retired,
                "duplicate or post-retirement token production")
        self.ready[token] = int(ready)
        self.live_peak = max(self.live_peak, len(self.ready))

    def resolve_group(self, dependencies: Sequence[str]) -> int:
        missing = [token for token in dependencies
                   if token not in self.ready]
        require(not missing, "unresolved or retired dependency: " + repr(missing))
        return max((self.ready[token] for token in dependencies), default=0)

    def consume_group(self, dependencies: Sequence[str], count: int,
                      *, force_early_retire: bool = False) -> None:
        count = int(count)
        require(count >= 1, "invalid dependency group count")
        missing = [token for token in dependencies
                   if token not in self.ready]
        require(not missing, "unresolved or retired dependency: " + repr(missing))
        for token in dependencies:
            decrement = count + (1 if force_early_retire else 0)
            require(self.remaining[token] >= decrement,
                    "premature terminal-token retirement attack")
            self.remaining[token] -= decrement
            if self.remaining[token] == 0:
                self.ready.pop(token)
                self.retired.add(token)

    def finish(self) -> None:
        require(all(value == 0 for value in self.remaining.values()),
                "unconsumed dependency uses remain")
        require(not self.ready, "last-use retirement leaked live tokens")


class PackedPriorityEvents:
    """Packed int64 endpoints for exact six-class priority reconstruction."""

    def __init__(self) -> None:
        self.issue = array("q")
        self.starts = {name: array("q") for name in CYCLE_CLASS_ORDER[1:-1]}
        self.ends = {name: array("q") for name in CYCLE_CLASS_ORDER[1:-1]}

    def observe(self, earliest: int, dependency: int, issue: int,
                returned: int, reason: str) -> None:
        earliest, dependency = int(earliest), int(dependency)
        issue, returned = int(issue), int(returned)
        require(0 <= earliest <= issue <= returned,
                "scheduled endpoint ordering drift")
        self.issue.append(issue)
        if issue < returned:
            self._interval("dependency_completion", issue, returned)
        if earliest < issue:
            if dependency > earliest:
                self._interval("dependency_completion", earliest,
                               min(issue, dependency))
            if reason == "dependency_completion":
                self._interval("dependency_completion", earliest, issue)
            elif reason in ("weight_bank", "psum_bank", "memory"):
                self._interval(reason, earliest, issue)
            else:
                require(reason in ("none", "compute"),
                        "unknown frozen wait reason")

    def _interval(self, name: str, start: int, end: int) -> None:
        if int(start) < int(end):
            self.starts[name].append(int(start))
            self.ends[name].append(int(end))

    @staticmethod
    def _merged(starts: array, ends: array,
                limit: int) -> List[Tuple[int, int]]:
        rows = sorted(zip(starts, ends))
        output: List[Tuple[int, int]] = []
        for start, end in rows:
            start = max(0, min(int(limit), int(start)))
            end = max(0, min(int(limit), int(end)))
            if start >= end:
                continue
            if output and start <= output[-1][1]:
                output[-1] = (output[-1][0], max(output[-1][1], end))
            else:
                output.append((start, end))
        return output

    def finalize(self, total_cycles: int) -> Dict[str, int]:
        total_cycles = int(total_cycles)
        require(total_cycles >= 1, "empty cycle domain")
        # Active-service is a set of cycles, exactly as frozen M768.
        active_points = sorted(set(int(value) for value in self.issue
                                   if 0 <= value < total_cycles))
        events: Dict[int, List[int]] = {0: [0] * 5,
                                        total_cycles: [0] * 5}
        for point in active_points:
            events.setdefault(point, [0] * 5)[0] += 1
            events.setdefault(point + 1, [0] * 5)[0] -= 1
        for priority, name in enumerate(CYCLE_CLASS_ORDER[1:-1], start=1):
            for start, end in self._merged(self.starts[name], self.ends[name],
                                           total_cycles):
                events.setdefault(start, [0] * 5)[priority] += 1
                events.setdefault(end, [0] * 5)[priority] -= 1
        result = {name: 0 for name in CYCLE_CLASS_ORDER}
        state = [0] * 5
        points = sorted(events)
        for ordinal, point in enumerate(points[:-1]):
            state = [value + change for value, change in
                     zip(state, events[point])]
            require(all(value >= 0 for value in state),
                    "packed priority event underflow")
            span = points[ordinal + 1] - point
            if span <= 0 or point >= total_cycles:
                continue
            chosen = "compute"
            for index, name in enumerate(CYCLE_CLASS_ORDER[:-1]):
                if state[index] > 0:
                    chosen = name
                    break
            result[chosen] += span
        require(sum(result.values()) == total_cycles,
                "packed priority timeline conservation failure")
        return result

    @property
    def packed_event_bytes(self) -> int:
        return (len(self.issue) + sum(len(value) for value in self.starts.values())
                + sum(len(value) for value in self.ends.values())) * 8


class GTLSScheduler:
    """Exact transaction scheduler with a narrowly proved closed form."""

    def __init__(self, resource) -> None:
        resource.validate()
        self.resource = resource
        self.next_port_cycle: Dict[Tuple[str, int, str], int] = {}
        self.outstanding_returns: Dict[Tuple[str, int], List[int]] = {}
        self.closed_form_transactions = 0
        self.fallback_transactions = 0

    def _resource(self, kind: str):
        if kind == "weight_write":
            return "weight", self.resource.weight, "write"
        # Call the frozen M768 implementation directly.  M785's override uses
        # ``super()`` and therefore cannot be borrowed by an unrelated class.
        return M785.M777.M768.AddressTimedScheduler._resource(self, kind)

    @staticmethod
    def _port_name(port, operation: str) -> str:
        return "rw" if port.port_mode == "1RW" else operation

    def _outstanding_bound(self, resource_name: str, banks: Tuple[int, ...],
                           candidate: int, limit: int) -> int:
        bound = int(candidate)
        changed = True
        while changed:
            changed = False
            for bank in banks:
                occupied = [value for value in self.outstanding_returns.get(
                    (resource_name, bank), []) if value > candidate]
                if len(occupied) >= limit:
                    proposed = sorted(occupied)[len(occupied) - limit]
                    if proposed > bound:
                        bound = proposed
                        changed = True
        return bound

    def _validate_range(self, kind: str, addresses: Sequence[int],
                        width: int) -> None:
        if kind in ("psum_read", "psum_write"):
            require(all(0 <= address and address + width <=
                        self.resource.psum_bytes_logical
                        for address in addresses), "psum physical range drift")
        if kind in ("weight_read", "weight_write"):
            per_bank = self.resource.weight_bytes_logical // M785.WEIGHT_BANKS
            require(all(0 <= address and address + width <= per_bank
                        for address in addresses), "weight physical range drift")

    def _can_closed_form(self, tx: CompressedTransaction,
                         resource_name: str, port_name: str) -> bool:
        if int(tx.count) < 4:
            return False
        for bank in tx.bank_pattern:
            if self.next_port_cycle.get((resource_name, bank, port_name), 0):
                return False
            if self.outstanding_returns.get((resource_name, bank), []):
                return False
        return True

    @staticmethod
    def _closed_form_issues(count: int, base: int, service: int,
                            distance: int, outstanding: int) -> List[int]:
        require(count >= 1 and service >= 1 and distance >= 0 and
                outstanding >= 1, "closed-form parameter drift")
        if distance <= outstanding * service:
            return [base + index * service for index in range(count)]
        return [base + (index // outstanding) * distance +
                (index % outstanding) * service for index in range(count)]

    def schedule(self, ir: PackedGroupIR, *, retain_details: bool,
                 retain_expanded_address_sha: bool,
                 force_fallback: bool = False) -> Dict[str, object]:
        liveness = TerminalLiveness(ir.dependency_uses)
        events = PackedPriorityEvents()
        address_digest = hashlib.sha256()
        commit_digest = hashlib.sha256()
        audit_terminal: Dict[str, int] = {}
        details: Optional[List[ScheduledRequest]] = [] if retain_details else None
        maximum_commit = -1
        expanded = commit_ordinal = 0
        populations, configs = set(), set()

        for tx in ir.transactions:
            resource_name, port, operation = self._resource(tx.kind)
            port_name = self._port_name(port, operation)
            dependencies_ready = liveness.resolve_group(tx.dependency_tokens)
            latency = port.read_latency if operation == "read" else port.write_latency
            beats = max(1, math.ceil(int(tx.width_bytes) /
                        (self.resource.external_bytes_per_cycle
                         if resource_name == "external" else port.row_bytes)))
            service = max(port.initiation_interval, beats)
            distance = latency + beats - 1
            use_closed = (not force_fallback and self._can_closed_form(
                tx, resource_name, port_name))
            if use_closed:
                base = max(int(tx.earliest_issue_cycle), dependencies_ready)
                issues = self._closed_form_issues(
                    int(tx.count), base, service, distance,
                    int(port.outstanding_per_bank))
                self.closed_form_transactions += 1
            else:
                issues = []
                self.fallback_transactions += 1

            offsets = (tuple(tx.address_offsets) if tx.address_offsets else
                       tuple(bank * int(tx.width_bytes)
                             for bank in tx.bank_pattern))
            for index in range(int(tx.count)):
                addresses = tuple(int(tx.base_address) +
                                  index * int(tx.address_stride_bytes) + offset
                                  for offset in offsets)
                self._validate_range(tx.kind, addresses, int(tx.width_bytes))
                port_bound = max((self.next_port_cycle.get(
                    (resource_name, bank, port_name), 0)
                                  for bank in tx.bank_pattern), default=0)
                initial = max(int(tx.earliest_issue_cycle),
                              dependencies_ready, port_bound)
                outstanding_bound = self._outstanding_bound(
                    resource_name, tuple(tx.bank_pattern), initial,
                    int(port.outstanding_per_bank))
                exact_issue = max(initial, outstanding_bound)
                issue = issues[index] if use_closed else exact_issue
                require(issue == exact_issue,
                        "closed-form endpoint diverged from frozen recurrence")
                if issue == int(tx.earliest_issue_cycle):
                    reason = "none"
                elif dependencies_ready == issue:
                    reason = "dependency_completion"
                elif outstanding_bound == issue and outstanding_bound > initial:
                    reason = "memory"
                elif port_bound == issue:
                    reason = ("weight_bank" if resource_name == "weight" else
                              "psum_bank" if resource_name == "psum" else
                              "memory" if resource_name == "external" else
                              "compute")
                else:
                    reason = "compute"
                returned = issue + distance
                for bank in tx.bank_pattern:
                    self.next_port_cycle[(resource_name, bank, port_name)] = (
                        issue + service)
                    key = (resource_name, bank)
                    current = [value for value in
                               self.outstanding_returns.get(key, [])
                               if value > issue]
                    current.append(returned)
                    self.outstanding_returns[key] = current
                produced = token_for(tx, index)
                liveness.produce(produced, returned)
                if produced and index == int(tx.count) - 1:
                    audit_terminal[produced] = returned
                request_id = "{}:{}".format(tx.transaction_id, index)
                if retain_expanded_address_sha:
                    for bank, address in zip(tx.bank_pattern, addresses):
                        address_digest.update(json.dumps(
                            [request_id, tx.kind, address, bank],
                            separators=(",", ":")).encode("utf-8"))
                if tx.kind == "commit":
                    for address in addresses:
                        commit_digest.update(json.dumps(
                            [commit_ordinal, address, int(tx.width_bytes)],
                            separators=(",", ":")).encode("utf-8"))
                        commit_ordinal += 1
                events.observe(int(tx.earliest_issue_cycle), dependencies_ready,
                               issue, returned, reason)
                if details is not None:
                    details.append(ScheduledRequest(
                        request_id=request_id,
                        transaction_id=tx.transaction_id,
                        population_id=tx.population_id,
                        config=tx.config,
                        kind=tx.kind,
                        addresses=addresses,
                        banks=tuple(tx.bank_pattern),
                        width_bytes=int(tx.width_bytes),
                        dependency_tokens=tuple(tx.dependency_tokens),
                        earliest_issue_cycle=int(tx.earliest_issue_cycle),
                        dependency_ready_cycle=dependencies_ready,
                        issue_cycle=issue,
                        return_cycle=returned,
                        commit_cycle=returned,
                        wait_reason=reason,
                        produces_token=produced))
                expanded += 1
                maximum_commit = max(maximum_commit, returned)
            # Release only after the final request in this consuming group.
            liveness.consume_group(tx.dependency_tokens, int(tx.count))
            populations.add(tx.population_id)
            configs.add(tx.config)

        require(expanded == ir.expanded_count and expanded > 0,
                "expanded request conservation failure")
        liveness.finish()
        total = maximum_commit + 1
        output: Dict[str, object] = {
            "total_cycles": total,
            "expanded_request_count": expanded,
            "compressed_transaction_count": len(ir.transactions),
            "compressed_group_ir_sha256": ir.compressed_group_ir_sha256,
            "expanded_address_sha256": (address_digest.hexdigest()
                                        if retain_expanded_address_sha else None),
            "transaction_address_sha256": (address_digest.hexdigest()
                                           if retain_expanded_address_sha else None),
            "commit_sequence_sha256": commit_digest.hexdigest(),
            "population_ids": sorted(populations),
            "configs": sorted(configs),
            "cycle_classes": events.finalize(total),
            "same_cycle_response_slot_reuse": True,
            "terminal_readiness": audit_terminal,
            "terminal_readiness_sha256": M785.canonical_sha256(audit_terminal),
            "live_token_peak": liveness.live_peak,
            "live_token_final": len(liveness.ready),
            "packed_event_bytes": events.packed_event_bytes,
            "packed_ir_bytes": ir.packed_ir_bytes,
            "closed_form_transactions": self.closed_form_transactions,
            "fallback_transactions": self.fallback_transactions,
            "port_calendars": calendar_identity(
                self.next_port_cycle, self.outstanding_returns),
            "detail_retained": details is not None,
        }
        if details is not None:
            output["scheduled_requests"] = [asdict(row) for row in details]
            output["compressed_schedule"] = (
                M785.M777.M768.compress_scheduled_rows(details))
        return output


def truncate_transactions(transactions: Iterable[CompressedTransaction],
                          expanded_limit: int) -> List[CompressedTransaction]:
    expanded_limit = int(expanded_limit)
    require(1 <= expanded_limit <= 100000,
            "bounded transaction prefix limit drift")
    output: List[CompressedTransaction] = []
    remaining = expanded_limit
    for tx in transactions:
        if remaining <= 0:
            break
        count = min(int(tx.count), remaining)
        output.append(tx if count == int(tx.count) else replace(tx, count=count))
        remaining -= count
    require(remaining == 0, "transaction source shorter than bounded prefix")
    return output


def real_prefix_transactions(limit: int) -> List[CompressedTransaction]:
    """Freeze the same first D0/A1/t0 identity as M861/M883."""
    require(1 <= int(limit) <= 100000, "real prefix is bounded to 100K")
    contract = M785.strict_json(HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    manifest = M785.strict_json(payload_root / "manifest.json")
    records = M785.normalized_population_records(
        manifest, "M686_ZURICH_CITY_09_A_S10")
    record = records[0]
    require(int(record["module_index"]) == 0 and int(record["sample_id"]) == 0,
            "first sealed decoder row identity drift")
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(HW / mapper_row["path"],
                                     mapper_row["sha256"], "m890_mapper")
    m712, m722, storage = (contract["inputs"][name] for name in
                           ("m712_oracle", "m722r2_oracle",
                            "m785_storage_oracle"))
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])
    stream = M785.iter_record_transactions(
        mapper, record, payload_root, "M686_ZURICH_CITY_09_A_S10",
        "A1_OSG", 0, oracles)
    return truncate_transactions(stream, int(limit))


def synthetic_transactions(count: int, *, q_probe: bool = False) -> List[CompressedTransaction]:
    count = int(count)
    require(1 <= count <= 10000, "synthetic transaction bound drift")
    rows: List[CompressedTransaction] = []
    produced = ""
    remaining = count
    ordinal = 0
    while remaining:
        width = min(remaining, 17 if q_probe else 64)
        kind = ("external_read", "weight_read", "psum_read", "compute",
                "psum_write", "external_write", "commit")[ordinal % 7]
        if kind.startswith("weight"):
            banks, item_width, base, stride = (ordinal % 8,), 16, 0, 16
        elif kind.startswith("psum"):
            banks, item_width, base, stride = (ordinal % 6,), 48, 0, 48
        elif kind == "compute":
            banks, item_width, base, stride = (0,), 288, 1 << 60, 288
        else:
            banks, item_width, base, stride = (0,), 192, 1 << 60, 192
        prefix = "m890:synthetic:{}".format(ordinal)
        dependency = (produced,) if produced else ()
        tx = CompressedTransaction(
            transaction_id=prefix,
            population_id="M890_SYNTHETIC",
            config="TYPED_SIGNED_K8",
            kind=kind,
            base_address=base,
            address_stride_bytes=stride,
            count=width,
            bank_pattern=banks,
            width_bytes=item_width,
            dependency_tokens=dependency,
            produces_token_prefix=prefix + ":done",
            earliest_issue_cycle=ordinal % 5)
        rows.append(tx)
        produced = terminal_token(tx)
        remaining -= width
        ordinal += 1
    return rows


def _reference_terminal_map(reference: Mapping[str, object],
                            ir: PackedGroupIR) -> Dict[str, int]:
    wanted = {terminal_token(tx) for tx in ir.transactions
              if tx.produces_token_prefix}
    return {str(row["produces_token"]): int(row["return_cycle"])
            for row in reference["scheduled_requests"]
            if row["produces_token"] in wanted}


def calendar_identity(next_port_cycle: Mapping[Tuple[str, int, str], int],
                      outstanding_returns: Mapping[
                          Tuple[str, int], Sequence[int]]) -> Dict[str, object]:
    """Canonical JSON-safe identity of every frozen port calendar."""
    return {
        "next_port_cycle": [
            [resource, int(bank), port, int(value)]
            for (resource, bank, port), value in sorted(next_port_cycle.items())
        ],
        "outstanding_returns": [
            [resource, int(bank), [int(value) for value in values]]
            for (resource, bank), values in sorted(outstanding_returns.items())
        ],
    }


def exact_miter(transactions: Sequence[CompressedTransaction],
                *, include_old: bool = True,
                force_fallback: bool = False) -> Dict[str, object]:
    ir = PackedGroupIR(transactions,
                       (transactions[0].population_id,
                        transactions[0].config, 0, 0, 0))
    requests = list(M785.expand_transactions(transactions))
    require(len(requests) <= 100000, "miter prefix exceeds 100K")
    resource = M861._synthetic_resource()
    # Real transactions require the frozen physical resource rather than the
    # deliberately tiny synthetic ranges.
    if transactions[0].population_id != "M890_SYNTHETIC":
        resource = M785.resource_from_contract(M785.strict_json(
            HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"))
    streaming_scheduler = M861.StreamingAddressTimedScheduler(resource)
    streaming = streaming_scheduler.schedule(iter(requests), retain_details=True)
    new_scheduler = GTLSScheduler(resource)
    new = new_scheduler.schedule(ir, retain_details=True,
                                 retain_expanded_address_sha=True,
                                 force_fallback=force_fallback)
    fields = (
        "total_cycles", "expanded_request_count",
        "compressed_transaction_count", "scheduled_requests",
        "compressed_schedule", "transaction_address_sha256",
        "commit_sequence_sha256", "population_ids", "configs",
        "cycle_classes", "same_cycle_response_slot_reuse",
    )
    for field in fields:
        require(streaming[field] == new[field],
                "M861/GTLS miter mismatch: " + field)
    reference_terminal = _reference_terminal_map(streaming, ir)
    require(reference_terminal == new["terminal_readiness"],
            "terminal readiness miter mismatch")
    require(calendar_identity(streaming_scheduler.next_port_cycle,
                              streaming_scheduler.outstanding_returns) ==
            new["port_calendars"], "M861/GTLS port-calendar mismatch")
    require(new["live_token_final"] == 0,
            "terminal liveness did not drain")
    if include_old:
        require(len(requests) <= 10000,
                "frozen old reference is bounded to 10K")
        old = M785.AddressTimedScheduler(resource).schedule(requests)
        for field in fields:
            require(old[field] == new[field],
                    "M768/GTLS miter mismatch: " + field)
    return {
        "status": "PASS_EXACT_M768_M861_GTLS_MITER" if include_old else
                  "PASS_EXACT_M861_GTLS_MITER",
        "expanded_requests": len(requests),
        "compressed_transactions": len(transactions),
        "closed_form_transactions": new["closed_form_transactions"],
        "fallback_transactions": new["fallback_transactions"],
        "live_token_peak": new["live_token_peak"],
        "packed_event_bytes": new["packed_event_bytes"],
        "compressed_group_ir_sha256": new["compressed_group_ir_sha256"],
        "terminal_readiness_sha256": new["terminal_readiness_sha256"],
        "fields": list(fields),
    }


def liveness_attack_self_test() -> Dict[str, object]:
    ledger = TerminalLiveness({"source": 3})
    ledger.produce("source", 7)
    require(ledger.resolve_group(("source",)) == 7,
            "multi-consumer readiness drift")
    ledger.consume_group(("source",), 2)
    attacked = False
    try:
        ledger.consume_group(("source",), 1, force_early_retire=True)
    except Failure:
        attacked = True
    require(attacked, "premature-retirement attack was not rejected")
    require(ledger.resolve_group(("source",)) == 7,
            "last legal consumer failed")
    ledger.consume_group(("source",), 1)
    reused = False
    try:
        ledger.resolve_group(("source",))
    except Failure:
        reused = True
    require(reused, "post-retirement reuse attack was not rejected")
    ledger.finish()
    return {"status": "PASS_TERMINAL_LIVENESS_ATTACKS",
            "premature_retirement_rejected": attacked,
            "post_retirement_reuse_rejected": reused}


def closed_form_boundary_self_test() -> Dict[str, object]:
    resource = M861._synthetic_resource()
    counts = set()
    for port in (resource.weight, resource.psum, resource.external,
                 resource.compute):
        q = int(port.outstanding_per_bank)
        counts.update((1, max(1, q - 1), q, q + 1))
    checks = 0
    for count in sorted(counts):
        rows = synthetic_transactions(max(32, count), q_probe=True)
        fast = exact_miter(rows, include_old=True, force_fallback=False)
        slow = exact_miter(rows, include_old=True, force_fallback=True)
        require(fast["terminal_readiness_sha256"] ==
                slow["terminal_readiness_sha256"],
                "closed-form/fallback readiness mismatch")
        checks += 1
    return {"status": "PASS_CLOSED_FORM_Q_BOUNDARIES",
            "boundary_cases": checks}


def validate_source_candidate(contract_path: Path) -> Dict[str, object]:
    contract = M785.strict_json(Path(contract_path))
    require(contract.get("schema") == CONTRACT_SCHEMA,
            "M890 contract schema drift")
    require(contract.get("status") ==
            "SOURCE_ONLY_GTLS__FRESH_HAMMER_REQUIRED" and
            contract.get("launch_now") is False and
            contract.get("full_first_row") is False and
            contract.get("full_population") is False,
            "M890 source candidate is not fail closed")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    for directory, filename, expected in (
            (M883_DIR, "review.json", M883_IDENTITY),
            (M886_DIR, "review.json", M886_IDENTITY),
            (M887_DIR, "handoff.json", M887_IDENTITY),
            (M888_DIR, "request.json", M888_IDENTITY)):
        identity = M785.verify_sealed_directory(directory)
        require(sha256(directory / filename) == expected[0] and
                identity["manifest_sha256"] == expected[1] and
                identity["outer_seal_file_sha256"] == expected[2],
                "bound authority drift: " + directory.name)
    for name, row in contract["source_identity"].items():
        path = HW / row["path"]
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == row["sha256"],
                "source identity drift: " + name)
    require(contract["future_full_row_gate"] == {
        "anchor_elapsed_seconds": 932.078357,
        "candidate_end_to_end_seconds_max": 9.320783571,
        "minimum_host_speedup": 100.0,
        "execution_authorized_now": False,
    }, "future 100x gate drift")
    return {
        "status": "PASS_M890_SOURCE_IDENTITY_ONLY__NO_FULL_ROW",
        "contract_sha256": sha256(Path(contract_path)),
        "launch_now": False,
        "full_first_row": False,
        "full_population": False,
        "production_cycles": None,
        "production_speedup": None,
    }


def source_self_test() -> Dict[str, object]:
    transactions = synthetic_transactions(1000)
    miter = exact_miter(transactions, include_old=True)
    attack = liveness_attack_self_test()
    boundary = closed_form_boundary_self_test()
    ir = PackedGroupIR(transactions,
                       ("M890_SYNTHETIC", "TYPED_SIGNED_K8", 0, 0, 0))
    require(ir.deterministic_shard(17) == ir.deterministic_shard(17),
            "row sharding is nondeterministic")
    return {
        "status": "PASS_M890_BOUNDED_SOURCE_SELF_TEST__NO_FULL_ROW",
        "synthetic_miter": miter,
        "liveness": attack,
        "closed_form": boundary,
        "deterministic_shard_17": ir.deterministic_shard(17),
        "full_first_row": False,
        "full_population": False,
        "cycles_or_speedup_citable": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source-candidate", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--real-prefix", type=int)
    parser.add_argument("--run-full-first-row", action="store_true")
    parser.add_argument("--run-production", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(not args.run_full_first_row and not args.run_production,
            "M890 source candidate refuses full-row/production replay")
    require(args.output is None,
            "M890 source candidate refuses result publication")
    if args.self_test:
        print(json.dumps(source_self_test(), sort_keys=True,
                         allow_nan=False))
        return 0
    if args.validate_source_candidate:
        require(args.contract is not None, "contract is required")
        print(json.dumps(validate_source_candidate(args.contract),
                         sort_keys=True, allow_nan=False))
        return 0
    if args.real_prefix is not None:
        require(args.real_prefix in (1000, 10000, 100000),
                "only sealed bounded real prefixes are allowed")
        rows = real_prefix_transactions(args.real_prefix)
        print(json.dumps(exact_miter(
            rows, include_old=args.real_prefix <= 10000),
            sort_keys=True, allow_nan=False))
        return 0
    raise Failure("only bounded source validation/test modes are authorized")


if __name__ == "__main__":
    raise SystemExit(main())
