#!/usr/bin/env python3
"""M896 bounded RUN-GTLS exact decoder scheduling source candidate.

This additive successor keeps M890's frozen M785/M768 endpoint equations and
changes only resident host state.  Per-request event endpoints are replaced by
maximal half-open run unions.  Closed-form transactions use counted arithmetic
progressions directly, without allocating an issue-cycle list.  Terminal
liveness is encoded by packed run-index arrays and retired online.

The source refuses full-row, population, production, publication, EDA, GPU and
remote modes.  It may execute only bounded synthetic or D0/A1/t0 prefixes up to
100K expanded requests for exact source miters and scaling preflight.
"""

import argparse
from array import array
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M890_PATH = HERE / "analyze_m890_decoder_gtls_source_candidate.py"
M890_SHA256 = "cacc118ea33616ae4284403ad69656bbeacaa7bc83d227c0d9b5a86c2ead459e"
M893_DIR = HW / "reviews/m893_m890_decoder_gtls_source_fresh_hammer_r1_20260829"
M893_IDENTITY = (
    "f883f68ca27aca654a558e2cb27ee3d9a56b490c4cba0e481523781ae4e7d102",
    "8642b26197cfbdf7f71e47d22c2ad92e3586f1555d975dd3dcb938f13709ced9",
    "a21108afcea9b0ed2e85314c20878338835370151b41923019e990827addaf3b",
)
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT_SCHEMA = "m896_decoder_run_gtls_source_candidate_v1"
FULL_ROW_REQUESTS = 38672612
STATE_GATE_BYTES = 512 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen(path: Path, expected: str, name: str):
    if sha256(path) != expected:
        raise RuntimeError(name + " identity drift")
    spec = importlib.util.spec_from_file_location("m896_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M890 = _load_frozen(M890_PATH, M890_SHA256, "frozen_m890")
M861 = M890.M861
M785 = M890.M785
Failure = M890.Failure
require = M890.require
CompressedTransaction = M890.CompressedTransaction
ScheduledRequest = M890.ScheduledRequest
CYCLE_CLASS_ORDER = M890.CYCLE_CLASS_ORDER


def _deep_size(value: object, seen: Optional[set] = None) -> int:
    """Measure live Python/container state; never use serialized byte size."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_deep_size(key, seen) + _deep_size(item, seen)
                    for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _deep_size(vars(value), seen)
    return size


class MaximalHalfOpenRuns:
    """Online exact union of integer half-open intervals.

    The retained state is one pair per maximal run, never one endpoint per
    request.  Contiguous active-service cycles therefore collapse into a
    single run even when requests arrive out of cycle order.
    """

    def __init__(self) -> None:
        # Interleaved int64 [start0,end0,start1,end1,...].  A Python tuple per
        # run would alone fail the conservative full-row projection gate.
        self.runs = array("q")

    def add(self, start: int, end: int) -> None:
        start, end = int(start), int(end)
        require(0 <= start <= end, "invalid half-open run")
        if start == end:
            return
        rows = self.runs
        low, high = 0, len(rows) // 2
        while low < high:
            middle = (low + high) // 2
            if int(rows[2 * middle]) < start:
                low = middle + 1
            else:
                high = middle
        index = low
        if index and int(rows[2 * (index - 1) + 1]) >= start:
            index -= 1
        merged_start, merged_end = start, end
        stop = index
        while stop < len(rows) // 2 and int(rows[2 * stop]) <= merged_end:
            merged_start = min(merged_start, int(rows[2 * stop]))
            merged_end = max(merged_end, int(rows[2 * stop + 1]))
            stop += 1
        rows[2 * index:2 * stop] = array("q", [merged_start, merged_end])

    def iter_runs(self):
        for index in range(0, len(self.runs), 2):
            yield int(self.runs[index]), int(self.runs[index + 1])

    def add_counted_progression(self, start: int, step: int,
                                count: int) -> None:
        """Consume an exact counted AP without allocating its point list."""
        start, step, count = int(start), int(step), int(count)
        require(start >= 0 and step >= 1 and count >= 1,
                "invalid counted arithmetic progression")
        if step == 1:
            self.add(start, start + count)
            return
        for index in range(count):
            point = start + index * step
            self.add(point, point + 1)

    @property
    def covered(self) -> int:
        return sum(end - start for start, end in self.iter_runs())

    @property
    def count(self) -> int:
        return len(self.runs) // 2

    @property
    def resident_bytes(self) -> int:
        return _deep_size(self.runs)


class OnlinePriorityRuns:
    """Six-class priority aggregation backed only by maximal run unions."""

    def __init__(self) -> None:
        self.classes = {name: MaximalHalfOpenRuns()
                        for name in CYCLE_CLASS_ORDER[:-1]}

    def observe(self, earliest: int, dependency: int, issue: int,
                returned: int, reason: str) -> None:
        earliest, dependency = int(earliest), int(dependency)
        issue, returned = int(issue), int(returned)
        require(0 <= earliest <= issue <= returned,
                "scheduled endpoint ordering drift")
        self.classes["active_service"].add(issue, issue + 1)
        if issue < returned:
            self.classes["dependency_completion"].add(issue, returned)
        if earliest < issue:
            if dependency > earliest:
                self.classes["dependency_completion"].add(
                    earliest, min(issue, dependency))
            if reason == "dependency_completion":
                self.classes["dependency_completion"].add(earliest, issue)
            elif reason in ("weight_bank", "psum_bank", "memory"):
                self.classes[reason].add(earliest, issue)
            else:
                require(reason in ("none", "compute"),
                        "unknown frozen wait reason")

    def finalize(self, total_cycles: int) -> Dict[str, int]:
        total_cycles = int(total_cycles)
        require(total_cycles >= 1, "empty cycle domain")
        # The sweep contains one boundary per maximal run, not per request.
        events: Dict[int, List[int]] = {0: [0] * 5,
                                        total_cycles: [0] * 5}
        for priority, name in enumerate(CYCLE_CLASS_ORDER[:-1]):
            for start, end in self.classes[name].iter_runs():
                start = max(0, min(total_cycles, int(start)))
                end = max(0, min(total_cycles, int(end)))
                if start >= end:
                    continue
                events.setdefault(start, [0] * 5)[priority] += 1
                events.setdefault(end, [0] * 5)[priority] -= 1
        result = {name: 0 for name in CYCLE_CLASS_ORDER}
        state = [0] * 5
        points = sorted(events)
        for ordinal, point in enumerate(points[:-1]):
            state = [value + change for value, change in
                     zip(state, events[point])]
            require(all(value >= 0 for value in state),
                    "online priority run underflow")
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
                "online priority run conservation failure")
        return result

    @property
    def run_counts(self) -> Dict[str, int]:
        return {name: value.count for name, value in self.classes.items()}

    @property
    def resident_bytes(self) -> int:
        return _deep_size(self.classes)


class RunGroupIR:
    """Transaction-run IR with packed dependency/liveness metadata."""

    def __init__(self, transactions: Sequence[CompressedTransaction],
                 shard_key: Tuple[str, str, int, int, int]):
        require(transactions, "empty RUN-GTLS IR")
        self.transactions = tuple(transactions)
        self.shard_key = tuple(shard_key)
        require(len(self.shard_key) == 5, "row shard key arity drift")
        count = len(self.transactions)
        self.expanded_count = 0
        self.dependency_offsets = array("q", [0])
        self.dependency_indices = array("q")
        self.dependency_uses = array("q", [0]) * count
        producer: Dict[str, int] = {}
        digest = hashlib.sha256()
        for ordinal, tx in enumerate(self.transactions):
            tx.validate()
            digest.update(M890._canonical_bytes(M890._transaction_dict(tx)))
            self.expanded_count += int(tx.count)
            for dependency in tx.dependency_tokens:
                require(dependency in producer,
                        "dependency is not a preceding terminal token: " +
                        dependency)
                producer_index = producer[dependency]
                self.dependency_indices.append(producer_index)
                self.dependency_uses[producer_index] += int(tx.count)
            self.dependency_offsets.append(len(self.dependency_indices))
            terminal = M890.terminal_token(tx)
            if terminal:
                require(terminal not in producer,
                        "duplicate terminal token production")
                producer[terminal] = ordinal
        self.compressed_group_ir_sha256 = digest.hexdigest()
        # The string dictionary is construction-only and is deliberately not
        # retained as scheduler liveness state.
        del producer

    def dependencies(self, ordinal: int) -> Tuple[int, ...]:
        start = int(self.dependency_offsets[ordinal])
        end = int(self.dependency_offsets[ordinal + 1])
        return tuple(int(value) for value in
                     self.dependency_indices[start:end])

    def deterministic_shard(self, shard_count: int) -> int:
        shard_count = int(shard_count)
        require(shard_count >= 1, "invalid shard count")
        value = int(hashlib.sha256(M890._canonical_bytes(
            list(self.shard_key))).hexdigest()[:16], 16)
        return value % shard_count

    @property
    def packed_control_bytes(self) -> int:
        return (_deep_size(self.dependency_offsets) +
                _deep_size(self.dependency_indices) +
                _deep_size(self.dependency_uses))


class CompactTerminalLiveness:
    """Packed run-index readiness with no retained token strings."""

    def __init__(self, uses: array):
        # Scheduling is deliberately one-shot: ownership of the packed use
        # ledger moves from the IR to liveness, avoiding a second O(runs)
        # resident copy.  A repeated schedule must construct a fresh IR.
        self.remaining = uses
        self.ready = array("q", [-1]) * len(uses)
        self.live = 0
        self.live_peak = 0

    def produce(self, ordinal: int, ready_cycle: int) -> None:
        if self.remaining[ordinal] == 0:
            return
        require(self.ready[ordinal] < 0,
                "duplicate live terminal production")
        self.ready[ordinal] = int(ready_cycle)
        self.live += 1
        self.live_peak = max(self.live_peak, self.live)

    def resolve(self, dependencies: Sequence[int]) -> int:
        require(all(self.remaining[index] > 0 and self.ready[index] >= 0
                    for index in dependencies),
                "unresolved or retired terminal dependency")
        return max((int(self.ready[index]) for index in dependencies),
                   default=0)

    def consume(self, dependencies: Sequence[int], count: int,
                *, force_early_retire: bool = False) -> None:
        count = int(count)
        require(count >= 1, "invalid dependency group count")
        for index in dependencies:
            decrement = count + (1 if force_early_retire else 0)
            require(self.remaining[index] >= decrement and
                    self.ready[index] >= 0,
                    "premature or post-retirement terminal attack")
            self.remaining[index] -= decrement
            if self.remaining[index] == 0:
                self.ready[index] = -1
                self.live -= 1

    def finish(self) -> None:
        require(self.live == 0 and all(value == 0 for value in self.remaining),
                "terminal liveness did not drain")

    @property
    def resident_bytes(self) -> int:
        return _deep_size(self.remaining) + _deep_size(self.ready)


class RUNGTLSScheduler:
    """M890-equivalent scheduler with run-compressed resident event state."""

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
                        for address in addresses),
                    "psum physical range drift")
        if kind in ("weight_read", "weight_write"):
            per_bank = self.resource.weight_bytes_logical // M785.WEIGHT_BANKS
            require(all(0 <= address and address + width <= per_bank
                        for address in addresses),
                    "weight physical range drift")

    def _can_closed_form(self, tx: CompressedTransaction,
                         resource_name: str, port_name: str) -> bool:
        if int(tx.count) < 4:
            return False
        return all(not self.next_port_cycle.get(
                       (resource_name, bank, port_name), 0) and
                   not self.outstanding_returns.get((resource_name, bank), [])
                   for bank in tx.bank_pattern)

    @staticmethod
    def _counted_ap_issue(index: int, base: int, service: int,
                          distance: int, outstanding: int) -> int:
        """Return one point from the closed-form counted AP, without a list."""
        if distance <= outstanding * service:
            return base + index * service
        return (base + (index // outstanding) * distance +
                (index % outstanding) * service)

    def schedule(self, ir: RunGroupIR, *, retain_details: bool,
                 retain_expanded_address_sha: bool,
                 retain_terminal_audit: bool = True) -> Dict[str, object]:
        liveness = CompactTerminalLiveness(ir.dependency_uses)
        events = OnlinePriorityRuns()
        address_digest = hashlib.sha256()
        commit_digest = hashlib.sha256()
        terminal_ready = (array("q", [-1]) * len(ir.transactions)
                          if retain_terminal_audit else None)
        details: Optional[List[ScheduledRequest]] = [] if retain_details else None
        maximum_commit = -1
        expanded = commit_ordinal = 0
        populations, configs = set(), set()

        for ordinal, tx in enumerate(ir.transactions):
            resource_name, port, operation = self._resource(tx.kind)
            port_name = self._port_name(port, operation)
            dependency_indices = ir.dependencies(ordinal)
            dependencies_ready = liveness.resolve(dependency_indices)
            latency = port.read_latency if operation == "read" else port.write_latency
            beats = max(1, math.ceil(int(tx.width_bytes) /
                        (self.resource.external_bytes_per_cycle
                         if resource_name == "external" else port.row_bytes)))
            service = max(port.initiation_interval, beats)
            distance = latency + beats - 1
            use_closed = self._can_closed_form(tx, resource_name, port_name)
            if use_closed:
                base = max(int(tx.earliest_issue_cycle), dependencies_ready)
                self.closed_form_transactions += 1
            else:
                base = -1
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
                issue = (self._counted_ap_issue(
                    index, base, service, distance,
                    int(port.outstanding_per_bank)) if use_closed else
                         exact_issue)
                require(issue == exact_issue,
                        "counted-AP endpoint diverged from frozen recurrence")
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
                if index == int(tx.count) - 1:
                    liveness.produce(ordinal, returned)
                if (terminal_ready is not None and tx.produces_token_prefix and
                        index == int(tx.count) - 1):
                    terminal_ready[ordinal] = returned
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
                        produces_token=M890.token_for(tx, index)))
                expanded += 1
                maximum_commit = max(maximum_commit, returned)
            liveness.consume(dependency_indices, int(tx.count))
            populations.add(tx.population_id)
            configs.add(tx.config)

        require(expanded == ir.expanded_count and expanded > 0,
                "expanded request conservation failure")
        liveness.finish()
        total = maximum_commit + 1
        audit_terminal = ({
            M890.terminal_token(tx): int(terminal_ready[ordinal])
            for ordinal, tx in enumerate(ir.transactions)
            if tx.produces_token_prefix
        } if terminal_ready is not None else None)
        # This is the measured in-process control state: compact dependency
        # arrays, live readiness, maximal event runs and resource calendars.
        # Shared objects are de-duplicated by _deep_size; no serialized size is
        # substituted for resident state.
        combined_state = _deep_size((
            ir.dependency_offsets, ir.dependency_indices,
            liveness.remaining, liveness.ready, events,
            self.next_port_cycle, self.outstanding_returns))
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
            "terminal_readiness_sha256": (M785.canonical_sha256(audit_terminal)
                                          if audit_terminal is not None else None),
            "live_token_peak": liveness.live_peak,
            "live_token_final": liveness.live,
            "event_run_counts": events.run_counts,
            "event_resident_state_bytes": events.resident_bytes,
            "compact_control_state_bytes": ir.packed_control_bytes,
            "liveness_resident_state_bytes": liveness.resident_bytes,
            "combined_live_event_state_bytes": combined_state,
            "closed_form_transactions": self.closed_form_transactions,
            "fallback_transactions": self.fallback_transactions,
            "port_calendars": M890.calendar_identity(
                self.next_port_cycle, self.outstanding_returns),
            "detail_retained": details is not None,
        }
        if details is not None:
            output["scheduled_requests"] = [asdict(row) for row in details]
            output["compressed_schedule"] = (
                M785.M777.M768.compress_scheduled_rows(details))
        return output


def exact_miter(transactions: Sequence[CompressedTransaction],
                *, include_old: bool) -> Dict[str, object]:
    require(1 <= sum(int(tx.count) for tx in transactions) <= 100000,
            "bounded miter exceeds 100K")
    ir = RunGroupIR(transactions,
                    (transactions[0].population_id,
                     transactions[0].config, 0, 0, 0))
    resource = M861._synthetic_resource()
    if transactions[0].population_id != "M890_SYNTHETIC":
        resource = M785.resource_from_contract(M785.strict_json(
            HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"))
    new = RUNGTLSScheduler(resource).schedule(
        ir, retain_details=True, retain_expanded_address_sha=True)
    m890_ir = M890.PackedGroupIR(
        transactions, (transactions[0].population_id,
                       transactions[0].config, 0, 0, 0))
    m890_scheduler = M890.GTLSScheduler(resource)
    reference = m890_scheduler.schedule(
        m890_ir, retain_details=True, retain_expanded_address_sha=True)
    fields = (
        "total_cycles", "expanded_request_count",
        "compressed_transaction_count", "scheduled_requests",
        "compressed_schedule", "transaction_address_sha256",
        "commit_sequence_sha256", "population_ids", "configs",
        "cycle_classes", "same_cycle_response_slot_reuse",
        "terminal_readiness", "terminal_readiness_sha256", "port_calendars",
    )
    for field in fields:
        require(reference[field] == new[field],
                "M890/RUN-GTLS exact miter mismatch: " + field)
    if include_old:
        require(sum(int(tx.count) for tx in transactions) <= 10000,
                "M768/M861 bounded reference exceeds 10K")
        prior = M890.exact_miter(transactions, include_old=True)
        require(prior["terminal_readiness_sha256"] ==
                new["terminal_readiness_sha256"],
                "M768/M861/RUN-GTLS terminal miter mismatch")
    return {
        "status": ("PASS_EXACT_M768_M861_M890_RUN_GTLS_MITER" if include_old
                   else "PASS_EXACT_M861_M890_RUN_GTLS_MITER"),
        "expanded_requests": new["expanded_request_count"],
        "compressed_transactions": new["compressed_transaction_count"],
        "live_token_peak": new["live_token_peak"],
        "terminal_readiness_sha256": new["terminal_readiness_sha256"],
        "event_run_counts": new["event_run_counts"],
        "fields": list(fields),
    }


def measure_real_100k_state() -> Dict[str, object]:
    transactions = M890.real_prefix_transactions(100000)
    ir = RunGroupIR(transactions,
                    (transactions[0].population_id,
                     transactions[0].config, 0, 0, 0))
    resource = M785.resource_from_contract(M785.strict_json(
        HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"))
    result = RUNGTLSScheduler(resource).schedule(
        ir, retain_details=False, retain_expanded_address_sha=False,
        retain_terminal_audit=False)
    measured = int(result["combined_live_event_state_bytes"])
    projected = (measured * FULL_ROW_REQUESTS + 100000 - 1) // 100000
    require(projected <= STATE_GATE_BYTES,
            "RUN-GTLS conservative combined-state projection exceeds 512 MiB")
    return {
        "status": "PASS_RUN_GTLS_100K_COMBINED_STATE_PROJECTION_GATE",
        "expanded_requests": 100000,
        "combined_live_event_state_bytes": measured,
        "event_resident_state_bytes": result["event_resident_state_bytes"],
        "compact_control_state_bytes": result["compact_control_state_bytes"],
        "liveness_resident_state_bytes": result["liveness_resident_state_bytes"],
        "event_run_counts": result["event_run_counts"],
        "future_full_row_requests": FULL_ROW_REQUESTS,
        "conservative_projection_bytes": projected,
        "gate_bytes": STATE_GATE_BYTES,
        "projection_mib": projected / float(1024 * 1024),
        "gate_mib": 512,
        "input_transaction_objects_excluded": True,
        "serialized_or_compressed_file_size_used": False,
        "full_row_authorized": False,
    }


def liveness_attack_self_test() -> Dict[str, object]:
    uses = array("q", [3])
    ledger = CompactTerminalLiveness(uses)
    ledger.produce(0, 7)
    require(ledger.resolve((0,)) == 7, "readiness drift")
    ledger.consume((0,), 2)
    attacked = False
    try:
        ledger.consume((0,), 1, force_early_retire=True)
    except Failure:
        attacked = True
    require(attacked and ledger.resolve((0,)) == 7,
            "premature-retirement attack not rejected")
    ledger.consume((0,), 1)
    reused = False
    try:
        ledger.resolve((0,))
    except Failure:
        reused = True
    require(reused, "post-retirement attack not rejected")
    ledger.finish()
    return {"status": "PASS_COMPACT_LIVENESS_ATTACKS",
            "premature_retirement_rejected": attacked,
            "post_retirement_rejected": reused}


def priority_run_self_test() -> Dict[str, object]:
    old = M890.PackedPriorityEvents()
    new = OnlinePriorityRuns()
    rows = (
        (0, 0, 0, 3, "none"),
        (0, 3, 3, 5, "dependency_completion"),
        (0, 0, 7, 9, "weight_bank"),
        (2, 2, 8, 11, "psum_bank"),
        (1, 1, 12, 15, "memory"),
        (4, 4, 4, 4, "none"),
    )
    for row in rows:
        old.observe(*row)
        new.observe(*row)
    require(old.finalize(18) == new.finalize(18),
            "priority-run directed miter mismatch")
    probe = MaximalHalfOpenRuns()
    probe.add_counted_progression(2, 1, 7)
    probe.add_counted_progression(20, 3, 4)
    require(probe.covered == 11 and next(probe.iter_runs()) == (2, 9),
            "counted AP/run compression drift")
    return {"status": "PASS_PRIORITY_RUN_AND_COUNTED_AP_SELF_TEST",
            "cycle_classes": new.finalize(18),
            "run_counts": new.run_counts}


def validate_source_candidate(contract_path: Path) -> Dict[str, object]:
    contract = M785.strict_json(Path(contract_path))
    require(contract.get("schema") == CONTRACT_SCHEMA,
            "M896 contract schema drift")
    require(contract.get("status") ==
            "SOURCE_ONLY_RUN_GTLS__FRESH_HAMMER_REQUIRED" and
            contract.get("launch_now") is False and
            contract.get("full_first_row") is False and
            contract.get("full_population") is False,
            "M896 contract is not fail closed")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    identity = M785.verify_sealed_directory(M893_DIR)
    require(sha256(M893_DIR / "review.json") == M893_IDENTITY[0] and
            identity["manifest_sha256"] == M893_IDENTITY[1] and
            identity["outer_seal_file_sha256"] == M893_IDENTITY[2],
            "M893 authority drift")
    for name, row in contract["source_identity"].items():
        path = HW / row["path"]
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == row["sha256"],
                "source identity drift: " + name)
    return {
        "status": "PASS_M896_SOURCE_IDENTITY_ONLY__NO_FULL_ROW",
        "contract_sha256": sha256(Path(contract_path)),
        "launch_now": False,
        "full_first_row": False,
        "full_population": False,
        "production_cycles": None,
        "production_speedup": None,
    }


def source_self_test() -> Dict[str, object]:
    synthetic = exact_miter(M890.synthetic_transactions(1000),
                            include_old=True)
    return {
        "status": "PASS_M896_BOUNDED_SOURCE_SELF_TEST__NO_FULL_ROW",
        "synthetic_1k": synthetic,
        "priority_runs": priority_run_self_test(),
        "liveness": liveness_attack_self_test(),
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
    parser.add_argument("--measure-real-100k-state", action="store_true")
    parser.add_argument("--run-full-first-row", action="store_true")
    parser.add_argument("--run-production", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(not args.run_full_first_row and not args.run_production,
            "M896 source candidate refuses full-row/production replay")
    require(args.output is None,
            "M896 source candidate refuses result publication")
    if args.self_test:
        print(json.dumps(source_self_test(), sort_keys=True, allow_nan=False))
        return 0
    if args.validate_source_candidate:
        require(args.contract is not None, "contract is required")
        print(json.dumps(validate_source_candidate(args.contract),
                         sort_keys=True, allow_nan=False))
        return 0
    if args.real_prefix is not None:
        require(args.real_prefix in (1000, 10000, 100000),
                "only sealed bounded real prefixes are allowed")
        print(json.dumps(exact_miter(
            M890.real_prefix_transactions(args.real_prefix),
            include_old=args.real_prefix <= 10000),
            sort_keys=True, allow_nan=False))
        return 0
    if args.measure_real_100k_state:
        print(json.dumps(measure_real_100k_state(), sort_keys=True,
                         allow_nan=False))
        return 0
    raise Failure("only bounded source validation/test modes are authorized")


if __name__ == "__main__":
    raise SystemExit(main())
