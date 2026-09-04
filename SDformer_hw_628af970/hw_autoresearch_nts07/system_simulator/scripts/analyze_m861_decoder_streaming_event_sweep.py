#!/usr/bin/env python3
"""M861 source-only streaming/event-sweep successor to frozen M785.

M785/M768 transaction, resource and scheduling semantics remain immutable.
This additive module removes only two non-scalable implementation choices:

* the production-facing path no longer materializes ``rows`` or the returned
  ``scheduled_requests``/``compressed_schedule`` populations; and
* mutually exclusive cycle classes are reconstructed from exact half-open
  interval unions instead of scanning every scheduled request at every cycle.

Detailed retention is deliberately available for bounded old-vs-new miters.
Production replay, decoder cycles, speedup, result publication, VCS and EDA
remain unauthorized by the M861 source candidate.
"""

import argparse
from bisect import bisect_left
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import resource as process_resource
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M785_PATH = HERE / "analyze_m785_h67_decoder_physical_residency_repair.py"
M785_SHA256 = "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1"
M857_DIR = HW / "reviews/m857_m836_decoder_controlled_scalability_failure_hammer_r1_20260829"
M857_REVIEW_SHA256 = "c2b244e4d6d56af6d81c028aa0cfe000517161e67e2866cc1ca782c9fd58e75a"
M857_MANIFEST_SHA256 = "32e804ca48f3274f52fe9ce87fde320d343df469197f54540ccc9e1fb032381d"
M857_OUTER_SEAL_FILE_SHA256 = "dee8b308df28d7ea1b6840def6e9ac73319fa3dc406c56ec5a26e8edf60fb8db"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT_SCHEMA = "m861_decoder_streaming_event_sweep_candidate_v1"
CYCLE_CLASS_ORDER = (
    "active_service",
    "dependency_completion",
    "weight_bank",
    "psum_bank",
    "memory",
    "compute",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_m785():
    if _sha256(M785_PATH) != M785_SHA256:
        raise RuntimeError("frozen M785 identity drift")
    spec = importlib.util.spec_from_file_location("m861_frozen_m785", M785_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen M785")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M785 = _load_frozen_m785()
Failure = M785.Failure
require = M785.require
Request = M785.Request
ScheduledRequest = M785.M777.M768.ScheduledRequest


class IntervalUnion:
    """Incrementally maintained exact union of integer half-open intervals.

    The common scheduling case is monotone and therefore O(1) amortized.  A
    bisect/merge fallback preserves exactness for adversarial out-of-order
    endpoints used by the miter suite.
    """

    def __init__(self) -> None:
        self._intervals: List[Tuple[int, int]] = []
        self.out_of_order_insertions = 0

    def add(self, start: int, end: int) -> None:
        start, end = int(start), int(end)
        require(0 <= start <= end, "invalid half-open interval")
        if start == end:
            return
        rows = self._intervals
        if not rows:
            rows.append((start, end))
            return
        last_start, last_end = rows[-1]
        if start >= last_start:
            if start <= last_end:
                rows[-1] = (last_start, max(last_end, end))
            else:
                rows.append((start, end))
            return
        self.out_of_order_insertions += 1
        index = bisect_left(rows, (start, -1))
        if index and rows[index - 1][1] >= start:
            index -= 1
            start = min(start, rows[index][0])
            end = max(end, rows[index][1])
        limit = index
        while limit < len(rows) and rows[limit][0] <= end:
            start = min(start, rows[limit][0])
            end = max(end, rows[limit][1])
            limit += 1
        rows[index:limit] = [(start, end)]

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self._intervals)

    @property
    def cardinality(self) -> int:
        return sum(end - start for start, end in self._intervals)


class ExactCycleClassSweep:
    """Streaming interval observer for frozen M768 cycle-class semantics."""

    def __init__(self) -> None:
        self.coverage = {
            name: IntervalUnion()
            for name in CYCLE_CLASS_ORDER[:-1]
        }

    def observe(self, row: ScheduledRequest) -> None:
        earliest = int(row.earliest_issue_cycle)
        dependency = int(row.dependency_ready_cycle)
        issue = int(row.issue_cycle)
        returned = int(row.return_cycle)
        require(0 <= earliest <= issue <= returned,
                "scheduled endpoint ordering drift")

        # E/D/I/R semantics are integer half-open.  Active issue has highest
        # priority and therefore masks every interval at [I,I+1).
        self.coverage["active_service"].add(issue, issue + 1)
        if issue < returned:
            self.coverage["dependency_completion"].add(issue, returned)
        if earliest < issue:
            if dependency > earliest:
                self.coverage["dependency_completion"].add(
                    earliest, min(issue, dependency))
            reason = str(row.wait_reason)
            if reason == "dependency_completion":
                self.coverage["dependency_completion"].add(earliest, issue)
            elif reason in ("weight_bank", "psum_bank", "memory"):
                self.coverage[reason].add(earliest, issue)
            else:
                require(reason in ("none", "compute"),
                        "unknown frozen wait reason")

    def finalize(self, total_cycles: int) -> Dict[str, int]:
        total_cycles = int(total_cycles)
        require(total_cycles >= 1, "empty cycle domain")
        events: Dict[int, List[int]] = {0: [0] * 5,
                                        total_cycles: [0] * 5}
        for index, name in enumerate(CYCLE_CLASS_ORDER[:-1]):
            for start, end in self.coverage[name].intervals:
                start = min(total_cycles, max(0, start))
                end = min(total_cycles, max(0, end))
                if start >= end:
                    continue
                events.setdefault(start, [0] * 5)[index] += 1
                events.setdefault(end, [0] * 5)[index] -= 1
        points = sorted(events)
        active = [0] * 5
        result = {name: 0 for name in CYCLE_CLASS_ORDER}
        for ordinal, point in enumerate(points[:-1]):
            delta = events[point]
            active = [value + change for value, change in zip(active, delta)]
            require(all(value >= 0 for value in active),
                    "event-sweep coverage underflow")
            next_point = points[ordinal + 1]
            span = next_point - point
            if span <= 0 or point >= total_cycles:
                continue
            chosen = "compute"
            for index, name in enumerate(CYCLE_CLASS_ORDER[:-1]):
                if active[index] > 0:
                    chosen = name
                    break
            result[chosen] += span
        require(sum(result.values()) == total_cycles,
                "event-sweep timeline conservation failure")
        return result

    def diagnostics(self) -> Dict[str, object]:
        return {
            name: {
                "merged_intervals": len(union.intervals),
                "covered_cycles_before_priority": union.cardinality,
                "out_of_order_insertions": union.out_of_order_insertions,
            }
            for name, union in self.coverage.items()
        }


class StreamingCompressionCounter:
    """Count frozen compressed-schedule groups with O(1) retained state."""

    def __init__(self) -> None:
        self.count = 0
        self.previous: Optional[Tuple[object, ...]] = None

    def observe(self, row: ScheduledRequest) -> None:
        key = (
            row.transaction_id,
            row.population_id,
            row.config,
            row.kind,
            tuple(row.banks),
            tuple(row.dependency_tokens),
            int(row.width_bytes),
        )
        if key != self.previous:
            self.count += 1
            self.previous = key


class StreamingAddressTimedScheduler(M785.AddressTimedScheduler):
    """Exact frozen scheduler core with a streaming aggregate observer."""

    def _validate_physical_range(self, request: Request) -> None:
        per_weight_bank = self.resource.weight_bytes_logical // M785.WEIGHT_BANKS
        require(self.resource.weight_bytes_logical % M785.WEIGHT_BANKS == 0,
                "weight partition is not evenly banked")
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

    def schedule(self, requests: Iterable[Request], *,
                 retain_details: bool = False) -> Dict[str, object]:
        address_digest = hashlib.sha256()
        commit_digest = hashlib.sha256()
        population_ids = set()
        configs = set()
        commit_ordinal = 0
        expanded_count = 0
        maximum_commit = -1
        compression = StreamingCompressionCounter()
        sweep = ExactCycleClassSweep()
        detailed: Optional[List[ScheduledRequest]] = [] if retain_details else None

        for request in requests:
            self._validate_physical_range(request)
            resource_name, port, operation = self._resource(request.kind)
            require(request.config in M785.CONFIGS,
                    "request configuration drift")
            require(request.banks and len(request.banks) == len(request.addresses),
                    "address/bank arity mismatch")
            require(all(0 <= bank < port.banks for bank in request.banks),
                    "bank index out of range")
            missing = [token for token in request.dependency_tokens
                       if token not in self.token_ready]
            require(not missing,
                    "unresolved dependency token: " + repr(missing))
            dependency_ready = max(
                (self.token_ready[token] for token in request.dependency_tokens),
                default=request.earliest_issue_cycle,
            )
            port_name = self._port_name(port, operation)
            port_bound = max(
                (self.next_port_cycle.get((resource_name, bank, port_name), 0)
                 for bank in request.banks), default=0)
            initial = max(request.earliest_issue_cycle,
                          dependency_ready, port_bound)
            outstanding_bound = self._outstanding_bound(
                resource_name, request.banks, initial,
                port.outstanding_per_bank)
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
                    else "compute")
            else:
                reason = "compute"
            latency = (port.read_latency if operation == "read"
                       else port.write_latency)
            beats = max(1, math.ceil(
                request.width_bytes /
                (self.resource.external_bytes_per_cycle
                 if resource_name == "external" else port.row_bytes)))
            return_cycle = issue + latency + beats - 1
            commit_cycle = return_cycle
            for bank in request.banks:
                self.next_port_cycle[(resource_name, bank, port_name)] = (
                    issue + max(port.initiation_interval, beats))
                key = (resource_name, bank)
                current = [value for value in
                           self.outstanding_returns.get(key, [])
                           if value > issue]
                current.append(return_cycle)
                self.outstanding_returns[key] = current
            if request.produces_token:
                require(request.produces_token not in self.token_ready,
                        "duplicate produced token")
                self.token_ready[request.produces_token] = return_cycle
            for bank, address in zip(request.banks, request.addresses):
                address_digest.update(json.dumps(
                    [request.request_id, request.kind, address, bank],
                    separators=(",", ":")).encode("utf-8"))
            if request.kind == "commit":
                for address in request.addresses:
                    commit_digest.update(json.dumps(
                        [commit_ordinal, address, request.width_bytes],
                        separators=(",", ":")).encode("utf-8"))
                    commit_ordinal += 1
            population_ids.add(request.population_id)
            configs.add(request.config)
            row = ScheduledRequest(
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
            )
            sweep.observe(row)
            compression.observe(row)
            if detailed is not None:
                detailed.append(row)
            expanded_count += 1
            maximum_commit = max(maximum_commit, commit_cycle)

        require(expanded_count > 0, "empty schedule")
        total_cycles = maximum_commit + 1
        cycle_classes = sweep.finalize(total_cycles)
        result: Dict[str, object] = {
            "total_cycles": total_cycles,
            "expanded_request_count": expanded_count,
            "compressed_transaction_count": compression.count,
            "transaction_address_sha256": address_digest.hexdigest(),
            "commit_sequence_sha256": commit_digest.hexdigest(),
            "population_ids": sorted(population_ids),
            "configs": sorted(configs),
            "cycle_classes": cycle_classes,
            "same_cycle_response_slot_reuse": True,
            "detail_retained": retain_details,
            "event_sweep_diagnostics": sweep.diagnostics(),
        }
        if detailed is not None:
            result["scheduled_requests"] = [asdict(row) for row in detailed]
            result["compressed_schedule"] = M785.M777.M768.compress_scheduled_rows(
                detailed)
        return result


M768_RESULT_FIELDS = (
    "total_cycles",
    "expanded_request_count",
    "compressed_transaction_count",
    "scheduled_requests",
    "compressed_schedule",
    "transaction_address_sha256",
    "commit_sequence_sha256",
    "population_ids",
    "configs",
    "cycle_classes",
    "same_cycle_response_slot_reuse",
)


def exact_old_new_miter(requests: Sequence[Request]) -> Dict[str, object]:
    """Run the frozen O(C*R) reference only on a bounded request sequence."""
    require(0 < len(requests) <= 10000,
            "reference miter is bounded to at most 10000 requests")
    old_scheduler = M785.AddressTimedScheduler(_synthetic_resource())
    new_scheduler = StreamingAddressTimedScheduler(_synthetic_resource())
    old = old_scheduler.schedule(list(requests))
    new = new_scheduler.schedule(iter(requests), retain_details=True)
    for field in M768_RESULT_FIELDS:
        require(old[field] == new[field], "old/new miter mismatch: " + field)
    require(old_scheduler.token_ready == new_scheduler.token_ready,
            "produced-token readiness miter mismatch")
    return {
        "requests": len(requests),
        "fields": list(M768_RESULT_FIELDS),
        "produced_token_readiness_sha256": M785.canonical_sha256(
            new_scheduler.token_ready),
        "status": "PASS_EXACT_OLD_NEW_MITER",
    }


def _synthetic_resource():
    return M785.CommonResource(
        lanes=96, accumulator_bits=24, clock_ns=3.0,
        external_bytes_per_cycle=192,
        onchip_budget_bytes_macro_rounded=245760,
        macro_round_bytes=128,
        weight_bytes_logical=13824,
        psum_bytes_logical=221184,
        descriptor_control_bytes_logical=8192,
        reserved_unallocated_bytes=2560,
        weight=M785.PortSpec(8, "1R1W", 16, 4, 1, 1, 2),
        psum=M785.PortSpec(6, "1RW", 48, 2, 1, 1, 2),
        external=M785.PortSpec(1, "1RW", 192, 4, 3, 1, 2),
        compute=M785.PortSpec(1, "1RW", 288, 1, 1, 1, 1),
    )


def _request(index: int, kind: str, *, earliest: int = 0,
             dependencies: Tuple[str, ...] = (), produce: bool = True,
             transaction: Optional[str] = None) -> Request:
    port_banks = {
        "weight_read": (index % 8,), "weight_write": (index % 8,),
        "psum_read": (index % 6,), "psum_write": (index % 6,),
        "external_read": (0,), "external_write": (0,),
        "commit": (0,), "compute": (0,),
    }
    widths = {
        "weight_read": 16, "weight_write": 16,
        "psum_read": 48, "psum_write": 48,
        "external_read": 192, "external_write": 192,
        "commit": 192, "compute": 288,
    }
    banks = port_banks[kind]
    if kind.startswith("weight"):
        addresses = tuple((index % 100) * 16 for _ in banks)
    elif kind.startswith("psum"):
        addresses = tuple((index % 100) * 48 for _ in banks)
    else:
        addresses = tuple((1 << 60) + index * 192 for _ in banks)
    return Request(
        request_id="r{}".format(index),
        transaction_id=(transaction if transaction is not None
                        else "tx{}".format(index)),
        population_id="M861_SYNTHETIC",
        config="TYPED_SIGNED_K8",
        kind=kind,
        addresses=addresses,
        banks=banks,
        width_bytes=widths[kind],
        dependency_tokens=tuple(dependencies),
        produces_token=("token{}".format(index) if produce else ""),
        earliest_issue_cycle=int(earliest),
    )


def deterministic_random_dag(count: int, seed: int = 861) -> List[Request]:
    require(1 <= count <= 10000, "bounded DAG size drift")
    rng = random.Random(seed)
    kinds = (
        "weight_read", "weight_write", "psum_read", "psum_write",
        "external_read", "external_write", "compute", "commit",
    )
    rows: List[Request] = []
    for index in range(count):
        dependency = ()
        if index and rng.randrange(3) != 0:
            dependency = ("token{}".format(rng.randrange(index)),)
        rows.append(_request(
            index, kinds[rng.randrange(len(kinds))],
            earliest=rng.randrange(0, 32), dependencies=dependency,
            transaction="group{}".format(index // 4)))
    return rows


def manual_endpoint_priority_miter() -> Dict[str, object]:
    """Exercise active>dependency>weight>psum>memory>compute and E/D/I/R."""
    def row(index: int, earliest: int, dependency: int, issue: int,
            returned: int, reason: str) -> ScheduledRequest:
        return ScheduledRequest(
            request_id="manual{}".format(index), transaction_id="manual",
            population_id="M861_MANUAL", config="TYPED_SIGNED_K8",
            kind="compute", addresses=(index,), banks=(0,), width_bytes=1,
            dependency_tokens=(), earliest_issue_cycle=earliest,
            dependency_ready_cycle=dependency, issue_cycle=issue,
            return_cycle=returned, commit_cycle=returned,
            wait_reason=reason, produces_token="")

    rows = [
        # Dependency wait plus inflight covers [0,5), masking the weight
        # interval at cycle 4; issue at 3 still wins over inflight.
        row(0, 0, 3, 3, 5, "dependency_completion"),
        # The staggered reason intervals expose every lower-priority class and
        # overlap pairwise so that precedence, not just membership, is tested.
        row(1, 4, 4, 8, 8, "weight_bank"),
        row(2, 6, 6, 10, 10, "psum_bank"),
        row(3, 9, 9, 12, 12, "memory"),
        # Cycle 13 is deliberately uncovered and must fall back to compute;
        # cycle 14 is a zero-latency active endpoint.
        row(4, 14, 14, 14, 14, "none"),
    ]
    sweep = ExactCycleClassSweep()
    for item in rows:
        sweep.observe(item)
    total = max(item.commit_cycle for item in rows) + 1
    observed = sweep.finalize(total)

    # Deliberately retained frozen reference definition for this tiny miter.
    expected = {name: 0 for name in CYCLE_CLASS_ORDER}
    issue_cycles = {item.issue_cycle for item in rows}
    for cycle in range(total):
        if cycle in issue_cycles:
            expected["active_service"] += 1
            continue
        waiting = [item for item in rows
                   if item.earliest_issue_cycle <= cycle < item.issue_cycle]
        inflight = [item for item in rows
                    if item.issue_cycle <= cycle < item.return_cycle]
        reasons = [item.wait_reason for item in waiting
                   if item.wait_reason != "none"]
        dependency_wait = any(item.dependency_ready_cycle > cycle
                              for item in waiting)
        if dependency_wait or "dependency_completion" in reasons or inflight:
            expected["dependency_completion"] += 1
        elif "weight_bank" in reasons:
            expected["weight_bank"] += 1
        elif "psum_bank" in reasons:
            expected["psum_bank"] += 1
        elif "memory" in reasons:
            expected["memory"] += 1
        else:
            expected["compute"] += 1
    require(observed == expected, "manual endpoint/priority miter mismatch")
    require(all(observed[name] > 0 for name in CYCLE_CLASS_ORDER),
            "manual miter did not expose every priority class")
    return {"status": "PASS_MANUAL_E_D_I_R_PRIORITY_MITER",
            "cycle_classes": observed}


def synthetic_prefix_requests(count: int) -> Iterator[Request]:
    kinds = ("weight_read", "psum_read", "external_read", "compute",
             "weight_write", "psum_write", "external_write", "commit")
    for index in range(int(count)):
        yield _request(index, kinds[index % len(kinds)],
                       earliest=index // 16, produce=False,
                       transaction="bulk{}".format(index // 64))


def run_scale_prefixes(prefixes: Sequence[int]) -> List[Dict[str, object]]:
    output = []
    for count in prefixes:
        require(count in (1000, 10000, 100000),
                "only frozen 1K/10K/100K synthetic prefixes are authorized")
        started = time.monotonic()
        scheduler = StreamingAddressTimedScheduler(_synthetic_resource())
        result = scheduler.schedule(synthetic_prefix_requests(count))
        elapsed = time.monotonic() - started
        output.append({
            "prefix_requests": count,
            "elapsed_seconds": elapsed,
            "process_max_rss_kib": int(process_resource.getrusage(
                process_resource.RUSAGE_SELF).ru_maxrss),
            "total_cycles": result["total_cycles"],
            "compressed_transaction_count":
                result["compressed_transaction_count"],
            "cycle_classes_sha256": M785.canonical_sha256(
                result["cycle_classes"]),
            "detail_retained": result["detail_retained"],
            "event_sweep_diagnostics": result["event_sweep_diagnostics"],
        })
    return output


def real_prefix_requests(limit: int) -> Iterator[Request]:
    """Yield only a bounded prefix of the exact first M854 row identity."""
    require(1 <= limit <= 100000,
            "real prefix is bounded to at most 100K expanded requests")
    contract_path = HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"
    contract = M785.strict_json(contract_path)
    resource = M785.resource_from_contract(contract)
    del resource
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    manifest = M785.strict_json(payload_root / "manifest.json")
    records = M785.normalized_population_records(
        manifest, "M686_ZURICH_CITY_09_A_S10")
    record = records[0]
    require(int(record["module_index"]) == 0 and int(record["sample_id"]) == 0,
            "first M854 row identity drift")
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(
        HW / mapper_row["path"], mapper_row["sha256"], "m861_m672_mapper")
    m712 = contract["inputs"]["m712_oracle"]
    m722 = contract["inputs"]["m722r2_oracle"]
    storage = contract["inputs"]["m785_storage_oracle"]
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])
    rows = M785.expand_transactions(M785.iter_record_transactions(
        mapper, record, payload_root, "M686_ZURICH_CITY_09_A_S10",
        "A1_OSG", 0, oracles))
    yield from itertools.islice(rows, limit)


def run_real_prefix(limit: int, *, miter_limit: int = 0) -> Dict[str, object]:
    require(1 <= limit <= 100000, "real prefix limit drift")
    started = time.monotonic()
    scheduler = StreamingAddressTimedScheduler(
        M785.resource_from_contract(M785.strict_json(
            HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")))
    result = scheduler.schedule(real_prefix_requests(limit))
    output = {
        "identity": "M854_FIRST_D0_A1_T0_BOUNDED_PREFIX_ONLY",
        "prefix_requests": limit,
        "elapsed_seconds": time.monotonic() - started,
        "process_max_rss_kib": int(process_resource.getrusage(
            process_resource.RUSAGE_SELF).ru_maxrss),
        "summary": {key: result[key] for key in (
            "total_cycles", "expanded_request_count",
            "compressed_transaction_count", "transaction_address_sha256",
            "commit_sequence_sha256", "cycle_classes")},
        "detail_retained": result["detail_retained"],
        "production_result": False,
        "cycles_or_speedup_citable": False,
    }
    if miter_limit:
        require(miter_limit <= min(limit, 10000), "real miter bound drift")
        bounded = list(real_prefix_requests(miter_limit))
        output["old_new_miter"] = exact_old_new_miter(bounded)
    return output


def validate_source_candidate(contract_path: Path) -> Dict[str, object]:
    contract = M785.strict_json(Path(contract_path))
    require(isinstance(contract, dict) and
            contract.get("schema") == CONTRACT_SCHEMA,
            "M861 contract schema drift")
    require(contract.get("status") ==
            "SOURCE_ONLY_STREAMING_EVENT_SWEEP__FRESH_HAMMER_REQUIRED" and
            contract.get("launch_now") is False and
            contract.get("production_replay") is False,
            "M861 candidate is not source-only")
    require(_sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    identity = M785.verify_sealed_directory(M857_DIR)
    require(_sha256(M857_DIR / "review.json") == M857_REVIEW_SHA256 and
            identity["manifest_sha256"] == M857_MANIFEST_SHA256 and
            identity["outer_seal_file_sha256"] ==
            M857_OUTER_SEAL_FILE_SHA256,
            "M857 failure authority drift")
    for name, row in contract["source_identity"].items():
        path = HW / row["path"]
        require(path.is_file() and not path.is_symlink() and
                _sha256(path) == row["sha256"],
                "M861 source identity drift: " + name)
    require(contract["bounds"] == {
        "old_reference_miter_max_requests": 10000,
        "synthetic_scale_prefixes": [1000, 10000, 100000],
        "real_prefix_max_requests": 100000,
        "full_first_row": False,
        "full_population": False,
    }, "M861 bounded-work contract drift")
    return {
        "status": "PASS_M861_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
        "contract_sha256": _sha256(Path(contract_path)),
        "m857_outer_seal_file_sha256":
            identity["outer_seal_file_sha256"],
        "launch_now": False,
        "production_cycles": None,
        "production_speedup": None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source-candidate", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--scale-prefixes", action="store_true")
    parser.add_argument("--real-prefix", type=int)
    parser.add_argument("--real-miter-prefix", type=int, default=0)
    parser.add_argument("--run-production", action="store_true")
    args = parser.parse_args(argv)
    require(not args.run_production,
            "M861 source candidate refuses production replay")
    if args.self_test:
        value = {
            "manual": manual_endpoint_priority_miter(),
            "random_dag": exact_old_new_miter(
                deterministic_random_dag(512)),
            "production_replay": False,
        }
    elif args.validate_source_candidate:
        require(args.contract is not None, "M861 contract is required")
        value = validate_source_candidate(args.contract)
    elif args.scale_prefixes:
        value = {
            "schema": "m861_bounded_synthetic_scale_diagnostic_v1",
            "rows": run_scale_prefixes((1000, 10000, 100000)),
            "production_result": False,
            "cycles_or_speedup_citable": False,
        }
    elif args.real_prefix is not None:
        value = run_real_prefix(args.real_prefix,
                                miter_limit=args.real_miter_prefix)
    else:
        raise Failure("select one source-only M861 action")
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
